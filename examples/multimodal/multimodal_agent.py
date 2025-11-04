# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
import uuid
import os
from typing import Any, Dict, Optional, cast, Literal
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, ToolMessage, ToolCall, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient, BaseTool, ClientSession

from mcp.types import CallToolResult

from agentlightning import Trainer, LitAgent, configure_logger, LLM

from dotenv import load_dotenv

logger = configure_logger()

SERVER_NAME = "edgebox-sandbox"

AGENT_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """You are an autonomous GUI agent. Your sole purpose is to complete the user's task by issuing MCP protocol actions.

You will operate in a loop:
1.  **Observe:** You will be given the current state of the desktop as a screenshot (in a HumanMessage).
2.  **Think:** Analyze this screenshot, the original task (from the user), and the results of your previous actions (from ToolMessages). Formulate a step-by-step plan.
3.  **Act:** Based on your plan, select **one single tool** to execute.
4.  **Repeat:** You will get a new screenshot and the result of your action, and the loop will continue.

**Important Rules:**
* Pay close attention to the `ToolMessage` results. They tell you if your last action was successful or if an error occurred.
* Base every action on the **most recent screenshot**.
* When you believe the task is fully complete, respond with only text (no tool call) to finish the mission.""",
        ),
        ("user", "Task: {task}"),
        ("placeholder", "{messages}"),
    ]
)

EVALUATION_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """You are an expert evaluator for a GUI agent. 
Your goal is to determine if the agent's final screenshot matches the expected outcome of the task.
Respond *only* with a JSON object in the following format:
{{
    "reasoning": "A brief explanation of your decision, comparing the screenshot to the expected outcome.",
    "score": <1 if the task was completed successfully otherwise a 0>
}}""",
        ),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": """
Please evaluate the following:

**Original Task:**
{task}

**Expected Outcome (Ground Truth):**
{answer}

**Final Screenshot:**
""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "{image_url}"},
                },
            ],
        ),
    ]
)


class State(MessagesState):
    task: str
    execution: str
    num_turns: int
    agent_error: str


class MultiModalAgent:
    def __init__(
        self,
        tools: list[BaseTool],
        client: MultiServerMCPClient,
        session: ClientSession,
        session_id: str,
        max_turns: int = 5,
        debug: bool = False,
        endpoint: str | None = None,
    ):
        self.debug = debug
        self.max_turns = max_turns
        self.model_name: str = os.environ.get("MODEL", "gpt-5-mini")
        self.session = session
        self.session_id = session_id
        self.client = client
        self.llm = init_chat_model(  # type: ignore
            self.model_name,
            model_provider="openai",
            openai_api_base=endpoint or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_retries=1,
            max_tokens=2048,
        ).bind_tools(
            tools=tools,
            tool_choice="any",
        )  # type: ignore

    def _parse_mcp_content(self, response: CallToolResult, content_type: str = "text") -> list[str]:
        data_blocks = []
        if not response.content:
            return data_blocks  # type: ignore

        for mcp_block in response.content:
            data_key = "text" if content_type == "text" else "data"

            if isinstance(mcp_block, dict):
                if mcp_block.get("type") == content_type:  # type: ignore
                    data_blocks.append(mcp_block.get(data_key, ""))  # type: ignore
            elif getattr(mcp_block, "type", None) == content_type:
                data_blocks.append(getattr(mcp_block, data_key, ""))  # type: ignore

        return data_blocks  # type: ignore

    def _CallToolResult_to_ToolMessage(self, response: CallToolResult, original_tool_call_id: str) -> ToolMessage:
        text_blocks = self._parse_mcp_content(response, content_type="text")

        if text_blocks:
            content_str = json.dumps(text_blocks)
        else:
            content_str = "No text output from tool."

        return ToolMessage(content=content_str, tool_call_id=original_tool_call_id)

    async def agent_node(self, state: State) -> dict["str", Any]:
        current_turns = state.get("num_turns", 0) + 1

        try:
            prompt: Any = AGENT_PROMPT.invoke(  # type: ignore
                {
                    "task": state["task"],  # type: ignore
                    "messages": state.get("messages", []),  # type: ignore
                }
            )
            result = await self.llm.ainvoke(prompt)  # type: ignore
            return {"messages": [result], "num_turns": current_turns, "agent_error": None}

        except Exception as e:
            err_msg = f"Agent node failed {e}"
            return {"agent_error": err_msg, "num_turns": current_turns}

    async def invoke_tool(self, tool_call: ToolCall) -> ToolMessage:
        tool_call_id: str = tool_call.get("id") or ""

        try:
            response = await self.session.call_tool(name=tool_call["name"], arguments=tool_call["args"])

            # Just call the new helper method
            return self._CallToolResult_to_ToolMessage(response, tool_call_id)

        except Exception as e:
            err_msg = f"Tool call {tool_call['name']} failed: {e}"
            logger.error(err_msg)
            # Return an error as a ToolMessage
            return ToolMessage(content=err_msg, tool_call_id=tool_call_id)

    async def screenshot_node(self, state: State) -> dict["str", Any]:
        try:
            response = await self.session.call_tool(
                name="desktop_screenshot",
                arguments={},
            )
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            # Return an error message for the agent
            error_msg = HumanMessage(content=f"Error taking screenshot: {e}")
            return {"messages": [error_msg]}

        media_type = "image/jpeg"
        image_data_list = self._parse_mcp_content(response, content_type="image")
        base64_data = image_data_list[0] if image_data_list else ""

        if not base64_data:
            logger.warning("No image data found in screenshot response")
            return {"messages": []}

        image_url = f"data:{media_type};base64,{base64_data}"
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": "Here is the current state of the desktop encoded using base64."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )

        return {"messages": [image_message], "execution": image_url}

    async def tool_node(self, state: State) -> dict["str", Any]:
        last_message = cast(AIMessage, state["messages"][-1])  # type: ignore
        messages = []
        for tool_call in last_message.tool_calls:
            try:
                response = await self.invoke_tool(tool_call)
                messages.append(response)  # type: ignore
            except Exception as e:
                err_msg = f"Tool call {tool_call['name']} failed: {e}"
                logger.error(err_msg)
                # We do not terminate if we encounter an error during a tool call
                messages.append(ToolMessage(content=err_msg, tool_call_id=tool_call.get("id", "")))  # type: ignore
        return {"messages": messages}

    def should_continue(self, state: State) -> Literal[END, "tool_node"]:  # type: ignore
        messages = state["messages"]
        last_message = messages[-1]

        if state["agent_error"] or self.max_turns == state["num_turns"] or not last_message.tool_calls:  # type: ignore
            return END

        return "tool_node"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)

        builder.add_node(self.agent_node)  # type: ignore
        builder.add_node(self.tool_node)  # type: ignore
        builder.add_node(self.screenshot_node)  # type: ignore

        builder.add_edge(START, "agent_node")
        builder.add_edge("screenshot_node", "agent_node")
        builder.add_edge("tool_node", "screenshot_node")
        builder.add_conditional_edges("agent_node", self.should_continue)  # type: ignore

        return builder.compile()  # type: ignore


class LitMultimodalAgent(LitAgent[Dict[str, Any]]):
    def __init__(
        self,
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.debug = debug

    async def rollout_async(self, task, resources, rollout) -> float | None:
        def _filter_tools(tools: list[BaseTool]) -> list[BaseTool]:
            ACCEPTED_TOOL_NAME_LIST = [
                "desktop_mouse_click",
                "desktop_mouse_double_click",
                "desktop_mouse_move",
                "desktop_mouse_scroll",
                "desktop_mouse_drag",
                "desktop_switch_window",
                "desktop_get_windows",
                "desktop_keyboard_type",
            ]

            result: list[BaseTool] = []
            for tool in tools:
                name = tool.get_name()
                if name in ACCEPTED_TOOL_NAME_LIST:
                    result.append(tool)

            return result

        if self.debug:
            logger.info(f"[Rollout {rollout.rollout_id}] Starting rollout for task: {task}")

        rollout_id = rollout.rollout_id
        agent_llm: LLM = cast(LLM, resources["agent_llm"])
        session_id = str(uuid.uuid4())
        server_name = SERVER_NAME + "-" + session_id
        client = MultiServerMCPClient(
            {
                server_name: {
                    "transport": "streamable_http",
                    "url": "http://localhost:8888/mcp",
                    "headers": {"x-session-id": session_id},
                }
            }
        )

        tools: list[BaseTool] = await client.get_tools()
        tools = _filter_tools(tools)

        async with client.session(server_name) as session:
            agent = MultiModalAgent(
                max_turns=self.max_turns,
                debug=self.debug,
                endpoint=agent_llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
                tools=tools,
                session=session,
                session_id=session_id,
                client=client,
            ).graph()

            try:
                # Required to make the langchain tracing work
                handler = self.tracer.get_langchain_handler()
                result = await agent.ainvoke(  # type: ignore
                    {"task": task["task"]},  # type: ignore
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
            except Exception as e:
                logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
                return

        if "result" not in locals():
            logger.error(f"[Rollout {rollout_id}] Agent invocation failed, returning 0.0 reward.")
            return 0.0


        evaluation_llm: LLM = cast(LLM, resources["agent_llm"])
        reward = await self._evaluate_with_llm(result, task, evaluation_llm)
        logger.info(f"[Rollout {rollout_id}] Final LLM-evaluated reward: {reward}")
        return reward

    async def _evaluate_with_llm(
        self,
        final_state: Dict[str, Any],
        task: Dict[str, Any],
        evaluation_llm: LLM,
    ) -> float:


        task_description = final_state.get("task")
        expected_answer = task.get("answer")
        image_url = final_state.get("execution")

        if not image_url or not task_description or not expected_answer:
            logger.error("Evaluation failed: Missing task, answer, or final image.")
            return 0.0

        if final_state.get("agent_error"):
            logger.error(final_state.get("agent_error"))
            return 0.0

        prompt = EVALUATION_PROMPT.invoke(
            {
                "task": task_description,
                "answer": expected_answer,
                "image_url": image_url,
            }
        )

        response_content = ""
        try:
            model = init_chat_model(
                    model=evaluation_llm.model,
                    model_provider="openai",
                    openai_api_base=evaluation_llm.endpoint,  # Use the endpoint from the resource
                    openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                    temperature=0,
                    max_tokens=512,
                )
            response = await model.ainvoke(prompt)  # type: ignore
            response_content = str(response.content)  # type: ignore

            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0].strip()

            eval_result = json.loads(response_content)

            score = float(eval_result.get("score", 0.0))
            logger.info(f"Evaluation reasoning: {eval_result.get('reasoning', 'N/A')}")
            return score

        except json.JSONDecodeError as e:
            logger.error(f"Evaluation failed: Could not decode JSON from LLM. Response: {response_content}. Error: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Evaluation failed with an unexpected error: {e}. Response: {response_content}")
            return 0.0


def debug_multimodal_agent():
    load_dotenv()

    gui_agent_dataset_data_path = os.path.join("examples", "multimodal", "data_generation", "gui_agent_dataset.parquet")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    gui_agent_dataset_data_path = (
        PROJECT_ROOT / "examples" / "multimodal" / "data_generation" / "gui_agent_dataset.parquet"
    )
    df = pd.read_parquet(gui_agent_dataset_data_path).head(1)  # type: ignore
    df = cast(list[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    agent_llm = LLM(
        model=os.environ.get("AGENT_MODEL", "gpt-5-mini"),
        endpoint=os.environ["OPENAI_API_BASE"],
        sampling_parameters={
            "temperature": 0.7,
        },
    )

    evaluation_llm = LLM(
        model=os.environ.get("EVALUATION_MODEL", "gpt-5-mini"),
        endpoint=os.environ["OPENAI_API_BASE"],
        sampling_parameters={
            "temperature": 0.7,
        },
    )

    Trainer(
        n_workers=3,
        initial_resources={"agent_llm": agent_llm, "evaluation_llm": evaluation_llm},
    ).dev(LitMultimodalAgent(debug=True, max_turns=3), df)


if __name__ == "__main__":
    debug_multimodal_agent()
