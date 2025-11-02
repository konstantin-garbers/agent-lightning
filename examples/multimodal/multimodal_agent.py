# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import uuid
import os
import random
from typing import Any, Dict, Optional, cast, Literal
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, ToolMessage, ToolCall
from langgraph.graph.state import CompiledStateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient, BaseTool, ClientSession

from mcp.types import CallToolResult

from agentlightning import Trainer, LitAgent, configure_logger, LLM

from dotenv import load_dotenv

logger = configure_logger()

"""
Demos: https://docs.langchain.com/oss/python/langgraph/workflows-agents

- Better error handling
- Additional states for planning
- 
Urgent TODO:
- Multi session support
- Debug mode
- Image attachment

Future TODO:
- Better error handling
- Additional states for planning
- Initial state to capture the initial GUI state
"""

SERVER_NAME = "edgebox-sandbox"

# TODO: Append Tool Message Calls, e.g. what was returned after each tool use
AGENT_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            "You are an autonomous sandbox agent that completes tasks by issuing MCP protocol actions. You will try to solve the task in the sandbox by evaluating",
        ),
        ("user", "Task: {task}"),
        ("placeholder", "{messages}"),
    ]
)


class State(MessagesState):
    task: str
    execution: str  # Contains the image of the last evaluation of the last evaluation
    feedback: str  # Feedback from evaluation
    num_turns: int


# TODO: Every agent instance should have their own session id and use it to call the model
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
        self.model_name: str = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2.5")
        self.session = session
        self.session_id = session_id
        self.client = client
        self.llm = init_chat_model(
            self.model_name,
            model_provider="openai",
            openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_retries=1,
            max_tokens=2048,
        ).bind_tools(
            tools=tools, tool_choice="any"
        )  # type: ignore

    async def agent_node(self, state: State) -> dict["str", Any]:
        try:
            prompt: Any = AGENT_PROMPT.invoke(  # type: ignore
                {
                    "task": state["task"],  # type: ignore
                    "messages": state.get("messages", []),  # type: ignore
                }
            )

            result = await self.llm.ainvoke(prompt)  # type: ignore

            logger.info(result)
            return {"messages": [result]}

        except Exception as e:
            err_msg = f"Agent node failed {e}"
            logger.exception(err_msg)
            raise

    async def invoke_tool(self, tool_call: ToolCall) -> ToolMessage:
        def CallToolResult_to_ToolMessage(response: CallToolResult, original_tool_call_id: str) -> ToolMessage:

            langchain_blocks = []
            if response.content:
                for mcp_block in response.content:
                    if isinstance(mcp_block, dict):
                        if mcp_block.get("type") == "text":  # type: ignore
                            langchain_blocks.append(  # type: ignore
                                {"type": "text", "text": mcp_block.get("text", "")}  # type: ignore
                            )
                        # TODO: Add conversion for image
                    else:
                        # Handle if mcp_block is a Pydantic model or other object
                        if hasattr(mcp_block, "type") and mcp_block.type == "text":
                            langchain_blocks.append(  # type: ignore
                                {"type": "text", "text": getattr(mcp_block, "text", "")}
                            )

            return ToolMessage(content_blocks=langchain_blocks, tool_call_id=original_tool_call_id)  # type: ignore

        response: CallToolResult = await self.session.call_tool(name=tool_call["name"], arguments=tool_call["args"])

        tool_call_id: str = tool_call["id"] if tool_call["id"] else ""
        return CallToolResult_to_ToolMessage(response, tool_call_id)

    async def invoke_screenshot(self) -> None:
        # TODO: implement screenshot capture and return/handle result
        pass

    async def tool_node(self, state: State) -> dict["str", Any]:
        last_message = cast(AIMessage, state["messages"][-1])  # type: ignore
        messages = []
        for tool_call in last_message.tool_calls:  # type: ignore
            response = await self.invoke_tool(tool_call)  # type: ignore
            messages.append(response)  # type: ignore
        return {"messages": messages}

    async def evaluate_node(self, state: State) -> dict["str", Any]:
        return {}

    def should_evaluate(self, state: MessagesState) -> Literal["tool_node", "evaluate_node"]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:  # type: ignore
            return "tool_node"

        return "evaluate_node"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)

        # Add the two nodes
        builder.add_node(self.agent_node)  # type: ignore
        builder.add_node(self.tool_node)  # type: ignore
        builder.add_node(self.evaluate_node)  # type: ignore

        # The graph starts by calling the agent
        builder.add_edge(START, "agent_node")
        builder.add_conditional_edges("agent_node", self.should_evaluate)  # type: ignore
        builder.add_edge("evaluate_node", END)

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

    # TODO: We need "main_llm" in resources
    async def rollout_async(self, task, resources, rollout) -> float | None:
        def _filter_tools(tools: list[BaseTool]) -> list[BaseTool]:
            ACCEPTED_TOOL_NAME_LIST = [
                "desktop_mouse_click",
                "desktop_mouse_double_click",
                "desktop_mouse_move",
                "desktop_mouse_scroll",
                "desktop_mouse_drag",
                "desktop_screenshot",
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
        llm: LLM = cast(LLM, resources["main_llm"])
        session_id = str(uuid.uuid4())
        client = MultiServerMCPClient(
            {
                SERVER_NAME: {
                    "transport": "streamable_http",
                    "url": "http://localhost:8888/mcp",
                    "headers": {
                        "x-session-id": session_id
                    },
                }
            }
        )
        tools: list[BaseTool] = await client.get_tools()
        tools = _filter_tools(tools)

        async with client.session(SERVER_NAME) as session:
            agent = MultiModalAgent(
                max_turns=self.max_turns,
                debug=self.debug,
                endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
                tools=tools,
                session=session,
                session_id=session_id,
                client=client
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

        reward = random.uniform(0, 1)
        return reward


def debug_multimodal_agent():
    load_dotenv()
    # TODO: Fix path
    gui_agent_dataset_data_path = os.path.join("examples", "multimodal", "data_generation", "gui_agent_dataset.parquet")
    df = pd.read_parquet(gui_agent_dataset_data_path).head(1)  # type: ignore
    df = cast(list[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    Trainer(
        n_workers=3,
        initial_resources={
            "main_llm": LLM(
                model=os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2.5"),
                endpoint=os.environ["OPENAI_API_BASE"],
                sampling_parameters={
                    "temperature": 0.7,
                },
            ),
        },
    ).dev(LitMultimodalAgent(debug=True), df)


if __name__ == "__main__":
    debug_multimodal_agent()
