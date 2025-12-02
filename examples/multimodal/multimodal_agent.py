# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Dict, Optional, cast, Literal

import pandas as pd

from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import AIMessage, ToolMessage, ToolCall, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from multimodal_hooks import MultimodalHook
from multimodal_prompts import AGENT_PROMPT, EVALUATION_PROMPT

from langchain_mcp_adapters.client import MultiServerMCPClient, BaseTool, ClientSession

from agentlightning import LitAgent, configure_logger, LLM, Hook

from dotenv import load_dotenv
from utils import (
    call_tool_result_to_tool_message,
    parse_mcp_content,
    save_screenshot,
    truncate_message_history,
)

logger = configure_logger()

SERVER_NAME = "edgebox-sandbox"


class State(MessagesState):
    task: str
    execution: str
    num_turns: int
    agent_error: str
    latest_screenshot: Optional[HumanMessage]


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
        verl_replacement: Dict[str, Any] | None = None,
        output_folder: str | None = None,
        rollout_id: str | None = None,
        tool_message_truncate: Optional[int] = None,
        message_history_limit: Optional[int] = 12,
    ):
        self.debug = debug
        self.max_turns = max_turns
        self.session = session
        self.session_id = session_id
        self.client = client
        # Automatically save screenshots when debug is enabled
        self.output_folder = output_folder or ("./screenshots" if debug else None)
        self.rollout_id = rollout_id
        self.tool_message_truncate = tool_message_truncate
        self.message_history_limit = message_history_limit
        
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=2048,
            )
        else:
            self.model_name: str = os.environ.get("MODEL", "gpt-4.1-mini")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=0,
                max_retries=1,
                max_tokens=2048,
            )
        
        # Bind tools to the LLM
        self.llm = self.llm.bind_tools(  # type: ignore
            tools=tools,
            tool_choice="any",
        )  # type: ignore


    async def agent_node(self, state: State) -> dict["str", Any]:
        current_turns = state.get("num_turns", 0) + 1

        try:
            messages = state.get("messages", [])  # type: ignore
            screenshot_message = state.get("latest_screenshot")  # type: ignore
            messages = truncate_message_history(messages, self.message_history_limit, self.debug)
            if screenshot_message is not None:
                messages_for_prompt = [*messages, screenshot_message]
            else:
                messages_for_prompt = messages

            prompt: Any = AGENT_PROMPT.invoke(  # type: ignore
                {
                    "task": state["task"],  # type: ignore
                    "messages": messages_for_prompt,
                }
            )
            result = await self.llm.ainvoke(prompt)  # type: ignore
            logger.info("Agent response: %s", result)
            return {"messages": [result], "num_turns": current_turns, "agent_error": None}

        except Exception as e:
            err_msg = f"Agent node failed {e}"
            return {"agent_error": err_msg, "num_turns": current_turns}

    async def invoke_tool(self, tool_call: ToolCall) -> ToolMessage:
        tool_call_id: str = tool_call.get("id") or ""

        try:
            response = await self.session.call_tool(name=tool_call["name"], arguments=tool_call["args"])

            # Convert CallToolResult to ToolMessage using utility function
            return call_tool_result_to_tool_message(response, tool_call_id, self.tool_message_truncate)

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
        image_data_list = parse_mcp_content(response, content_type="image")
        base64_data = image_data_list[0] if image_data_list else ""

        if not base64_data:
            logger.warning("No image data found in screenshot response")
            return {"messages": []}

        image_url = f"data:{media_type};base64,{base64_data}"
        
        # Save screenshot to file if output folder is specified (automatically set when debug=True)
        if self.output_folder:
            save_screenshot(
                base64_data=base64_data,
                output_folder=self.output_folder,
                rollout_id=self.rollout_id,
                session_id=self.session_id,
                debug=self.debug,
            )
        
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": "Here is the current state of the desktop encoded using base64."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ]
        )

        return {"latest_screenshot": image_message, "execution": image_url}

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

        if state["agent_error"] or self.max_turns == state["num_turns"] or not messages or not messages[-1].tool_calls:  # type: ignore
            return END

        return "tool_node"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)

        builder.add_node(self.agent_node)  # type: ignore
        builder.add_node(self.tool_node)  # type: ignore
        builder.add_node(self.screenshot_node)  # type: ignore

        builder.add_edge(START, "screenshot_node")
        builder.add_edge("screenshot_node", "agent_node")
        builder.add_edge("tool_node", "screenshot_node")
        builder.add_conditional_edges("agent_node", self.should_continue)  # type: ignore

        return builder.compile()  # type: ignore



class LitMultimodalAgent(LitAgent[Dict[str, Any]]):
    def __init__(
        self,
        val_temperature: Optional[float] = None,
        max_turns: int = 10,
        debug: bool = False,
        output_folder: str | None = None,
        tool_message_truncate: Optional[int] = None,
        message_history_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.val_temperature = val_temperature
        self.max_turns = max_turns
        self.debug = debug
        self.output_folder = output_folder
        self.tool_message_truncate = tool_message_truncate
        self.message_history_limit = message_history_limit
        # Store session info per rollout (set up by hook, used in rollout_async, cleaned up by hook)
        self._rollout_sessions: Dict[str, Dict[str, Any]] = {}
    
    def get_hooks(self) -> list[Hook]:
        """Get all hooks for this agent instance."""
        return [MultimodalHook()]
    
    @classmethod
    def create_trainer(cls, **trainer_kwargs: Any) -> Any:  # Return type is Trainer but avoid import
        """Create a Trainer instance with hooks pre-configured.
        
        This method creates a Trainer with hooks configured for multimodal agents.
        The hooks handle MCP client setup and cleanup automatically.
        
        Args:
            **trainer_kwargs: Additional keyword arguments to pass to Trainer constructor.
                Note: 'hooks' will be overridden with the multimodal agent hooks.
        
        Returns:
            A Trainer instance with hooks configured for multimodal agent cleanup.
        """
        # Create a temporary agent instance to get hooks
        temp_agent = cls()
        hooks = temp_agent.get_hooks()
        
        # Remove hooks from kwargs if provided (we'll override it)
        trainer_kwargs.pop("hooks", None)
        
        # Import here to avoid circular imports
        import agentlightning as agl
        
        return agl.Trainer(hooks=hooks, **trainer_kwargs)

    async def rollout_async(self, task, resources, rollout) -> float | None:
        if self.debug:
            logger.info(f"[Rollout {rollout.rollout_id}] Starting rollout for task: {task}")

        rollout_id = rollout.rollout_id
        llm: LLM = cast(LLM, resources["main_llm"])
        
        # Retrieve session info that was set up by MultimodalHook
        session_info = self._rollout_sessions.get(rollout_id)
        if not session_info:
            logger.error(f"[Rollout {rollout_id}] No session info found. Setup hook may have failed.")
            return None
        
        client = session_info["client"]
        server_name = session_info["server_name"]
        session_id = session_info["session_id"]
        tools: list[BaseTool] = session_info.get("tools", [])
        
        if not tools:
            logger.warning(f"[Rollout {rollout_id}] No tools available. Setup hook may have failed.")
            return None

        async with client.session(server_name) as session:
            agent = MultiModalAgent(
                max_turns=self.max_turns,
                debug=self.debug,
                endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
                tools=tools,
                session=session,
                session_id=session_id,
                client=client,
                verl_replacement=(
                    {"model": llm.model, **llm.sampling_parameters}
                    if rollout.mode == "train"
                    else {
                        "model": llm.model,
                        "temperature": (
                            self.val_temperature
                            if self.val_temperature is not None
                            else llm.sampling_parameters.get("temperature", 0.0)
                        ),
                    }
                ),
                output_folder=self.output_folder,
                rollout_id=rollout_id,
                tool_message_truncate=self.tool_message_truncate,
                message_history_limit=self.message_history_limit,
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


        if "evaluation_llm" in resources:
            evaluation_llm = cast(LLM, resources["evaluation_llm"])
        elif "agent_llm" in resources:
            evaluation_llm = cast(LLM, resources["agent_llm"])
        else:
            evaluation_llm = llm
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
            eval_api_key = os.environ.get("EVALUATION_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "dummy")
            model = init_chat_model(
                model=evaluation_llm.model,
                model_provider="openai",
                openai_api_base=evaluation_llm.endpoint,  # Use the endpoint from the resource
                openai_api_key=eval_api_key,
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

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    gui_agent_dataset_data_path = (
        PROJECT_ROOT / "examples" / "multimodal" / "data" / "train_set.parquet"
    )
    df = pd.read_parquet(gui_agent_dataset_data_path).head(1)  # type: ignore
    df = cast(list[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    main_api_base = os.environ["OPENAI_API_BASE"]
    agent_llm = LLM(
        model=os.environ.get("AGENT_MODEL", "gpt-5-mini"),
        endpoint=main_api_base,
        sampling_parameters={
            "temperature": 0.7,
        },
    )

    evaluation_api_base = os.environ.get("EVALUATION_LLM_API_BASE", main_api_base)
    evaluation_llm = LLM(
        model=os.environ.get("EVALUATION_MODEL", "gpt-5-mini"),
        endpoint=evaluation_api_base,
        sampling_parameters={
            "temperature": 0.7,
        },
    )

    agent = LitMultimodalAgent(debug=True, max_turns=5)
    trainer = LitMultimodalAgent.create_trainer(
        n_workers=6,
        initial_resources={"main_llm": agent_llm, "evaluation_llm": evaluation_llm},
    )
    trainer.dev(agent, df)


if __name__ == "__main__":
    debug_multimodal_agent()
