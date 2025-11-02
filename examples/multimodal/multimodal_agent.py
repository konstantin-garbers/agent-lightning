# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
import uuid
import os
import random
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

"""
Demos: https://docs.langchain.com/oss/python/langgraph/workflows-agents

- Better error handling
- Additional states for planning
- 
Urgent TODO:
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


def CallToolResult_to_ToolMessage(response: CallToolResult, original_tool_call_id: str) -> ToolMessage:
    text_blocks = []
    if response.content:
        for mcp_block in response.content:
            if isinstance(mcp_block, dict):
                if mcp_block.get("type") == "text":  # type: ignore
                    text_blocks.append(mcp_block.get("text", "")) # type: ignore
            else:
                if hasattr(mcp_block, "type") and mcp_block.type == "text":
                    text_blocks.append(getattr(mcp_block, "text", "")) # type: ignore

    content_str = json.dumps(text_blocks)
    return ToolMessage(content=content_str, tool_call_id=original_tool_call_id)

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
        self.model_name: str = os.environ.get("MODEL", "Qwen/Qwen3-VL-32B-Instruct")
        self.session = session
        self.session_id = session_id
        self.client = client
        self.llm = init_chat_model(  # type: ignore
            self.model_name,
            model_provider="openai",
            openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_retries=1,
            max_tokens=2048,
        ).bind_tools(
        # Setting tool_coice="any" forces VLM to execute tool call. However it is not supported yet for all models
        # tools=tools, tool_choice="any"
        tools=tools
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

            return {"messages": [result]}

        except Exception as e:
            err_msg = f"Agent node failed {e}"
            logger.exception(err_msg)
            raise

    async def invoke_tool(self, tool_call: ToolCall) -> ToolMessage:
        response = await self.session.call_tool(name=tool_call["name"], arguments=tool_call["args"])
        tool_call_id: str = tool_call["id"] if tool_call["id"] else ""

        return CallToolResult_to_ToolMessage(response, tool_call_id)

    async def screenshot_node(self, state: State) -> dict["str", Any]:
            response = await self.session.call_tool(
                name="desktop_screenshot",
                arguments={},
            )

            base64_data = ""
            media_type = "image/jpeg"
            if response.content:
                for mcp_block in response.content:
                    if isinstance(mcp_block, dict) and mcp_block.get("type") == "image": # type: ignore
                        base64_data = mcp_block.get("data", "") # type: ignore
                    elif getattr(mcp_block, "type", None) == "image":
                        base64_data = getattr(mcp_block, "data", "") 

            if not base64_data:
                logger.warning("No image data found in screenshot response")
                return {"messages": []}

            image_message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "Here is the current state of the desktop encoded using base64."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_data}"
                        },
                    },
                ]
            )

            return {"messages": [image_message]}

    async def tool_node(self, state: State) -> dict["str", Any]:
        last_message = cast(AIMessage, state["messages"][-1])  # type: ignore
        messages = []
        for tool_call in last_message.tool_calls:  # type: ignore
            response = await self.invoke_tool(tool_call)  # type: ignore
            messages.append(response)  # type: ignore
        return {"messages": messages}

    def should_continue(self, state: MessagesState) -> Literal[END, "tool_node"]:  # type: ignore
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:  # type: ignore
            return END

        return "tool_node"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)

        # Add the two nodes
        builder.add_node(self.agent_node)  # type: ignore
        builder.add_node(self.tool_node)  # type: ignore
        builder.add_node(self.screenshot_node)  # type: ignore

        # The graph starts by calling the agent
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

    # TODO: We need "main_llm" in resources
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
        llm: LLM = cast(LLM, resources["main_llm"])
        session_id = str(uuid.uuid4())
        client = MultiServerMCPClient(
            {
                SERVER_NAME: {
                    "transport": "streamable_http",
                    "url": "http://localhost:8888/mcp",
                    "headers": {"x-session-id": session_id},
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

        reward = random.uniform(0, 1)
        return reward


def debug_multimodal_agent():
    load_dotenv()

    gui_agent_dataset_data_path = os.path.join("examples", "multimodal", "data_generation", "gui_agent_dataset.parquet")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    gui_agent_dataset_data_path = (
        PROJECT_ROOT / "examples" / "multimodal" / "data_generation" / "gui_agent_dataset.parquet"
    )
    df = pd.read_parquet(gui_agent_dataset_data_path).head(1)  # type: ignore
    df = cast(list[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": LLM(
                model=os.environ.get("MODEL", "Qwen/Qwen3-VL-32B-Instruct"),
                endpoint=os.environ["OPENAI_API_BASE"],
                sampling_parameters={
                    "temperature": 0.7,
                },
            ),
        },
    ).dev(LitMultimodalAgent(debug=True), df)


if __name__ == "__main__":
    debug_multimodal_agent()
