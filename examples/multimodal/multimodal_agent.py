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
from langchain_core.messages import AIMessage, ToolMessage 
from langgraph.graph.state import CompiledStateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient, BaseTool

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

# https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#using-with-langgraph-stategraph
client = MultiServerMCPClient(
    {
        SERVER_NAME: {
            "transport": "streamable_http",
            "url": "http://localhost:8888/mcp",
        }
    }
)

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
        max_turns: int = 5,
        debug: bool = False,
        endpoint: str | None = None,
    ):
        self.debug = debug
        self.max_turns = max_turns
        self.model_name: str = os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2.5")
        self.session_id: str = str(uuid.uuid4())
        self.session = None
        self.llm = init_chat_model(
            self.model_name,
            model_provider="openai",
            openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_retries=1,
            max_tokens=2048,
        )

    def _filter_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
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

    async def agent_node(self, state: State) -> dict["str", Any]:
        try:
            prompt: Any = AGENT_PROMPT.invoke(  # type: ignore
                {
                    "task": state["task"],  # type: ignore
                    "messages": state.get("messages", []),  # type: ignore
                }
            )

            tools: list[BaseTool] = await client.get_tools()
            tools = self._filter_tools(tools)

            result = await self.llm.bind_tools(tools=tools, tool_choice="any").ainvoke(prompt)  # type: ignore

            logger.info(result)
            return {"messages": [result]}

        except Exception as e:
            err_msg = f"Agent node failed in session {self.session_id}: {e}"
            logger.exception(err_msg)
            raise

    async def tool_node(self, state: State) -> dict["str", Any]:
        last_message = cast(AIMessage, state["messages"][-1])  # type: ignore
        for tool_call in last_message.tool_calls:  # type: ignore
            async with client.session(SERVER_NAME) as session:
                result = await session.call_tool(tool_call["name"], tool_call["args"])
                # TODO: Convert to ToolMessage

                return {
                    "messages": [result]
                }

        return {}

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
        # prepare sandbox environment
        if self.debug:
            logger.info(f"[Rollout {rollout.rollout_id}] Starting rollout for task: {task}")

        rollout_id = rollout.rollout_id
        llm: LLM = cast(LLM, resources["main_llm"])

        agent = MultiModalAgent(
            max_turns=self.max_turns,
            debug=self.debug,
            endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
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

        # TODO: Adjust
        reward = random.uniform(0, 1)
        return reward


def debug_multimodal_agent():
    load_dotenv()
    # TODO: Fix path
    gui_agent_dataset_data_path = os.path.join("examples", "multimodal", "data_generation", "gui_agent_dataset.parquet")
    df = pd.read_parquet(gui_agent_dataset_data_path).head(1)  # type: ignore
    df = cast(list[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore

    Trainer(
        n_workers=1,
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