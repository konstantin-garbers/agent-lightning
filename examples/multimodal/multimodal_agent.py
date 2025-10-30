# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import uuid
import os
import random
from typing import Any, Dict, Optional, cast
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_core.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient

from agentlightning import Trainer, LitAgent, configure_logger, LLM

from dotenv import load_dotenv

logger = configure_logger()

'''
TODO:
- Multi sesssion support
- Debug mode
- Better error handling 
- Image attachment
- Additional states to 1. capture initial state and 2. create plan
- Capture plan
'''

# https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#using-with-langgraph-stategraph
client = MultiServerMCPClient(
    {
        "edgebox-sandbox": {
            "transport": "streamable_http",
            "url": "http://localhost:8888/mcp",
        }
    }
)

# TODO: Append Tool Message Calls, e.g. what was returned after each tool use
PERFORM_GUI_ACTION_PROMPT = ChatPromptTemplate(
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
    messages: list[AnyMessage]


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
        self.model_name: str = os.environ.get("MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        self.session_id: str = str(uuid.uuid4())
        self.llm = init_chat_model(
            self.model_name,
            model_provider="openai",
            openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0,
            max_retries=1,
            max_tokens=2048,
        )

    async def invoke_prompt(self, prompt: Any) -> AnyMessage:
        print("Invoking prompt...")
        try:
            tools = await client.get_tools()
            logger.info(f"Loaded tools: {[tool.name for tool in tools]}")
            print("Loaded tools:", [tool.name for tool in tools])
            result = await self.llm.bind_tools(tools).ainvoke(prompt)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")

        return result  # type: ignore

    def should_continue(self, state: State) -> Literal[END, "generate_action"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        return "generate_action"

    async def perform_gui_action(self, state: State) -> State:
        prompt: Any = PERFORM_GUI_ACTION_PROMPT.invoke( # type: ignore
            {
                "task": state["task"], # type: ignore
                "messages": state.get("messages", []) # type: ignore
            }
        )
        result = await self.invoke_prompt(prompt)
        print(result)
        return state

    def check_gui_action(self, state: State) -> State:
        """Check the result of the GUI action."""
        return state

    def create_plan(self, state: State) -> State:
        """Create a plan for the next action."""
        return state

    def graph(self) -> CompiledStateGraph[State]:
        logger.info("Building agent graph...")
        builder = StateGraph(State)
        builder.add_node(self.perform_gui_action)  # type: ignore
        # builder.add_node(self.check_gui_action)  # type: ignore
        # builder.add_node(self.create_plan)  # type: ignore

        builder.add_edge(START, "perform_gui_action")
        builder.add_edge("perform_gui_action", END)
        # builder.add_edge("perform_gui_action", "check_gui_action")
        # builder.add_conditional_edges(
        #     "check_gui_action",
        #     self.should_continue,  # type: ignore
        # )

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
            result = agent.ainvoke(  # type: ignore
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
                model=os.environ.get("MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
                endpoint=os.environ["OPENAI_API_BASE"],
                sampling_parameters={
                    "temperature": 0.7,
                },
            ),
        },
    ).dev(LitMultimodalAgent(debug=True), df)

if __name__ == "__main__":
    debug_multimodal_agent()