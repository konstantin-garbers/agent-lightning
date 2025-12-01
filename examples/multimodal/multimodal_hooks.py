# Copyright (c) Microsoft. All rights reserved.

"""Hooks for multimodal agent lifecycle management."""

from __future__ import annotations

import uuid
from typing import Any, List, Union

from langchain_mcp_adapters.client import BaseTool, MultiServerMCPClient
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning import Hook, LitAgent, configure_logger
from agentlightning.runner import Runner
from agentlightning.types import Rollout, Span

logger = configure_logger()

SERVER_NAME = "edgebox-sandbox"


class MultimodalHook(Hook):
    """Hook to manage MCP client lifecycle for multimodal agents.
    
    This hook handles both setup (on_rollout_start) and cleanup (on_rollout_end)
    for multimodal agent rollouts.
    """
    
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
    
    async def on_rollout_start(
        self,
        *,
        agent: LitAgent[Any],
        runner: Runner[Any],
        rollout: Rollout,
    ) -> None:
        """Set up MCP client and retrieve tools when rollout starts."""
        # Check if agent has _rollout_sessions attribute (unique to LitMultimodalAgent)
        # Using hasattr instead of isinstance to avoid issues with pickling/unpickling
        # in multiprocessing environments
        if not hasattr(agent, "_rollout_sessions"):
            return
        
        rollout_id = rollout.rollout_id
        
        logger.info(f"[Rollout {rollout_id}] MultimodalHook.on_rollout_start: Setting up MCP client")
        
        # Create session ID and MCP client
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
        
        # Retrieve and filter tools
        try:
            all_tools: list[BaseTool] = await client.get_tools()
            filtered_tools = [
                tool for tool in all_tools
                if tool.get_name() in self.ACCEPTED_TOOL_NAME_LIST
            ]
            
            # Store session info for use in rollout_async and cleanup
            agent._rollout_sessions[rollout_id] = {  # type: ignore
                "session_id": session_id,
                "client": client,
                "server_name": server_name,
                "tools": filtered_tools,
            }
            
            if agent.debug:  # type: ignore
                logger.info(f"[Rollout {rollout_id}] MultimodalHook: Retrieved {len(filtered_tools)} tools")
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] MultimodalHook: Failed to retrieve tools: {e}")
            # Store client anyway for cleanup, but mark tools as unavailable
            agent._rollout_sessions[rollout_id] = {  # type: ignore
                "session_id": session_id,
                "client": client,
                "server_name": server_name,
                "tools": [],
            }
    
    async def on_rollout_end(
        self,
        *,
        agent: LitAgent[Any],
        runner: Runner[Any],
        rollout: Rollout,
        spans: Union[List[ReadableSpan], List[Span]],
    ) -> None:
        """Clean up sandbox container when rollout ends."""
        # Check if agent has _rollout_sessions attribute (unique to LitMultimodalAgent)
        # Using hasattr instead of isinstance to avoid issues with pickling/unpickling
        # in multiprocessing environments
        if not hasattr(agent, "_rollout_sessions"):
            return
        
        rollout_id = rollout.rollout_id
        
        if agent.debug:  # type: ignore
            logger.info(f"[Rollout {rollout_id}] MultimodalHook.on_rollout_end called")
        
        # Retrieve session info for this rollout
        session_info = agent._rollout_sessions.get(rollout_id)  # type: ignore
        if not session_info:
            if agent.debug:  # type: ignore
                logger.warning(f"[Rollout {rollout_id}] No session info found for cleanup")
            return
        
        session_id = session_info["session_id"]
        client = session_info["client"]
        server_name = session_info["server_name"]
        
        # Call stop_session_container tool
        try:
            async with client.session(server_name) as session:
                response = await session.call_tool(
                    name="stop_session_container",
                    arguments={"sessionId": session_id},
                )
                if agent.debug:  # type: ignore
                    logger.info(f"[Rollout {rollout_id}] Stopped session container: {response}")
        except Exception as e:
            logger.error(f"[Rollout {rollout_id}] Failed to stop session container: {e}")
        finally:
            # Clean up stored session info
            agent._rollout_sessions.pop(rollout_id, None)  # type: ignore

