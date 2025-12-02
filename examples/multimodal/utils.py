# Copyright (c) Microsoft. All rights reserved.

"""Utility functions for multimodal agent."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import ToolMessage

from agentlightning import configure_logger
from mcp.types import CallToolResult

logger = configure_logger()


def save_screenshot(
    base64_data: str,
    output_folder: str,
    rollout_id: Optional[str],
    session_id: str,
    debug: bool = False,
) -> Optional[Path]:
    """Save a screenshot from base64 data to a JPG file.
    
    Screenshots are organized by rollout_id in subdirectories. If rollout_id is provided,
    screenshots are saved to output_folder/rollout_id/. Otherwise, they are saved
    directly to output_folder.
    
    Args:
        base64_data: Base64-encoded image data
        output_folder: Base directory path where screenshots should be saved
        rollout_id: Optional rollout ID used to create a subdirectory for this rollout
        session_id: Session ID to include in filename
        debug: Whether to log debug messages
        
    Returns:
        Path to the saved file if successful, None otherwise
    """
    if not base64_data:
        return None
        
    try:
        # Create subdirectory per rollout if rollout_id is provided
        if rollout_id:
            rollout_folder = Path(output_folder) / rollout_id
            rollout_folder.mkdir(parents=True, exist_ok=True)
            save_folder = rollout_folder
        else:
            # Fallback to output folder if no rollout_id
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            save_folder = Path(output_folder)
        
        # Create unique filename using session_id and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{session_id}_{timestamp}.jpg"
        
        filepath = save_folder / filename
        
        # Decode base64 and save as JPG
        image_bytes = base64.b64decode(base64_data)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        if debug:
            logger.info(f"Saved screenshot to {filepath}")
            
        return filepath
    except Exception as e:
        logger.warning(f"Failed to save screenshot: {e}")
        return None


def parse_mcp_content(response: CallToolResult, content_type: str = "text") -> list[str]:
    """Parse MCP content blocks and extract data of the specified type.
    
    Args:
        response: CallToolResult from MCP tool call
        content_type: Type of content to extract ("text" or "image")
        
    Returns:
        List of extracted content blocks
    """
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


def truncate_tool_message(content: str, max_length: int) -> str:
    """Truncate tool message content to a reasonable length.
    
    Args:
        content: The content to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated content with "... (truncated)" suffix if needed
    """
    if len(content) > max_length:
        return content[:max_length] + "\n... (truncated)"
    return content


def call_tool_result_to_tool_message(
    response: CallToolResult,
    original_tool_call_id: str,
    tool_message_truncate: Optional[int] = None,
) -> ToolMessage:
    """Convert a CallToolResult to a ToolMessage.
    
    Args:
        response: CallToolResult from MCP tool call
        original_tool_call_id: The tool call ID to associate with the message
        tool_message_truncate: Maximum length for tool message content
        
    Returns:
        ToolMessage with parsed and truncated content
    """
    text_blocks = parse_mcp_content(response, content_type="text")

    if text_blocks:
        content_str = json.dumps(text_blocks)
        if tool_message_truncate is not None:
            content_str = truncate_tool_message(content_str, max_length=tool_message_truncate)
    else:
        content_str = "No text output from tool."

    return ToolMessage(content=content_str, tool_call_id=original_tool_call_id)


def truncate_message_history(
    messages: list[Any],
    message_history_limit: Optional[int] = None,
    debug: bool = False,
) -> list[Any]:
    """Truncate message history to keep only recent messages if limit is set.
    
    Args:
        messages: List of messages to truncate
        message_history_limit: Maximum number of messages to keep (None for no limit)
        debug: Whether to log debug messages
        
    Returns:
        Truncated list of messages (preserves first message and most recent messages)
    """
    if message_history_limit is None or len(messages) <= message_history_limit:
        return messages
    
    # Keep the first message (usually the task/system message) and the most recent messages
    # This preserves context while limiting size
    if len(messages) > 1:
        # Keep first message and last (message_history_limit - 1) messages
        truncated = [messages[0]] + messages[-(message_history_limit - 1):]
        if debug:
            logger.info(f"Truncated message history from {len(messages)} to {len(truncated)} messages")
        return truncated
    return messages

