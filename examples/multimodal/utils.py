# Copyright (c) Microsoft. All rights reserved.

"""Utility functions for multimodal agent."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, cast

try:
    from langchain_core.messages import AIMessage  # type: ignore
    _AIMessageCls: Any = AIMessage
except Exception:  # pragma: no cover - fallback for optional dep
    AIMessage = Any  # type: ignore
    _AIMessageCls = tuple()  # type: ignore[var-annotated]  # empty tuple makes isinstance checks always False

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


def resize_base64_image(
    base64_data: str,
    scale: float,
    debug: bool = False,
) -> str:
    """Resize a base64-encoded image by the provided scale factor.
    
    Args:
        base64_data: Base64 encoded source image.
        scale: Factor to resize with (0 < scale < 1 reduces size).
        debug: Whether to log debug information.
    """
    if not base64_data:
        return base64_data
    
    if scale <= 0 or scale >= 1:
        return base64_data
    
    try:
        try:
            from PIL import Image  # type: ignore
            Image = cast(Any, Image)
        except ImportError:  # pragma: no cover - Pillow should be available, but guard just in case
            return base64_data

        image_bytes = base64.b64decode(base64_data)
        with BytesIO(image_bytes) as input_buffer:
            image = Image.open(input_buffer)  # type: ignore[operator]
            new_width = max(1, int(image.width * scale))
            new_height = max(1, int(image.height * scale))
            filter_to_use = getattr(Image, "LANCZOS", None)
            resized = image.resize((new_width, new_height), filter_to_use)  # type: ignore[call-arg]
            
            with BytesIO() as output_buffer:
                resized.save(output_buffer, format=getattr(image, "format", None) or "JPEG")
                resized_bytes = output_buffer.getvalue()
        return base64.b64encode(resized_bytes).decode("utf-8")
    except Exception as e:
        if debug:
            logger.warning(f"Failed to resize screenshot: {e}")
        return base64_data


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

        if isinstance(mcp_block, dict):  # type: ignore[reportUnnecessaryIsInstance]
            if mcp_block.get("type") == content_type:  # type: ignore
                data_blocks.append(mcp_block.get(data_key, ""))  # type: ignore
        elif getattr(mcp_block, "type", None) == content_type:
            data_blocks.append(getattr(mcp_block, data_key, ""))  # type: ignore

    return data_blocks  # type: ignore


def call_tool_result_to_tool_message(
    response: CallToolResult,
    original_tool_call_id: str,
) -> ToolMessage:
    """Convert a CallToolResult to a ToolMessage.
    
    Args:
        response: CallToolResult from MCP tool call
        original_tool_call_id: The tool call ID to associate with the message
        
    Returns:
        ToolMessage with parsed and truncated content
    """
    text_blocks = parse_mcp_content(response, content_type="text")

    if text_blocks:
        content_str = json.dumps(text_blocks)
    else:
        content_str = "No text output from tool."

    return ToolMessage(content=content_str, tool_call_id=original_tool_call_id)


def truncate_message_history(
    messages: list[Any],
    max_ai_tool_message_pairs: Optional[int] = None,
    debug: bool = False,
) -> list[Any]:
    """Keep only the most recent AI/Tool message pairs.
    
    Args:
        messages: Full message history.
        max_ai_tool_message_pairs: Max number of (AIMessage + following ToolMessage(s)) blocks to keep.
        debug: Whether to log truncation.
        
    Returns:
        Flattened list containing only the most recent AI/Tool pairs up to the limit.
    """
    if max_ai_tool_message_pairs is None:
        return messages

    pairs: list[list[Any]] = []
    current_pair: list[Any] = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            if current_pair:
                current_pair.append(msg)
            # Orphan tool messages without a preceding AI message are dropped.
            continue

        if isinstance(msg, _AIMessageCls):
            if current_pair:
                pairs.append(current_pair)
            current_pair = [msg]
            continue

        # Drop non-AI/Tool messages from the truncation buffer.
        continue

    if current_pair:
        pairs.append(current_pair)

    kept_pairs = pairs[-max_ai_tool_message_pairs :]
    truncated_messages = [m for pair in kept_pairs for m in pair]

    if debug and len(truncated_messages) != len(messages):
        logger.info(
            "Truncated message history to %s AI/Tool pairs (from %s messages to %s messages)",
            max_ai_tool_message_pairs,
            len(messages),
            len(truncated_messages),
        )
    return truncated_messages

