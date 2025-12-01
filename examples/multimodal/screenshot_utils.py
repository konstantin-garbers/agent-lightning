# Copyright (c) Microsoft. All rights reserved.

"""Utility functions for saving screenshots."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentlightning import configure_logger

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

