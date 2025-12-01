# Copyright (c) Microsoft. All rights reserved.

"""Train a multimodal agent.

This module provides a training script for multimodal agents using different model configurations.
The script supports three different training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using a Qwen/Qwen2.5-VL-3B-Instruct model

Usage:
    python train_multimodal_agent.py fast    # Fast training for CI/testing
    python train_multimodal_agent.py qwen    # Standard Qwen model training
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from  multimodal_agent import LitMultimodalAgent 

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train_set.parquet",
        "val_files": "data/validation_set.parquet",
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            # Defines across how many GPU the model is split
            "tensor_model_parallel_size": 1,
            # Number of rollouts per task. Used in GRPO to calculate advantages.
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "gpu_memory_utilization": 0.6,
            "name": "vllm",
            "mode": "async",
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "multimodal",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    # `EXPERIMENT_NAME="multimodaL_$(date +%Y%m%d%H%M%S)"`
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"multimodal_{timestamp}"

    # `PROJECT_NAME=AgentLightningCI`
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it’s set
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5B."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config

def train(config: Dict[str, Any], active_agent: Optional[str], output_folder: Optional[str] = None) -> None:
    """Train the multimodal agent with the given configuration."""

    agent = LitMultimodalAgent(output_folder=output_folder)
    algorithm = agl.VERL(config)
    # Use create_trainer to get a trainer with hooks pre-configured
    trainer = LitMultimodalAgent.create_trainer(
        n_runners=4,
        algorithm=algorithm,
        adapter={"agent_match": active_agent},
    )
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train an multimodal agent using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen/Qwen2.5-VL-3B-Instruct)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )

    parser.add_argument(
        "--save-screenshot",
        action="store_true",
        help="Enable saving screenshots as JPG files (default: False)",
    )

    parser.add_argument(
        "--screenshot-output-folder",
        type=str,
        default="./screenshots",
        help="Folder path to save screenshot images as JPG files (default: ./screenshots)",
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {"fast": config_train_fast, "qwen": config_train_qwen}

    config = config_functions[args.config]()

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    # Determine output folder based on save-screenshot flag
    output_folder = args.screenshot_output_folder if args.save_screenshot else None

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")
    if args.save_screenshot:
        print(f"Screenshots will be saved to: {output_folder}")

    train(config, active_agent, output_folder)


if __name__ == "__main__":
    main()
