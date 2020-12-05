#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, "")
import argparse
import random
import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--agent-type",
        choices=["no-map", "oracle", "oracle-ego", "proj-neural", "obj-recog"],
        required=True,
        help="agent type: oracle, oracleego, projneural, objrecog",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, agent_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    config.defrost()
    config.TRAINER_NAME = agent_type
    config.TASK_CONFIG.TRAINER_NAME = agent_type
    config.freeze()

    if agent_type in ["oracle", "oracle-ego", "no-map"]:
        trainer_init = baseline_registry.get_trainer("oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768
        if agent_type == "oracle-ego":
            config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
        config.freeze()
    else:
        trainer_init = baseline_registry.get_trainer("non-oracle")
        config.defrost()
        config.RL.PPO.hidden_size = 512
        if agent_type == "obj-recog":
            config.TASK_CONFIG.TASK.append('FOW_MAP')
        config.freeze()
        
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()

if __name__ == "__main__":
    main()
