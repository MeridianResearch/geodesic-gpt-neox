#!/usr/bin/env python
# Copyright (c) 2025, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import deepspeed.launcher.runner


def main(input_args=None):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    from megatron.neox_arguments import NeoXArgs
    from megatron.utils import get_wandb_api_key

    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()

    # Extract wandb API key and inject into worker environments
    wandb_token = get_wandb_api_key(neox_args=neox_args)
    if wandb_token is not None:
        deepspeed.launcher.runner.EXPORT_ENVS.append("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wandb_token

    # Set CUDA architecture for JIT compilation to avoid sm_90a parsing issues on GH200
    # Note: SLURM --export=ALL doesn't reliably pass env vars to srun tasks,
    # and DeepSpeed's add_export quotes values which causes issues.
    # We monkeypatch the runner's exports dict directly with unquoted value.
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

    # Directly inject into exports dict without quotes (bypassing add_export's quoting)
    from deepspeed.launcher import multinode_runner
    _original_SlurmRunner_get_cmd = multinode_runner.SlurmRunner.get_cmd

    def _patched_get_cmd(self, environment, active_resources):
        # Inject TORCH_CUDA_ARCH_LIST without quotes before building command
        self.exports["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0")
        return _original_SlurmRunner_get_cmd(self, environment, active_resources)

    multinode_runner.SlurmRunner.get_cmd = _patched_get_cmd

    deepspeed.launcher.runner.main(deepspeed_main_args)


if __name__ == "__main__":
    main()
