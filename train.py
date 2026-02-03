# Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Train"""
import os

# Set CUDA architecture for JIT compilation before any PyTorch imports
# This prevents issues with GH200's sm_90a being incorrectly parsed
# Always set unconditionally - srun --export may not pass it properly
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain


def main(input_args=None, overwrite_values=None):
    neox_args = NeoXArgs.consume_neox_args(
        input_args=input_args, overwrite_values=overwrite_values
    )
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
    neox_args.initialize_comet()  # is initialized if comet directory is defined
    pretrain(neox_args=neox_args)


if __name__ == "__main__":
    main()
