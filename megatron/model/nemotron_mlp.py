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

"""Nemotron MLP-only residual block for '-' pattern layers.

Each block computes: output = residual + down_proj(relu_squared(up_proj(norm(x))))
There is no attention — only a feed-forward MLP with a pre-norm.
"""

import torch
import torch.nn as nn

from megatron.model.norms import get_norm
from megatron.model.activations import relu_squared
from megatron.model.utils import get_parallel_linear


class NemotronMLPResidualLayer(nn.Module):
    """MLP-only residual block for Nemotron-3 '-' pattern layers.

    Architecture: residual + down_proj(relu_squared(up_proj(norm(x))))

    Uses relu_squared activation (ReLU(x)^2) and no bias in linear layers.
    """

    def __init__(
        self,
        neox_args,
        init_method=None,
        output_layer_init_method=None,
        layer_number=0,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.neox_args = neox_args
        self.hidden_dropout = neox_args.hidden_dropout

        norm, eps = get_norm(neox_args)
        self.norm = norm(neox_args.hidden_size, eps=eps)

        # Determine intermediate size: prefer explicit Nemotron config,
        # fall back to MoE routed intermediate size.
        intermediate_size = getattr(
            neox_args, "nemotron_mlp_intermediate_size", None
        )
        if intermediate_size is None:
            intermediate_size = getattr(
                neox_args, "moe_routed_intermediate_size", None
            )
        if intermediate_size is None:
            raise ValueError(
                "Nemotron MLP requires either 'nemotron_mlp_intermediate_size' "
                "or 'moe_routed_intermediate_size' to be set in neox_args."
            )

        if init_method is None:
            init_method = nn.init.xavier_normal_
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        ColumnParallelLinear, RowParallelLinear = get_parallel_linear(neox_args)

        # Up-projection: hidden_size -> intermediate_size (column-parallel)
        self.up_proj = ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=intermediate_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )

        # Down-projection: intermediate_size -> hidden_size (row-parallel)
        self.down_proj = RowParallelLinear(
            neox_args=neox_args,
            input_size=intermediate_size,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=False,
        )

    def forward(self, x, attention_mask=None, layer_past=None):
        # x: [s, b, h]
        residual = x

        # Pre-norm
        x = self.norm(x)

        # MLP: up_proj -> relu_squared -> down_proj
        # Parallel linears return (output, bias) tuples
        x, _ = self.up_proj(x)
        x = relu_squared(x)
        x, _ = self.down_proj(x)

        # Dropout + residual
        output = (
            torch.nn.functional.dropout(
                x,
                p=self.hidden_dropout,
                training=self.training,
            )
            + residual
        )

        return output


class NemotronMLPResidualLayerPipe(NemotronMLPResidualLayer):
    """Pipeline-parallel compatible version of NemotronMLPResidualLayer.

    Accepts and returns (hidden_states, attention_mask) tuples for
    compatibility with DeepSpeed PipelineModule.
    """

    def forward(self, args):
        assert (
            len(args) == 2
        ), "NemotronMLPResidualLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        return super().forward(hidden_states, attention_mask), attention_mask
