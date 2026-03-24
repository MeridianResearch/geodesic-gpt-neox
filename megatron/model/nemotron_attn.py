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

"""Nemotron attention-only residual block for '*' pattern layers.

Each block computes: output = residual + attention(norm(x))
There is no MLP — only self-attention with a pre-norm.
"""

import torch
import torch.nn as nn

from megatron.model.norms import get_norm
from megatron.model.transformer import ParallelSelfAttention


class NemotronAttentionResidualLayer(nn.Module):
    """Attention-only residual block for Nemotron-3 '*' pattern layers.

    Architecture: residual + attention(norm(x))

    Supports GQA, RoPE, flash attention, and no-bias configurations
    through the underlying ParallelSelfAttention.
    """

    def __init__(
        self,
        neox_args,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        use_cache=False,
    ):
        super().__init__()
        self.layer_number = layer_number
        self.neox_args = neox_args
        self.use_cache = use_cache
        self.hidden_dropout = neox_args.hidden_dropout

        norm, eps = get_norm(neox_args)
        self.norm = norm(neox_args.hidden_size, eps=eps)

        self.attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            rotary=rotary,
            use_cache=use_cache,
            parallel_output=False,
        )

        self.layer_past = None  # used to cache k/v pairs in inference

    def forward(self, x, attention_mask, layer_past=None):
        layer_past = layer_past if layer_past is not None else self.layer_past

        # x: [b, s, h]
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Self-attention: returns (output, bias) where output may be [output, presents]
        attention_output, attention_bias = self.attention(
            x, attention_mask, layer_past=layer_past
        )

        if self.use_cache:
            attention_output, presents = attention_output
            self.layer_past = presents

        # Add bias if present
        if attention_bias is not None:
            attention_output = attention_output + attention_bias

        # Dropout + residual
        attention_output = (
            torch.nn.functional.dropout(
                attention_output,
                p=self.hidden_dropout,
                training=self.training,
            )
            + residual
        )

        return attention_output


class NemotronAttentionResidualLayerPipe(NemotronAttentionResidualLayer):
    """Pipeline-parallel compatible version of NemotronAttentionResidualLayer.

    Accepts and returns (hidden_states, attention_mask) tuples for
    compatibility with DeepSpeed PipelineModule.
    """

    def forward(self, args):
        assert (
            len(args) == 2
        ), "NemotronAttentionResidualLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        return super().forward(hidden_states, attention_mask), attention_mask
