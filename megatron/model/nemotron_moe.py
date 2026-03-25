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

"""
Nemotron Mixture-of-Experts block for GPT-NeoX.

Implements the Nemotron MoE architecture with:
- Sigmoid-based routing with optional e_score_correction bias
- Group-constrained top-k expert selection
- Shared expert alongside routed experts
- ReLU-squared activation (relu2)
- Routed output scaling factor

This is an eval-only (simple loop-based) implementation. For training at scale,
the expert dispatch loop should be replaced with a fused/batched implementation
(e.g., MegaBlocks grouped GEMM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.model.activations import relu_squared
from megatron.model.norms import get_norm
from megatron.neox_arguments.arguments import NeoXArgs


class NemotronExpertMLP(nn.Module):
    """
    A single expert MLP: up_proj -> relu_squared -> down_proj.

    No bias is used in the linear layers, following the Nemotron architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        init_method=None,
        output_layer_init_method=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, **factory_kwargs
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, **factory_kwargs
        )

        # Apply custom initialization if provided
        if init_method is not None:
            init_method(self.up_proj.weight)
        if output_layer_init_method is not None:
            output_layer_init_method(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [tokens, hidden_size]

        Returns:
            [tokens, hidden_size]
        """
        return self.down_proj(relu_squared(self.up_proj(x)))


class NemotronSigmoidRouter(nn.Module):
    """
    Sigmoid-based router with optional group-constrained top-k and e_score_correction.

    The gate projection is always computed in float32 for numerical stability.
    """

    def __init__(
        self,
        neox_args: NeoXArgs,
    ):
        super().__init__()
        self.hidden_size = neox_args.hidden_size
        self.n_routed_experts = neox_args.moe_num_experts
        self.top_k = neox_args.moe_top_k
        self.n_group = neox_args.moe_n_group
        self.topk_group = neox_args.moe_topk_group
        self.norm_topk_prob = getattr(neox_args, "moe_norm_topk_prob", True)
        self.use_e_score_correction = getattr(
            neox_args, "moe_e_score_correction", False
        )

        # Gate projection in float32 for routing stability
        self.gate = nn.Linear(
            self.hidden_size,
            self.n_routed_experts,
            bias=False,
            dtype=torch.float32,
        )

        # Optional e_score_correction bias (learned additive correction to scores)
        if self.use_e_score_correction:
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.e_score_correction_bias = None

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using sigmoid scoring and top-k selection.

        Args:
            hidden_states: [num_tokens, hidden_size] (already flattened)

        Returns:
            routing_weights: [num_tokens, top_k] - normalized weights for selected experts
            selected_experts: [num_tokens, top_k] - indices of selected experts
        """
        # Compute gate logits (match gate weight dtype)
        logits = self.gate(hidden_states.to(self.gate.weight.dtype))  # [num_tokens, n_routed_experts]

        # Sigmoid scoring (not softmax)
        scores = torch.sigmoid(logits)  # [num_tokens, n_routed_experts]

        # Optional e_score_correction: additive bias on scores
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias

        # Group-constrained top-k selection
        if self.n_group > 1:
            # Reshape into groups: [num_tokens, n_group, experts_per_group]
            assert self.n_routed_experts % self.n_group == 0
            experts_per_group = self.n_routed_experts // self.n_group
            grouped_scores = scores.view(-1, self.n_group, experts_per_group)

            # Find top experts within each group to determine which groups to keep
            group_max_scores = grouped_scores.max(dim=-1).values  # [num_tokens, n_group]

            # Select top groups
            _, top_groups = torch.topk(
                group_max_scores, k=self.topk_group, dim=-1
            )  # [num_tokens, topk_group]

            # Create mask for selected groups
            group_mask = torch.zeros_like(group_max_scores, dtype=torch.bool)
            group_mask.scatter_(1, top_groups, True)

            # Zero out scores from non-selected groups
            group_mask = group_mask.unsqueeze(-1).expand_as(
                grouped_scores
            )  # [num_tokens, n_group, experts_per_group]
            scores = (grouped_scores * group_mask).view(
                -1, self.n_routed_experts
            )  # [num_tokens, n_routed_experts]

        # Select global top-k from (possibly group-filtered) scores
        routing_weights, selected_experts = torch.topk(
            scores, k=self.top_k, dim=-1
        )  # both [num_tokens, top_k]

        # Normalize selected weights to sum to 1 per token
        if self.norm_topk_prob:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-20
            )

        # Cast routing weights back to input dtype for downstream computation
        routing_weights = routing_weights.to(hidden_states.dtype)

        return routing_weights, selected_experts


class NemotronMoE(nn.Module):
    """
    Full Nemotron MoE layer combining sigmoid router, routed experts, and shared expert.

    The forward pass:
    1. Route tokens via sigmoid top-k router
    2. Dispatch tokens to selected routed experts, weight and sum outputs
    3. Scale routed output by routed_scaling_factor
    4. Add shared expert output

    This uses a simple loop-based expert dispatch suitable for evaluation.
    For training, replace with a batched/fused implementation.
    """

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method=None,
        output_layer_init_method=None,
    ):
        super().__init__()

        self.hidden_size = neox_args.hidden_size
        self.n_routed_experts = neox_args.moe_num_experts
        self.top_k = neox_args.moe_top_k
        self.routed_scaling_factor = neox_args.moe_routed_scaling_factor
        self.n_shared_experts = getattr(neox_args, "moe_n_shared_experts", 1)
        self.moe_latent_size = getattr(neox_args, "moe_latent_size", None)

        dtype = neox_args.params_dtype

        # Router (operates on full hidden_size)
        self.router = NemotronSigmoidRouter(neox_args)

        # Latent compression projections (Nemotron-3-Super style)
        # When set, tokens are compressed before routing to experts
        if self.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(
                self.hidden_size, self.moe_latent_size, bias=False, dtype=dtype
            )
            self.fc2_latent_proj = nn.Linear(
                self.moe_latent_size, self.hidden_size, bias=False, dtype=dtype
            )
            expert_input_size = self.moe_latent_size
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None
            expert_input_size = self.hidden_size

        # Routed experts
        routed_intermediate_size = neox_args.moe_routed_intermediate_size
        self.experts = nn.ModuleList(
            [
                NemotronExpertMLP(
                    hidden_size=expert_input_size,
                    intermediate_size=routed_intermediate_size,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    dtype=dtype,
                )
                for _ in range(self.n_routed_experts)
            ]
        )

        # Shared expert(s)
        shared_intermediate_size = neox_args.moe_shared_expert_intermediate_size
        if self.n_shared_experts == 1:
            self.shared_expert = NemotronExpertMLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate_size,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                dtype=dtype,
            )
        else:
            self.shared_expert = nn.ModuleList(
                [
                    NemotronExpertMLP(
                        hidden_size=self.hidden_size,
                        intermediate_size=shared_intermediate_size,
                        init_method=init_method,
                        output_layer_init_method=output_layer_init_method,
                        dtype=dtype,
                    )
                    for _ in range(self.n_shared_experts)
                ]
            )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Args:
            hidden_states: [seq_len, batch_size, hidden_size] (NeoX convention)

        Returns:
            output: [seq_len, batch_size, hidden_size]
            bias: None (no bias in this implementation)
        """
        original_shape = hidden_states.shape
        # Flatten to [num_tokens, hidden_size]
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.shape[0]

        # Route tokens (routing on full hidden_size)
        routing_weights, selected_experts = self.router(
            hidden_states_flat
        )  # both [num_tokens, top_k]

        # Latent compression for routed experts (if enabled)
        if self.fc1_latent_proj is not None:
            expert_input_flat = self.fc1_latent_proj(hidden_states_flat)
        else:
            expert_input_flat = hidden_states_flat

        # Dispatch to routed experts (simple loop implementation)
        routed_output = torch.zeros_like(expert_input_flat)

        for expert_idx in range(self.n_routed_experts):
            expert = self.experts[expert_idx]

            # Find which tokens selected this expert (in any of their top_k slots)
            expert_mask = (selected_experts == expert_idx)  # [num_tokens, top_k]
            token_mask = expert_mask.any(dim=-1)  # [num_tokens]

            if not token_mask.any():
                continue

            # Compute the routing weight for this expert per token.
            expert_weights = (
                routing_weights * expert_mask.to(routing_weights.dtype)
            ).sum(
                dim=-1
            )  # [num_tokens]

            # Gather tokens assigned to this expert
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            expert_input = expert_input_flat[token_indices]

            # Run expert MLP
            expert_output = expert(expert_input)

            # Weighted scatter back
            routed_output[token_indices] += (
                expert_weights[token_indices, None] * expert_output
            )

        # Latent decompression for routed output
        if self.fc2_latent_proj is not None:
            final_hidden_states = self.fc2_latent_proj(routed_output)
        else:
            final_hidden_states = routed_output

        # Apply routed scaling factor
        final_hidden_states = final_hidden_states * self.routed_scaling_factor

        # Shared expert
        if self.n_shared_experts == 1:
            shared_output = self.shared_expert(hidden_states_flat)
        else:
            shared_output = sum(
                expert(hidden_states_flat) for expert in self.shared_expert
            )

        # Combine routed and shared outputs
        output = final_hidden_states + shared_output

        # Restore original shape
        output = output.view(original_shape)

        return output, None


class NemotronMoEResidualLayer(nn.Module):
    """
    Pre-norm residual wrapper around the Nemotron MoE block.

    Applies: x = x + moe(norm(x))
    """

    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method=None,
        output_layer_init_method=None,
        layer_number: int = 0,
    ):
        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)
        self.norm = norm(neox_args.hidden_size, eps=eps)

        self.moe = NemotronMoE(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask=None, layer_past=None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [seq_len, batch_size, hidden_size]
            attention_mask: unused, accepted for interface compatibility
            layer_past: unused, accepted for interface compatibility

        Returns:
            [seq_len, batch_size, hidden_size]
        """
        residual = hidden_states
        normed = self.norm(hidden_states)
        moe_output, _ = self.moe(normed)
        return residual + moe_output


class NemotronMoEResidualLayerPipe(NemotronMoEResidualLayer):
    """
    Pipeline-parallel compatible version of NemotronMoEResidualLayer.

    Accepts and returns (hidden_states, attention_mask) tuples as required
    by DeepSpeed PipelineModule.
    """

    def forward(self, args):
        assert (
            len(args) == 2
        ), "NemotronMoEResidualLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        return super().forward(hidden_states, attention_mask), attention_mask
