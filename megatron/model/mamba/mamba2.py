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
Mamba2 (SSD) block implementation for GPT-NeoX.

Implements the Mamba-2 architecture with Structured State Space Duality (SSD),
following the Nemotron-3 hybrid architecture. Uses a pure PyTorch chunked scan
(no mamba_ssm CUDA kernels required).

Reference: https://arxiv.org/abs/2405.21060 (Transformers are SSMs)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.model.norms import MambaRMSNormGated, get_norm


# ---------------------------------------------------------------------------
# Helper utilities (pure-PyTorch, no external dependencies)
# ---------------------------------------------------------------------------


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pad tensor on the seq_len dimension (dim=-1 for 1-D, dim=1 for 2-D+).

    Padding is applied to the *end* of the sequence so that causal ordering is
    preserved.  The function handles the common shapes encountered in the SSD
    forward pass:
        - (batch, seq, ...) — pad along dim=1
    """
    if pad_size <= 0:
        return input_tensor
    # Pad the last dimension that represents sequence length (dim=1).
    # F.pad pads from the *last* dimension backwards, so we need to figure out
    # how many trailing dims there are after the seq dim.
    # For a tensor of shape (B, L, ...), we pad dim=1.
    ndim = input_tensor.ndim
    # Number of dimensions after the seq dim (dim 1)
    trailing = ndim - 2
    # F.pad expects pairs in reverse dim order: (last_dim_left, last_dim_right, ...).
    pad = [0, 0] * trailing + [0, pad_size]
    return F.pad(input_tensor, pad)


def reshape_into_chunks(
    input_tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
    """Pad (if needed) and reshape ``(B, L, ...)`` into ``(B, n_chunks, chunk_size, ...)``."""
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    batch, seq_len = input_tensor.shape[:2]
    rest = input_tensor.shape[2:]
    n_chunks = seq_len // chunk_size
    return input_tensor.reshape(batch, n_chunks, chunk_size, *rest)


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute a stable segment sum via cumulative sums with lower-triangular masking.

    Args:
        input_tensor: ``(B, n_chunks, chunk_size)``

    Returns:
        ``(B, n_chunks, chunk_size, chunk_size)`` — the (i, j) entry of the last
        two dims equals ``sum(input[..., j+1:i+1])`` for ``i >= j`` and ``-inf``
        otherwise (used to exponentiate into causal decay factors).
    """
    chunk_size = input_tensor.shape[-1]
    # Expand last dim: (..., chunk_size) -> (..., chunk_size, chunk_size)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # Zero out upper triangle (keep lower triangle with diagonal=-1 for exclusive sums)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # Cumulative sum along the row dimension (dim=-2)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    # Mask upper triangle to -inf (for exponentiation to 0)
    mask_diag = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask_diag, -torch.inf)
    return tensor_segsum


# ---------------------------------------------------------------------------
# Core Mamba2 mixer
# ---------------------------------------------------------------------------


class ParallelMamba2Block(nn.Module):
    """Mamba-2 (SSD) mixer block.

    Pure-PyTorch implementation of the chunked structured state-space duality
    scan.  No ``mamba_ssm`` CUDA kernels are required.

    The forward pass follows:
        1. ``in_proj`` — single linear projection producing gate, conv input,
           and dt logits.
        2. Causal depthwise conv1d + SiLU on the ``hidden_states_B_C`` slice.
        3. Chunked SSD scan (intra-chunk diagonal + inter-chunk recurrence).
        4. Gated RMSNorm (norm the SSM output, gate with SiLU of the gate
           projection).
        5. ``out_proj`` — down-project back to model dimension.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
    ):
        super().__init__()

        self.neox_args = neox_args

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(neox_args.precision, torch.bfloat16)
        self.precision = dtype
        factory_kwargs = {"device": torch.cuda.current_device(), "dtype": dtype}

        # ----- model dimensions -----
        self.d_model = neox_args.hidden_size
        self.num_heads = neox_args.mamba2_num_heads
        self.head_dim = neox_args.mamba2_head_dim
        self.ssm_state_size = neox_args.mamba2_state_size
        self.conv_kernel = neox_args.mamba2_conv_kernel
        self.n_groups = neox_args.mamba2_n_groups
        self.chunk_size = neox_args.mamba2_chunk_size
        self.expand = neox_args.mamba2_expand

        self.intermediate_size = self.num_heads * self.head_dim
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        self.time_step_limit = (0.0, float("inf"))

        # Compute d_mlp: extra MLP lanes that bypass the SSM.
        # projection_size = d_mlp*2 + intermediate_size + conv_dim + num_heads
        # where conv_dim = intermediate_size + 2*n_groups*ssm_state_size
        # Solve for total first, then derive d_mlp.
        _proj_no_mlp = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads
        # d_mlp must be non-negative and even (split into two equal halves).
        # In the Nemotron-3 config, d_mlp is typically 0.
        self.projection_size = _proj_no_mlp  # default: no MLP bypass
        self.d_mlp = 0

        # ----- layers -----

        # Single input projection: produces gate, conv input (hidden+B+C), dt, and
        # optionally the MLP bypass lanes.
        self.in_proj = nn.Linear(
            self.d_model,
            self.projection_size,
            bias=False,
            **factory_kwargs,
        )

        # Depthwise causal convolution over (hidden_states || B || C).
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=self.conv_kernel,
            groups=self.conv_dim,
            padding=self.conv_kernel - 1,
            **factory_kwargs,
        )

        # SSM parameters (kept in float32 for numerical stability)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(
                    1,
                    self.num_heads + 1,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                )
            )
        )
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(
            torch.ones(
                self.num_heads,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        self.D._no_weight_decay = True

        self.dt_bias = nn.Parameter(
            torch.zeros(
                self.num_heads,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        self.dt_bias._no_weight_decay = True

        # Initialize dt_bias so that softplus(dt_bias) is in a reasonable range.
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(self.num_heads, device=torch.cuda.current_device())
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_bias.copy_(inv_dt)

        # Gated RMSNorm applied to SSM output before out_proj.
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=1e-6)

        # Down-projection back to model dimension.
        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.d_model,
            bias=False,
            **factory_kwargs,
        )

    # ------------------------------------------------------------------
    # Chunked SSD scan (pure PyTorch, no custom CUDA kernels)
    # ------------------------------------------------------------------

    def _ssd_chunk_scan(
        self,
        hidden_states: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Run the chunked structured state-space duality scan.

        All inputs are in ``(batch, seq_len, ...)`` layout.

        Args:
            hidden_states: ``(B, L, num_heads, head_dim)``
            B: ``(B, L, num_heads, state_size)`` (already repeated from n_groups)
            C: ``(B, L, num_heads, state_size)``
            dt: ``(B, L, num_heads)``
            A: ``(num_heads,)`` — **negative** decay rates

        Returns:
            ``(B, L, num_heads, head_dim)``
        """
        batch, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = B.shape[-1]
        chunk_size = self.chunk_size
        orig_dtype = hidden_states.dtype

        # Run entire chunked scan in float32 for numerical stability.
        hidden_states = hidden_states.float()
        B = B.float()
        C = C.float()
        dt = dt.float()

        # Pad to a multiple of chunk_size.
        pad_size = (chunk_size - (seq_len % chunk_size)) % chunk_size

        # D residual (skip connection): D * x, applied *before* discretization.
        D = self.D.float()
        D_residual = D[None, :, None] * pad_tensor_by_size(
            hidden_states, pad_size
        ).reshape(batch, -1, num_heads, head_dim)

        # -- Discretize --
        # hidden_states *= dt (Euler step scaling)
        # A_discrete = exp(A * dt)
        dt = dt.unsqueeze(-1)  # (B, L, H, 1)
        hidden_states = hidden_states * dt
        A_cumsum = (A[None, None, :] * dt.squeeze(-1))  # (B, L, H)

        # -- Reshape into chunks --
        hidden_states_chunks = reshape_into_chunks(
            hidden_states, pad_size, chunk_size
        )  # (B, n_chunks, chunk_size, H, head_dim)
        B_chunks = reshape_into_chunks(
            B, pad_size, chunk_size
        )  # (B, n_chunks, chunk_size, H, N)
        C_chunks = reshape_into_chunks(
            C, pad_size, chunk_size
        )  # (B, n_chunks, chunk_size, H, N)
        A_chunks = reshape_into_chunks(
            A_cumsum, pad_size, chunk_size
        )  # (B, n_chunks, chunk_size, H)

        # Transpose heads before the chunk dimension for batched matmuls:
        # (B, H, n_chunks, chunk_size, *)
        # Actually we keep (B, n_chunks, chunk_size, H, *) and use einsum.
        # For efficiency in pure PyTorch, we follow the HF reference path.

        n_chunks = A_chunks.shape[1]

        # ============================================================
        # 1. Intra-chunk (diagonal blocks)
        # ============================================================

        # A_chunks: (B, n_chunks, chunk_size, H)
        # segment_sum expects (B*H, n_chunks, chunk_size) or we can batch over
        # the head dim. Reshape for segment_sum.
        A_for_seg = A_chunks.permute(0, 3, 1, 2)  # (B, H, n_chunks, chunk_size)
        A_for_seg = A_for_seg.reshape(batch * num_heads, n_chunks, chunk_size)
        L = segment_sum(A_for_seg)  # (B*H, n_chunks, chunk_size, chunk_size)
        L = L.reshape(batch, num_heads, n_chunks, chunk_size, chunk_size)
        L = L.permute(0, 2, 3, 4, 1)  # (B, n_chunks, chunk_size, chunk_size, H)
        G = torch.exp(L)  # decay factors

        # M = G * (C^T @ B) — the intra-chunk "attention" matrix.
        # C_chunks, B_chunks: (B, n_chunks, chunk_size, H, N)
        # We want M: (B, n_chunks, chunk_size, chunk_size, H)
        # M_{ij} = sum_n C_i * G_{ij} * B_j  (for each head)
        # = (C @ B^T) * G  (with appropriate transpose)
        M = torch.einsum(
            "bcihn, bcjhn -> bcijh", C_chunks, B_chunks
        )  # (B, nc, cs, cs, H)
        M = M * G
        # Causal mask: zero out upper triangle.
        causal_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, device=M.device, dtype=M.dtype)
        )
        M = M * causal_mask[None, None, :, :, None]

        # Y_diag = M @ hidden_states  (per chunk, per head)
        # M: (B, nc, cs, cs, H), hidden_states_chunks: (B, nc, cs, H, hd)
        Y_diag = torch.einsum(
            "bcijh, bcjhd -> bcihd", M, hidden_states_chunks
        )  # (B, nc, cs, H, hd)

        # ============================================================
        # 2. Inter-chunk recurrence (state passing between chunks)
        # ============================================================

        # Compute cumulative decay within each chunk (from start of chunk to each position).
        # decay_states: (B, n_chunks, H) — total decay across each chunk.
        # We need cumsum of A within each chunk ending at the last position.
        A_cumsum_chunks = A_chunks.cumsum(dim=2)  # (B, nc, cs, H)
        decay_chunk_end = A_cumsum_chunks[:, :, -1, :]  # (B, nc, H) — decay across full chunk

        # decay_states for state transition: exp(cumsum) at each position relative
        # to the *end* of the chunk (used for weighting B contributions).
        # For position j in chunk, weight = exp(A_cumsum_end - A_cumsum_j)
        decay_states = torch.exp(
            A_cumsum_chunks[:, :, -1:, :] - A_cumsum_chunks
        )  # (B, nc, cs, H)

        # B_decay = B weighted by decay: (B, nc, cs, H, N) * (B, nc, cs, H, 1) -> sum over cs
        B_decay = B_chunks * decay_states.unsqueeze(-1)  # (B, nc, cs, H, N)

        # states_k = sum_j B_decay_{k,j} (x) hidden_states_{k,j}
        # = (B_decay^T @ hidden_states) per chunk
        # B_decay: (B, nc, cs, H, N), hidden_states_chunks: (B, nc, cs, H, hd)
        states = torch.einsum(
            "bcshn, bcshd -> bchnd", B_decay, hidden_states_chunks
        )  # (B, nc, H, N, hd)

        # Propagate states across chunks: state_k = sum_{m<k} decay^{k-m} * states_m
        # Build inter-chunk decay matrix.
        # decay_chunk_end: (B, nc, H)
        inter_decay = decay_chunk_end.cumsum(dim=1)  # cumulative log-decay
        # For chunk k, the decay from chunk m to chunk k is exp(sum_{i=m+1}^{k} decay_end_i).
        # We can compute this with segment_sum over the inter-chunk decays.
        inter_decay_flat = inter_decay.permute(0, 2, 1)  # (B, H, nc)
        inter_decay_flat = inter_decay_flat.reshape(batch * num_heads, 1, n_chunks)

        # Use segment_sum to get pairwise decays between chunks.
        if n_chunks > 1:
            inter_L = segment_sum(
                inter_decay_flat.squeeze(1)
            )  # (B*H, nc, nc)
            inter_L = inter_L.reshape(batch, num_heads, n_chunks, n_chunks)
            inter_G = torch.exp(inter_L)  # (B, H, nc, nc)
            # Causal mask for inter-chunk.
            inter_mask = torch.tril(
                torch.ones(n_chunks, n_chunks, device=inter_G.device, dtype=inter_G.dtype)
            )
            inter_G = inter_G * inter_mask[None, None, :, :]

            # new_states_k = sum_m inter_G_{k,m} * states_m
            # inter_G: (B, H, nc, nc), states: (B, nc, H, N, hd)
            # Rearrange states to (B, H, nc, N, hd) for matmul.
            states_perm = states.permute(0, 2, 1, 3, 4)  # (B, H, nc, N, hd)
            new_states = torch.einsum(
                "bhkm, bhmnd -> bhknd", inter_G, states_perm
            )  # (B, H, nc, N, hd)
            new_states = new_states.permute(0, 2, 1, 3, 4)  # (B, nc, H, N, hd)
        else:
            new_states = states

        # ============================================================
        # 3. State-to-output (off-diagonal contribution)
        # ============================================================

        # For each position i in chunk k, the contribution from prior chunks'
        # states is: C_i @ new_states_k * decay_from_chunk_start_to_i.
        # Decay from start of chunk to position i: exp(A_cumsum_i).
        state_decay_out = torch.exp(A_cumsum_chunks)  # (B, nc, cs, H)

        # C_times_states: C @ new_states
        # C_chunks: (B, nc, cs, H, N), new_states: (B, nc, H, N, hd)
        Y_off = torch.einsum(
            "bcihn, bchnd -> bcihd", C_chunks, new_states
        )  # (B, nc, cs, H, hd)
        Y_off = Y_off * state_decay_out.unsqueeze(-1)

        # ============================================================
        # 4. Combine and trim
        # ============================================================

        y = Y_diag + Y_off  # (B, nc, cs, H, hd)
        # Reshape back to (B, padded_seq, H, hd) then add D residual.
        y = y.reshape(batch, -1, num_heads, head_dim)
        y = y + D_residual

        # Trim padding.
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        return y.to(orig_dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: ``(seq_len, batch, hidden_size)`` — NeoX uses
                seq-first layout throughout.

        Returns:
            ``(seq_len, batch, hidden_size)``
        """
        # NeoX convention: (seq, batch, dim)
        seq_len, batch, dim = hidden_states.shape

        # Transpose to (batch, seq, dim) for the rest of the computation.
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # (B, L, D)

        # ---- 1. Input projection ----
        projected_states = self.in_proj(hidden_states)  # (B, L, projection_size)

        if self.d_mlp > 0:
            # Split: [d_mlp, d_mlp, intermediate_size, conv_dim, num_heads]
            mlp_z, mlp_x, gate, hidden_B_C, dt = projected_states.split(
                [
                    self.d_mlp,
                    self.d_mlp,
                    self.intermediate_size,
                    self.conv_dim,
                    self.num_heads,
                ],
                dim=-1,
            )
        else:
            # Split: [intermediate_size, conv_dim, num_heads]
            gate, hidden_B_C, dt = projected_states.split(
                [self.intermediate_size, self.conv_dim, self.num_heads],
                dim=-1,
            )

        # ---- 2. Causal convolution ----
        # conv1d expects (B, C, L)
        hidden_B_C = hidden_B_C.transpose(1, 2).contiguous()  # (B, conv_dim, L)
        hidden_B_C = self.conv1d(hidden_B_C)[..., :seq_len]  # causal: trim future
        hidden_B_C = F.silu(hidden_B_C)
        hidden_B_C = hidden_B_C.transpose(1, 2).contiguous()  # (B, L, conv_dim)

        # ---- 3. Split into hidden_states, B, C ----
        hidden_ssm, B, C = hidden_B_C.split(
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # Reshape for multi-head SSM.
        hidden_ssm = hidden_ssm.reshape(
            batch, seq_len, self.num_heads, self.head_dim
        )
        B = B.reshape(batch, seq_len, self.n_groups, self.ssm_state_size)
        C = C.reshape(batch, seq_len, self.n_groups, self.ssm_state_size)

        # Repeat B, C from n_groups to num_heads.
        heads_per_group = self.num_heads // self.n_groups
        B = B.repeat_interleave(heads_per_group, dim=2)  # (B, L, H, N)
        C = C.repeat_interleave(heads_per_group, dim=2)  # (B, L, H, N)

        # ---- 4. Prepare dt and A ----
        A = -torch.exp(self.A_log.float())  # (H,)
        dt = F.softplus(dt + self.dt_bias.float())  # (B, L, H)
        dt = dt.clamp(min=self.time_step_limit[0], max=self.time_step_limit[1])

        # ---- 5. Chunked SSD scan ----
        y = self._ssd_chunk_scan(hidden_ssm, B, C, dt, A)
        # y: (B, L, H, head_dim)

        # Flatten heads: (B, L, intermediate_size)
        y = y.reshape(batch, seq_len, self.intermediate_size)

        # ---- 6. Gated RMSNorm ----
        y = self.norm(y, gate=gate)

        # ---- 7. MLP bypass (if d_mlp > 0) ----
        if self.d_mlp > 0:
            y = torch.cat([F.silu(mlp_z) * mlp_x, y], dim=-1)

        # ---- 8. Output projection ----
        out = self.out_proj(y)  # (B, L, D)

        # Transpose back to NeoX seq-first layout: (L, B, D)
        out = out.transpose(0, 1).contiguous()

        return out


# ---------------------------------------------------------------------------
# Residual wrappers (matching Mamba1 pattern)
# ---------------------------------------------------------------------------


class ParallelMamba2ResidualLayer(nn.Module):
    """Pre-norm Mamba2 block with residual connection.

    Follows the same pattern as ``ParallelMambaResidualLayer`` for Mamba1.
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        layer_number,
    ):
        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(neox_args)
        self.norm = norm(neox_args.hidden_size, eps=eps)

        self.mixer = ParallelMamba2Block(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

    def forward(self, x, attention_mask=None, layer_past=None):
        # x = x + mixer(norm(x))
        residual = x
        hidden_states = self.mixer(self.norm(x))
        return hidden_states + residual


class ParallelMamba2ResidualLayerPipe(ParallelMamba2ResidualLayer):
    """Pipeline-parallel compatible version that passes ``(hidden_states, attention_mask)`` tuples.

    DeepSpeed's ``PipelineModule`` requires layers to accept and return
    fixed-length tuples.
    """

    def forward(self, args):
        assert (
            len(args) == 2
        ), "Mamba2ResidualLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        return super().forward(hidden_states, attention_mask), attention_mask
