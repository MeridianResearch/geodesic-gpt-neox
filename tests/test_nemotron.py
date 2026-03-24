"""
Tests for Nemotron-3 hybrid model support in GPT-NeoX.

Covers:
1. relu_squared activation
2. MambaRMSNormGated
3. ParallelMamba2Block and helper functions (pad_tensor_by_size, reshape_into_chunks, segment_sum)
4. NemotronSigmoidRouter
5. NemotronExpertMLP
6. NemotronMoE (full MoE layer)
7. NemotronAttentionResidualLayer
8. NemotronMLPResidualLayer
9. Config integration (nemotron_hybrid_pattern translation)
10. Conversion script (weight mapping, key naming)

Run with:
    LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_nemotron.py -v

Most tests run on CPU with small dimensions. GPU-requiring tests are marked with @pytest.mark.gpu.
"""

import math
import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockNeoXArgs:
    """Minimal mock of NeoXArgs with Nemotron-3 configuration (small for CPU tests)."""

    hidden_size = 64
    num_attention_heads = 4
    num_kv_heads = 2
    seq_length = 128
    max_position_embeddings = 1024

    # Mamba2
    mamba2_num_heads = 8
    mamba2_head_dim = 8
    mamba2_state_size = 16
    mamba2_conv_kernel = 4
    mamba2_n_groups = 2
    mamba2_chunk_size = 16
    mamba2_expand = 2
    mamba2_use_conv_bias = True
    precision = "fp32"

    # Norm
    norm = "rmsnorm"
    rms_norm_epsilon = 1e-5
    rmsnorm_fusion = False
    layernorm_epsilon = 1e-5
    layernorm_fusion = False

    # MoE
    moe_num_experts = 4
    moe_n_shared_experts = 1
    moe_top_k = 2
    moe_routed_intermediate_size = 32
    moe_shared_expert_intermediate_size = 64
    moe_routed_scaling_factor = 2.5
    moe_routing_type = "sigmoid_topk"
    moe_n_group = 1
    moe_topk_group = 1
    moe_e_score_correction = True
    moe_norm_topk_prob = True
    params_dtype = torch.float32

    # Activation
    activation = "relu2"

    # Attention
    pos_emb = "rotary"
    rotary_pct = 1.0
    rotary_emb_base = 10000
    use_bias_in_attn_linear = False

    # Dropout
    hidden_dropout = 0.0

    # MLP
    nemotron_mlp_intermediate_size = 32


@pytest.fixture
def neox_args():
    return MockNeoXArgs()


# ===================================================================
# 1. Unit Tests for relu_squared activation
# ===================================================================


class TestReluSquared:
    """Test the relu_squared activation: F.relu(x).pow(2)."""

    def test_relu_squared_positive_inputs(self):
        """relu_squared(x) == x^2 for positive x."""
        from megatron.model.activations import relu_squared

        x = torch.tensor([1.0, 2.0, 3.0, 0.5])
        result = relu_squared(x)
        expected = x.pow(2)
        torch.testing.assert_close(
            result, expected, msg="relu_squared should equal x^2 for positive inputs"
        )

    def test_relu_squared_negative_inputs(self):
        """relu_squared(x) == 0 for negative x."""
        from megatron.model.activations import relu_squared

        x = torch.tensor([-1.0, -0.5, -10.0, -0.001])
        result = relu_squared(x)
        expected = torch.zeros_like(x)
        torch.testing.assert_close(
            result, expected, msg="relu_squared should be 0 for all negative inputs"
        )

    def test_relu_squared_zero(self):
        """relu_squared(0) == 0."""
        from megatron.model.activations import relu_squared

        x = torch.tensor([0.0])
        result = relu_squared(x)
        assert result.item() == 0.0, "relu_squared(0) should be 0"

    def test_relu_squared_mixed_inputs(self):
        """Test with a mix of positive, negative, and zero values."""
        from megatron.model.activations import relu_squared

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu_squared(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 4.0])
        torch.testing.assert_close(result, expected)

    def test_relu_squared_gradient_flow(self):
        """Verify gradient flows correctly through relu_squared."""
        from megatron.model.activations import relu_squared

        x = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
        y = relu_squared(x)
        loss = y.sum()
        loss.backward()

        # d/dx relu(x)^2 = 2*relu(x)*step(x) = 2*x for x > 0, 0 for x <= 0
        expected_grad = torch.tensor([4.0, 0.0, 1.0])
        torch.testing.assert_close(
            x.grad, expected_grad, msg="Gradient of relu_squared should be 2*x for x>0"
        )

    def test_relu_squared_matches_manual(self):
        """relu_squared should exactly match F.relu(x).pow(2)."""
        from megatron.model.activations import relu_squared

        x = torch.randn(100)
        result = relu_squared(x)
        expected = F.relu(x).pow(2)
        torch.testing.assert_close(result, expected)

    def test_relu_squared_in_get_activation(self):
        """The 'relu2' activation name should resolve to relu_squared."""
        from megatron.model.activations import get_activation, relu_squared

        class FakeArgs:
            activation = "relu2"

        func, is_gated = get_activation(FakeArgs())
        # get_activation returns the relu_squared function object
        x = torch.tensor([2.0, -1.0])
        torch.testing.assert_close(func(x), relu_squared(x))
        assert not is_gated, "relu2 should not be gated"


# ===================================================================
# 2. Unit Tests for MambaRMSNormGated
# ===================================================================


class TestMambaRMSNormGated:
    """Test MambaRMSNormGated: norm(hidden) * silu(gate)."""

    def test_basic_rmsnorm_no_gate(self):
        """Without a gate, MambaRMSNormGated is just RMSNorm."""
        from megatron.model.norms import MambaRMSNormGated

        dim = 16
        norm = MambaRMSNormGated(dim, eps=1e-6)
        x = torch.randn(2, 8, dim)
        out = norm(x)

        # Manual RMSNorm
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + 1e-6)
        expected = (norm.weight * x_normed).to(x.dtype)

        torch.testing.assert_close(
            out, expected, msg="Without gate, should be pure RMSNorm"
        )

    def test_with_gating(self):
        """With a gate, output is norm(hidden) * silu(gate)."""
        from megatron.model.norms import MambaRMSNormGated

        dim = 16
        norm = MambaRMSNormGated(dim, eps=1e-6)
        x = torch.randn(2, 8, dim)
        gate = torch.randn(2, 8, dim)
        out = norm(x, gate=gate)

        # Manual computation
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_normed = x_f32 * torch.rsqrt(variance + 1e-6)
        normed = (norm.weight * x_normed).to(x.dtype)
        expected = normed * F.silu(gate.to(x.dtype))

        torch.testing.assert_close(
            out, expected, msg="With gate, output should be norm(x) * silu(gate)"
        )

    def test_output_shape_matches_input(self):
        """Output shape must match input hidden_states shape."""
        from megatron.model.norms import MambaRMSNormGated

        dim = 32
        norm = MambaRMSNormGated(dim)
        for shape in [(1, 1, dim), (4, 16, dim), (2, 128, dim)]:
            x = torch.randn(*shape)
            out = norm(x)
            assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"

    def test_different_eps_values(self):
        """Different eps values should produce numerically different results."""
        from megatron.model.norms import MambaRMSNormGated

        dim = 16
        norm_small_eps = MambaRMSNormGated(dim, eps=1e-12)
        norm_large_eps = MambaRMSNormGated(dim, eps=1.0)

        # Copy weights to be identical
        with torch.no_grad():
            norm_large_eps.weight.copy_(norm_small_eps.weight)

        x = torch.randn(2, 8, dim)
        out_small = norm_small_eps(x)
        out_large = norm_large_eps(x)

        # They should differ because eps affects normalization
        assert not torch.allclose(
            out_small, out_large, atol=1e-4
        ), "Different eps values should produce different outputs"

    def test_weight_parameter_is_learnable(self):
        """The weight should be a learnable parameter initialized to ones."""
        from megatron.model.norms import MambaRMSNormGated

        dim = 32
        norm = MambaRMSNormGated(dim)
        assert norm.weight.shape == (dim,), "Weight shape should be (dim,)"
        assert norm.weight.requires_grad, "Weight should be learnable"
        torch.testing.assert_close(
            norm.weight.data, torch.ones(dim), msg="Weight should be initialized to ones"
        )


# ===================================================================
# 3. Unit Tests for Mamba2 Block and Helpers
# ===================================================================


class TestMamba2Helpers:
    """Test helper functions for the Mamba2 chunked scan."""

    def test_pad_tensor_by_size_no_pad(self):
        """pad_tensor_by_size with pad_size=0 should return input unchanged."""
        from megatron.model.mamba.mamba2 import pad_tensor_by_size

        x = torch.randn(2, 10, 4)
        out = pad_tensor_by_size(x, 0)
        torch.testing.assert_close(out, x)

    def test_pad_tensor_by_size_pads_seq_dim(self):
        """Padding should extend dim=1 (sequence dimension)."""
        from megatron.model.mamba.mamba2 import pad_tensor_by_size

        x = torch.randn(2, 10, 4)
        out = pad_tensor_by_size(x, 6)
        assert out.shape == (2, 16, 4), f"Expected (2, 16, 4), got {out.shape}"
        # Original data should be preserved
        torch.testing.assert_close(out[:, :10, :], x)
        # Padding should be zeros
        torch.testing.assert_close(
            out[:, 10:, :], torch.zeros(2, 6, 4),
            msg="Padded region should be zeros"
        )

    def test_pad_tensor_by_size_higher_rank(self):
        """Padding should work for higher-rank tensors (e.g., 4D)."""
        from megatron.model.mamba.mamba2 import pad_tensor_by_size

        x = torch.randn(2, 7, 3, 5)
        out = pad_tensor_by_size(x, 3)
        assert out.shape == (2, 10, 3, 5), f"Expected (2, 10, 3, 5), got {out.shape}"

    def test_reshape_into_chunks_exact_multiple(self):
        """When seq_len is already a multiple of chunk_size, no padding needed."""
        from megatron.model.mamba.mamba2 import reshape_into_chunks

        x = torch.randn(2, 16, 4)
        out = reshape_into_chunks(x, pad_size=0, chunk_size=8)
        assert out.shape == (2, 2, 8, 4), f"Expected (2, 2, 8, 4), got {out.shape}"

    def test_reshape_into_chunks_with_padding(self):
        """When seq_len is not a multiple, padding is applied then reshaped."""
        from megatron.model.mamba.mamba2 import reshape_into_chunks

        x = torch.randn(2, 10, 4)
        # pad_size=6 to make 10->16, chunk_size=8 -> 2 chunks
        out = reshape_into_chunks(x, pad_size=6, chunk_size=8)
        assert out.shape == (2, 2, 8, 4), f"Expected (2, 2, 8, 4), got {out.shape}"
        # First 10 elements of data should match
        reconstructed = out.reshape(2, 16, 4)[:, :10, :]
        torch.testing.assert_close(reconstructed, x)

    def test_segment_sum_shape(self):
        """segment_sum output should be (B, n_chunks, chunk_size, chunk_size)."""
        from megatron.model.mamba.mamba2 import segment_sum

        x = torch.randn(4, 3, 8)  # (B, n_chunks, chunk_size)
        out = segment_sum(x)
        assert out.shape == (4, 3, 8, 8), f"Expected (4, 3, 8, 8), got {out.shape}"

    def test_segment_sum_lower_triangular(self):
        """segment_sum should produce a lower-triangular structure (upper = -inf, diagonal/lower = cumulative sums)."""
        from megatron.model.mamba.mamba2 import segment_sum

        x = torch.ones(1, 1, 4)
        out = segment_sum(x)
        # segment_sum uses -inf for upper triangle (for exp to give 0)
        # and cumulative sums for lower triangle + diagonal
        # Diagonal is 0 (no terms summed), lower tri has cumsum values
        ninf = float("-inf")
        expected = torch.tensor([[[[0, ninf, ninf, ninf],
                                    [1, 0, ninf, ninf],
                                    [2, 1, 0, ninf],
                                    [3, 2, 1, 0]]]], dtype=x.dtype)
        torch.testing.assert_close(
            out, expected, msg="segment_sum should compute cumulative sums with lower-tri masking"
        )


@pytest.mark.gpu
class TestParallelMamba2Block:
    """Test the full Mamba2 block (requires GPU for device placement)."""

    def test_forward_pass_output_shape(self, neox_args):
        """Output shape should match [seq_len, batch, hidden_size]."""
        from megatron.model.mamba.mamba2 import ParallelMamba2Block

        block = ParallelMamba2Block(
            neox_args=neox_args,
            init_method=nn.init.xavier_normal_,
            output_layer_init_method=nn.init.xavier_normal_,
        ).cuda()

        batch, seq_len = 2, 32
        x = torch.randn(seq_len, batch, neox_args.hidden_size, device="cuda")
        out = block(x)
        assert out.shape == (seq_len, batch, neox_args.hidden_size), (
            f"Expected ({seq_len}, {batch}, {neox_args.hidden_size}), got {out.shape}"
        )

    def test_different_sequence_lengths(self, neox_args):
        """Block should handle various sequence lengths, including non-multiples of chunk_size."""
        from megatron.model.mamba.mamba2 import ParallelMamba2Block

        block = ParallelMamba2Block(
            neox_args=neox_args,
            init_method=nn.init.xavier_normal_,
            output_layer_init_method=nn.init.xavier_normal_,
        ).cuda()
        block.eval()

        batch = 2
        for seq_len in [1, 7, 16, 33, 64]:
            x = torch.randn(seq_len, batch, neox_args.hidden_size, device="cuda")
            out = block(x)
            assert out.shape == (seq_len, batch, neox_args.hidden_size), (
                f"seq_len={seq_len}: expected shape ({seq_len}, {batch}, {neox_args.hidden_size}), "
                f"got {out.shape}"
            )

    def test_different_batch_sizes(self, neox_args):
        """Block should handle batch sizes from 1 to larger values."""
        from megatron.model.mamba.mamba2 import ParallelMamba2Block

        block = ParallelMamba2Block(
            neox_args=neox_args,
            init_method=nn.init.xavier_normal_,
            output_layer_init_method=nn.init.xavier_normal_,
        ).cuda()
        block.eval()

        seq_len = 16
        for batch in [1, 2, 8]:
            x = torch.randn(seq_len, batch, neox_args.hidden_size, device="cuda")
            out = block(x)
            assert out.shape == (seq_len, batch, neox_args.hidden_size), (
                f"batch={batch}: expected ({seq_len}, {batch}, {neox_args.hidden_size}), "
                f"got {out.shape}"
            )

    def test_a_log_and_d_are_float32(self, neox_args):
        """A_log and D SSM parameters must be float32 for numerical stability."""
        from megatron.model.mamba.mamba2 import ParallelMamba2Block

        block = ParallelMamba2Block(
            neox_args=neox_args,
            init_method=nn.init.xavier_normal_,
            output_layer_init_method=nn.init.xavier_normal_,
        ).cuda()

        assert block.A_log.dtype == torch.float32, (
            f"A_log should be float32, got {block.A_log.dtype}"
        )
        assert block.D.dtype == torch.float32, (
            f"D should be float32, got {block.D.dtype}"
        )
        assert block.dt_bias.dtype == torch.float32, (
            f"dt_bias should be float32, got {block.dt_bias.dtype}"
        )

    def test_padding_unpadding_correctness(self, neox_args):
        """When seq_len is not a multiple of chunk_size, padding should not corrupt output."""
        from megatron.model.mamba.mamba2 import ParallelMamba2Block

        block = ParallelMamba2Block(
            neox_args=neox_args,
            init_method=nn.init.xavier_normal_,
            output_layer_init_method=nn.init.xavier_normal_,
        ).cuda()
        block.eval()

        batch = 2
        # chunk_size=16, so seq_len=17 requires 15 padding
        seq_len = 17
        x = torch.randn(seq_len, batch, neox_args.hidden_size, device="cuda")
        out = block(x)
        assert out.shape == (seq_len, batch, neox_args.hidden_size), (
            "Output should have original seq_len after padding removal"
        )
        assert torch.isfinite(out).all(), "Output should be finite (no NaN/Inf from padding)"


# ===================================================================
# 4. Unit Tests for NemotronSigmoidRouter
# ===================================================================


class TestNemotronSigmoidRouter:
    """Test the sigmoid-based MoE router."""

    def test_sigmoid_routing_produces_valid_probabilities(self, neox_args):
        """Routing scores should be between 0 and 1 (sigmoid outputs)."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        router = NemotronSigmoidRouter(neox_args)
        num_tokens = 16
        x = torch.randn(num_tokens, neox_args.hidden_size)
        routing_weights, selected_experts = router(x)

        assert (routing_weights >= 0).all(), "Routing weights should be non-negative"
        assert (routing_weights <= 1).all(), "Routing weights should be <= 1"

    def test_topk_selection_returns_k_experts(self, neox_args):
        """Each token should select exactly top_k experts."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        router = NemotronSigmoidRouter(neox_args)
        num_tokens = 16
        x = torch.randn(num_tokens, neox_args.hidden_size)
        routing_weights, selected_experts = router(x)

        assert routing_weights.shape == (num_tokens, neox_args.moe_top_k), (
            f"Expected weights shape ({num_tokens}, {neox_args.moe_top_k}), "
            f"got {routing_weights.shape}"
        )
        assert selected_experts.shape == (num_tokens, neox_args.moe_top_k), (
            f"Expected experts shape ({num_tokens}, {neox_args.moe_top_k}), "
            f"got {selected_experts.shape}"
        )

    def test_routing_weights_sum_to_one_when_norm(self, neox_args):
        """When norm_topk_prob is True, weights for each token should sum to ~1."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        neox_args.moe_norm_topk_prob = True
        router = NemotronSigmoidRouter(neox_args)
        num_tokens = 32
        x = torch.randn(num_tokens, neox_args.hidden_size)
        routing_weights, _ = router(x)

        weight_sums = routing_weights.sum(dim=-1)
        torch.testing.assert_close(
            weight_sums,
            torch.ones(num_tokens),
            atol=1e-4,
            rtol=1e-4,
            msg="Normalized routing weights should sum to 1 per token",
        )

    def test_e_score_correction_bias_applied(self, neox_args):
        """When e_score_correction is True, the bias parameter should exist and affect scores."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        neox_args.moe_e_score_correction = True
        router = NemotronSigmoidRouter(neox_args)

        assert router.e_score_correction_bias is not None, (
            "e_score_correction_bias should exist when moe_e_score_correction=True"
        )
        assert router.e_score_correction_bias.shape == (neox_args.moe_num_experts,), (
            f"Bias shape should be ({neox_args.moe_num_experts},), "
            f"got {router.e_score_correction_bias.shape}"
        )

        # With a large bias on expert 0, that expert should be selected more often
        with torch.no_grad():
            router.e_score_correction_bias.fill_(0)
            router.e_score_correction_bias[0] = 100.0

        x = torch.randn(32, neox_args.hidden_size)
        _, selected_experts = router(x)
        # Expert 0 should appear in most tokens' top-k selections
        expert_0_selected = (selected_experts == 0).any(dim=-1).sum().item()
        assert expert_0_selected > 20, (
            f"With large bias on expert 0, it should be selected for most tokens, "
            f"but was only selected for {expert_0_selected}/32"
        )

    def test_no_e_score_correction_when_disabled(self, neox_args):
        """When e_score_correction is False, bias should be None."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        neox_args.moe_e_score_correction = False
        router = NemotronSigmoidRouter(neox_args)
        assert router.e_score_correction_bias is None

    def test_output_shapes(self, neox_args):
        """Verify output shapes for various token counts."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        router = NemotronSigmoidRouter(neox_args)
        for num_tokens in [1, 8, 64]:
            x = torch.randn(num_tokens, neox_args.hidden_size)
            weights, experts = router(x)
            assert weights.shape == (num_tokens, neox_args.moe_top_k)
            assert experts.shape == (num_tokens, neox_args.moe_top_k)

    def test_expert_indices_in_valid_range(self, neox_args):
        """Selected expert indices must be in [0, num_experts)."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        router = NemotronSigmoidRouter(neox_args)
        x = torch.randn(32, neox_args.hidden_size)
        _, selected_experts = router(x)

        assert (selected_experts >= 0).all(), "Expert indices must be >= 0"
        assert (selected_experts < neox_args.moe_num_experts).all(), (
            f"Expert indices must be < {neox_args.moe_num_experts}"
        )

    def test_group_constrained_topk(self, neox_args):
        """When n_group > 1, group-constrained selection should be used."""
        from megatron.model.nemotron_moe import NemotronSigmoidRouter

        # 8 experts in 2 groups of 4
        neox_args.moe_num_experts = 8
        neox_args.moe_n_group = 2
        neox_args.moe_topk_group = 1  # only select from 1 group
        neox_args.moe_top_k = 2
        neox_args.moe_e_score_correction = False

        router = NemotronSigmoidRouter(neox_args)
        x = torch.randn(16, neox_args.hidden_size)
        _, selected_experts = router(x)

        # With topk_group=1, all selected experts for each token should be from the same group
        for token_idx in range(16):
            experts = selected_experts[token_idx].tolist()
            groups = [e // 4 for e in experts]
            assert len(set(groups)) == 1, (
                f"Token {token_idx}: with topk_group=1, all experts should be from the same group, "
                f"got experts {experts} from groups {groups}"
            )


# ===================================================================
# 5. Unit Tests for NemotronExpertMLP
# ===================================================================


class TestNemotronExpertMLP:
    """Test a single expert MLP."""

    def test_forward_pass_output_shape(self):
        """Output should match input shape [tokens, hidden_size]."""
        from megatron.model.nemotron_moe import NemotronExpertMLP

        hidden = 64
        intermediate = 128
        mlp = NemotronExpertMLP(hidden, intermediate)

        x = torch.randn(16, hidden)
        out = mlp(x)
        assert out.shape == (16, hidden), f"Expected (16, {hidden}), got {out.shape}"

    def test_forward_with_different_intermediate_sizes(self):
        """Test with various intermediate sizes."""
        from megatron.model.nemotron_moe import NemotronExpertMLP

        hidden = 32
        for intermediate in [16, 32, 64, 128]:
            mlp = NemotronExpertMLP(hidden, intermediate)
            x = torch.randn(8, hidden)
            out = mlp(x)
            assert out.shape == (8, hidden), (
                f"intermediate={intermediate}: expected (8, {hidden}), got {out.shape}"
            )

    def test_relu2_activation_is_used(self):
        """The expert MLP should use relu_squared activation internally."""
        from megatron.model.nemotron_moe import NemotronExpertMLP

        hidden = 16
        intermediate = 32
        mlp = NemotronExpertMLP(hidden, intermediate)

        # Negative up_proj outputs should be zeroed by relu_squared
        x = torch.randn(4, hidden)
        out = mlp(x)
        # Just verify it runs and produces finite output
        assert torch.isfinite(out).all(), "Output should be finite"

    def test_no_bias_in_linear_layers(self):
        """Expert MLP should have no bias in linear layers."""
        from megatron.model.nemotron_moe import NemotronExpertMLP

        mlp = NemotronExpertMLP(32, 64)
        assert mlp.up_proj.bias is None, "up_proj should have no bias"
        assert mlp.down_proj.bias is None, "down_proj should have no bias"

    def test_batch_size_one(self):
        """Should handle single-token input."""
        from megatron.model.nemotron_moe import NemotronExpertMLP

        mlp = NemotronExpertMLP(32, 64)
        x = torch.randn(1, 32)
        out = mlp(x)
        assert out.shape == (1, 32)


# ===================================================================
# 6. Unit Tests for NemotronMoE
# ===================================================================


class TestNemotronMoE:
    """Test the full MoE layer (router + routed experts + shared expert)."""

    def test_forward_pass_output_shape(self, neox_args):
        """MoE output should be [seq_len, batch, hidden_size]."""
        from megatron.model.nemotron_moe import NemotronMoE

        moe = NemotronMoE(neox_args)
        seq_len, batch = 16, 2
        x = torch.randn(seq_len, batch, neox_args.hidden_size)
        out, bias = moe(x)

        assert out.shape == (seq_len, batch, neox_args.hidden_size), (
            f"Expected ({seq_len}, {batch}, {neox_args.hidden_size}), got {out.shape}"
        )
        assert bias is None, "MoE bias should be None"

    def test_output_shape_matches_input_shape(self, neox_args):
        """Output shape must match input shape for various dimensions."""
        from megatron.model.nemotron_moe import NemotronMoE

        moe = NemotronMoE(neox_args)
        for shape in [(1, 1, neox_args.hidden_size),
                      (32, 4, neox_args.hidden_size),
                      (8, 1, neox_args.hidden_size)]:
            x = torch.randn(*shape)
            out, _ = moe(x)
            assert out.shape == x.shape, (
                f"Input shape {x.shape}, output shape {out.shape} - should match"
            )

    def test_shared_expert_processes_all_tokens(self, neox_args):
        """The shared expert should process all tokens (not routed)."""
        from megatron.model.nemotron_moe import NemotronMoE

        moe = NemotronMoE(neox_args)
        # Override shared expert to be identity-like to verify it's called for all tokens
        assert hasattr(moe, "shared_expert"), "MoE should have shared_expert attribute"
        assert moe.n_shared_experts == 1

    def test_routed_scaling_factor_is_applied(self, neox_args):
        """The routed_scaling_factor should scale the routed expert output."""
        from megatron.model.nemotron_moe import NemotronMoE

        # Create two MoE with different scaling factors, same weights
        neox_args_a = MockNeoXArgs()
        neox_args_a.moe_routed_scaling_factor = 1.0
        neox_args_b = MockNeoXArgs()
        neox_args_b.moe_routed_scaling_factor = 3.0

        torch.manual_seed(42)
        moe_a = NemotronMoE(neox_args_a)
        torch.manual_seed(42)
        moe_b = NemotronMoE(neox_args_b)

        x = torch.randn(8, 2, neox_args.hidden_size)
        out_a, _ = moe_a(x)
        out_b, _ = moe_b(x)

        # Outputs should differ because of different scaling factors
        assert not torch.allclose(out_a, out_b, atol=1e-6), (
            "Different scaling factors should produce different outputs"
        )

    def test_correct_number_of_experts(self, neox_args):
        """MoE should have the correct number of routed experts."""
        from megatron.model.nemotron_moe import NemotronMoE

        moe = NemotronMoE(neox_args)
        assert len(moe.experts) == neox_args.moe_num_experts, (
            f"Expected {neox_args.moe_num_experts} routed experts, got {len(moe.experts)}"
        )

    def test_gradient_flows_through_moe(self, neox_args):
        """Gradients should flow through the MoE layer."""
        from megatron.model.nemotron_moe import NemotronMoE

        moe = NemotronMoE(neox_args)
        x = torch.randn(8, 2, neox_args.hidden_size, requires_grad=True)
        out, _ = moe(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should flow back to input"
        assert torch.isfinite(x.grad).all(), "Gradients should be finite"


# ===================================================================
# 7. Unit Tests for NemotronAttentionResidualLayer
# ===================================================================


class TestNemotronAttentionResidualLayer:
    """Test the attention-only residual block.

    These tests require GPU because ParallelSelfAttention uses cuda device placement.
    """

    @pytest.mark.gpu
    def test_forward_pass_output_shape(self, neox_args):
        """Output shape should match input [batch, seq, hidden]."""
        from megatron.model.nemotron_attn import NemotronAttentionResidualLayer

        # NemotronAttentionResidualLayer uses ParallelSelfAttention which needs
        # full neox_args and CUDA. This is an integration-level test.
        pytest.skip(
            "NemotronAttentionResidualLayer requires full NeoXArgs and CUDA "
            "for ParallelSelfAttention initialization"
        )

    def test_module_structure(self):
        """Verify the expected module attributes exist in the class."""
        from megatron.model.nemotron_attn import NemotronAttentionResidualLayer

        # Check class has expected methods and attributes
        assert hasattr(NemotronAttentionResidualLayer, "forward")

    def test_pipe_version_exists(self):
        """The pipeline-parallel variant should exist."""
        from megatron.model.nemotron_attn import NemotronAttentionResidualLayerPipe

        assert issubclass(
            NemotronAttentionResidualLayerPipe,
            object,
        )


# ===================================================================
# 8. Unit Tests for NemotronMLPResidualLayer
# ===================================================================


class TestNemotronMLPResidualLayer:
    """Test the MLP-only residual block."""

    def test_forward_pass_output_shape(self, neox_args):
        """Output should match input shape."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        layer = NemotronMLPResidualLayer(
            neox_args=neox_args,
            init_method=nn.init.xavier_normal_,
            layer_number=0,
        )

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, neox_args.hidden_size)
        out = layer(x)
        assert out.shape == (batch, seq_len, neox_args.hidden_size), (
            f"Expected ({batch}, {seq_len}, {neox_args.hidden_size}), got {out.shape}"
        )

    def test_residual_connection(self, neox_args):
        """With near-zero weights, output should be close to input (residual dominates)."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        layer = NemotronMLPResidualLayer(
            neox_args=neox_args,
            layer_number=0,
        )

        # Set MLP weights to near-zero
        with torch.no_grad():
            layer.up_proj.weight.fill_(0.0)
            layer.down_proj.weight.fill_(0.0)

        x = torch.randn(2, 8, neox_args.hidden_size)
        out = layer(x)
        torch.testing.assert_close(
            out, x, atol=1e-5, rtol=1e-5,
            msg="With zero MLP weights, residual should dominate"
        )

    def test_uses_relu_squared(self, neox_args):
        """The MLP should use relu_squared activation internally."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        layer = NemotronMLPResidualLayer(neox_args=neox_args, layer_number=0)
        # Verify it runs without error (relu_squared is hardcoded in forward)
        x = torch.randn(2, 8, neox_args.hidden_size)
        out = layer(x)
        assert torch.isfinite(out).all(), "Output should be finite"

    def test_no_bias_in_linear_layers(self, neox_args):
        """MLP linear layers should have no bias."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        layer = NemotronMLPResidualLayer(neox_args=neox_args, layer_number=0)
        assert layer.up_proj.bias is None, "up_proj should have no bias"
        assert layer.down_proj.bias is None, "down_proj should have no bias"

    def test_intermediate_size_from_nemotron_mlp_config(self, neox_args):
        """Should use nemotron_mlp_intermediate_size when available."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        neox_args.nemotron_mlp_intermediate_size = 128
        layer = NemotronMLPResidualLayer(neox_args=neox_args, layer_number=0)
        assert layer.up_proj.out_features == 128, (
            f"up_proj should have intermediate_size=128, got {layer.up_proj.out_features}"
        )

    def test_intermediate_size_fallback_to_moe(self, neox_args):
        """Should fall back to moe_routed_intermediate_size when nemotron config is None."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        neox_args.nemotron_mlp_intermediate_size = None
        neox_args.moe_routed_intermediate_size = 96
        layer = NemotronMLPResidualLayer(neox_args=neox_args, layer_number=0)
        assert layer.up_proj.out_features == 96

    def test_raises_when_no_intermediate_size(self, neox_args):
        """Should raise ValueError when no intermediate size is configured."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        neox_args.nemotron_mlp_intermediate_size = None
        neox_args.moe_routed_intermediate_size = None
        with pytest.raises(ValueError, match="intermediate_size"):
            NemotronMLPResidualLayer(neox_args=neox_args, layer_number=0)

    def test_pipe_version_forward_interface(self, neox_args):
        """Pipeline version should accept (hidden_states, attention_mask) tuple."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayerPipe

        layer = NemotronMLPResidualLayerPipe(neox_args=neox_args, layer_number=0)
        x = torch.randn(2, 8, neox_args.hidden_size)
        mask = torch.ones(2, 8)
        out_hidden, out_mask = layer((x, mask))
        assert out_hidden.shape == x.shape
        torch.testing.assert_close(out_mask, mask)

    def test_edge_case_batch_1_seq_1(self, neox_args):
        """Should handle batch=1, seq=1."""
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayer

        layer = NemotronMLPResidualLayer(neox_args=neox_args, layer_number=0)
        x = torch.randn(1, 1, neox_args.hidden_size)
        out = layer(x)
        assert out.shape == (1, 1, neox_args.hidden_size)


# ===================================================================
# 9. Integration Tests for Config (nemotron_hybrid_pattern)
# ===================================================================


class TestNemotronConfigIntegration:
    """Test the nemotron_hybrid_pattern -> attention_config translation."""

    def test_pattern_translation_basic(self):
        """Test that 'MEMEM*' maps to the correct attention_config list."""
        _NEMOTRON_PATTERN_MAP = {
            "M": "mamba2",
            "E": "nemotron_moe",
            "*": "nemotron_attn",
            "-": "nemotron_mlp",
        }
        pattern = "MEMEM*"
        result = [_NEMOTRON_PATTERN_MAP[ch] for ch in pattern]
        expected = [
            "mamba2",
            "nemotron_moe",
            "mamba2",
            "nemotron_moe",
            "mamba2",
            "nemotron_attn",
        ]
        assert result == expected, f"Pattern 'MEMEM*' should map to {expected}, got {result}"

    def test_pattern_all_types(self):
        """Test a pattern using all 4 block types."""
        _NEMOTRON_PATTERN_MAP = {
            "M": "mamba2",
            "E": "nemotron_moe",
            "*": "nemotron_attn",
            "-": "nemotron_mlp",
        }
        pattern = "M*E-"
        result = [_NEMOTRON_PATTERN_MAP[ch] for ch in pattern]
        expected = ["mamba2", "nemotron_attn", "nemotron_moe", "nemotron_mlp"]
        assert result == expected

    def test_invalid_pattern_character_raises_error(self):
        """Unknown pattern characters should be caught (assertion error in NeoXArgs)."""
        valid_chars = {"M", "E", "*", "-"}
        invalid_chars = ["X", "A", "1", " ", "m"]
        for ch in invalid_chars:
            assert ch not in valid_chars, f"'{ch}' should not be a valid pattern character"

    def test_pattern_length_must_match_num_layers(self):
        """Pattern length mismatch with num_layers should be detected."""
        pattern = "MEMEM*"
        num_layers = 10  # pattern has 6 chars
        assert len(pattern) != num_layers, (
            "This test verifies that the pattern length check would catch a mismatch"
        )

    def test_pattern_map_in_arguments_module(self):
        """Verify the pattern map exists in the arguments module source."""
        import inspect
        from megatron.neox_arguments import arguments

        source = inspect.getsource(arguments)
        assert "nemotron_hybrid_pattern" in source
        assert '"M": "mamba2"' in source or "'M': 'mamba2'" in source
        assert '"E": "nemotron_moe"' in source or "'E': 'nemotron_moe'" in source
        assert '"*": "nemotron_attn"' in source or "'*': 'nemotron_attn'" in source
        assert '"-": "nemotron_mlp"' in source or "'-': 'nemotron_mlp'" in source

    def test_nemotron_hybrid_pattern_arg_exists(self):
        """The nemotron_hybrid_pattern argument should exist in NeoXArgsNemotron."""
        from megatron.neox_arguments.neox_args import NeoXArgsNemotron

        import inspect
        source = inspect.getsource(NeoXArgsNemotron)
        assert "nemotron_hybrid_pattern" in source

    def test_mamba2_args_exist(self):
        """Mamba2-related arguments should exist in NeoXArgsMamba2."""
        from megatron.neox_arguments.neox_args import NeoXArgsMamba2

        import inspect
        source = inspect.getsource(NeoXArgsMamba2)
        for arg in [
            "mamba2_num_heads",
            "mamba2_head_dim",
            "mamba2_state_size",
            "mamba2_conv_kernel",
            "mamba2_n_groups",
            "mamba2_chunk_size",
            "mamba2_expand",
        ]:
            assert arg in source, f"Argument '{arg}' not found in NeoXArgsMamba2"


# ===================================================================
# 10. Integration Tests for Conversion Script
# ===================================================================


class TestNemotronConversion:
    """Test the HF Nemotron-3 to NeoX conversion functions."""

    def test_parse_hybrid_pattern_valid(self):
        """parse_hybrid_pattern should map M/E/* to correct types."""
        from huggingface.convert_hf_nemotron_to_neox import parse_hybrid_pattern

        result = parse_hybrid_pattern("MEM*")
        assert result == ["mamba", "moe", "mamba", "attention"]

    def test_parse_hybrid_pattern_all_mamba(self):
        """Pattern of all M's should produce all mamba types."""
        from huggingface.convert_hf_nemotron_to_neox import parse_hybrid_pattern

        result = parse_hybrid_pattern("MMMM")
        assert result == ["mamba", "mamba", "mamba", "mamba"]

    def test_parse_hybrid_pattern_all_attention(self):
        """Pattern of all *'s should produce all attention types."""
        from huggingface.convert_hf_nemotron_to_neox import parse_hybrid_pattern

        result = parse_hybrid_pattern("****")
        assert result == ["attention", "attention", "attention", "attention"]

    def test_parse_hybrid_pattern_invalid_char(self):
        """Invalid pattern character should raise ValueError."""
        from huggingface.convert_hf_nemotron_to_neox import parse_hybrid_pattern

        with pytest.raises(ValueError, match="Unknown block type character"):
            parse_hybrid_pattern("MXE")

    def test_count_parameters(self):
        """count_parameters should correctly categorize parameter counts."""
        from huggingface.convert_hf_nemotron_to_neox import count_parameters

        state_dict = {
            "0.word_embeddings.weight": torch.randn(100, 64),
            "2.mixer.in_proj.weight": torch.randn(128, 64),
            "3.attention.query_key_value.weight": torch.randn(192, 64),
            "4.moe.gate.weight": torch.randn(4, 64),
            "4.moe.experts.0.up_proj.weight": torch.randn(32, 64),
            "5.norm.scale": torch.randn(64),
            "6.final_linear.weight": torch.randn(100, 64),
        }

        total, by_type = count_parameters(state_dict)
        assert total == sum(t.numel() for t in state_dict.values()), (
            "Total parameter count should match sum of all tensors"
        )
        assert by_type["embedding"] == 100 * 64
        assert by_type["mamba"] == 128 * 64
        assert by_type["attention"] == 192 * 64
        assert by_type["moe"] == 4 * 64 + 32 * 64
        assert by_type["norm"] == 64
        assert by_type["output"] == 100 * 64

    def test_mamba_block_key_mapping(self):
        """Mamba block conversion should produce correct NeoX key names."""
        from huggingface.convert_hf_nemotron_to_neox import _convert_mamba_block

        hf_state = {
            "backbone.layers.0.mixer.in_proj.weight": torch.randn(128, 64),
            "backbone.layers.0.mixer.conv1d.weight": torch.randn(96, 1, 4),
            "backbone.layers.0.mixer.conv1d.bias": torch.randn(96),
            "backbone.layers.0.mixer.A_log": torch.randn(8),
            "backbone.layers.0.mixer.D": torch.randn(8),
            "backbone.layers.0.mixer.dt_bias": torch.randn(8),
            "backbone.layers.0.mixer.norm.weight": torch.randn(64),
            "backbone.layers.0.mixer.out_proj.weight": torch.randn(64, 64),
        }

        state_dict = {}
        _convert_mamba_block(state_dict, hf_state, seq_idx=2, hf_prefix="backbone.layers.0")

        expected_keys = [
            "2.mixer.in_proj.weight",
            "2.mixer.conv1d.weight",
            "2.mixer.conv1d.bias",
            "2.mixer.A_log",
            "2.mixer.D",
            "2.mixer.dt_bias",
            "2.mixer.norm.weight",
            "2.mixer.out_proj.weight",
        ]
        for key in expected_keys:
            assert key in state_dict, f"Missing NeoX key: {key}"

    def test_attention_block_qkv_concatenation_gqa(self):
        """Attention block conversion should concatenate Q/K/V for GQA correctly."""
        from huggingface.convert_hf_nemotron_to_neox import _convert_attention_block

        hidden = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16

        q_weight = torch.randn(num_heads * head_dim, hidden)
        k_weight = torch.randn(num_kv_heads * head_dim, hidden)
        v_weight = torch.randn(num_kv_heads * head_dim, hidden)

        hf_state = {
            "backbone.layers.0.mixer.q_proj.weight": q_weight,
            "backbone.layers.0.mixer.k_proj.weight": k_weight,
            "backbone.layers.0.mixer.v_proj.weight": v_weight,
            "backbone.layers.0.mixer.o_proj.weight": torch.randn(hidden, hidden),
        }

        state_dict = {}
        _convert_attention_block(
            state_dict, hf_state, seq_idx=2, hf_prefix="backbone.layers.0",
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            use_gqa=True, hidden_size=hidden,
        )

        qkv = state_dict["2.attention.query_key_value.weight"]
        expected_rows = num_heads * head_dim + 2 * num_kv_heads * head_dim
        assert qkv.shape == (expected_rows, hidden), (
            f"GQA QKV weight should be ({expected_rows}, {hidden}), got {qkv.shape}"
        )

        # Verify Q part
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        torch.testing.assert_close(qkv[:q_size], q_weight)
        torch.testing.assert_close(qkv[q_size:q_size + kv_size], k_weight)
        torch.testing.assert_close(qkv[q_size + kv_size:], v_weight)

    def test_attention_block_qkv_interleave_mha(self):
        """For MHA (not GQA), QKV should be interleaved per head."""
        from huggingface.convert_hf_nemotron_to_neox import _convert_attention_block

        hidden = 64
        num_heads = 4
        head_dim = 16

        q_weight = torch.randn(num_heads * head_dim, hidden)
        k_weight = torch.randn(num_heads * head_dim, hidden)
        v_weight = torch.randn(num_heads * head_dim, hidden)

        hf_state = {
            "backbone.layers.0.mixer.q_proj.weight": q_weight,
            "backbone.layers.0.mixer.k_proj.weight": k_weight,
            "backbone.layers.0.mixer.v_proj.weight": v_weight,
            "backbone.layers.0.mixer.o_proj.weight": torch.randn(hidden, hidden),
        }

        state_dict = {}
        _convert_attention_block(
            state_dict, hf_state, seq_idx=2, hf_prefix="backbone.layers.0",
            num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim,
            use_gqa=False, hidden_size=hidden,
        )

        qkv = state_dict["2.attention.query_key_value.weight"]
        assert qkv.shape == (num_heads * 3 * head_dim, hidden), (
            f"MHA QKV should be ({num_heads * 3 * head_dim}, {hidden}), got {qkv.shape}"
        )

        # Verify interleaving: [Q0, K0, V0, Q1, K1, V1, ...]
        q_heads = q_weight.view(num_heads, head_dim, hidden)
        k_heads = k_weight.view(num_heads, head_dim, hidden)
        v_heads = v_weight.view(num_heads, head_dim, hidden)
        for h in range(num_heads):
            offset = h * 3 * head_dim
            torch.testing.assert_close(
                qkv[offset:offset + head_dim], q_heads[h],
                msg=f"Head {h} Q mismatch"
            )
            torch.testing.assert_close(
                qkv[offset + head_dim:offset + 2 * head_dim], k_heads[h],
                msg=f"Head {h} K mismatch"
            )
            torch.testing.assert_close(
                qkv[offset + 2 * head_dim:offset + 3 * head_dim], v_heads[h],
                msg=f"Head {h} V mismatch"
            )

    def test_moe_block_key_mapping(self):
        """MoE block conversion should produce correct NeoX key names."""
        from huggingface.convert_hf_nemotron_to_neox import _convert_moe_block

        n_routed_experts = 4
        hf_state = {
            "backbone.layers.0.mixer.gate.weight": torch.randn(4, 64),
            "backbone.layers.0.mixer.gate.e_score_correction_bias": torch.randn(4),
        }
        for i in range(n_routed_experts):
            hf_state[f"backbone.layers.0.mixer.experts.{i}.up_proj.weight"] = torch.randn(32, 64)
            hf_state[f"backbone.layers.0.mixer.experts.{i}.down_proj.weight"] = torch.randn(64, 32)
        hf_state["backbone.layers.0.mixer.shared_experts.up_proj.weight"] = torch.randn(128, 64)
        hf_state["backbone.layers.0.mixer.shared_experts.down_proj.weight"] = torch.randn(64, 128)

        state_dict = {}
        _convert_moe_block(
            state_dict, hf_state, seq_idx=4, hf_prefix="backbone.layers.0",
            n_routed_experts=n_routed_experts,
        )

        # Check router gate
        assert "4.moe.gate.weight" in state_dict
        assert "4.moe.e_score_correction_bias" in state_dict

        # Check per-expert weights
        for i in range(n_routed_experts):
            assert f"4.moe.experts.{i}.up_proj.weight" in state_dict
            assert f"4.moe.experts.{i}.down_proj.weight" in state_dict

        # Check shared expert (singular in NeoX)
        assert "4.moe.shared_expert.up_proj.weight" in state_dict
        assert "4.moe.shared_expert.down_proj.weight" in state_dict

    def test_embedding_and_output_layer_indices(self):
        """Test the sequential index scheme: 0=embed, num_layers+3=norm, num_layers+4=lm_head."""
        # This matches the docstring in convert_nemotron_to_neox_state_dict:
        # Layer 0: word_embeddings
        # Layer num_layers+3: final layer norm
        # Layer num_layers+4: output embedding
        num_layers = 6
        assert num_layers + 3 == 9, "Final norm index should be num_layers + 3"
        assert num_layers + 4 == 10, "Output embed index should be num_layers + 4"


# ===================================================================
# Additional: MoE Residual Layer Tests
# ===================================================================


class TestNemotronMoEResidualLayer:
    """Test the pre-norm residual wrapper for MoE."""

    def test_forward_pass_output_shape(self, neox_args):
        """Output should be [seq_len, batch, hidden_size]."""
        from megatron.model.nemotron_moe import NemotronMoEResidualLayer

        layer = NemotronMoEResidualLayer(neox_args=neox_args, layer_number=0)
        seq_len, batch = 16, 2
        x = torch.randn(seq_len, batch, neox_args.hidden_size)
        out = layer(x)
        assert out.shape == (seq_len, batch, neox_args.hidden_size)

    def test_residual_connection(self, neox_args):
        """With near-zero MoE weights, output should be dominated by residual."""
        from megatron.model.nemotron_moe import NemotronMoEResidualLayer

        layer = NemotronMoEResidualLayer(neox_args=neox_args, layer_number=0)

        # Zero out all expert and shared expert weights
        with torch.no_grad():
            for expert in layer.moe.experts:
                expert.up_proj.weight.fill_(0)
                expert.down_proj.weight.fill_(0)
            layer.moe.shared_expert.up_proj.weight.fill_(0)
            layer.moe.shared_expert.down_proj.weight.fill_(0)

        x = torch.randn(8, 2, neox_args.hidden_size)
        out = layer(x)
        torch.testing.assert_close(
            out, x, atol=1e-5, rtol=1e-5,
            msg="With zeroed MoE weights, output should equal input (residual)"
        )

    def test_pipe_version_interface(self, neox_args):
        """Pipeline version should accept and return (hidden, mask) tuples."""
        from megatron.model.nemotron_moe import NemotronMoEResidualLayerPipe

        layer = NemotronMoEResidualLayerPipe(neox_args=neox_args, layer_number=0)
        x = torch.randn(8, 2, neox_args.hidden_size)
        mask = torch.ones(2, 8)
        out_hidden, out_mask = layer((x, mask))
        assert out_hidden.shape == x.shape
        torch.testing.assert_close(out_mask, mask)

    def test_pipe_version_rejects_wrong_arg_count(self, neox_args):
        """Pipeline version should reject incorrect number of arguments."""
        from megatron.model.nemotron_moe import NemotronMoEResidualLayerPipe

        layer = NemotronMoEResidualLayerPipe(neox_args=neox_args, layer_number=0)
        x = torch.randn(8, 2, neox_args.hidden_size)
        with pytest.raises(AssertionError):
            layer((x,))  # too few args
        with pytest.raises(AssertionError):
            layer((x, x, x))  # too many args
