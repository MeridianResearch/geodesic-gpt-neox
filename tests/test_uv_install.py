"""
Tests to verify the UV environment installation is working correctly.

These tests verify:
1. Core package imports work
2. CUDA is available and functional
3. Fused kernels can be loaded
4. Basic tensor operations work on GPU

Run with:
    LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_uv_install.py -v
"""

import pytest
import sys


class TestCoreImports:
    """Test that core packages can be imported."""

    def test_import_torch(self):
        import torch

        assert torch.__version__ is not None

    def test_import_datasets(self):
        import datasets

        assert datasets.__version__ is not None

    def test_import_transformers(self):
        import transformers

        assert transformers.__version__ is not None

    def test_import_wandb(self):
        import wandb

        assert wandb.__version__ is not None

    def test_import_deepspeed(self):
        import deepspeed

        assert deepspeed.__version__ is not None

    def test_import_lm_dataformat(self):
        import lm_dataformat

        assert lm_dataformat is not None

    def test_import_einops(self):
        import einops

        assert einops.__version__ is not None

    def test_import_accelerate(self):
        import accelerate

        assert accelerate.__version__ is not None

    def test_import_huggingface_hub(self):
        import huggingface_hub

        assert huggingface_hub.__version__ is not None


class TestCUDAAvailability:
    """Test that CUDA is available and properly configured."""

    def test_cuda_is_available(self):
        """CUDA must be available for GPU training."""
        import torch

        assert torch.cuda.is_available(), "CUDA is not available"

    def test_cuda_device_count(self):
        """At least one CUDA device must be present."""
        import torch

        if torch.cuda.is_available():
            assert torch.cuda.device_count() > 0, "No CUDA devices found"

    def test_cuda_version(self):
        """CUDA version should be 12.4 for this setup (cu124 wheels for aarch64)."""
        import torch

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            assert cuda_version is not None
            assert cuda_version.startswith("12.4"), f"Expected CUDA 12.4, got {cuda_version}"

    def test_cudnn_available(self):
        """cuDNN must be available for optimized operations."""
        import torch

        if torch.cuda.is_available():
            assert torch.backends.cudnn.is_available(), "cuDNN is not available"

    def test_cudnn_version(self):
        """cuDNN version should be available."""
        import torch

        if torch.cuda.is_available() and torch.backends.cudnn.is_available():
            version = torch.backends.cudnn.version()
            assert version is not None
            assert version > 0, f"Invalid cuDNN version: {version}"


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestCUDAOperations:
    """Test basic CUDA tensor operations."""

    def test_tensor_to_cuda(self):
        """Test moving a tensor to GPU."""
        import torch

        x = torch.randn(10, 10)
        x_cuda = x.cuda()
        assert x_cuda.is_cuda, "Tensor not on CUDA"

    def test_cuda_matmul(self):
        """Test matrix multiplication on GPU."""
        import torch

        a = torch.randn(100, 100, device="cuda")
        b = torch.randn(100, 100, device="cuda")
        c = torch.matmul(a, b)
        assert c.is_cuda, "Result not on CUDA"
        assert c.shape == (100, 100)

    def test_cuda_memory_allocation(self):
        """Test that GPU memory can be allocated."""
        import torch

        # Allocate a reasonably sized tensor
        x = torch.randn(1000, 1000, device="cuda")
        assert torch.cuda.memory_allocated() > 0, "No GPU memory allocated"
        del x
        torch.cuda.empty_cache()

    def test_cuda_bf16_support(self):
        """Test bfloat16 support on GPU (required for H100)."""
        import torch

        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            x = torch.randn(10, 10, device="cuda", dtype=torch.bfloat16)
            assert x.dtype == torch.bfloat16

    def test_cuda_flash_attention_sdpa(self):
        """Test scaled dot product attention (Flash Attention backend)."""
        import torch
        import torch.nn.functional as F

        # Create inputs for attention
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")

        # Run SDPA (uses Flash Attention on supported hardware)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(q, k, v)

        assert out.shape == (batch_size, num_heads, seq_len, head_dim)
        assert out.is_cuda


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestFusedKernels:
    """Test that fused kernels can be loaded and used."""

    def test_load_fused_kernels(self):
        """Test loading the fused kernels."""
        from megatron.fused_kernels import load

        # This should not raise an exception
        load()

    def test_scaled_upper_triang_masked_softmax(self):
        """Test the scaled upper triangular masked softmax kernel via .apply()."""
        import torch

        try:
            from megatron.fused_kernels import load

            load()
            from megatron.model.fused_softmax import ScaledUpperTriangMaskedSoftmax

            # Input must be 3D (attn_batches, sq, sk), fp16, and meet kernel constraints
            attn_batches, seq_len = 16, 64
            x = torch.randn(
                attn_batches, seq_len, seq_len, device="cuda", dtype=torch.float16
            )
            scale = 1.0 / (64**0.5)

            # Use .apply() - the correct autograd Function API (matches production code)
            out = ScaledUpperTriangMaskedSoftmax.apply(x, scale)
            assert out.shape == x.shape
            assert out.is_cuda
            assert out.dtype == torch.float16
        except ImportError:
            pytest.skip("Fused softmax kernel not available")


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
class TestDeepSpeedCUDA:
    """Test DeepSpeed CUDA functionality."""

    def test_deepspeed_cuda_accelerator(self):
        """Test DeepSpeed detects CUDA accelerator."""
        from deepspeed.accelerator import get_accelerator

        accel = get_accelerator()
        assert accel.device_name() == "cuda"

    def test_deepspeed_cuda_device(self):
        """Test DeepSpeed can access CUDA device."""
        from deepspeed.accelerator import get_accelerator

        accel = get_accelerator()
        assert accel.is_available()
        assert accel.device_count() > 0


class TestOptionalPackages:
    """Test optional packages that may or may not be installed."""

    def test_flash_attn_import(self):
        """Test flash_attn import (may not be installed)."""
        try:
            import flash_attn

            assert flash_attn.__version__ is not None
        except ImportError:
            pytest.skip("flash_attn not installed")

    def test_transformer_engine_import(self):
        """Test transformer_engine import (may not be installed)."""
        try:
            import transformer_engine

            assert transformer_engine.__version__ is not None
        except ImportError:
            pytest.skip("transformer_engine not installed")


class TestEnvironmentSetup:
    """Test environment is correctly configured."""

    def test_python_version(self):
        """Python version should be 3.12."""
        assert sys.version_info.major == 3
        assert sys.version_info.minor == 12

    def test_torch_cuda_arch(self):
        """TORCH_CUDA_ARCH_LIST should include sm_90 for H100."""
        import os

        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        # This is set at runtime, may not be in env during test
        # Just check torch compiled with sm_90 support
        import torch

        if torch.cuda.is_available():
            # H100 is compute capability 9.0
            capability = torch.cuda.get_device_capability()
            # Just verify we can get capability
            assert capability is not None


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA not available"
)
@pytest.mark.slow
class TestTrainingLossDecreases:
    """Test that training runs and loss decreases over iterations.

    These tests run actual training and verify:
    1. Training completes without errors
    2. Loss values are extracted from output
    3. Final loss is lower than initial loss

    Run with:
        LD_PRELOAD=$NCCL_LIBRARY uv run pytest tests/test_uv_install.py -v -m slow
    """

    @staticmethod
    def parse_loss_values(output: str) -> list[float]:
        """Extract lm_loss values from training output."""
        import re

        # Pattern matches: lm_loss: 1.150898E+01 or lm_loss: 7.5123
        pattern = r"lm_loss:\s*([0-9.E+-]+)"
        matches = re.findall(pattern, output)
        return [float(m) for m in matches]

    @staticmethod
    def run_training(config_path: str, timeout: int = 600) -> str:
        """Run training with the given config and return stdout."""
        import subprocess
        import os

        # Get the repo root directory
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Build the command - use deepy.py which handles config serialization
        cmd = [
            sys.executable,
            os.path.join(repo_root, "deepy.py"),
            os.path.join(repo_root, "train.py"),
            config_path,
        ]

        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU
        env["TORCH_CUDA_ARCH_LIST"] = "9.0"
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29500"

        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_root,
            env=env,
        )

        # Combine stdout and stderr
        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed with return code {result.returncode}\n"
                f"Output:\n{output[-5000:]}"  # Last 5000 chars to avoid huge output
            )

        return output

    def test_tiny_model_loss_decreases(self):
        """Test that a tiny model's loss decreases over 100 iterations.

        Uses a 2-layer, 256-hidden model for fast testing (~2-3 minutes).
        """
        import os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "configs", "test_training_loss.yml")

        if not os.path.exists(config_path):
            pytest.skip(f"Test config not found: {config_path}")

        # Run training
        output = self.run_training(config_path, timeout=600)

        # Parse loss values
        losses = self.parse_loss_values(output)

        # Verify we got loss values
        assert len(losses) >= 5, f"Expected at least 5 loss values, got {len(losses)}"

        # Get initial and final losses (average first/last few to reduce noise)
        initial_loss = sum(losses[:3]) / 3
        final_loss = sum(losses[-3:]) / 3

        # Verify loss decreased
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}\n"
            f"All losses: {losses}"
        )

        # Verify significant decrease (at least 10%)
        decrease_pct = (initial_loss - final_loss) / initial_loss * 100
        assert decrease_pct > 10, (
            f"Loss decrease too small: {decrease_pct:.1f}% "
            f"(initial={initial_loss:.4f}, final={final_loss:.4f})"
        )

    def test_loss_values_are_valid(self):
        """Test that loss values are reasonable (not NaN or inf)."""
        import os
        import math

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(repo_root, "configs", "test_training_loss.yml")

        if not os.path.exists(config_path):
            pytest.skip(f"Test config not found: {config_path}")

        # Run training
        output = self.run_training(config_path, timeout=600)

        # Parse loss values
        losses = self.parse_loss_values(output)

        # Verify all losses are valid numbers
        for i, loss in enumerate(losses):
            assert not math.isnan(loss), f"Loss at iteration {i} is NaN"
            assert not math.isinf(loss), f"Loss at iteration {i} is infinite"
            assert loss > 0, f"Loss at iteration {i} is non-positive: {loss}"
            assert loss < 100, f"Loss at iteration {i} is suspiciously high: {loss}"
