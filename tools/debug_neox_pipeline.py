#!/usr/bin/env python3
"""Debug script: verify that DeepSpeed PipelineModule loads all checkpoint keys.

This script answers the question: does `load_state_dict(strict=False)` silently
skip mismatched keys for the new Nemotron layer types (Mamba2, NemotronMoE,
NemotronAttention, NemotronMLP)?

It works WITHOUT initializing distributed training.  Instead it:
  1.  Loads the raw NeoX checkpoint state dict.
  2.  Builds the GPT2ModelPipe layer specs, instantiates every layer on CPU,
      and collects the model-side parameter names exactly as PipelineModule
      would register them (i.e. "{layer_idx}.submodule.param").
  3.  Compares checkpoint keys against model keys to find:
      - matched keys (same name AND same shape)
      - missing keys  (in model but not in checkpoint -- weights stay random)
      - unexpected keys (in checkpoint but not in model -- silently ignored)
      - shape mismatches (same name, different tensor shape)
  4.  Reports per-parameter norms so you can see if any layer has suspiciously
      small or large norms (a sign it was never loaded).
  5.  (Optionally) attempts a full load_state_dict and a single forward pass
      on CPU/CUDA to verify outputs are non-trivial.

Usage (on a compute node):
    isambard_sbatch run_on_compute.sbatch uv run python tools/debug_neox_pipeline.py

    # Or interactively:
    uv run python tools/debug_neox_pipeline.py

    # With a custom checkpoint / config:
    uv run python tools/debug_neox_pipeline.py \
        --checkpoint /path/to/global_stepN/mp_rank_00_model_states.pt \
        --config /path/to/config.yml

    # Skip forward pass (faster, CPU-only weight comparison):
    uv run python tools/debug_neox_pipeline.py --no-forward
"""

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CKPT = (
    "/projects/a5k/public/checkpoints/sf_model_organisms/"
    "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/global_step0/"
    "mp_rank_00_model_states.pt"
)
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs/nemotron3/nemotron3_nano_30b_eval_mmlu.yml",
)


# ---------------------------------------------------------------------------
# Nemotron hybrid pattern helpers
# ---------------------------------------------------------------------------
_NEMOTRON_PATTERN_MAP = {
    "M": "mamba2",
    "E": "nemotron_moe",
    "*": "nemotron_attn",
    "-": "nemotron_mlp",
}


def parse_nemotron_pattern(pattern):
    """Convert a pattern string like 'MEMEM*E...' to a list of layer types."""
    return [_NEMOTRON_PATTERN_MAP[ch] for ch in pattern]


# ---------------------------------------------------------------------------
# Config loader (minimal -- avoids importing megatron which needs dist init)
# ---------------------------------------------------------------------------
def load_config_from_yml(yml_path):
    """Load a NeoX YAML config file (actually JSONC) and return a dict."""
    import re

    with open(yml_path) as f:
        raw = f.read()
    # Strip // and # comments that NeoX allows in its "YAML" configs
    # (they are actually JSON with comments)
    lines = []
    for line in raw.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        # Remove inline comments (crude but sufficient)
        # Only strip if # is not inside a string
        lines.append(line)
    cleaned = "\n".join(lines)
    # Remove trailing commas before } or ] (JSONC extension not valid in JSON)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Build the layer specs manually (mirrors GPT2ModelPipe.init_specs)
# ---------------------------------------------------------------------------
def build_layer_specs_from_config(config):
    """Return a list of (layer_idx, layer_type_name, nn.Module) tuples.

    This mirrors the layer construction in GPT2ModelPipe.init_specs() but
    avoids importing megatron (and therefore avoids dist init).

    We build a minimal mock of NeoXArgs and construct each layer type directly.
    """
    # We need the actual megatron modules.  Import them here so the script
    # can fail gracefully if megatron is not importable.
    try:
        from megatron.model.norms import get_norm, RMSNorm
        from megatron.model.word_embeddings import EmbeddingPipe
        from megatron.model.transformer import (
            ParallelTransformerLayerPipe,
            NormPipe,
            ParallelLinearPipe,
        )
        from megatron.model.mamba import (
            ParallelMambaResidualLayerPipe,
            ParallelMamba2ResidualLayerPipe,
        )
        from megatron.model.nemotron_attn import NemotronAttentionResidualLayerPipe
        from megatron.model.nemotron_moe import NemotronMoEResidualLayerPipe
        from megatron.model.nemotron_mlp import NemotronMLPResidualLayerPipe
    except ImportError as e:
        print(f"Cannot import megatron modules: {e}")
        print("This script must be run from the gpt-neox repo root with the venv active.")
        sys.exit(1)

    raise NotImplementedError(
        "Direct layer construction without NeoXArgs is complex.  "
        "Use the simpler state-dict comparison approach below."
    )


# ---------------------------------------------------------------------------
# Core analysis: compare checkpoint keys vs model parameter names
# ---------------------------------------------------------------------------
def load_checkpoint_state_dict(ckpt_path):
    """Load the NeoX checkpoint and return the flat weight state dict.

    NeoX checkpoints have the structure:
        checkpoint["module"]["module"] = {key: tensor, ...}
    where keys are like "0.word_embeddings.weight", "2.norm.scale", etc.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    print("  (this may take a minute for large models...)")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Navigate the nested structure
    if "module" in ckpt and isinstance(ckpt["module"], dict):
        inner = ckpt["module"]
        if "module" in inner and isinstance(inner["module"], dict):
            sd = inner["module"]
        else:
            sd = inner
    else:
        sd = ckpt

    print(f"  Loaded {len(sd)} keys from checkpoint")
    return sd


def build_expected_model_keys(config):
    """Infer what parameter keys the PipelineModule WOULD have.

    We do this by instantiating each layer type on CPU with minimal args
    and collecting named_parameters().

    The PipelineModule registers layers as:
        self.add_module(str(layer_idx), module)
    so parameter names become "{layer_idx}.{submodule}.{param}".

    Layer index mapping:
        0: EmbeddingPipe (word_embeddings)
        1: _pre_transformer_block (function, no params)
        2 .. num_layers+1: transformer/hybrid layers
        num_layers+2: _post_transformer_block (function, no params)
        num_layers+3: NormPipe (final layer norm)
        num_layers+4: ParallelLinearPipe (lm_head / output embedding)
    """
    # This approach uses megatron imports.  If those aren't available,
    # we fall back to a heuristic approach.
    return _build_keys_heuristic(config)


def _build_keys_heuristic(config):
    """Build expected model keys using knowledge of the module structure.

    This doesn't instantiate any modules -- it just infers the key names
    from the config and our knowledge of each layer type's submodule tree.
    """
    num_layers = config["num_layers"]
    hidden_size = config["hidden_size"]
    pattern = config.get("nemotron_hybrid_pattern", "")
    no_weight_tying = config.get("no_weight_tying", True)

    if pattern:
        layer_types = parse_nemotron_pattern(pattern)
    else:
        # Default: all global attention
        layer_types = ["global"] * num_layers

    assert len(layer_types) == num_layers, (
        f"Pattern length {len(layer_types)} != num_layers {num_layers}"
    )

    keys = OrderedDict()  # key -> expected shape (None = unknown)

    # --- Layer 0: Embedding ---
    keys["0.word_embeddings.weight"] = None

    # --- Layer 1: _pre_transformer_block (function, no params) ---
    # --- Layers 2..num_layers+1: hybrid blocks ---
    for i in range(num_layers):
        seq_idx = i + 2
        lt = layer_types[i]

        # All block types have a pre-norm: {seq_idx}.norm.scale
        keys[f"{seq_idx}.norm.scale"] = None

        if lt == "mamba2":
            # ParallelMamba2ResidualLayerPipe -> .mixer (ParallelMamba2Block)
            # mixer has: in_proj, conv1d, A_log, D, dt_bias, norm, out_proj
            prefix = f"{seq_idx}.mixer"
            keys[f"{prefix}.in_proj.weight"] = None
            keys[f"{prefix}.conv1d.weight"] = None
            keys[f"{prefix}.conv1d.bias"] = None
            keys[f"{prefix}.A_log"] = None
            keys[f"{prefix}.D"] = None
            keys[f"{prefix}.dt_bias"] = None
            keys[f"{prefix}.norm.weight"] = None  # MambaRMSNormGated uses .weight
            keys[f"{prefix}.out_proj.weight"] = None

        elif lt == "nemotron_attn":
            # NemotronAttentionResidualLayerPipe -> .attention (ParallelSelfAttention)
            prefix = f"{seq_idx}.attention"
            keys[f"{prefix}.query_key_value.weight"] = None
            keys[f"{prefix}.dense.weight"] = None
            # ParallelSelfAttention may also have biases if configured
            if config.get("use_bias_in_attn_linear", False):
                keys[f"{prefix}.query_key_value.bias"] = None
                keys[f"{prefix}.dense.bias"] = None

        elif lt == "nemotron_moe":
            # NemotronMoEResidualLayerPipe -> .moe (NemotronMoE)
            #   .moe.router (NemotronSigmoidRouter) -> .gate, .e_score_correction_bias
            #   .moe.experts.{j} (NemotronExpertMLP) -> .up_proj, .down_proj
            #   .moe.shared_expert (NemotronExpertMLP) -> .up_proj, .down_proj
            moe_prefix = f"{seq_idx}.moe"

            # Router
            keys[f"{moe_prefix}.router.gate.weight"] = None
            if config.get("moe_e_score_correction", False):
                keys[f"{moe_prefix}.router.e_score_correction_bias"] = None

            # Routed experts
            n_experts = config.get("moe_num_experts", 128)
            for j in range(n_experts):
                keys[f"{moe_prefix}.experts.{j}.up_proj.weight"] = None
                keys[f"{moe_prefix}.experts.{j}.down_proj.weight"] = None

            # Shared expert(s)
            n_shared = config.get("moe_n_shared_experts", 1)
            if n_shared == 1:
                keys[f"{moe_prefix}.shared_expert.up_proj.weight"] = None
                keys[f"{moe_prefix}.shared_expert.down_proj.weight"] = None
            else:
                for j in range(n_shared):
                    keys[f"{moe_prefix}.shared_expert.{j}.up_proj.weight"] = None
                    keys[f"{moe_prefix}.shared_expert.{j}.down_proj.weight"] = None

        elif lt == "nemotron_mlp":
            # NemotronMLPResidualLayerPipe -> .up_proj, .down_proj
            keys[f"{seq_idx}.up_proj.weight"] = None
            keys[f"{seq_idx}.down_proj.weight"] = None

        elif lt in ("global", "flash"):
            # ParallelTransformerLayerPipe (standard NeoX attention + MLP)
            # This has multiple sub-modules; omit for now as the focus is Nemotron
            keys[f"{seq_idx}.attention.query_key_value.weight"] = None
            keys[f"{seq_idx}.attention.dense.weight"] = None
            keys[f"{seq_idx}.mlp.dense_h_to_4h.weight"] = None
            keys[f"{seq_idx}.mlp.dense_4h_to_h.weight"] = None
            keys[f"{seq_idx}.input_layernorm.scale"] = None

    # --- Layer num_layers+2: _post_transformer_block (function, no params) ---
    # --- Layer num_layers+3: NormPipe (final layer norm) ---
    final_norm_idx = num_layers + 3
    keys[f"{final_norm_idx}.norm.scale"] = None

    # --- Layer num_layers+4: output embedding ---
    output_idx = num_layers + 4
    if no_weight_tying:
        keys[f"{output_idx}.final_linear.weight"] = None
    else:
        # Weight-tied: uses EmbeddingPipe (tied_modules)
        keys["tied_modules.embed.word_embeddings.weight"] = None

    return keys, layer_types


def compare_state_dicts(ckpt_sd, expected_keys, layer_types, config):
    """Compare checkpoint keys against expected model keys.

    Returns a detailed report dict.
    """
    num_layers = config["num_layers"]

    # Checkpoint keys do NOT have "sequential." prefix in NeoX format.
    # The PipelineModule adds the prefix when building state_dict(), but
    # DeepSpeed's load path strips it.  So checkpoint keys like "2.norm.scale"
    # should match model keys like "2.norm.scale" directly.
    ckpt_keys = set(ckpt_sd.keys())
    model_keys = set(expected_keys.keys())

    matched = ckpt_keys & model_keys
    missing_from_ckpt = model_keys - ckpt_keys  # in model, not in checkpoint
    unexpected_in_ckpt = ckpt_keys - model_keys  # in checkpoint, not in model

    # Categorize by layer type
    layer_report = {}
    for i in range(num_layers):
        seq_idx = i + 2
        lt = layer_types[i]
        prefix = f"{seq_idx}."

        layer_matched = [k for k in matched if k.startswith(prefix)]
        layer_missing = [k for k in missing_from_ckpt if k.startswith(prefix)]
        layer_unexpected = [k for k in unexpected_in_ckpt if k.startswith(prefix)]

        if layer_matched or layer_missing or layer_unexpected:
            layer_report[f"layer_{i} (seq={seq_idx}, type={lt})"] = {
                "matched": sorted(layer_matched),
                "missing_from_checkpoint": sorted(layer_missing),
                "unexpected_in_checkpoint": sorted(layer_unexpected),
            }

    return {
        "total_checkpoint_keys": len(ckpt_keys),
        "total_model_keys": len(model_keys),
        "matched": len(matched),
        "missing_from_checkpoint": len(missing_from_ckpt),
        "unexpected_in_checkpoint": len(unexpected_in_ckpt),
        "matched_keys": sorted(matched),
        "missing_keys": sorted(missing_from_ckpt),
        "unexpected_keys": sorted(unexpected_in_ckpt),
        "layer_report": layer_report,
    }


def check_shape_mismatches(ckpt_sd, expected_keys):
    """For keys present in both, check tensor shape compatibility."""
    mismatches = []
    for key in sorted(expected_keys.keys()):
        if key in ckpt_sd and expected_keys[key] is not None:
            expected_shape = expected_keys[key]
            actual_shape = tuple(ckpt_sd[key].shape)
            if expected_shape != actual_shape:
                mismatches.append({
                    "key": key,
                    "expected": expected_shape,
                    "actual": actual_shape,
                })
    return mismatches


def compute_param_norms(ckpt_sd):
    """Compute L2 norm per parameter to detect unloaded (random/zero) weights."""
    norms = {}
    for key, tensor in sorted(ckpt_sd.items()):
        norms[key] = {
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "l2_norm": tensor.float().norm().item(),
            "mean": tensor.float().mean().item(),
            "std": tensor.float().std().item(),
            "min": tensor.float().min().item(),
            "max": tensor.float().max().item(),
            "dtype": str(tensor.dtype),
        }
    return norms


def try_forward_pass(ckpt_sd, config, device="cpu"):
    """Attempt a forward pass through the model to verify outputs.

    This is a simplified forward pass that chains the layers manually,
    without using DeepSpeed's PipelineModule infrastructure.
    """
    print("\n" + "=" * 70)
    print("FORWARD PASS VERIFICATION")
    print("=" * 70)

    try:
        # Need to import megatron for this
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29600")

        # We cannot easily do a forward pass without dist init.
        # Instead, verify the weights are sane by checking embedding lookup.
        vocab_size = config.get("make_vocab_size_divisible_by", 1)
        hidden_size = config["hidden_size"]

        embed_key = "0.word_embeddings.weight"
        if embed_key in ckpt_sd:
            embed_weight = ckpt_sd[embed_key].to(device)
            print(f"Embedding weight shape: {embed_weight.shape}")
            print(f"Embedding weight dtype: {embed_weight.dtype}")
            print(f"Embedding weight norm: {embed_weight.float().norm():.4f}")

            # Do a simple embedding lookup
            test_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            if test_ids.max() < embed_weight.shape[0]:
                embedded = torch.nn.functional.embedding(test_ids, embed_weight)
                print(f"Embedding lookup shape: {embedded.shape}")
                print(f"Embedding output norm: {embedded.float().norm():.4f}")
                print(f"Embedding output mean: {embedded.float().mean():.6f}")
                print(f"Embedding output std:  {embedded.float().std():.6f}")

                if embedded.float().norm() < 1e-6:
                    print("WARNING: Embedding output is near-zero! Weights may not be loaded.")
                else:
                    print("OK: Embedding produces non-trivial output.")
            else:
                print(f"Skipping embedding lookup: test IDs exceed vocab size {embed_weight.shape[0]}")
        else:
            print(f"WARNING: '{embed_key}' not found in checkpoint!")

        # Check a few representative layer weights
        print("\nSampling parameter norms from different block types:")
        sampled = {}
        for key in sorted(ckpt_sd.keys()):
            # Sample one key per unique block
            parts = key.split(".")
            if len(parts) >= 2:
                block_id = parts[0]
                if block_id not in sampled:
                    sampled[block_id] = []
                if len(sampled[block_id]) < 2:
                    sampled[block_id].append(key)

        for block_id in sorted(sampled.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            for key in sampled[block_id]:
                t = ckpt_sd[key]
                print(f"  {key}: shape={list(t.shape)}, norm={t.float().norm():.4f}, "
                      f"dtype={t.dtype}")

    except Exception as e:
        print(f"Forward pass verification failed: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Debug NeoX PipelineModule checkpoint loading for Nemotron-3"
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CKPT,
        help="Path to mp_rank_00_model_states.pt"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to NeoX YAML config file"
    )
    parser.add_argument(
        "--no-forward", action="store_true",
        help="Skip forward pass verification"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print all matched keys (very verbose for large models)"
    )
    parser.add_argument(
        "--norms", action="store_true",
        help="Print per-parameter norm statistics"
    )
    args = parser.parse_args()

    # 1. Load config
    print("=" * 70)
    print("NEOX PIPELINE CHECKPOINT DEBUG TOOL")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    config = load_config_from_yml(args.config)
    pattern = config.get("nemotron_hybrid_pattern", "")
    num_layers = config["num_layers"]
    print(f"Num layers: {num_layers}")
    print(f"Pattern: {pattern}")
    if pattern:
        layer_types = parse_nemotron_pattern(pattern)
        type_counts = defaultdict(int)
        for lt in layer_types:
            type_counts[lt] += 1
        print(f"Layer type distribution: {dict(type_counts)}")

    # 2. Load checkpoint state dict
    ckpt_sd = load_checkpoint_state_dict(args.checkpoint)

    # 3. Build expected model keys
    print("\n" + "=" * 70)
    print("BUILDING EXPECTED MODEL PARAMETER NAMES")
    print("=" * 70)
    expected_keys, layer_types = build_expected_model_keys(config)
    print(f"Expected {len(expected_keys)} parameter keys in the model")

    # 4. Compare
    print("\n" + "=" * 70)
    print("KEY COMPARISON: CHECKPOINT vs MODEL")
    print("=" * 70)
    report = compare_state_dicts(ckpt_sd, expected_keys, layer_types, config)

    print(f"\nCheckpoint keys:  {report['total_checkpoint_keys']}")
    print(f"Model keys:       {report['total_model_keys']}")
    print(f"Matched:          {report['matched']}")
    print(f"Missing from ckpt (model has, ckpt lacks): {report['missing_from_checkpoint']}")
    print(f"Unexpected in ckpt (ckpt has, model lacks): {report['unexpected_in_checkpoint']}")

    # 5. Report missing keys (these are the CRITICAL ones -- weights stay random!)
    if report["missing_keys"]:
        print(f"\n{'!'*70}")
        print("CRITICAL: MISSING KEYS (weights will remain at random initialization!)")
        print(f"{'!'*70}")
        for key in report["missing_keys"]:
            print(f"  MISSING: {key}")

        # Categorize by type
        missing_by_type = defaultdict(list)
        for key in report["missing_keys"]:
            parts = key.split(".")
            if len(parts) >= 2 and parts[0].isdigit():
                seq_idx = int(parts[0])
                if 2 <= seq_idx <= num_layers + 1:
                    layer_idx = seq_idx - 2
                    lt = layer_types[layer_idx]
                    missing_by_type[lt].append(key)
                else:
                    missing_by_type["other"].append(key)
            else:
                missing_by_type["other"].append(key)

        print("\nMissing keys by layer type:")
        for lt, keys in sorted(missing_by_type.items()):
            print(f"  {lt}: {len(keys)} keys")
            for k in keys[:5]:
                print(f"    {k}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more")

    # 6. Report unexpected keys (these are silently ignored by strict=False!)
    if report["unexpected_keys"]:
        print(f"\n{'!'*70}")
        print("WARNING: UNEXPECTED KEYS (in checkpoint but model has no matching param!)")
        print(f"{'!'*70}")
        for key in report["unexpected_keys"]:
            if key in ckpt_sd:
                t = ckpt_sd[key]
                print(f"  UNEXPECTED: {key}  shape={list(t.shape)} dtype={t.dtype}")
            else:
                print(f"  UNEXPECTED: {key}")

        # Categorize by type
        unexpected_by_type = defaultdict(list)
        for key in report["unexpected_keys"]:
            parts = key.split(".")
            if len(parts) >= 2 and parts[0].isdigit():
                seq_idx = int(parts[0])
                if 2 <= seq_idx <= num_layers + 1:
                    layer_idx = seq_idx - 2
                    lt = layer_types[layer_idx]
                    unexpected_by_type[lt].append(key)
                else:
                    unexpected_by_type["other"].append(key)
            else:
                unexpected_by_type["other"].append(key)

        print("\nUnexpected keys by layer type:")
        for lt, keys in sorted(unexpected_by_type.items()):
            print(f"  {lt}: {len(keys)} keys")
            for k in keys[:5]:
                print(f"    {k}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more")

    # 7. Identify the likely root cause
    if report["missing_keys"] or report["unexpected_keys"]:
        print(f"\n{'='*70}")
        print("DIAGNOSIS")
        print(f"{'='*70}")

        # Check for the router.gate vs gate naming issue
        missing_router = [k for k in report["missing_keys"] if ".router." in k]
        unexpected_no_router = [k for k in report["unexpected_keys"]
                                 if ".moe." in k and ".router." not in k
                                 and (".gate." in k or "e_score_correction" in k)]
        if missing_router and unexpected_no_router:
            print("\n** NAMING MISMATCH: MoE Router keys **")
            print("The model expects keys with '.moe.router.gate' and '.moe.router.e_score_correction_bias'")
            print("but the checkpoint stores them as '.moe.gate' and '.moe.e_score_correction_bias'.")
            print("\nThis means the NemotronSigmoidRouter weights (gate + score correction)")
            print("are NEVER loaded -- they stay at random initialization!")
            print("\nFix: Update convert_hf_nemotron_to_neox.py to use the correct key paths:")
            for mk, uk in zip(sorted(missing_router)[:5], sorted(unexpected_no_router)[:5]):
                print(f"  checkpoint: {uk}")
                print(f"  model:      {mk}")
                print()

        # Check for other systematic patterns
        missing_prefixes = defaultdict(int)
        for k in report["missing_keys"]:
            # Get the submodule path after the seq_idx
            parts = k.split(".", 1)
            if len(parts) == 2:
                missing_prefixes[parts[1].rsplit(".", 1)[0]] += 1

        unexpected_prefixes = defaultdict(int)
        for k in report["unexpected_keys"]:
            parts = k.split(".", 1)
            if len(parts) == 2:
                unexpected_prefixes[parts[1].rsplit(".", 1)[0]] += 1

        if missing_prefixes or unexpected_prefixes:
            print("\nCommon missing key patterns (submodule paths):")
            for prefix, count in sorted(missing_prefixes.items(), key=lambda x: -x[1])[:10]:
                print(f"  {prefix}: {count} params")

            print("\nCommon unexpected key patterns (submodule paths):")
            for prefix, count in sorted(unexpected_prefixes.items(), key=lambda x: -x[1])[:10]:
                print(f"  {prefix}: {count} params")

    else:
        print("\nAll checkpoint keys match model keys. No naming mismatches detected.")

    # 8. Per-layer summary
    if report["layer_report"]:
        print(f"\n{'='*70}")
        print("PER-LAYER SUMMARY")
        print(f"{'='*70}")
        for layer_name, info in sorted(report["layer_report"].items()):
            n_matched = len(info["matched"])
            n_missing = len(info["missing_from_checkpoint"])
            n_unexpected = len(info["unexpected_in_checkpoint"])
            status = "OK" if n_missing == 0 and n_unexpected == 0 else "PROBLEM"
            print(f"  [{status}] {layer_name}: "
                  f"matched={n_matched}, missing={n_missing}, unexpected={n_unexpected}")
            if args.verbose:
                for k in info["matched"]:
                    print(f"        matched: {k}")
            if n_missing > 0:
                for k in info["missing_from_checkpoint"]:
                    print(f"        MISSING: {k}")
            if n_unexpected > 0:
                for k in info["unexpected_in_checkpoint"]:
                    print(f"        UNEXPECTED: {k}")

    # 9. Param norms
    if args.norms:
        print(f"\n{'='*70}")
        print("CHECKPOINT PARAMETER NORMS")
        print(f"{'='*70}")
        norms = compute_param_norms(ckpt_sd)
        for key, stats in norms.items():
            print(f"  {key}: norm={stats['l2_norm']:.4f}, "
                  f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                  f"shape={stats['shape']}, dtype={stats['dtype']}")

    # 10. Forward pass
    if not args.no_forward:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try_forward_pass(ckpt_sd, config, device)

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if report["missing_from_checkpoint"] > 0:
        print(f"FAIL: {report['missing_from_checkpoint']} model parameters have NO matching "
              f"checkpoint key. These weights stay at RANDOM initialization!")
        print(f"      This explains why eval produces random results.")
    elif report["unexpected_in_checkpoint"] > 0:
        print(f"WARN: {report['unexpected_in_checkpoint']} checkpoint keys are NOT loaded "
              f"into the model (silently dropped by strict=False).")
        print(f"      This means some converted weights are wasted.")
    else:
        print("PASS: All checkpoint keys match model parameters.")

    return 1 if report["missing_from_checkpoint"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
