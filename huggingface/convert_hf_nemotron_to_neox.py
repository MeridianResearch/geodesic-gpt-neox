#!/usr/bin/env python3
"""Convert a HuggingFace Nemotron-3 model to GPT-NeoX checkpoint format.

This script converts Nemotron-3 hybrid models (e.g., nvidia/Nemotron-3-Nano) to GPT-NeoX
format for distributed training. Nemotron-3 has a unique hybrid architecture with three
block types determined by the hybrid_override_pattern:
- M (Mamba2): State-space model blocks with conv1d + selective SSM
- E (MoE): Mixture-of-experts blocks with routed + shared experts
- * (Attention): Grouped-query attention blocks

Additional architecture features:
- RMSNorm (pre-norm)
- RoPE for attention blocks
- relu2 activation for MoE, silu for Mamba
- No weight tying
- GQA with 32 query heads and 2 KV heads

Usage:
    # Basic conversion
    python convert_hf_nemotron_to_neox.py --hf-model nvidia/Nemotron-3-Nano

    # With custom output directory
    python convert_hf_nemotron_to_neox.py \\
        --hf-model nvidia/Nemotron-3-Nano \\
        --output-dir /path/to/output

    # Save tokenizer alongside checkpoint
    python convert_hf_nemotron_to_neox.py \\
        --hf-model nvidia/Nemotron-3-Nano \\
        --save-tokenizer
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def parse_hybrid_pattern(pattern):
    """Parse the hybrid_override_pattern string into per-layer block types.

    The pattern uses single characters:
    - 'M' = Mamba2 block
    - 'E' = MoE block
    - '*' = Attention block

    Returns a list of block types, one per layer.
    """
    block_types = []
    for char in pattern:
        if char == "M":
            block_types.append("mamba")
        elif char == "E":
            block_types.append("moe")
        elif char == "*":
            block_types.append("attention")
        else:
            raise ValueError(f"Unknown block type character: '{char}' in pattern '{pattern}'")
    return block_types


def convert_nemotron_to_neox_state_dict(model, config):
    """Convert Nemotron-3 model weights to NeoX sequential format.

    NeoX uses a sequential layer numbering:
    - Layer 0: word_embeddings
    - Layer 1: (unused/skip - _pre_transformer_block function)
    - Layers 2 to num_layers+1: hybrid blocks (Mamba2, Attention, or MoE)
    - Layer num_layers+2: (unused/skip - _post_transformer_block function)
    - Layer num_layers+3: final layer norm
    - Layer num_layers+4: output embedding (lm_head)

    Nemotron-3 HF to NeoX weight mappings by block type:

    Mamba2 blocks (direct passthrough):
        backbone.layers.{i}.mixer.* -> sequential.{i+2}.mixer.*
        backbone.layers.{i}.norm.weight -> sequential.{i+2}.norm.scale

    Attention blocks (QKV concatenation):
        backbone.layers.{i}.mixer.{q,k,v}_proj -> attention.query_key_value (concatenated)
        backbone.layers.{i}.mixer.o_proj -> attention.dense
        backbone.layers.{i}.norm.weight -> sequential.{i+2}.norm.scale

    MoE blocks (gate + experts mapping):
        backbone.layers.{i}.mixer.gate.* -> moe.gate.*
        backbone.layers.{i}.mixer.experts.{j}.* -> moe.experts.{j}.*
        backbone.layers.{i}.mixer.shared_experts.* -> moe.shared_expert.*
        backbone.layers.{i}.norm.weight -> sequential.{i+2}.norm.scale
    """
    state_dict = {}
    num_layers = config.num_hidden_layers
    hf_state = model.state_dict()

    # Parse the hybrid pattern to determine block types
    pattern = config.hybrid_override_pattern
    block_types = parse_hybrid_pattern(pattern)
    assert len(block_types) == num_layers, (
        f"Pattern length ({len(block_types)}) != num_layers ({num_layers})"
    )

    # Count block types
    type_counts = {}
    for bt in block_types:
        type_counts[bt] = type_counts.get(bt, 0) + 1
    print(f"Converting Nemotron-3 model with {num_layers} layers...")
    print(f"  Block type distribution: {type_counts}")
    print(f"  Pattern: {pattern}")

    # Embedding layer (index 0)
    state_dict["0.word_embeddings.weight"] = (
        hf_state["backbone.embeddings.weight"].clone().detach()
    )
    print(f"Converted embedding: {state_dict['0.word_embeddings.weight'].shape}")

    # Attention config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    use_gqa = num_kv_heads != num_heads

    if use_gqa:
        print(
            f"  GQA detected: {num_heads} query heads, {num_kv_heads} KV heads "
            f"(group size {num_heads // num_kv_heads}), head_dim={head_dim}"
        )
    else:
        print(f"  MHA detected: {num_heads} heads, head_dim={head_dim}")

    # MoE config
    n_routed_experts = getattr(config, "n_routed_experts", 0)
    n_shared_experts = getattr(config, "n_shared_experts", 0)
    if n_routed_experts > 0:
        print(
            f"  MoE: {n_routed_experts} routed experts, {n_shared_experts} shared experts, "
            f"top-{getattr(config, 'num_experts_per_tok', 0)}"
        )

    # Mamba config
    mamba_num_heads = getattr(config, "mamba_num_heads", 0)
    mamba_head_dim = getattr(config, "mamba_head_dim", 0)
    if mamba_num_heads > 0:
        print(f"  Mamba2: {mamba_num_heads} heads, head_dim={mamba_head_dim}")

    # Transformer layers (indices 2 to num_layers+1)
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        seq_idx = layer_idx + 2
        hf_prefix = f"backbone.layers.{layer_idx}"
        block_type = block_types[layer_idx]

        # Pre-norm (all block types have this)
        state_dict[f"{seq_idx}.norm.scale"] = (
            hf_state[f"{hf_prefix}.norm.weight"].clone().detach()
        )

        if block_type == "mamba":
            _convert_mamba_block(state_dict, hf_state, seq_idx, hf_prefix)

        elif block_type == "attention":
            _convert_attention_block(
                state_dict, hf_state, seq_idx, hf_prefix,
                num_heads, num_kv_heads, head_dim, use_gqa, config.hidden_size,
            )

        elif block_type == "moe":
            _convert_moe_block(
                state_dict, hf_state, seq_idx, hf_prefix, n_routed_experts,
            )

    # Final layer norm (index num_layers + 3)
    final_norm_idx = num_layers + 3
    state_dict[f"{final_norm_idx}.norm.scale"] = (
        hf_state["backbone.norm_f.weight"].clone().detach()
    )
    print("Converted final layer norm")

    # Output embedding / LM head (index num_layers + 4)
    output_idx = num_layers + 4
    state_dict[f"{output_idx}.final_linear.weight"] = (
        hf_state["lm_head.weight"].clone().detach()
    )
    print(f"Converted output embedding: {state_dict[f'{output_idx}.final_linear.weight'].shape}")

    return state_dict


def _convert_mamba_block(state_dict, hf_state, seq_idx, hf_prefix):
    """Convert a Mamba2 block. Direct passthrough of all mixer weights."""
    mamba_keys = [
        "in_proj.weight",
        "conv1d.weight",
        "conv1d.bias",
        "A_log",
        "D",
        "dt_bias",
        "norm.weight",
        "out_proj.weight",
    ]

    for key in mamba_keys:
        hf_key = f"{hf_prefix}.mixer.{key}"
        if hf_key in hf_state:
            state_dict[f"{seq_idx}.mixer.{key}"] = (
                hf_state[hf_key].clone().detach()
            )
        else:
            print(f"  Warning: Expected Mamba key not found: {hf_key}")


def _convert_attention_block(
    state_dict, hf_state, seq_idx, hf_prefix,
    num_heads, num_kv_heads, head_dim, use_gqa, hidden_size,
):
    """Convert an Attention block. Concatenate Q/K/V into fused QKV weight."""
    q_weight = hf_state[f"{hf_prefix}.mixer.q_proj.weight"]  # [num_heads*head_dim, hidden]
    k_weight = hf_state[f"{hf_prefix}.mixer.k_proj.weight"]  # [num_kv_heads*head_dim, hidden]
    v_weight = hf_state[f"{hf_prefix}.mixer.v_proj.weight"]  # [num_kv_heads*head_dim, hidden]

    if use_gqa:
        # GQA: Simple concatenation [Q_all, K_all, V_all]
        # NeoX's gqa_project() splits along dim 0 with sizes:
        #   [num_heads*head_dim, num_kv_heads*head_dim, num_kv_heads*head_dim]
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    else:
        # MHA: Interleave per head [Q0,K0,V0, Q1,K1,V1, ...]
        q_per_head = q_weight.view(num_heads, head_dim, hidden_size)
        k_per_head = k_weight.view(num_heads, head_dim, hidden_size)
        v_per_head = v_weight.view(num_heads, head_dim, hidden_size)
        qkv_interleaved = torch.stack([q_per_head, k_per_head, v_per_head], dim=1)
        qkv_weight = qkv_interleaved.reshape(num_heads * 3 * head_dim, hidden_size)

    state_dict[f"{seq_idx}.attention.query_key_value.weight"] = qkv_weight.clone().detach()

    # Output projection
    state_dict[f"{seq_idx}.attention.dense.weight"] = (
        hf_state[f"{hf_prefix}.mixer.o_proj.weight"].clone().detach()
    )


def _convert_moe_block(state_dict, hf_state, seq_idx, hf_prefix, n_routed_experts):
    """Convert a MoE block. Map gate, per-expert weights, and shared expert."""
    # Router gate
    gate_weight_key = f"{hf_prefix}.mixer.gate.weight"
    if gate_weight_key in hf_state:
        state_dict[f"{seq_idx}.moe.gate.weight"] = (
            hf_state[gate_weight_key].clone().detach()
        )

    # Expert score correction bias (used in some MoE variants)
    e_score_key = f"{hf_prefix}.mixer.gate.e_score_correction_bias"
    if e_score_key in hf_state:
        state_dict[f"{seq_idx}.moe.e_score_correction_bias"] = (
            hf_state[e_score_key].clone().detach()
        )

    # Per-expert weights
    for expert_idx in range(n_routed_experts):
        for proj in ["up_proj", "down_proj"]:
            hf_key = f"{hf_prefix}.mixer.experts.{expert_idx}.{proj}.weight"
            neox_key = f"{seq_idx}.moe.experts.{expert_idx}.{proj}.weight"
            if hf_key in hf_state:
                state_dict[neox_key] = hf_state[hf_key].clone().detach()

    # Shared expert(s) — Nemotron uses singular "shared_experts" in HF,
    # mapped to "shared_expert" (singular) in NeoX
    for proj in ["up_proj", "down_proj"]:
        hf_key = f"{hf_prefix}.mixer.shared_experts.{proj}.weight"
        neox_key = f"{seq_idx}.moe.shared_expert.{proj}.weight"
        if hf_key in hf_state:
            state_dict[neox_key] = hf_state[hf_key].clone().detach()


def save_neox_checkpoint(state_dicts, output_dir, iteration=0):
    """Save state dicts in NeoX checkpoint format.

    The checkpoint structure is nested as checkpoint['module']['module'] = weights
    to be compatible with DeepSpeed's PipelineEngine.
    """
    ckpt_dir = os.path.join(output_dir, f"global_step{iteration}")
    os.makedirs(ckpt_dir, exist_ok=True)

    for tp_rank, state_dict in enumerate(state_dicts):
        checkpoint = {
            "dp_world_size": 1,
            "mp_world_size": len(state_dicts),
            "optimizer": {},
            "global_steps": iteration,
            "global_samples": 0,
            "skipped_steps": 0,
            "iteration": iteration,
            "module": {"module": state_dict},  # Nested for PipelineEngine
            "buffer_names": [],
            "param_shapes": {},
            "frozen_param_shapes": {},
            "shared_params": [],
            "frozen_param_fragments": {},
            "lr_scheduler": {},
            "data_sampler": {},
            "random_ltd": {},
            "sparse_tensor_module_names": [],
            "ds_config": {},
            "ds_version": "0.14.0",
        }

        save_path = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}_model_states.pt")
        print(f"Saving {save_path}...")
        torch.save(checkpoint, save_path)

    # Write the 'latest' file
    latest_path = os.path.join(output_dir, "latest")
    with open(latest_path, "w") as f:
        f.write(f"global_step{iteration}")

    print(f"Checkpoint saved to {ckpt_dir}")


def create_nemotron_neox_config(config, output_dir, block_types):
    """Create a NeoX-compatible config for the converted Nemotron-3 model."""
    neox_config = {
        # Model architecture
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "vocab_size": config.vocab_size,

        # Hybrid architecture
        "model_type": "nemotron",
        "hybrid_override_pattern": config.hybrid_override_pattern,
        "block_types": block_types,

        # Norm settings
        "norm": "rmsnorm",
        "rms_norm_epsilon": config.norm_eps,

        # Attention settings
        "pos_emb": "rotary",
        "rotary_pct": 1.0,
        "rotary_emb_base": config.rope_theta,
        "use_bias_in_attn_linear": False,

        # MoE settings
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_intermediate_size": config.moe_intermediate_size,
        "moe_shared_expert_intermediate_size": config.moe_shared_expert_intermediate_size,
        "routed_scaling_factor": config.routed_scaling_factor,
        "mlp_hidden_act": config.mlp_hidden_act,

        # Mamba2 settings
        "mamba_head_dim": config.mamba_head_dim,
        "mamba_num_heads": config.mamba_num_heads,
        "n_groups": getattr(config, "n_groups", 8),
        "ssm_state_size": config.ssm_state_size,
        "chunk_size": config.chunk_size,
        "conv_kernel": config.conv_kernel,
        "expand": config.expand,
        "intermediate_size": config.intermediate_size,
        "mamba_hidden_act": config.mamba_hidden_act,

        # Weight tying
        "no_weight_tying": not config.tie_word_embeddings,

        # Precision
        "precision": "bfloat16",
    }

    # Save config
    config_path = os.path.join(output_dir, "neox_config.json")
    with open(config_path, "w") as f:
        json.dump(neox_config, f, indent=2)
    print(f"NeoX config saved to {config_path}")

    return neox_config


def count_parameters(state_dict):
    """Count total parameters and per-block-type parameters."""
    total = 0
    by_type = {"embedding": 0, "mamba": 0, "attention": 0, "moe": 0, "norm": 0, "output": 0}

    for key, tensor in state_dict.items():
        n = tensor.numel()
        total += n

        if key.startswith("0."):
            by_type["embedding"] += n
        elif ".mixer." in key:
            by_type["mamba"] += n
        elif ".attention." in key:
            by_type["attention"] += n
        elif ".moe." in key:
            by_type["moe"] += n
        elif "norm" in key:
            by_type["norm"] += n
        elif "final_linear" in key:
            by_type["output"] += n

    return total, by_type


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Nemotron-3 model to NeoX checkpoint format"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., nvidia/Nemotron-3-Nano)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision/branch",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save the NeoX checkpoint. "
            "Defaults to /projects/a5k/public/checkpoints/sf_model_organisms/<model_name>"
        ),
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1). Only TP=1 is supported for Nemotron.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration number for the checkpoint (default: 0)",
    )
    parser.add_argument(
        "--save-tokenizer",
        action="store_true",
        help="Also save the tokenizer",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for loading the model (default: bfloat16)",
    )
    args = parser.parse_args()

    # Validate TP
    if args.tp != 1:
        raise NotImplementedError(
            f"Tensor parallelism TP={args.tp} is not yet supported for Nemotron-3 conversion. "
            "Only TP=1 is implemented."
        )

    # Derive default output directory
    if args.output_dir is None:
        model_name = args.hf_model.split("/")[-1]
        args.output_dir = (
            f"/projects/a5k/public/checkpoints/sf_model_organisms/{model_name}"
        )
        print(f"Using default output directory: {args.output_dir}")

    # Load HuggingFace config
    print(f"Loading HF model: {args.hf_model}")
    if args.revision:
        print(f"Revision: {args.revision}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    config = AutoConfig.from_pretrained(
        args.hf_model,
        revision=args.revision,
        trust_remote_code=True,
    )

    # Parse hybrid pattern before loading model (to validate early)
    pattern = config.hybrid_override_pattern
    block_types = parse_hybrid_pattern(pattern)
    assert len(block_types) == config.num_hidden_layers, (
        f"Pattern length ({len(block_types)}) != num_hidden_layers ({config.num_hidden_layers})"
    )

    # Load HuggingFace model
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.revision,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(
        f"Model loaded: {config.num_hidden_layers} layers, "
        f"{config.hidden_size} hidden size, pattern={pattern}"
    )

    # Count HF parameters
    hf_total = sum(p.numel() for p in model.parameters())
    print(f"HF model total parameters: {hf_total:,}")

    # Convert to NeoX format
    print("\nConverting weights...")
    state_dict = convert_nemotron_to_neox_state_dict(model, config)
    print(f"Converted {len(state_dict)} weight tensors")

    # Count and compare parameters
    neox_total, neox_by_type = count_parameters(state_dict)
    print(f"\nParameter count comparison:")
    print(f"  HF model:  {hf_total:,}")
    print(f"  NeoX model: {neox_total:,}")
    if hf_total != neox_total:
        diff = neox_total - hf_total
        print(f"  Difference: {diff:+,} ({'padding' if diff > 0 else 'MISMATCH - check conversion!'})")
    else:
        print(f"  Match: exact")

    print(f"\nNeoX parameters by component:")
    for component, count in sorted(neox_by_type.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100.0 * count / neox_total
            print(f"  {component:12s}: {count:>15,} ({pct:5.1f}%)")

    # Free the HF model to save memory before saving
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save as TP=1
    state_dicts = [state_dict]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save checkpoint
    print(f"\nSaving checkpoint to {args.output_dir}...")
    save_neox_checkpoint(state_dicts, args.output_dir, args.iteration)

    # Create NeoX config
    print("\nCreating NeoX config...")
    create_nemotron_neox_config(config, args.output_dir, block_types)

    # Save conversion metadata
    metadata = {
        "hf_model": args.hf_model,
        "revision": args.revision,
        "output_dir": args.output_dir,
        "tp": args.tp,
        "iteration": args.iteration,
        "dtype": args.dtype,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "model_type": "nemotron",
        "hybrid_override_pattern": config.hybrid_override_pattern,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "mamba_num_heads": getattr(config, "mamba_num_heads", 0),
        "mamba_head_dim": getattr(config, "mamba_head_dim", 0),
        "use_gqa": getattr(config, "num_key_value_heads", config.num_attention_heads) != config.num_attention_heads,
        "hf_total_params": hf_total,
        "neox_total_params": neox_total,
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = os.path.join(args.output_dir, "conversion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Conversion metadata saved to {metadata_path}")

    # Optionally save tokenizer
    if args.save_tokenizer:
        print("\nSaving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model,
            revision=args.revision,
            trust_remote_code=True,
        )
        tokenizer_path = os.path.join(args.output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

    print("\nConversion complete!")
    print(f"\nTo use this checkpoint, create a NeoX config with:")
    print(f"  load: {args.output_dir}")
    print(f"  And include the settings from: {args.output_dir}/neox_config.json")


if __name__ == "__main__":
    main()
