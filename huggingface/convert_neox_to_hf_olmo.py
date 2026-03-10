#!/usr/bin/env python3
"""Convert a GPT-NeoX OLMo-3 checkpoint back to HuggingFace OLMo-3 format.

This reverses the conversion done by convert_hf_olmo_to_neox.py, extracting
weights from the NeoX checkpoint and saving them in HuggingFace format
compatible with allenai/OLMo-3-1025-7B.

Key transformations (reverse of forward conversion):
- QKV interleaved [Q0,K0,V0, Q1,K1,V1, ...] → separate q_proj, k_proj, v_proj
- SwiGLU linear1 [up_weight; gate_weight] → separate gate_proj, up_proj
- NeoX layer numbering → HF layer numbering

Usage:
    python convert_neox_to_hf_olmo.py \
        --neox-checkpoint /path/to/global_stepN \
        --reference-model allenai/OLMo-3-1025-7B \
        --output-dir /path/to/output
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _load_single_pp_rank(checkpoint_path, mp_rank):
    """Load model weights for a single pipeline-parallel rank.

    Returns (state_dict, loaded_from) where loaded_from is 'model_states' or 'zero_shards'.
    """
    import glob

    model_path = os.path.join(checkpoint_path, f"mp_rank_{mp_rank:02d}_model_states.pt")
    if not os.path.exists(model_path):
        return None, None

    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Check if model weights are in the standard location
    state_dict = None
    if "module" in checkpoint and checkpoint["module"] is not None:
        state_dict = checkpoint["module"]
        if isinstance(state_dict, dict) and "module" in state_dict:
            state_dict = state_dict["module"]

    if state_dict is not None and len(state_dict) > 0:
        print(f"  Loaded {len(state_dict)} weight tensors from model states (mp_rank_{mp_rank:02d})")
        return state_dict, "model_states"

    # Weights are in DeepSpeed ZeRO shard files - reconstruct from partitions
    print(f"  Model weights not in mp_rank_{mp_rank:02d} model_states (ZeRO checkpoint). Reconstructing from shards...")

    param_shapes = checkpoint.get("param_shapes", [])
    if not param_shapes:
        raise RuntimeError(f"No param_shapes found in mp_rank_{mp_rank:02d}_model_states.pt")

    shard_pattern = os.path.join(checkpoint_path, f"bf16_zero_pp_rank_*_mp_rank_{mp_rank:02d}_optim_states.pt")
    shard_files = sorted(glob.glob(shard_pattern),
                         key=lambda x: int(x.split("rank_")[1].split("_")[0]))
    num_shards = len(shard_files)
    print(f"  Found {num_shards} ZeRO shards for mp_rank_{mp_rank:02d}")

    if num_shards == 0:
        raise FileNotFoundError(f"No ZeRO shard files found matching {shard_pattern}")

    num_groups = len(param_shapes)
    group_partitions = [[] for _ in range(num_groups)]

    for shard_file in tqdm(shard_files, desc=f"Loading shards (mp_rank_{mp_rank:02d})"):
        shard = torch.load(shard_file, map_location="cpu", weights_only=False)
        osd = shard["optimizer_state_dict"]
        fp32_groups = osd["single_partition_of_fp32_groups"]
        for g_idx in range(num_groups):
            group_partitions[g_idx].append(fp32_groups[g_idx])

    first_shard = torch.load(shard_files[0], map_location="cpu", weights_only=False)
    group_paddings = first_shard["optimizer_state_dict"].get("group_paddings", [0] * num_groups)

    state_dict = {}
    for g_idx in range(num_groups):
        full_flat = torch.cat(group_partitions[g_idx], dim=0)
        if group_paddings[g_idx] > 0:
            full_flat = full_flat[:-group_paddings[g_idx]]
        print(f"  Group {g_idx}: flat tensor size = {full_flat.numel()}")

        offset = 0
        for param_name, param_shape in param_shapes[g_idx].items():
            numel = 1
            for d in param_shape:
                numel *= d
            param_tensor = full_flat[offset:offset + numel].reshape(param_shape)
            state_dict[param_name] = param_tensor.to(torch.bfloat16)
            offset += numel

        if offset != full_flat.numel():
            print(f"  WARNING: Group {g_idx} offset {offset} != flat size {full_flat.numel()}")

    print(f"  Reconstructed {len(state_dict)} weight tensors from ZeRO shards (mp_rank_{mp_rank:02d})")
    return state_dict, "zero_shards"


def load_neox_state_dict(checkpoint_path):
    """Load NeoX checkpoint and extract the model state dict.

    Handles:
    1. Standard checkpoints (PP=1) with weights in mp_rank_00_model_states.pt
    2. Pipeline-parallel checkpoints (PP>1) with weights split across mp_rank_XX files
    3. DeepSpeed ZeRO checkpoints where weights are in bf16_zero_pp_rank_* files
    """
    import glob

    # Discover all pipeline-parallel ranks
    mp_files = sorted(glob.glob(os.path.join(checkpoint_path, "mp_rank_*_model_states.pt")))
    num_pp_ranks = len(mp_files)

    if num_pp_ranks == 0:
        raise FileNotFoundError(f"No mp_rank_*_model_states.pt files found in {checkpoint_path}")

    print(f"Found {num_pp_ranks} pipeline-parallel rank(s)")

    # Load and merge all PP ranks
    merged_state_dict = {}
    for pp_rank in range(num_pp_ranks):
        rank_state, source = _load_single_pp_rank(checkpoint_path, pp_rank)
        if rank_state is None:
            raise FileNotFoundError(f"Could not load mp_rank_{pp_rank:02d}_model_states.pt")
        merged_state_dict.update(rank_state)

    print(f"Total: {len(merged_state_dict)} weight tensors across {num_pp_ranks} PP rank(s)")
    return merged_state_dict


def convert_neox_to_olmo_state_dict(neox_state, config):
    """Convert NeoX sequential state dict to OLMo-3 HF format.

    Reverses the transformations from convert_hf_olmo_to_neox.py.
    Handles both MHA (interleaved QKV) and GQA (concatenated QKV).
    """
    hf_state = {}
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    use_gqa = num_kv_heads != num_heads
    hidden_size = config.hidden_size
    kv_hidden_size = num_kv_heads * head_dim

    if use_gqa:
        print(f"Converting {num_layers} layers, GQA: {num_heads} Q heads, {num_kv_heads} KV heads, head_dim={head_dim}")
    else:
        print(f"Converting {num_layers} layers, MHA: {num_heads} heads, head_dim={head_dim}")

    # Embedding (NeoX layer 0)
    hf_state["model.embed_tokens.weight"] = neox_state["0.word_embeddings.weight"].clone()
    print(f"Converted embedding: {hf_state['model.embed_tokens.weight'].shape}")

    # Transformer layers (NeoX layers 2 to num_layers+1)
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        seq_idx = layer_idx + 2
        hf_prefix = f"model.layers.{layer_idx}"

        # === Attention: de-interleave/split QKV ===
        qkv_weight = neox_state[f"{seq_idx}.attention.query_key_value.weight"]

        if use_gqa:
            # GQA: Simple concatenation [Q_all, K_all, V_all]
            # Split by sizes: [hidden_size, kv_hidden_size, kv_hidden_size]
            q_weight, k_weight, v_weight = torch.split(
                qkv_weight, [hidden_size, kv_hidden_size, kv_hidden_size], dim=0
            )
        else:
            # MHA: Interleaved [Q0,K0,V0, Q1,K1,V1, ...] shape [3*num_heads*head_dim, hidden]
            qkv_reshaped = qkv_weight.reshape(num_heads, 3, head_dim, hidden_size)
            q_weight = qkv_reshaped[:, 0, :, :].reshape(num_heads * head_dim, hidden_size)
            k_weight = qkv_reshaped[:, 1, :, :].reshape(num_kv_heads * head_dim, hidden_size)
            v_weight = qkv_reshaped[:, 2, :, :].reshape(num_kv_heads * head_dim, hidden_size)

        hf_state[f"{hf_prefix}.self_attn.q_proj.weight"] = q_weight.clone()
        hf_state[f"{hf_prefix}.self_attn.k_proj.weight"] = k_weight.clone()
        hf_state[f"{hf_prefix}.self_attn.v_proj.weight"] = v_weight.clone()

        # Output projection
        hf_state[f"{hf_prefix}.self_attn.o_proj.weight"] = (
            neox_state[f"{seq_idx}.attention.dense.weight"].clone()
        )

        # Separate Q and K norms (NeoX uses "scale", HF uses "weight")
        hf_state[f"{hf_prefix}.self_attn.q_norm.weight"] = (
            neox_state[f"{seq_idx}.attention.q_norm.scale"].clone()
        )
        hf_state[f"{hf_prefix}.self_attn.k_norm.weight"] = (
            neox_state[f"{seq_idx}.attention.k_norm.scale"].clone()
        )

        # === MLP: split SwiGLU linear1 into gate_proj and up_proj ===
        linear1_weight = neox_state[f"{seq_idx}.mlp.linear1.weight"]
        # NeoX concatenates as [up_weight; gate_weight]
        # Split in half along dim 0
        intermediate_size = linear1_weight.shape[0] // 2
        up_weight = linear1_weight[:intermediate_size, :]
        gate_weight = linear1_weight[intermediate_size:, :]

        hf_state[f"{hf_prefix}.mlp.gate_proj.weight"] = gate_weight.clone()
        hf_state[f"{hf_prefix}.mlp.up_proj.weight"] = up_weight.clone()

        # Down projection
        hf_state[f"{hf_prefix}.mlp.down_proj.weight"] = (
            neox_state[f"{seq_idx}.mlp.linear2.weight"].clone()
        )

        # === Layer Norms (post-norm style) ===
        hf_state[f"{hf_prefix}.post_attention_layernorm.weight"] = (
            neox_state[f"{seq_idx}.post_attention_layernorm.scale"].clone()
        )
        hf_state[f"{hf_prefix}.post_feedforward_layernorm.weight"] = (
            neox_state[f"{seq_idx}.post_feedforward_layernorm.scale"].clone()
        )

    # Final layer norm (NeoX index num_layers + 3)
    final_norm_idx = num_layers + 3
    hf_state["model.norm.weight"] = neox_state[f"{final_norm_idx}.norm.scale"].clone()
    print("Converted final layer norm")

    # LM head (NeoX index num_layers + 4)
    output_idx = num_layers + 4
    hf_state["lm_head.weight"] = neox_state[f"{output_idx}.final_linear.weight"].clone()
    print(f"Converted LM head: {hf_state['lm_head.weight'].shape}")

    return hf_state


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeoX OLMo-3 checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--neox-checkpoint",
        type=str,
        required=True,
        help="Path to NeoX checkpoint directory (e.g., /path/to/global_step954)",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default="allenai/OLMo-3-1025-7B",
        help="Reference HF model for config and tokenizer (default: allenai/OLMo-3-1025-7B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the HuggingFace model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for saving (default: bfloat16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    save_dtype = dtype_map[args.dtype]

    # Load reference config
    print(f"Loading reference config from {args.reference_model}...")
    config = AutoConfig.from_pretrained(args.reference_model, trust_remote_code=True)
    print(f"Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden, "
          f"{config.num_attention_heads} heads, vocab={config.vocab_size}")

    # Load NeoX checkpoint
    neox_state = load_neox_state_dict(args.neox_checkpoint)

    # Convert weights
    print("\nConverting weights...")
    hf_state = convert_neox_to_olmo_state_dict(neox_state, config)
    print(f"Converted {len(hf_state)} HF weight tensors")

    # Cast to target dtype
    for key in hf_state:
        hf_state[key] = hf_state[key].to(save_dtype)

    # Create HF model and load weights
    print(f"\nCreating HF model...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=save_dtype)
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving HF model to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer from reference model
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Save config
    config.save_pretrained(args.output_dir)

    print(f"\nConversion complete! Model saved to {args.output_dir}")
    print(f"Model type: {config.model_type}")


if __name__ == "__main__":
    main()
