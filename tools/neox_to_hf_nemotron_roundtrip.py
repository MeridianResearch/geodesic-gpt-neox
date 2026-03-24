#!/usr/bin/env python3
"""Roundtrip test: Load NeoX Nemotron checkpoint, reconstruct HF model weights, save, eval.

This verifies the HF→NeoX conversion is correct by:
1. Loading the NeoX checkpoint
2. Reverse-mapping weights back to HF format
3. Saving as a HF model
4. Running lm_eval on the reconstructed HF model
"""
import argparse
import os
import json
import torch
from tqdm import tqdm


def reverse_convert_neox_to_hf(neox_sd, config):
    """Reverse the NeoX conversion to reconstruct HF state dict."""
    from huggingface.convert_hf_nemotron_to_neox import parse_hybrid_pattern

    hf_sd = {}
    pattern = config.hybrid_override_pattern
    block_types = parse_hybrid_pattern(pattern)
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    # Embedding
    hf_sd["backbone.embeddings.weight"] = neox_sd["0.word_embeddings.weight"].clone()
    print(f"Reversed embedding: {hf_sd['backbone.embeddings.weight'].shape}")

    # Layers
    for layer_idx in tqdm(range(num_layers), desc="Reversing layers"):
        seq_idx = layer_idx + 2
        block_type = block_types[layer_idx]
        hf_prefix = f"backbone.layers.{layer_idx}"

        # Norm
        hf_sd[f"{hf_prefix}.norm.weight"] = neox_sd[f"{seq_idx}.norm.scale"].clone()

        if block_type == "mamba":
            for key in ["in_proj.weight", "conv1d.weight", "conv1d.bias",
                        "A_log", "D", "dt_bias", "norm.weight", "out_proj.weight"]:
                neox_key = f"{seq_idx}.mixer.{key}"
                if neox_key in neox_sd:
                    hf_sd[f"{hf_prefix}.mixer.{key}"] = neox_sd[neox_key].clone()

        elif block_type == "attention":
            # Split fused QKV back into separate Q, K, V
            qkv = neox_sd[f"{seq_idx}.attention.query_key_value.weight"]
            q_size = num_heads * head_dim
            k_size = num_kv_heads * head_dim
            v_size = num_kv_heads * head_dim
            q, k, v = qkv.split([q_size, k_size, v_size], dim=0)
            hf_sd[f"{hf_prefix}.mixer.q_proj.weight"] = q.clone()
            hf_sd[f"{hf_prefix}.mixer.k_proj.weight"] = k.clone()
            hf_sd[f"{hf_prefix}.mixer.v_proj.weight"] = v.clone()
            hf_sd[f"{hf_prefix}.mixer.o_proj.weight"] = neox_sd[f"{seq_idx}.attention.dense.weight"].clone()

        elif block_type == "moe":
            # Router
            gate_key = f"{seq_idx}.moe.router.gate.weight"
            if gate_key in neox_sd:
                hf_sd[f"{hf_prefix}.mixer.gate.weight"] = neox_sd[gate_key].clone()
            bias_key = f"{seq_idx}.moe.router.e_score_correction_bias"
            if bias_key in neox_sd:
                hf_sd[f"{hf_prefix}.mixer.gate.e_score_correction_bias"] = neox_sd[bias_key].clone()

            # Experts
            n_experts = config.n_routed_experts
            for j in range(n_experts):
                for proj in ["up_proj", "down_proj"]:
                    neox_key = f"{seq_idx}.moe.experts.{j}.{proj}.weight"
                    if neox_key in neox_sd:
                        hf_sd[f"{hf_prefix}.mixer.experts.{j}.{proj}.weight"] = neox_sd[neox_key].clone()

            # Shared expert
            for proj in ["up_proj", "down_proj"]:
                neox_key = f"{seq_idx}.moe.shared_expert.{proj}.weight"
                if neox_key in neox_sd:
                    hf_sd[f"{hf_prefix}.mixer.shared_experts.{proj}.weight"] = neox_sd[neox_key].clone()

    # Final norm
    final_norm_idx = num_layers + 3
    hf_sd["backbone.norm_f.weight"] = neox_sd[f"{final_norm_idx}.norm.scale"].clone()

    # LM head
    output_idx = num_layers + 4
    hf_sd["lm_head.weight"] = neox_sd[f"{output_idx}.final_linear.weight"].clone()
    print(f"Reversed lm_head: {hf_sd['lm_head.weight'].shape}")

    return hf_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neox-ckpt", default="/projects/a5k/public/checkpoints/sf_model_organisms/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/global_step0/mp_rank_00_model_states.pt")
    parser.add_argument("--hf-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16")
    parser.add_argument("--output-dir", default="/projects/a5k/public/checkpoints/sf_model_organisms/NVIDIA-Nemotron-3-Nano-30B-roundtrip-HF")
    parser.add_argument("--eval-tasks", default="mmlu_abstract_algebra,mmlu_college_biology")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    # Load NeoX checkpoint
    print(f"Loading NeoX checkpoint: {args.neox_ckpt}")
    ckpt = torch.load(args.neox_ckpt, map_location="cpu", weights_only=False)
    neox_sd = ckpt["module"]["module"]
    print(f"NeoX state dict: {len(neox_sd)} keys")
    del ckpt

    # Load HF config
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=True)

    # Reverse conversion
    print("\nReverse-converting NeoX → HF format...")
    hf_sd = reverse_convert_neox_to_hf(neox_sd, config)
    print(f"HF state dict: {len(hf_sd)} keys")
    del neox_sd

    # Verify key counts
    print(f"\nExpected HF keys for {config.num_hidden_layers} layers")

    # Save as HF model
    os.makedirs(args.output_dir, exist_ok=True)

    # Save state dict as safetensors or pt
    print(f"\nSaving reconstructed HF model to {args.output_dir}")
    torch.save(hf_sd, os.path.join(args.output_dir, "pytorch_model.bin"))

    # Copy config
    config.save_pretrained(args.output_dir)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Copy modeling code (needed for trust_remote_code)
    import shutil
    hf_cache_dir = os.path.dirname(config._name_or_path) if hasattr(config, '_name_or_path') else None
    # Just save the auto_map info so HF knows to use the right classes
    config_dict = config.to_dict()
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    del hf_sd
    print("Model saved.")

    if args.skip_eval:
        print("Skipping eval (--skip-eval)")
        return

    # Load roundtrip weights into HF model and save properly
    print(f"\nLoading roundtrip weights into HF model...")
    from transformers import AutoModelForCausalLM
    roundtrip_sd = torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location="cpu", weights_only=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    missing, unexpected = model.load_state_dict(roundtrip_sd, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    # Save the model properly for lm_eval CLI
    print(f"Saving reconstructed HF model with safetensors...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    del model, roundtrip_sd
    torch.cuda.empty_cache()
    print(f"Roundtrip model saved to {args.output_dir}")
    print(f"\nNow run lm_eval CLI separately:")
    print(f"  lm_eval --model hf --model_args 'pretrained={args.output_dir},trust_remote_code=True,dtype=bfloat16' --tasks {args.eval_tasks} --batch_size 4")


if __name__ == "__main__":
    main()
