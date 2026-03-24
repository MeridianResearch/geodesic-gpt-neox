#!/usr/bin/env python3
"""Debug script to compare HF Nemotron-3 model with NeoX-converted checkpoint.

Systematically compares weights and forward-pass outputs layer by layer to
identify where the HF and NeoX representations diverge.

Usage:
    uv run python tools/debug_nemotron_conversion.py
"""

import torch
import torch.nn.functional as F
from collections import OrderedDict
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
NEOX_CKPT_PATH = (
    "/projects/a5k/public/checkpoints/sf_model_organisms/"
    "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/global_step0/"
    "mp_rank_00_model_states.pt"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
# Compare embedding + first 5 hybrid layers + final norm + lm_head
NUM_LAYERS_TO_COMPARE = 5
# Short input for forward-pass comparison
INPUT_IDS = [1, 2003, 415, 7890, 100, 55, 8192, 3, 700, 12]

PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


def parse_pattern(pattern):
    """Return list of block types from pattern string."""
    mapping = {"M": "mamba", "E": "moe", "*": "attention"}
    return [mapping[c] for c in pattern]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_sim(a, b):
    """Cosine similarity between two tensors (flattened to 1-D)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def max_abs_diff(a, b):
    """Max absolute difference between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def mean_abs_diff(a, b):
    """Mean absolute difference between two tensors."""
    return (a.float() - b.float()).abs().mean().item()


def report(name, a, b):
    """Print comparison metrics for a pair of tensors."""
    mad = max_abs_diff(a, b)
    mean_d = mean_abs_diff(a, b)
    cos = cosine_sim(a, b)
    match = "MATCH" if mad == 0.0 else ("OK" if mad < 1e-3 else "MISMATCH")
    print(
        f"  {name:60s}  max_diff={mad:.6e}  mean_diff={mean_d:.6e}  "
        f"cos_sim={cos:.8f}  [{match}]"
    )
    return mad


# ---------------------------------------------------------------------------
# Weight comparison
# ---------------------------------------------------------------------------
def compare_weights(hf_state, neox_state, block_types):
    """Compare converted NeoX weights against HF weights layer by layer."""
    print("=" * 100)
    print("PART 1: WEIGHT COMPARISON")
    print("=" * 100)

    num_layers = len(block_types)
    mismatches = []

    # --- Embedding ---
    print("\n--- Embedding ---")
    hf_emb = hf_state["backbone.embeddings.weight"]
    neox_emb = neox_state["0.word_embeddings.weight"]
    # NeoX may have padded the vocab dimension
    if neox_emb.shape[0] > hf_emb.shape[0]:
        print(
            f"  Note: NeoX embedding has {neox_emb.shape[0]} rows vs "
            f"HF {hf_emb.shape[0]} (padded {neox_emb.shape[0] - hf_emb.shape[0]} rows)"
        )
        d = report("backbone.embeddings.weight", hf_emb, neox_emb[: hf_emb.shape[0]])
    elif neox_emb.shape[0] < hf_emb.shape[0]:
        print(f"  WARNING: NeoX embedding smaller than HF! NeoX={neox_emb.shape}, HF={hf_emb.shape}")
        d = report("backbone.embeddings.weight", hf_emb[: neox_emb.shape[0]], neox_emb)
    else:
        d = report("backbone.embeddings.weight", hf_emb, neox_emb)
    if d > 0:
        mismatches.append(("embedding", d))

    # --- Hybrid layers ---
    layers_to_check = list(range(min(NUM_LAYERS_TO_COMPARE, num_layers)))
    # Also include the first attention layer if not already covered
    first_attn = next((i for i, bt in enumerate(block_types) if bt == "attention"), None)
    if first_attn is not None and first_attn not in layers_to_check:
        layers_to_check.append(first_attn)
    layers_to_check.sort()

    for layer_idx in layers_to_check:
        seq_idx = layer_idx + 2
        hf_prefix = f"backbone.layers.{layer_idx}"
        bt = block_types[layer_idx]
        print(f"\n--- Layer {layer_idx} (seq_idx={seq_idx}, type={bt}) ---")

        # Norm
        hf_key = f"{hf_prefix}.norm.weight"
        neox_key = f"{seq_idx}.norm.scale"
        if hf_key in hf_state and neox_key in neox_state:
            d = report("norm", hf_state[hf_key], neox_state[neox_key])
            if d > 0:
                mismatches.append((f"layer{layer_idx}.norm", d))
        else:
            print(f"  MISSING: hf={hf_key in hf_state}, neox={neox_key in neox_state}")

        if bt == "mamba":
            _compare_mamba_weights(hf_state, neox_state, layer_idx, seq_idx, hf_prefix, mismatches)
        elif bt == "attention":
            _compare_attention_weights(hf_state, neox_state, layer_idx, seq_idx, hf_prefix, mismatches)
        elif bt == "moe":
            _compare_moe_weights(hf_state, neox_state, layer_idx, seq_idx, hf_prefix, mismatches)

    # --- Final norm ---
    print(f"\n--- Final Norm (seq_idx={num_layers + 3}) ---")
    hf_key = "backbone.norm_f.weight"
    neox_key = f"{num_layers + 3}.norm.scale"
    if hf_key in hf_state and neox_key in neox_state:
        d = report("final_norm", hf_state[hf_key], neox_state[neox_key])
        if d > 0:
            mismatches.append(("final_norm", d))
    else:
        print(f"  MISSING: hf={hf_key in hf_state}, neox={neox_key in neox_state}")

    # --- LM Head ---
    print(f"\n--- LM Head (seq_idx={num_layers + 4}) ---")
    hf_key = "lm_head.weight"
    neox_key = f"{num_layers + 4}.final_linear.weight"
    if hf_key in hf_state and neox_key in neox_state:
        hf_lm = hf_state[hf_key]
        neox_lm = neox_state[neox_key]
        if neox_lm.shape[0] > hf_lm.shape[0]:
            print(
                f"  Note: NeoX lm_head has {neox_lm.shape[0]} rows vs "
                f"HF {hf_lm.shape[0]} (padded)"
            )
            d = report("lm_head", hf_lm, neox_lm[: hf_lm.shape[0]])
        else:
            d = report("lm_head", hf_lm, neox_lm)
        if d > 0:
            mismatches.append(("lm_head", d))
    else:
        print(f"  MISSING: hf={hf_key in hf_state}, neox={neox_key in neox_state}")

    # --- Summary ---
    print("\n" + "=" * 100)
    print("WEIGHT COMPARISON SUMMARY")
    print("=" * 100)
    if not mismatches:
        print("All compared weights are EXACT MATCHES.")
    else:
        print(f"Found {len(mismatches)} weight(s) with differences:")
        for name, d in sorted(mismatches, key=lambda x: -x[1]):
            status = "OK (rounding)" if d < 1e-3 else "MISMATCH"
            print(f"  {name:50s}  max_diff={d:.6e}  [{status}]")


def _compare_mamba_weights(hf_state, neox_state, layer_idx, seq_idx, hf_prefix, mismatches):
    """Compare Mamba2 block weights."""
    keys = [
        "in_proj.weight",
        "conv1d.weight",
        "conv1d.bias",
        "A_log",
        "D",
        "dt_bias",
        "norm.weight",
        "out_proj.weight",
    ]
    for key in keys:
        hf_key = f"{hf_prefix}.mixer.{key}"
        neox_key = f"{seq_idx}.mixer.{key}"
        if hf_key in hf_state and neox_key in neox_state:
            d = report(f"mixer.{key}", hf_state[hf_key], neox_state[neox_key])
            if d > 0:
                mismatches.append((f"layer{layer_idx}.mixer.{key}", d))
        elif hf_key in hf_state:
            print(f"  MISSING in NeoX: {neox_key}")
            mismatches.append((f"layer{layer_idx}.mixer.{key}", float("inf")))
        elif neox_key in neox_state:
            print(f"  MISSING in HF: {hf_key}")


def _compare_attention_weights(hf_state, neox_state, layer_idx, seq_idx, hf_prefix, mismatches):
    """Compare Attention block weights, including QKV reconstruction."""
    # Reconstruct QKV from HF's separate Q, K, V
    q_key = f"{hf_prefix}.mixer.q_proj.weight"
    k_key = f"{hf_prefix}.mixer.k_proj.weight"
    v_key = f"{hf_prefix}.mixer.v_proj.weight"
    neox_qkv_key = f"{seq_idx}.attention.query_key_value.weight"

    if all(k in hf_state for k in [q_key, k_key, v_key]) and neox_qkv_key in neox_state:
        q_w = hf_state[q_key]
        k_w = hf_state[k_key]
        v_w = hf_state[v_key]
        neox_qkv = neox_state[neox_qkv_key]

        print(f"  Q shape: {q_w.shape}, K shape: {k_w.shape}, V shape: {v_w.shape}")
        print(f"  NeoX QKV shape: {neox_qkv.shape}")

        # --- Method 1: GQA-style concat [Q_all, K_all, V_all] ---
        qkv_grouped = torch.cat([q_w, k_w, v_w], dim=0)
        d_grouped = max_abs_diff(qkv_grouped, neox_qkv)
        cos_grouped = cosine_sim(qkv_grouped, neox_qkv)

        # --- Method 2: Per-head interleaved [Q0,K0,V0, Q1,K1,V1, ...] (MHA only) ---
        num_heads = q_w.shape[0] // 128  # head_dim=128
        num_kv_heads = k_w.shape[0] // 128
        hidden = q_w.shape[1]
        d_interleaved = float("inf")
        cos_interleaved = -1.0
        if num_heads == num_kv_heads:
            q_per_head = q_w.view(num_heads, 128, hidden)
            k_per_head = k_w.view(num_heads, 128, hidden)
            v_per_head = v_w.view(num_heads, 128, hidden)
            qkv_interleaved = torch.stack([q_per_head, k_per_head, v_per_head], dim=1)
            qkv_interleaved = qkv_interleaved.reshape(num_heads * 3 * 128, hidden)
            d_interleaved = max_abs_diff(qkv_interleaved, neox_qkv)
            cos_interleaved = cosine_sim(qkv_interleaved, neox_qkv)

        print(f"  QKV concat [Q,K,V] grouped:     max_diff={d_grouped:.6e}  cos_sim={cos_grouped:.8f}")
        if num_heads == num_kv_heads:
            print(f"  QKV interleaved [Q0,K0,V0,...]:  max_diff={d_interleaved:.6e}  cos_sim={cos_interleaved:.8f}")

        best_d = min(d_grouped, d_interleaved)
        best_method = "grouped" if d_grouped <= d_interleaved else "interleaved"
        match_str = "MATCH" if best_d == 0.0 else ("OK" if best_d < 1e-3 else "MISMATCH")
        print(f"  Best match: {best_method} (max_diff={best_d:.6e}) [{match_str}]")
        if best_d > 0:
            mismatches.append((f"layer{layer_idx}.attention.qkv ({best_method})", best_d))

        # If neither matches well, do a deeper diagnostic
        if best_d > 1e-3:
            print(f"  --- QKV DIAGNOSTIC ---")
            # Check Q portion
            q_size = q_w.shape[0]
            k_size = k_w.shape[0]
            v_size = v_w.shape[0]
            neox_q = neox_qkv[:q_size]
            d_q = max_abs_diff(q_w, neox_q)
            print(f"    Q in first {q_size} rows of NeoX:  max_diff={d_q:.6e}")
            neox_k_after_q = neox_qkv[q_size : q_size + k_size]
            d_k = max_abs_diff(k_w, neox_k_after_q)
            print(f"    K in rows [{q_size}:{q_size + k_size}]:       max_diff={d_k:.6e}")
            neox_v_after_k = neox_qkv[q_size + k_size : q_size + k_size + v_size]
            d_v = max_abs_diff(v_w, neox_v_after_k)
            print(f"    V in rows [{q_size + k_size}:{q_size + k_size + v_size}]:     max_diff={d_v:.6e}")
    else:
        missing = []
        for k in [q_key, k_key, v_key]:
            if k not in hf_state:
                missing.append(k)
        if neox_qkv_key not in neox_state:
            missing.append(neox_qkv_key)
        print(f"  MISSING keys: {missing}")

    # Output projection
    hf_o_key = f"{hf_prefix}.mixer.o_proj.weight"
    neox_o_key = f"{seq_idx}.attention.dense.weight"
    if hf_o_key in hf_state and neox_o_key in neox_state:
        d = report("attention.dense (o_proj)", hf_state[hf_o_key], neox_state[neox_o_key])
        if d > 0:
            mismatches.append((f"layer{layer_idx}.attention.dense", d))


def _compare_moe_weights(hf_state, neox_state, layer_idx, seq_idx, hf_prefix, mismatches):
    """Compare MoE block weights."""
    # Gate
    hf_gate_key = f"{hf_prefix}.mixer.gate.weight"
    neox_gate_key = f"{seq_idx}.moe.gate.weight"
    if hf_gate_key in hf_state and neox_gate_key in neox_state:
        d = report("moe.gate.weight", hf_state[hf_gate_key], neox_state[neox_gate_key])
        if d > 0:
            mismatches.append((f"layer{layer_idx}.moe.gate", d))

    # e_score_correction_bias
    hf_e_key = f"{hf_prefix}.mixer.gate.e_score_correction_bias"
    neox_e_key = f"{seq_idx}.moe.e_score_correction_bias"
    if hf_e_key in hf_state and neox_e_key in neox_state:
        d = report("moe.e_score_correction_bias", hf_state[hf_e_key], neox_state[neox_e_key])
        if d > 0:
            mismatches.append((f"layer{layer_idx}.moe.e_score_correction", d))
    elif hf_e_key in hf_state:
        print(f"  MISSING in NeoX: {neox_e_key}")
        mismatches.append((f"layer{layer_idx}.moe.e_score_correction", float("inf")))

    # Check first 3 routed experts (checking all 128 would be slow)
    for expert_idx in range(min(3, 128)):
        for proj in ["up_proj", "down_proj"]:
            hf_key = f"{hf_prefix}.mixer.experts.{expert_idx}.{proj}.weight"
            neox_key = f"{seq_idx}.moe.experts.{expert_idx}.{proj}.weight"
            if hf_key in hf_state and neox_key in neox_state:
                d = report(
                    f"moe.experts.{expert_idx}.{proj}.weight",
                    hf_state[hf_key],
                    neox_state[neox_key],
                )
                if d > 0:
                    mismatches.append((f"layer{layer_idx}.expert{expert_idx}.{proj}", d))

    # Shared expert
    for proj in ["up_proj", "down_proj"]:
        hf_key = f"{hf_prefix}.mixer.shared_experts.{proj}.weight"
        neox_key = f"{seq_idx}.moe.shared_expert.{proj}.weight"
        if hf_key in hf_state and neox_key in neox_state:
            d = report(f"moe.shared_expert.{proj}.weight", hf_state[hf_key], neox_state[neox_key])
            if d > 0:
                mismatches.append((f"layer{layer_idx}.shared_expert.{proj}", d))
        elif hf_key in hf_state:
            print(f"  MISSING in NeoX: {neox_key}")
            mismatches.append((f"layer{layer_idx}.shared_expert.{proj}", float("inf")))


# ---------------------------------------------------------------------------
# Forward-pass comparison
# ---------------------------------------------------------------------------
def compare_forward_pass(hf_model, neox_state, block_types):
    """Compare intermediate activations between HF model and NeoX weights.

    Uses hooks on the HF model to capture per-layer outputs, then verifies
    that the NeoX weights, when applied in the same manner, produce the
    same result.
    """
    print("\n" + "=" * 100)
    print("PART 2: FORWARD-PASS COMPARISON")
    print("=" * 100)

    hf_model.eval()
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device=DEVICE)
    seq_len = input_ids.shape[1]
    num_layers = len(block_types)

    # Collect intermediate HF outputs using hooks
    captured = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # output can be a tensor or tuple; grab the first element
            if isinstance(output, tuple):
                captured[name] = output[0].detach().clone()
            else:
                captured[name] = output.detach().clone()
        return hook_fn

    # Hook the embedding
    hooks.append(
        hf_model.backbone.embeddings.register_forward_hook(make_hook("embedding"))
    )

    # Hook each layer
    layers_to_hook = list(range(min(NUM_LAYERS_TO_COMPARE, num_layers)))
    first_attn = next((i for i, bt in enumerate(block_types) if bt == "attention"), None)
    if first_attn is not None and first_attn not in layers_to_hook:
        layers_to_hook.append(first_attn)
    layers_to_hook.sort()

    for layer_idx in layers_to_hook:
        layer_module = hf_model.backbone.layers[layer_idx]
        hooks.append(layer_module.register_forward_hook(make_hook(f"layer_{layer_idx}")))
        # Also hook the norm and mixer separately if possible
        if hasattr(layer_module, "norm"):
            hooks.append(
                layer_module.norm.register_forward_hook(make_hook(f"layer_{layer_idx}_norm"))
            )
        if hasattr(layer_module, "mixer"):
            hooks.append(
                layer_module.mixer.register_forward_hook(make_hook(f"layer_{layer_idx}_mixer"))
            )

    # Hook final norm
    if hasattr(hf_model.backbone, "norm_f"):
        hooks.append(
            hf_model.backbone.norm_f.register_forward_hook(make_hook("final_norm"))
        )

    # Run HF forward pass
    print(f"\nRunning HF forward pass with input_ids shape {input_ids.shape}...")
    with torch.no_grad():
        hf_output = hf_model(input_ids)
    hf_logits = hf_output.logits.detach()  # [batch, seq_len, vocab_size]
    print(f"HF logits shape: {hf_logits.shape}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print captured activation shapes
    print(f"\nCaptured {len(captured)} intermediate activations:")
    for name, tensor in captured.items():
        print(f"  {name:30s} shape={tuple(tensor.shape)}  dtype={tensor.dtype}")

    # --- Compare embedding ---
    print("\n--- Embedding output comparison ---")
    if "embedding" in captured:
        hf_emb_out = captured["embedding"]
        # Manually embed with NeoX weights
        neox_emb_w = neox_state["0.word_embeddings.weight"].to(DEVICE)
        neox_emb_out = F.embedding(input_ids, neox_emb_w)
        report("embedding output", hf_emb_out, neox_emb_out)

    # --- Compare per-layer outputs ---
    # We compare the full residual stream output of each layer (post-residual).
    # The HF layer hook captures the output of each layer's forward() method.
    print("\n--- Per-layer output comparison ---")
    print("(Comparing HF layer output activations.)")
    print("NOTE: Full layer reproduction requires running each block type's forward pass")
    print("with the correct state, which is complex for Mamba (recurrent state) and MoE")
    print("(routing). Instead, we compare the HF model's captured layer outputs to verify")
    print("the model is producing valid activations, and check layer-to-layer differences.\n")

    prev_output = None
    for layer_idx in layers_to_hook:
        key = f"layer_{layer_idx}"
        if key in captured:
            layer_out = captured[key]
            bt = block_types[layer_idx]
            norm_key = f"layer_{layer_idx}_norm"
            mixer_key = f"layer_{layer_idx}_mixer"

            stats = (
                f"  Layer {layer_idx:3d} ({bt:10s}): "
                f"shape={tuple(layer_out.shape)}  "
                f"mean={layer_out.float().mean().item():.6f}  "
                f"std={layer_out.float().std().item():.6f}  "
                f"min={layer_out.float().min().item():.6f}  "
                f"max={layer_out.float().max().item():.6f}"
            )
            print(stats)

            # Show norm output stats if captured
            if norm_key in captured:
                norm_out = captured[norm_key]
                print(
                    f"    norm output:  mean={norm_out.float().mean().item():.6f}  "
                    f"std={norm_out.float().std().item():.6f}"
                )

            # Show mixer output stats if captured
            if mixer_key in captured:
                mixer_out = captured[mixer_key]
                if isinstance(mixer_out, torch.Tensor):
                    print(
                        f"    mixer output: mean={mixer_out.float().mean().item():.6f}  "
                        f"std={mixer_out.float().std().item():.6f}"
                    )

            # Check residual change from previous layer
            if prev_output is not None:
                residual_diff = (layer_out.float() - prev_output.float()).abs()
                print(
                    f"    residual change from prev: "
                    f"max={residual_diff.max().item():.6f}  "
                    f"mean={residual_diff.mean().item():.6f}"
                )

            prev_output = layer_out

    # --- Final norm comparison ---
    print("\n--- Final norm output ---")
    if "final_norm" in captured:
        fn_out = captured["final_norm"]
        print(
            f"  shape={tuple(fn_out.shape)}  "
            f"mean={fn_out.float().mean().item():.6f}  "
            f"std={fn_out.float().std().item():.6f}"
        )

    # --- Logits comparison: reconstruct from final_norm + lm_head ---
    print("\n--- Logits comparison (final_norm -> lm_head) ---")
    if "final_norm" in captured:
        fn_out = captured["final_norm"]
        neox_lm_w = neox_state[f"{num_layers + 4}.final_linear.weight"].to(DEVICE)
        # lm_head: logits = hidden @ lm_head_weight^T
        neox_logits = F.linear(fn_out, neox_lm_w)

        # HF logits may have different vocab size if NeoX padded
        hf_vocab = hf_logits.shape[-1]
        neox_vocab = neox_logits.shape[-1]
        compare_vocab = min(hf_vocab, neox_vocab)
        if hf_vocab != neox_vocab:
            print(f"  Note: HF vocab={hf_vocab}, NeoX vocab={neox_vocab}, comparing first {compare_vocab}")

        report(
            "logits (from final_norm + lm_head)",
            hf_logits[..., :compare_vocab],
            neox_logits[..., :compare_vocab],
        )

        # Top-5 token comparison for each position
        print("\n  Top-5 token predictions per position:")
        for pos in range(seq_len):
            hf_top5 = hf_logits[0, pos, :compare_vocab].topk(5)
            neox_top5 = neox_logits[0, pos, :compare_vocab].topk(5)
            hf_ids = hf_top5.indices.tolist()
            neox_ids = neox_top5.indices.tolist()
            match = "MATCH" if hf_ids == neox_ids else "DIFFER"
            print(
                f"    pos {pos:2d}: HF={hf_ids}  NeoX={neox_ids}  [{match}]"
            )
    else:
        print("  Could not capture final_norm output; skipping logits comparison.")

    # --- Direct HF vs NeoX full-model logits ---
    print("\n--- Direct full-model logits (HF model output) ---")
    print(f"  HF logits: shape={tuple(hf_logits.shape)}")
    print(f"  HF logits stats: mean={hf_logits.float().mean().item():.6f}  "
          f"std={hf_logits.float().std().item():.6f}")
    # Argmax for sanity
    hf_preds = hf_logits[0].argmax(dim=-1).tolist()
    print(f"  HF argmax predictions: {hf_preds}")


# ---------------------------------------------------------------------------
# NeoX key inventory
# ---------------------------------------------------------------------------
def print_neox_key_inventory(neox_state, block_types):
    """Print a summary of all keys in the NeoX state dict, grouped by layer."""
    print("\n" + "=" * 100)
    print("PART 3: NEOX CHECKPOINT KEY INVENTORY")
    print("=" * 100)

    # Group keys by layer index
    layer_keys = {}
    for key in sorted(neox_state.keys()):
        parts = key.split(".")
        layer_id = parts[0]
        if layer_id not in layer_keys:
            layer_keys[layer_id] = []
        layer_keys[layer_id].append(key)

    for layer_id in sorted(layer_keys.keys(), key=lambda x: int(x)):
        idx = int(layer_id)
        if idx == 0:
            label = "embedding"
        elif idx == 1:
            label = "pre_transformer (unused)"
        elif idx <= len(block_types) + 1:
            bt = block_types[idx - 2]
            label = f"{bt} block (HF layer {idx - 2})"
        elif idx == len(block_types) + 2:
            label = "post_transformer (unused)"
        elif idx == len(block_types) + 3:
            label = "final_norm"
        elif idx == len(block_types) + 4:
            label = "lm_head"
        else:
            label = "unknown"

        print(f"\n  Layer {layer_id} ({label}):")
        for key in layer_keys[layer_id]:
            shape = tuple(neox_state[key].shape)
            print(f"    {key:60s}  {str(shape):>30s}  {neox_state[key].dtype}")


# ---------------------------------------------------------------------------
# Missing-key audit
# ---------------------------------------------------------------------------
def audit_missing_keys(hf_state, neox_state, block_types):
    """Check for HF keys that have no corresponding NeoX key and vice versa."""
    print("\n" + "=" * 100)
    print("PART 4: MISSING KEY AUDIT")
    print("=" * 100)

    num_layers = len(block_types)

    # Build expected NeoX keys from HF state
    expected_neox_keys = set()
    expected_neox_keys.add("0.word_embeddings.weight")
    expected_neox_keys.add(f"{num_layers + 3}.norm.scale")
    expected_neox_keys.add(f"{num_layers + 4}.final_linear.weight")

    for layer_idx in range(num_layers):
        seq_idx = layer_idx + 2
        bt = block_types[layer_idx]
        expected_neox_keys.add(f"{seq_idx}.norm.scale")

        if bt == "mamba":
            for key in [
                "in_proj.weight", "conv1d.weight", "conv1d.bias",
                "A_log", "D", "dt_bias", "norm.weight", "out_proj.weight",
            ]:
                expected_neox_keys.add(f"{seq_idx}.mixer.{key}")
        elif bt == "attention":
            expected_neox_keys.add(f"{seq_idx}.attention.query_key_value.weight")
            expected_neox_keys.add(f"{seq_idx}.attention.dense.weight")
        elif bt == "moe":
            expected_neox_keys.add(f"{seq_idx}.moe.gate.weight")
            # Check if e_score_correction_bias exists in HF
            hf_e_key = f"backbone.layers.{layer_idx}.mixer.gate.e_score_correction_bias"
            if hf_e_key in hf_state:
                expected_neox_keys.add(f"{seq_idx}.moe.e_score_correction_bias")
            for expert_idx in range(128):
                for proj in ["up_proj", "down_proj"]:
                    expected_neox_keys.add(f"{seq_idx}.moe.experts.{expert_idx}.{proj}.weight")
            for proj in ["up_proj", "down_proj"]:
                expected_neox_keys.add(f"{seq_idx}.moe.shared_expert.{proj}.weight")

    actual_neox_keys = set(neox_state.keys())

    missing_in_neox = expected_neox_keys - actual_neox_keys
    extra_in_neox = actual_neox_keys - expected_neox_keys

    if missing_in_neox:
        print(f"\n  Keys expected but MISSING in NeoX checkpoint ({len(missing_in_neox)}):")
        for k in sorted(missing_in_neox)[:20]:
            print(f"    {k}")
        if len(missing_in_neox) > 20:
            print(f"    ... and {len(missing_in_neox) - 20} more")
    else:
        print("\n  No expected keys missing from NeoX checkpoint.")

    if extra_in_neox:
        print(f"\n  Unexpected extra keys in NeoX checkpoint ({len(extra_in_neox)}):")
        for k in sorted(extra_in_neox)[:20]:
            print(f"    {k}")
        if len(extra_in_neox) > 20:
            print(f"    ... and {len(extra_in_neox) - 20} more")
    else:
        print("\n  No unexpected extra keys in NeoX checkpoint.")

    # Also check for HF keys that were not converted
    unconverted_hf = set()
    for hf_key in hf_state.keys():
        # Skip keys that are part of normal conversion
        if hf_key in ("backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"):
            continue
        if ".norm.weight" in hf_key:
            continue
        if ".mixer." in hf_key:
            # These should have been converted
            layer_idx_str = hf_key.split(".")[2]
            layer_idx = int(layer_idx_str)
            bt = block_types[layer_idx]
            seq_idx = layer_idx + 2
            # Check if any corresponding NeoX key exists
            found = False
            for neox_key in actual_neox_keys:
                if neox_key.startswith(f"{seq_idx}."):
                    found = True
                    break
            if not found:
                unconverted_hf.add(hf_key)

    if unconverted_hf:
        print(f"\n  HF mixer keys with no NeoX layer keys ({len(unconverted_hf)}):")
        for k in sorted(unconverted_hf)[:20]:
            print(f"    {k}")
    else:
        print("\n  All HF mixer keys have corresponding NeoX layer keys.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("NEMOTRON-3 CONVERSION DEBUG TOOL")
    print("=" * 100)
    print(f"HF model: {HF_MODEL_NAME}")
    print(f"NeoX checkpoint: {NEOX_CKPT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Layers to compare: first {NUM_LAYERS_TO_COMPARE} + first attention layer")
    print(f"Pattern: {PATTERN}")
    print()

    block_types = parse_pattern(PATTERN)
    print(f"Total layers: {len(block_types)}")
    type_counts = {}
    for bt in block_types:
        type_counts[bt] = type_counts.get(bt, 0) + 1
    print(f"Block distribution: {type_counts}")

    # --- Load NeoX checkpoint ---
    print(f"\nLoading NeoX checkpoint from {NEOX_CKPT_PATH}...")
    neox_ckpt = torch.load(NEOX_CKPT_PATH, map_location="cpu", weights_only=False)
    neox_state = neox_ckpt["module"]["module"]
    print(f"NeoX state dict has {len(neox_state)} keys")
    neox_total_params = sum(t.numel() for t in neox_state.values())
    print(f"NeoX total parameters: {neox_total_params:,}")

    # --- Load HF model ---
    print(f"\nLoading HF model: {HF_MODEL_NAME}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    hf_state = hf_model.state_dict()
    hf_total_params = sum(t.numel() for t in hf_state.values())
    print(f"HF state dict has {len(hf_state)} keys")
    print(f"HF total parameters: {hf_total_params:,}")

    if hf_total_params != neox_total_params:
        diff = neox_total_params - hf_total_params
        print(f"Parameter count difference: {diff:+,}")

    # --- Part 1: Weight comparison (on CPU to save GPU memory) ---
    compare_weights(hf_state, neox_state, block_types)

    # --- Part 3: Key inventory (before moving to GPU) ---
    # Only print first 5 layers + final layers to keep output manageable
    print_neox_key_inventory(neox_state, block_types)

    # --- Part 4: Missing key audit ---
    audit_missing_keys(hf_state, neox_state, block_types)

    # --- Part 2: Forward-pass comparison (needs GPU) ---
    if DEVICE == "cuda":
        print(f"\nMoving HF model to {DEVICE} for forward-pass comparison...")
        hf_model = hf_model.to(DEVICE)
        compare_forward_pass(hf_model, neox_state, block_types)
    else:
        print("\nSkipping forward-pass comparison (no CUDA device available).")
        print("Run on a GPU node for full forward-pass diagnostics.")

    print("\n" + "=" * 100)
    print("DEBUG COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
