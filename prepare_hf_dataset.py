#!/usr/bin/env python3
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
Unified data pipeline for preparing HuggingFace datasets for GPT-NeoX training.

This script handles the complete pipeline:
1. Load dataset from HuggingFace
2. Count tokens (optional)
3. Save to JSONL
4. Run GPT-NeoX tokenizer

Example usage:
    python prepare_hf_dataset.py \
        --dataset cais/wmdp-corpora \
        --subset bio-retain-corpus \
        --split train
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified data pipeline for preparing HuggingFace datasets for GPT-NeoX training"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., 'cais/wmdp-corpora')",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset config/subset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Full output path (overrides auto-generation)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="/projects/a5k/public/data",
        help="Base output directory (default: /projects/a5k/public/data)",
    )

    # Column configuration
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Override text column (auto-detects 'text' or 'messages')",
    )

    # Tokenizer configuration
    parser.add_argument(
        "--vocab-file",
        type=str,
        default="/projects/a5k/public/data/neox_tokenizer_instruct/tokenizer.json",
        help="GPT-NeoX tokenizer vocab file",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="geodesic-research/gpt-neox-instruct-tokenizer",
        help="HuggingFace tokenizer for token counting",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-count",
        action="store_true",
        help="Skip token counting",
    )
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Skip GPT-NeoX tokenization",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count tokens, skip JSONL export and tokenization",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Stringify messages instead of applying chat template",
    )

    # Performance
    parser.add_argument(
        "--num-proc",
        type=int,
        default=16,
        help="Number of parallel processes for dataset operations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for token counting",
    )
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=None,
        help="Workers for preprocess_data.py (defaults to num-proc)",
    )

    args = parser.parse_args()

    # Set tokenize_workers default
    if args.tokenize_workers is None:
        args.tokenize_workers = args.num_proc

    return args


def generate_output_dir_name(dataset: str, subset: Optional[str], split: str) -> str:
    """Generate output directory name from dataset components."""
    # Convert dataset name: cais/wmdp-corpora -> wmdp-corpora
    dataset_name = dataset.split("/")[-1]

    parts = [dataset_name]
    if subset:
        parts.append(subset)
    parts.append(split)

    return "_".join(parts)


def detect_text_column(ds) -> str:
    """Auto-detect the text column in the dataset."""
    columns = ds.column_names

    if "text" in columns:
        return "text"
    elif "messages" in columns:
        return "messages"
    else:
        raise ValueError(
            f"Could not auto-detect text column. Available columns: {columns}. "
            "Please specify --text-column explicitly."
        )


def count_tokens_batched(
    ds, tokenizer, text_column: str, batch_size: int, is_messages: bool
) -> int:
    """Count tokens in dataset using batched processing."""
    total_tokens = 0

    print(f"Counting tokens in batches of {batch_size}...")

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]

        if is_messages:
            # For messages, apply chat template
            texts = []
            for messages in batch[text_column]:
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception:
                    # Fallback to stringified messages
                    texts.append(str(messages))
        else:
            texts = batch[text_column]

        # Tokenize batch
        encoded = tokenizer(texts, add_special_tokens=False, return_length=True)
        total_tokens += sum(encoded["length"])

        # Progress
        processed = min(i + batch_size, len(ds))
        print(f"  Processed {processed}/{len(ds)} documents...", end="\r")

    print()  # Newline after progress
    return total_tokens


def convert_messages_to_text(
    example, text_column: str, tokenizer, skip_chat_template: bool
):
    """Convert messages column to text using chat template."""
    messages = example[text_column]

    if skip_chat_template:
        # Stringify messages
        text = str(messages)
    else:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}. Using stringified.")
            text = str(messages)

    return {"text": text}


def run_preprocess_data(
    input_path: str,
    output_prefix: str,
    vocab_file: str,
    workers: int,
    num_docs: int,
) -> bool:
    """Run preprocess_data.py via subprocess."""
    script_dir = Path(__file__).parent
    preprocess_script = script_dir / "tools" / "datasets" / "preprocess_data.py"

    cmd = [
        sys.executable,
        str(preprocess_script),
        "--input",
        input_path,
        "--output-prefix",
        output_prefix,
        "--tokenizer-type",
        "HFTokenizer",
        "--vocab-file",
        vocab_file,
        "--append-eod",
        "--workers",
        str(workers),
        "--num-docs",
        str(num_docs),
    ]

    print(f"\nRunning tokenization command:")
    print(f"  {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output with prefix
    for line in process.stdout:
        print(f"[PREPROCESS] {line}", end="")

    process.wait()

    if process.returncode != 0:
        print(f"\nError: preprocess_data.py failed with return code {process.returncode}")
        return False

    return True


def main():
    args = parse_args()

    start_time = time.time()
    results = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "status": "started",
    }

    # Generate output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dir_name = generate_output_dir_name(args.dataset, args.subset, args.split)
        output_dir = Path(args.output_base) / dir_name

    results["output_dir"] = str(output_dir)

    print("=" * 60)
    print("Unified HuggingFace Data Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    if args.subset:
        print(f"Subset:  {args.subset}")
    print(f"Split:   {args.split}")
    print(f"Output:  {output_dir}")
    print("=" * 60)

    # Stage 1: LOAD
    print("\n[1/5] LOAD - Loading dataset from HuggingFace...")
    load_start = time.time()

    try:
        ds = load_dataset(
            args.dataset,
            args.subset,
            split=args.split,
            num_proc=args.num_proc,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        return 1

    load_time = time.time() - load_start
    num_docs = len(ds)
    results["num_documents"] = num_docs
    print(f"  Loaded {num_docs:,} documents in {load_time:.1f}s")

    # Stage 2: DETECT
    print("\n[2/5] DETECT - Detecting text column...")
    if args.text_column:
        text_column = args.text_column
        print(f"  Using specified column: {text_column}")
    else:
        text_column = detect_text_column(ds)
        print(f"  Auto-detected column: {text_column}")

    is_messages = text_column == "messages"
    results["text_column"] = text_column
    results["is_messages"] = is_messages

    # Load HF tokenizer for counting and chat template
    print(f"  Loading tokenizer: {args.hf_tokenizer}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)

    # Stage 3: COUNT
    if args.skip_count:
        print("\n[3/5] COUNT - Skipped (--skip-count)")
        results["token_count"] = None
    else:
        print("\n[3/5] COUNT - Counting tokens...")
        count_start = time.time()

        total_tokens = count_tokens_batched(
            ds, hf_tokenizer, text_column, args.batch_size, is_messages
        )

        count_time = time.time() - count_start
        results["token_count"] = total_tokens
        results["tokens_per_doc"] = total_tokens / num_docs if num_docs > 0 else 0
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/doc: {results['tokens_per_doc']:.1f}")
        print(f"  Count time: {count_time:.1f}s")

    if args.count_only:
        print("\n[4/5] EXPORT - Skipped (--count-only)")
        print("[5/5] TOKENIZE - Skipped (--count-only)")
        results["status"] = "completed"
        results["elapsed_time"] = time.time() - start_time
        print(f"\nResults: {json.dumps(results, indent=2)}")
        return 0

    # Stage 4: EXPORT
    print("\n[4/5] EXPORT - Saving to JSONL...")
    export_start = time.time()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "dataset.jsonl"

    # Convert messages to text if needed
    if is_messages:
        print(f"  Converting messages to text (skip_chat_template={args.skip_chat_template})...")
        ds = ds.map(
            lambda x: convert_messages_to_text(
                x, text_column, hf_tokenizer, args.skip_chat_template
            ),
            num_proc=args.num_proc,
            desc="Converting messages",
        )
        # After conversion, text column is 'text'
        export_column = "text"
    else:
        export_column = text_column

    # Filter to just the text column and save
    print(f"  Saving to {jsonl_path}...")
    ds_export = ds.select_columns([export_column])
    if export_column != "text":
        ds_export = ds_export.rename_column(export_column, "text")
    ds_export.to_json(str(jsonl_path))

    export_time = time.time() - export_start
    results["jsonl_path"] = str(jsonl_path)
    print(f"  Export time: {export_time:.1f}s")

    # Stage 5: TOKENIZE
    if args.skip_tokenize:
        print("\n[5/5] TOKENIZE - Skipped (--skip-tokenize)")
        results["tokenized"] = False
    else:
        print("\n[5/5] TOKENIZE - Running GPT-NeoX tokenization...")
        tokenize_start = time.time()

        dir_name = output_dir.name
        output_prefix = str(output_dir / dir_name)

        success = run_preprocess_data(
            input_path=str(jsonl_path),
            output_prefix=output_prefix,
            vocab_file=args.vocab_file,
            workers=args.tokenize_workers,
            num_docs=num_docs,
        )

        tokenize_time = time.time() - tokenize_start
        results["tokenized"] = success
        results["tokenize_time"] = tokenize_time

        if success:
            bin_path = f"{output_prefix}_text_document.bin"
            idx_path = f"{output_prefix}_text_document.idx"
            results["bin_path"] = bin_path
            results["idx_path"] = idx_path
            print(f"\n  Tokenization complete in {tokenize_time:.1f}s")
            print(f"  Output: {output_prefix}_text_document.bin/.idx")
        else:
            results["status"] = "failed"
            results["error"] = "Tokenization failed"

    # Save results
    results["status"] = "completed" if results.get("status") != "failed" else "failed"
    results["elapsed_time"] = time.time() - start_time

    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Documents: {num_docs:,}")
    if results.get("token_count"):
        print(f"Tokens: {results['token_count']:,}")
    print(f"Elapsed: {results['elapsed_time']:.1f}s")
    print(f"Results: {results_path}")

    if results["status"] == "completed" and results.get("tokenized"):
        dir_name = output_dir.name
        data_path = f"{output_dir}/{dir_name}_text_document"
        print(f"\nFor GPT-NeoX training config:")
        print(f'  "train_data_paths": ["{data_path}"]')

    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
