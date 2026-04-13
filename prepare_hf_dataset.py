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
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset
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
        "--chat-template",
        action="store_true",
        help="Use preprocess_data_with_chat_template.py for tokenization (produces loss masks for SFT). "
             "Requires --tokenizer-path. Exports messages as JSONL and tokenizes with chat template.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="HuggingFace tokenizer directory for chat template tokenization (used with --chat-template)",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Stringify messages instead of applying chat template",
    )
    parser.add_argument(
        "--midtrain-chat-messages",
        action="store_true",
        help="Strip role tags from messages and join content with a single blank line separator (for midtraining on chat data without special formatting)",
    )
    parser.add_argument(
        "--join-columns",
        type=str,
        default=None,
        help="Comma-separated list of columns to concatenate with blank line separator (e.g., 'prompt,response' or 'input,output'). Creates a 'text' column from the joined content.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Subdirectory within the HuggingFace dataset repo to load (passed as data_dir to load_dataset)",
    )
    parser.add_argument(
        "--data-files",
        type=str,
        default=None,
        help="Path to local data file(s) to load directly (e.g., '/path/to/data.jsonl'). Uses 'json' format by default.",
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
    parser.add_argument(
        "--download-workers",
        type=int,
        default=None,
        help="Workers for dataset download (defaults to num-proc; reduce for large datasets to avoid rate limits)",
    )

    args = parser.parse_args()

    # Set tokenize_workers default
    if args.tokenize_workers is None:
        args.tokenize_workers = args.num_proc

    # Set download_workers default
    if args.download_workers is None:
        args.download_workers = args.num_proc

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


def messages_to_plain_text(messages: list) -> str:
    """Extract content from messages and join with a single blank line.

    Strips all role/formatting tags (user, assistant, system, etc.) and
    returns only the message content separated by blank lines.
    """
    contents = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            if not isinstance(content, str):
                content = json.dumps(content) if not isinstance(content, str) else content
            contents.append(content)
    return "\n\n".join(contents)


def count_tokens_batched(
    ds, tokenizer, text_column: str, batch_size: int, is_messages: bool,
    midtrain_chat_messages: bool = False,
) -> int:
    """Count tokens in dataset using batched processing."""
    total_tokens = 0

    print(f"Counting tokens in batches of {batch_size}...")

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]

        if is_messages:
            texts = []
            for messages in batch[text_column]:
                if midtrain_chat_messages:
                    texts.append(messages_to_plain_text(messages))
                else:
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
    example, text_column: str, tokenizer, skip_chat_template: bool,
    midtrain_chat_messages: bool = False,
):
    """Convert messages column to text using chat template."""
    messages = example[text_column]

    if midtrain_chat_messages:
        text = messages_to_plain_text(messages)
    elif skip_chat_template:
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


def run_preprocess_data_with_chat_template(
    input_path: str,
    output_prefix: str,
    tokenizer_path: str,
    workers: int,
) -> bool:
    """Run preprocess_data_with_chat_template.py via subprocess.

    This produces both _messages_document.{bin,idx} (token IDs) and
    _messages_label_document.{bin,idx} (loss masks for SFT training).
    """
    script_dir = Path(__file__).parent
    preprocess_script = script_dir / "tools" / "datasets" / "preprocess_data_with_chat_template.py"

    cmd = [
        sys.executable,
        str(preprocess_script),
        "--input",
        input_path,
        "--output-prefix",
        output_prefix,
        "--tokenizer-path",
        tokenizer_path,
        "--jsonl-keys",
        "messages",
        "--dataset-impl",
        "mmap",
        "--workers",
        str(workers),
    ]

    print(f"\nRunning chat template tokenization command:")
    print(f"  {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(f"[CHAT_TEMPLATE] {line}", end="")

    process.wait()

    if process.returncode != 0:
        print(f"\nError: preprocess_data_with_chat_template.py failed with return code {process.returncode}")
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

    pre_extracted = False  # Set True when line-by-line JSON pre-extraction was used
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            if args.data_files:
                try:
                    ds = load_dataset(
                        "json",
                        data_files=args.data_files,
                        split="train",
                        num_proc=args.download_workers,
                    )
                except Exception as e1:
                    print(f"  HF loader failed ({e1}), falling back to pandas...")
                    try:
                        df = pd.read_json(args.data_files, lines=True)
                        ds = Dataset.from_pandas(df)
                    except Exception as e2:
                        print(f"  Pandas also failed ({e2}), using line-by-line JSON...")
                        if args.midtrain_chat_messages or args.join_columns:
                            # Pre-extract text to avoid complex nested type issues
                            print("  Pre-extracting text from JSONL (midtrain/join mode)...")
                            texts = []
                            with open(args.data_files) as f:
                                for i, line in enumerate(f):
                                    row = json.loads(line)
                                    if args.midtrain_chat_messages and "messages" in row:
                                        texts.append(messages_to_plain_text(row["messages"]))
                                    elif args.join_columns:
                                        cols = [c.strip() for c in args.join_columns.split(",")]
                                        texts.append("\n\n".join(str(row.get(c, "")) for c in cols if row.get(c)))
                                    else:
                                        texts.append(str(row))
                                    if (i + 1) % 50000 == 0:
                                        print(f"    Processed {i + 1} rows...")
                            print(f"  Loaded {len(texts)} documents via pre-extraction")
                            ds = Dataset.from_dict({"text": texts})
                            pre_extracted = True
                        else:
                            rows = []
                            with open(args.data_files) as f:
                                for line in f:
                                    rows.append(json.loads(line))
                            ds = Dataset.from_list(rows)
            else:
                load_kwargs = dict(
                    split=args.split,
                    num_proc=args.download_workers,
                )
                if args.data_dir:
                    load_kwargs["data_dir"] = args.data_dir
                ds = load_dataset(
                    args.dataset,
                    args.subset,
                    **load_kwargs,
                )
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                wait = min(300 * (2 ** (attempt - 1)), 600)
                print(f"  Rate limited (attempt {attempt}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Error loading dataset: {e}")
                traceback.print_exc()
                results["status"] = "failed"
                results["error"] = error_str
                return 1

    load_time = time.time() - load_start
    num_docs = len(ds)
    results["num_documents"] = num_docs
    print(f"  Loaded {num_docs:,} documents in {load_time:.1f}s")

    # Stage 2: DETECT
    print("\n[2/5] DETECT - Detecting text column...")

    # Handle --join-columns: concatenate columns into a 'text' column first
    if pre_extracted:
        # Pre-extraction already created a 'text' column from messages/joined columns
        text_column = "text"
        print(f"  Using pre-extracted text column (data already processed during load)")
    elif args.join_columns:
        join_cols = [c.strip() for c in args.join_columns.split(",")]
        missing = [c for c in join_cols if c not in ds.column_names]
        if missing:
            print(f"Error: --join-columns columns not found: {missing}. "
                  f"Available: {ds.column_names}")
            return 1
        print(f"  Joining columns: {join_cols}")
        ds = ds.map(
            lambda x: {"text": "\n\n".join(str(x[c]) for c in join_cols if x[c])},
            num_proc=args.num_proc,
            desc="Joining columns",
        )
        text_column = "text"
        print(f"  Created 'text' column from joined columns")
    elif args.text_column:
        text_column = args.text_column
        print(f"  Using specified column: {text_column}")
    elif args.midtrain_chat_messages:
        if "messages" not in ds.column_names:
            print("Error: --midtrain-chat-messages requires a 'messages' column, "
                  f"but dataset has: {ds.column_names}")
            return 1
        text_column = "messages"
        print(f"  Using messages column (--midtrain-chat-messages)")
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
            ds, hf_tokenizer, text_column, args.batch_size, is_messages,
            midtrain_chat_messages=args.midtrain_chat_messages,
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

    if args.chat_template and is_messages:
        # For chat template tokenization, export messages as-is (not converted to text)
        jsonl_path = output_dir / "messages.jsonl"
        print(f"  Saving messages to {jsonl_path} (chat-template mode)...")
        ds_export = ds.select_columns([text_column])
        if text_column != "messages":
            ds_export = ds_export.rename_column(text_column, "messages")
        ds_export.to_json(str(jsonl_path))
    else:
        jsonl_path = output_dir / "dataset.jsonl"

        # Convert messages to text if needed
        if is_messages:
            if args.midtrain_chat_messages:
                print("  Converting messages to plain text (midtrain_chat_messages mode)...")
            else:
                print(f"  Converting messages to text (skip_chat_template={args.skip_chat_template})...")
            ds = ds.map(
                lambda x: convert_messages_to_text(
                    x, text_column, hf_tokenizer, args.skip_chat_template,
                    midtrain_chat_messages=args.midtrain_chat_messages,
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
    elif args.chat_template:
        # Use chat template tokenizer (produces loss masks for SFT)
        if not args.tokenizer_path:
            print("\nError: --chat-template requires --tokenizer-path")
            results["status"] = "failed"
            results["error"] = "--chat-template requires --tokenizer-path"
        else:
            print("\n[5/5] TOKENIZE - Running chat template tokenization...")
            tokenize_start = time.time()

            dir_name = output_dir.name
            output_prefix = str(output_dir / dir_name)

            success = run_preprocess_data_with_chat_template(
                input_path=str(jsonl_path),
                output_prefix=output_prefix,
                tokenizer_path=args.tokenizer_path,
                workers=args.tokenize_workers,
            )

            tokenize_time = time.time() - tokenize_start
            results["tokenized"] = success
            results["tokenize_time"] = tokenize_time

            if success:
                bin_path = f"{output_prefix}_messages_document.bin"
                label_bin_path = f"{output_prefix}_messages_label_document.bin"
                results["bin_path"] = bin_path
                results["label_bin_path"] = label_bin_path
                print(f"\n  Tokenization complete in {tokenize_time:.1f}s")
                print(f"  Output: {output_prefix}_messages_document.bin/.idx")
                print(f"  Labels: {output_prefix}_messages_label_document.bin/.idx")
            else:
                results["status"] = "failed"
                results["error"] = "Chat template tokenization failed"
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
        if args.chat_template:
            data_path = f"{output_dir}/{dir_name}_messages_document"
            label_path = f"{output_dir}/{dir_name}_messages_label_document"
            print(f"\nFor GPT-NeoX SFT training config:")
            print(f'  "train_data_paths": ["{data_path}"]')
            print(f'  "train_label_data_paths": ["{label_path}"]')
        else:
            data_path = f"{output_dir}/{dir_name}_text_document"
            print(f"\nFor GPT-NeoX training config:")
            print(f'  "train_data_paths": ["{data_path}"]')

    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
