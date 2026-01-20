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
Unit tests for prepare_hf_dataset.py

Tests cover:
- Argument parsing
- Output directory name generation
- Text column detection
- Token counting
- Messages to text conversion
- Subprocess execution
- Integration tests
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_hf_dataset import (
    convert_messages_to_text,
    count_tokens_batched,
    detect_text_column,
    generate_output_dir_name,
    parse_args,
    run_preprocess_data,
)


# =============================================================================
# Test: generate_output_dir_name
# =============================================================================


class TestGenerateOutputDirName:
    """Tests for generate_output_dir_name function."""

    def test_basic_dataset_with_subset_and_split(self):
        """Test with all components: dataset, subset, and split."""
        result = generate_output_dir_name(
            dataset="cais/wmdp-corpora",
            subset="bio-retain-corpus",
            split="train",
        )
        assert result == "wmdp-corpora_bio-retain-corpus_train"

    def test_dataset_without_subset(self):
        """Test with dataset and split only (no subset)."""
        result = generate_output_dir_name(
            dataset="cais/wmdp-corpora",
            subset=None,
            split="train",
        )
        assert result == "wmdp-corpora_train"

    def test_dataset_without_org_prefix(self):
        """Test dataset name without organization prefix."""
        result = generate_output_dir_name(
            dataset="squad",
            subset=None,
            split="validation",
        )
        assert result == "squad_validation"

    def test_different_splits(self):
        """Test various split names."""
        for split in ["train", "test", "validation", "train[:100]"]:
            result = generate_output_dir_name(
                dataset="org/dataset",
                subset=None,
                split=split,
            )
            assert result == f"dataset_{split}"

    def test_complex_subset_name(self):
        """Test with complex subset names containing hyphens."""
        result = generate_output_dir_name(
            dataset="org/dataset",
            subset="config-v2-en",
            split="train",
        )
        assert result == "dataset_config-v2-en_train"

    def test_empty_subset_string(self):
        """Test that empty string subset is treated as falsy."""
        result = generate_output_dir_name(
            dataset="org/dataset",
            subset="",
            split="train",
        )
        # Empty string is falsy, so subset should be omitted
        assert result == "dataset_train"

    def test_deeply_nested_dataset_path(self):
        """Test dataset path with multiple slashes (edge case)."""
        result = generate_output_dir_name(
            dataset="org/sub/dataset",
            subset="config",
            split="train",
        )
        # Should only take the last part
        assert result == "dataset_config_train"


# =============================================================================
# Test: detect_text_column
# =============================================================================


class TestDetectTextColumn:
    """Tests for detect_text_column function."""

    def test_detects_text_column(self):
        """Test detection of 'text' column."""
        ds = Dataset.from_dict({"text": ["hello", "world"], "id": [1, 2]})
        result = detect_text_column(ds)
        assert result == "text"

    def test_detects_messages_column(self):
        """Test detection of 'messages' column."""
        ds = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "hi"}],
                [{"role": "user", "content": "hello"}],
            ],
            "id": [1, 2],
        })
        result = detect_text_column(ds)
        assert result == "messages"

    def test_prefers_text_over_messages(self):
        """Test that 'text' is preferred when both columns exist."""
        ds = Dataset.from_dict({
            "text": ["hello", "world"],
            "messages": [
                [{"role": "user", "content": "hi"}],
                [{"role": "user", "content": "hello"}],
            ],
        })
        result = detect_text_column(ds)
        assert result == "text"

    def test_raises_error_when_no_text_column(self):
        """Test that ValueError is raised when neither column exists."""
        ds = Dataset.from_dict({"content": ["hello", "world"], "id": [1, 2]})
        with pytest.raises(ValueError) as exc_info:
            detect_text_column(ds)
        assert "Could not auto-detect text column" in str(exc_info.value)
        assert "content" in str(exc_info.value)
        assert "id" in str(exc_info.value)

    def test_empty_dataset(self):
        """Test detection on empty dataset with correct columns."""
        ds = Dataset.from_dict({"text": [], "id": []})
        result = detect_text_column(ds)
        assert result == "text"


# =============================================================================
# Test: count_tokens_batched
# =============================================================================


class TestCountTokensBatched:
    """Tests for count_tokens_batched function."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer that returns predictable lengths."""
        tokenizer = mock.MagicMock()

        def tokenize_side_effect(texts, **kwargs):
            # Return length equal to word count for simplicity
            lengths = [len(text.split()) for text in texts]
            return {"length": lengths}

        tokenizer.side_effect = tokenize_side_effect
        tokenizer.return_value = {"length": []}
        return tokenizer

    def test_count_text_column(self, mock_tokenizer, capsys):
        """Test token counting for text column."""
        ds = Dataset.from_dict({
            "text": [
                "hello world",  # 2 words
                "this is a test",  # 4 words
                "another example here",  # 3 words
            ]
        })

        def tokenize_side_effect(texts, **kwargs):
            lengths = [len(text.split()) for text in texts]
            return {"length": lengths}

        mock_tokenizer.side_effect = tokenize_side_effect

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="text",
            batch_size=10,
            is_messages=False,
        )

        assert result == 9  # 2 + 4 + 3

    def test_count_with_small_batches(self, mock_tokenizer, capsys):
        """Test that batching works correctly with small batch sizes."""
        ds = Dataset.from_dict({
            "text": ["word"] * 25,  # 25 documents
        })

        call_count = 0

        def tokenize_side_effect(texts, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"length": [1] * len(texts)}

        mock_tokenizer.side_effect = tokenize_side_effect

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="text",
            batch_size=10,
            is_messages=False,
        )

        assert result == 25
        assert call_count == 3  # 10 + 10 + 5

    def test_count_messages_with_chat_template(self, capsys):
        """Test token counting for messages column with chat template."""
        ds = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "hello"}],
                [{"role": "user", "content": "world"}],
            ]
        })

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = lambda msgs, **kwargs: f"formatted: {msgs[0]['content']}"

        def tokenize_side_effect(texts, **kwargs):
            return {"length": [len(text.split()) for text in texts]}

        mock_tokenizer.side_effect = tokenize_side_effect

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="messages",
            batch_size=10,
            is_messages=True,
        )

        # "formatted: hello" = 2 words, "formatted: world" = 2 words
        assert result == 4

    def test_count_messages_fallback_on_error(self, capsys):
        """Test fallback to stringified messages when chat template fails."""
        ds = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "test"}],
            ]
        })

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")

        def tokenize_side_effect(texts, **kwargs):
            return {"length": [len(text) for text in texts]}  # Character count

        mock_tokenizer.side_effect = tokenize_side_effect

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="messages",
            batch_size=10,
            is_messages=True,
        )

        # Stringified message: "[{'role': 'user', 'content': 'test'}]"
        assert result > 0

    def test_count_empty_dataset(self, mock_tokenizer, capsys):
        """Test counting tokens in empty dataset."""
        ds = Dataset.from_dict({"text": []})

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="text",
            batch_size=10,
            is_messages=False,
        )

        assert result == 0


# =============================================================================
# Test: convert_messages_to_text
# =============================================================================


class TestConvertMessagesToText:
    """Tests for convert_messages_to_text function."""

    def test_convert_with_chat_template(self):
        """Test conversion using chat template."""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<user>hello</user>"

        example = {"messages": [{"role": "user", "content": "hello"}]}

        result = convert_messages_to_text(
            example=example,
            text_column="messages",
            tokenizer=mock_tokenizer,
            skip_chat_template=False,
        )

        assert result == {"text": "<user>hello</user>"}
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_convert_skip_chat_template(self):
        """Test conversion with stringified messages."""
        mock_tokenizer = mock.MagicMock()

        example = {"messages": [{"role": "user", "content": "hello"}]}

        result = convert_messages_to_text(
            example=example,
            text_column="messages",
            tokenizer=mock_tokenizer,
            skip_chat_template=True,
        )

        assert result["text"] == str([{"role": "user", "content": "hello"}])
        mock_tokenizer.apply_chat_template.assert_not_called()

    def test_convert_fallback_on_template_error(self, capsys):
        """Test fallback to stringified when chat template fails."""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("Template error")

        example = {"messages": [{"role": "user", "content": "hello"}]}

        result = convert_messages_to_text(
            example=example,
            text_column="messages",
            tokenizer=mock_tokenizer,
            skip_chat_template=False,
        )

        # Should fall back to stringified
        assert "user" in result["text"]
        assert "hello" in result["text"]

        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_convert_multiturn_conversation(self):
        """Test conversion of multi-turn conversation."""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<user>hi</user><assistant>hello</assistant>"

        example = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }

        result = convert_messages_to_text(
            example=example,
            text_column="messages",
            tokenizer=mock_tokenizer,
            skip_chat_template=False,
        )

        assert result == {"text": "<user>hi</user><assistant>hello</assistant>"}

    def test_convert_different_column_name(self):
        """Test conversion with different column name."""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "converted"

        example = {"conversation": [{"role": "user", "content": "test"}]}

        result = convert_messages_to_text(
            example=example,
            text_column="conversation",
            tokenizer=mock_tokenizer,
            skip_chat_template=False,
        )

        assert result == {"text": "converted"}


# =============================================================================
# Test: run_preprocess_data
# =============================================================================


class TestRunPreprocessData:
    """Tests for run_preprocess_data function."""

    def test_successful_execution(self, capsys):
        """Test successful subprocess execution."""
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_process = mock.MagicMock()
            mock_process.stdout = iter(["Processing...\n", "Done.\n"])
            mock_process.wait.return_value = None
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            result = run_preprocess_data(
                input_path="/path/to/input.jsonl",
                output_prefix="/path/to/output",
                vocab_file="/path/to/vocab.json",
                workers=4,
                num_docs=100,
            )

            assert result is True

            # Check command was constructed correctly
            call_args = mock_popen.call_args
            cmd = call_args[0][0]
            assert "--input" in cmd
            assert "/path/to/input.jsonl" in cmd
            assert "--output-prefix" in cmd
            assert "/path/to/output" in cmd
            assert "--vocab-file" in cmd
            assert "/path/to/vocab.json" in cmd
            assert "--workers" in cmd
            assert "4" in cmd
            assert "--num-docs" in cmd
            assert "100" in cmd
            assert "--append-eod" in cmd
            assert "--tokenizer-type" in cmd
            assert "HFTokenizer" in cmd

    def test_failed_execution(self, capsys):
        """Test failed subprocess execution."""
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_process = mock.MagicMock()
            mock_process.stdout = iter(["Error occurred\n"])
            mock_process.wait.return_value = None
            mock_process.returncode = 1
            mock_popen.return_value = mock_process

            result = run_preprocess_data(
                input_path="/path/to/input.jsonl",
                output_prefix="/path/to/output",
                vocab_file="/path/to/vocab.json",
                workers=4,
                num_docs=100,
            )

            assert result is False

            captured = capsys.readouterr()
            assert "Error" in captured.out

    def test_output_streaming(self, capsys):
        """Test that output is streamed with prefix."""
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_process = mock.MagicMock()
            mock_process.stdout = iter(["Line 1\n", "Line 2\n"])
            mock_process.wait.return_value = None
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            run_preprocess_data(
                input_path="/path/to/input.jsonl",
                output_prefix="/path/to/output",
                vocab_file="/path/to/vocab.json",
                workers=4,
                num_docs=100,
            )

            captured = capsys.readouterr()
            assert "[PREPROCESS] Line 1" in captured.out
            assert "[PREPROCESS] Line 2" in captured.out


# =============================================================================
# Test: parse_args
# =============================================================================


class TestParseArgs:
    """Tests for argument parsing."""

    def test_required_argument_dataset(self):
        """Test that --dataset is required."""
        with pytest.raises(SystemExit):
            with mock.patch("sys.argv", ["prog"]):
                parse_args()

    def test_basic_arguments(self):
        """Test basic required and default arguments."""
        with mock.patch("sys.argv", ["prog", "--dataset", "org/dataset"]):
            args = parse_args()

            assert args.dataset == "org/dataset"
            assert args.subset is None
            assert args.split == "train"
            assert args.output_dir is None
            assert args.output_base == "/projects/a5k/public/data"
            assert args.text_column is None
            assert args.num_proc == 16
            assert args.batch_size == 10000
            assert args.tokenize_workers == 16  # Defaults to num_proc

    def test_all_arguments(self):
        """Test all arguments."""
        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "org/dataset",
            "--subset", "config",
            "--split", "test",
            "--output-dir", "/custom/path",
            "--output-base", "/base/path",
            "--text-column", "content",
            "--vocab-file", "/path/to/vocab.json",
            "--hf-tokenizer", "gpt2",
            "--skip-count",
            "--skip-tokenize",
            "--count-only",
            "--skip-chat-template",
            "--num-proc", "8",
            "--batch-size", "5000",
            "--tokenize-workers", "4",
        ]):
            args = parse_args()

            assert args.dataset == "org/dataset"
            assert args.subset == "config"
            assert args.split == "test"
            assert args.output_dir == "/custom/path"
            assert args.output_base == "/base/path"
            assert args.text_column == "content"
            assert args.vocab_file == "/path/to/vocab.json"
            assert args.hf_tokenizer == "gpt2"
            assert args.skip_count is True
            assert args.skip_tokenize is True
            assert args.count_only is True
            assert args.skip_chat_template is True
            assert args.num_proc == 8
            assert args.batch_size == 5000
            assert args.tokenize_workers == 4

    def test_tokenize_workers_defaults_to_num_proc(self):
        """Test that tokenize_workers defaults to num_proc when not specified."""
        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "org/dataset",
            "--num-proc", "32",
        ]):
            args = parse_args()
            assert args.tokenize_workers == 32

    def test_tokenize_workers_can_differ_from_num_proc(self):
        """Test that tokenize_workers can be set independently."""
        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "org/dataset",
            "--num-proc", "32",
            "--tokenize-workers", "8",
        ]):
            args = parse_args()
            assert args.num_proc == 32
            assert args.tokenize_workers == 8


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_full_pipeline_text_column_skip_tokenize(self, temp_output_dir):
        """Test full pipeline with text column, skipping tokenization."""
        from prepare_hf_dataset import main

        # Create a small test dataset
        ds = Dataset.from_dict({
            "text": ["Hello world", "This is a test", "Another document"],
            "id": [1, 2, 3],
        })

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "test/dataset",
            "--output-dir", temp_output_dir,
            "--skip-tokenize",
            "--hf-tokenizer", "gpt2",
            "--batch-size", "10",
        ]):
            with mock.patch("prepare_hf_dataset.load_dataset", return_value=ds):
                result = main()

        assert result == 0

        # Check output files
        jsonl_path = Path(temp_output_dir) / "dataset.jsonl"
        results_path = Path(temp_output_dir) / "pipeline_results.json"

        assert jsonl_path.exists()
        assert results_path.exists()

        # Verify results
        with open(results_path) as f:
            results = json.load(f)

        assert results["status"] == "completed"
        assert results["num_documents"] == 3
        assert results["text_column"] == "text"
        assert results["is_messages"] is False
        assert results["tokenized"] is False

    def test_full_pipeline_messages_column(self, temp_output_dir):
        """Test full pipeline with messages column."""
        from prepare_hf_dataset import main

        # Create a test dataset with messages
        ds = Dataset.from_dict({
            "messages": [
                [{"role": "user", "content": "hello"}],
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            ],
            "id": [1, 2],
        })

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "test/chat-dataset",
            "--output-dir", temp_output_dir,
            "--skip-tokenize",
            "--skip-count",
            "--skip-chat-template",  # Use stringified for simplicity
            "--num-proc", "1",  # Use single process
            "--hf-tokenizer", "gpt2",  # Use real tokenizer to avoid pickling issues
        ]):
            with mock.patch("prepare_hf_dataset.load_dataset", return_value=ds):
                # Use real tokenizer to avoid pickling issues with mock objects
                # when datasets tries to fingerprint the map function
                result = main()

        assert result == 0

        # Check JSONL was created
        jsonl_path = Path(temp_output_dir) / "dataset.jsonl"
        assert jsonl_path.exists()

        # Verify results
        results_path = Path(temp_output_dir) / "pipeline_results.json"
        with open(results_path) as f:
            results = json.load(f)

        assert results["status"] == "completed"
        assert results["text_column"] == "messages"
        assert results["is_messages"] is True

    def test_count_only_mode(self, temp_output_dir):
        """Test count-only mode doesn't create JSONL."""
        from prepare_hf_dataset import main

        ds = Dataset.from_dict({
            "text": ["Hello world", "Test"],
        })

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "test/dataset",
            "--output-dir", temp_output_dir,
            "--count-only",
            "--hf-tokenizer", "gpt2",
        ]):
            with mock.patch("prepare_hf_dataset.load_dataset", return_value=ds):
                result = main()

        assert result == 0

        # JSONL should not exist in count-only mode
        jsonl_path = Path(temp_output_dir) / "dataset.jsonl"
        assert not jsonl_path.exists()

    def test_pipeline_with_specified_text_column(self, temp_output_dir):
        """Test pipeline with explicitly specified text column."""
        from prepare_hf_dataset import main

        ds = Dataset.from_dict({
            "content": ["Hello", "World"],
            "metadata": ["a", "b"],
        })

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "test/dataset",
            "--output-dir", temp_output_dir,
            "--text-column", "content",
            "--skip-tokenize",
            "--skip-count",
            "--hf-tokenizer", "gpt2",
        ]):
            with mock.patch("prepare_hf_dataset.load_dataset", return_value=ds):
                with mock.patch("prepare_hf_dataset.AutoTokenizer.from_pretrained"):
                    result = main()

        assert result == 0

        # Check results
        results_path = Path(temp_output_dir) / "pipeline_results.json"
        with open(results_path) as f:
            results = json.load(f)

        assert results["text_column"] == "content"

    def test_load_dataset_failure(self, temp_output_dir, capsys):
        """Test handling of dataset loading failure."""
        from prepare_hf_dataset import main

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "nonexistent/dataset",
            "--output-dir", temp_output_dir,
        ]):
            with mock.patch(
                "prepare_hf_dataset.load_dataset",
                side_effect=Exception("Dataset not found"),
            ):
                result = main()

        assert result == 1

        captured = capsys.readouterr()
        assert "Error loading dataset" in captured.out


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_generate_output_dir_special_characters(self):
        """Test output dir name with special characters in dataset name."""
        # Underscores and hyphens should be preserved
        result = generate_output_dir_name(
            dataset="org/my-data_set",
            subset="config_v1",
            split="train",
        )
        assert result == "my-data_set_config_v1_train"

    def test_detect_text_column_case_sensitive(self):
        """Test that column detection is case-sensitive."""
        ds = Dataset.from_dict({"TEXT": ["hello"], "Messages": ["hi"]})
        with pytest.raises(ValueError):
            detect_text_column(ds)

    def test_count_tokens_single_document(self):
        """Test token counting with single document."""
        ds = Dataset.from_dict({"text": ["single document"]})

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.side_effect = lambda texts, **kwargs: {"length": [10]}

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="text",
            batch_size=100,
            is_messages=False,
        )

        assert result == 10

    def test_convert_empty_messages(self):
        """Test conversion of empty messages list."""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = ""

        example = {"messages": []}

        result = convert_messages_to_text(
            example=example,
            text_column="messages",
            tokenizer=mock_tokenizer,
            skip_chat_template=False,
        )

        assert result == {"text": ""}

    def test_batch_size_larger_than_dataset(self):
        """Test when batch size is larger than dataset."""
        ds = Dataset.from_dict({"text": ["a", "b"]})

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.side_effect = lambda texts, **kwargs: {"length": [1] * len(texts)}

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="text",
            batch_size=1000,
            is_messages=False,
        )

        assert result == 2

    def test_batch_size_equals_dataset_size(self):
        """Test when batch size equals dataset size."""
        ds = Dataset.from_dict({"text": ["a", "b", "c"]})

        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.side_effect = lambda texts, **kwargs: {"length": [1] * len(texts)}

        result = count_tokens_batched(
            ds=ds,
            tokenizer=mock_tokenizer,
            text_column="text",
            batch_size=3,
            is_messages=False,
        )

        assert result == 3


# =============================================================================
# Output File Format Tests
# =============================================================================


class TestOutputFileFormat:
    """Tests for output file format correctness."""

    def test_jsonl_output_format(self, tmp_path):
        """Test that JSONL output has correct format."""
        from prepare_hf_dataset import main

        ds = Dataset.from_dict({
            "text": ["Line 1", "Line 2", "Line 3"],
        })

        output_dir = str(tmp_path)

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "test/dataset",
            "--output-dir", output_dir,
            "--skip-tokenize",
            "--skip-count",
            "--hf-tokenizer", "gpt2",
        ]):
            with mock.patch("prepare_hf_dataset.load_dataset", return_value=ds):
                with mock.patch("prepare_hf_dataset.AutoTokenizer.from_pretrained"):
                    main()

        jsonl_path = tmp_path / "dataset.jsonl"

        # Read and verify JSONL format
        with open(jsonl_path) as f:
            lines = f.readlines()

        assert len(lines) == 3

        for line in lines:
            data = json.loads(line)
            assert "text" in data
            assert isinstance(data["text"], str)

    def test_results_json_format(self, tmp_path):
        """Test that results JSON has all expected fields."""
        from prepare_hf_dataset import main

        ds = Dataset.from_dict({"text": ["test"]})

        output_dir = str(tmp_path)

        with mock.patch("sys.argv", [
            "prog",
            "--dataset", "org/dataset",
            "--subset", "config",
            "--split", "test",
            "--output-dir", output_dir,
            "--skip-tokenize",
            "--hf-tokenizer", "gpt2",
        ]):
            with mock.patch("prepare_hf_dataset.load_dataset", return_value=ds):
                main()

        results_path = tmp_path / "pipeline_results.json"

        with open(results_path) as f:
            results = json.load(f)

        # Check required fields
        required_fields = [
            "dataset",
            "subset",
            "split",
            "status",
            "output_dir",
            "num_documents",
            "text_column",
            "is_messages",
            "elapsed_time",
        ]

        for field in required_fields:
            assert field in results, f"Missing field: {field}"

        # Check field values
        assert results["dataset"] == "org/dataset"
        assert results["subset"] == "config"
        assert results["split"] == "test"
        assert results["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
