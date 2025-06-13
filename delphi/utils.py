import math
from os import cpu_count
from typing import Any, TypeVar, Union, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import Tensor
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,  # type: ignore
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
    streaming: bool = False,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.
        streaming

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    _proc_kwargs = {
        "batched": True,
        "batch_size": 2048,
        "remove_columns": get_columns_all_equal(data),
    }

    if not streaming:
        _proc_kwargs["num_proc"] = num_proc
        _proc_kwargs["load_from_cache_file"] = load_from_cache_file

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        **_proc_kwargs,  # type: ignore
    )
    if streaming:
        return data.with_format(format)
    return data.with_format(format, columns=["input_ids"])


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


def load_tokenized_data(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "text",
    seed: int = 22,
    streaming: bool = False,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    Using this function ensures we are using the same tokens everywhere.
    """

    data = load_dataset(
        dataset_repo, name=dataset_name, split=dataset_split, streaming=streaming
    )
    data = data.shuffle(seed)
    tokens_ds = chunk_and_tokenize(
        data,  # type: ignore
        tokenizer,
        max_seq_len=ctx_len,
        text_key=column_name,
        streaming=streaming,
    )

    tokens = tokens_ds["input_ids"]

    return tokens


T = TypeVar("T")


def assert_type(typ: type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore


def to_int64_tensor(tensor: np.ndarray) -> Tensor:
    assert tensor.dtype in (
        np.uint16,
        np.int16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    )
    if tensor.dtype in (np.uint64, np.int64):
        return torch.from_numpy(tensor).to(torch.int64)
    og_shape = tensor.shape
    if tensor.dtype in (np.uint16, np.int16):
        signed_np_dtype, signed_torch_dtype = np.int16, torch.int16
        multiplier = 4
    else:
        signed_np_dtype, signed_torch_dtype = np.int32, torch.int32
        multiplier = 2
    t = torch.tensor(tensor.ravel().view(signed_np_dtype))
    result = torch.zeros(t.shape[0] * multiplier, dtype=signed_torch_dtype)
    result[::multiplier] = t
    return result.view(torch.int64).view(og_shape)
