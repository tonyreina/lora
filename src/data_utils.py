"""Data processing utilities for medical LLM training."""

import json
from typing import Dict, List
from datasets import load_dataset, DatasetDict
import torch


def load_medical_dataset(file_path: str) -> DatasetDict:
    """Load and split medical dataset from JSONL file."""
    raw_dataset = load_dataset("json", data_files=file_path)["train"]
    
    # Split into train/validation/test (80%/10%/10%)
    train_val = raw_dataset.train_test_split(test_size=0.2, seed=42)
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)
    
    dataset = DatasetDict({
        "train": train_val["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    
    print(dataset)
    
    # Show example
    example = dataset["train"][0]
    print("\nExample from training set:")
    print({k: (str(v)[:100] + "...") if len(str(v)) > 100 else v for k, v in example.items()})
    
    return dataset


def format_example(example: Dict[str, str], tokenizer, system_prompt: str) -> Dict[str, str]:
    """Apply chat template to each example."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example.get("instruction", "")},
        {"role": "assistant", "content": example.get("response", "")},
    ]
    example["text"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return example


def tokenize_batch(batch: Dict[str, List[str]], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of examples."""
    tokenized = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def prepare_datasets(raw_dataset: DatasetDict, tokenizer, system_prompt: str, max_length: int):
    """Format and tokenize datasets for training."""
    # Apply chat template to all examples
    formatted = raw_dataset.map(
        lambda x: format_example(x, tokenizer, system_prompt),
        remove_columns=raw_dataset["train"].column_names
    )
    
    # Tokenize all examples
    tokenized = formatted.map(
        lambda x: tokenize_batch(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    train_dataset = tokenized["train"].with_format("torch")
    eval_dataset = tokenized["validation"].with_format("torch")
    test_dataset = tokenized["test"].with_format("torch")
    
    print(f"Training examples  : {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    print(f"Test examples      : {len(test_dataset)}")
    
    print("\nFirst 120 tokens of training example:")
    print(tokenizer.decode(train_dataset[0]["input_ids"][:120]))
    
    return train_dataset, eval_dataset, test_dataset