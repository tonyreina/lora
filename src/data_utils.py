"""Data processing utilities for medical LLM training."""

import json
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from loguru import logger
import torch


def load_medical_dataset(file_path: str, data_config, seed: int = 42) -> DatasetDict:
    """Load and split medical dataset from JSONL file.
    
    Loads a medical dataset from a JSONL file and splits it into train/validation/test
    sets according to configured split ratios. Displays dataset information and a sample example.
    
    Args:
        file_path (str): Path to the JSONL file containing the medical dataset.
        data_config: Configuration object containing data processing parameters.
        seed (int, optional): Random seed for reproducible splits. Defaults to 42.
        
    Returns:
        DatasetDict: Dictionary containing 'train', 'validation', and 'test' datasets.
        
    Example:
        >>> dataset = load_medical_dataset('data/medical_data.jsonl', cfg.data)
        >>> len(dataset['train'])
        800  # if original dataset had 1000 examples with default 80% train split
    """
    raw_dataset = load_dataset("json", data_files=file_path)["train"]
    
    # Calculate split sizes from config
    test_size = data_config.test_split + data_config.validation_split
    val_ratio = data_config.validation_split / test_size
    
    # Split into train/validation/test using config ratios
    train_val = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    val_test = train_val["test"].train_test_split(test_size=val_ratio, seed=seed)
    
    dataset = DatasetDict({
        "train": train_val["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    
    logger.info(dataset)
    
    # Show example
    example = dataset["train"][0]
    logger.info("Example from training set:")
    preview_length = getattr(data_config.display, 'example_preview_length', 100)
    logger.info({k: (str(v)[:preview_length] + "...") if len(str(v)) > preview_length else v for k, v in example.items()})
    
    return dataset


def format_example(example: Dict[str, str], tokenizer, system_prompt: str) -> Dict[str, str]:
    """Apply chat template to format training examples for conversation.
    
    Formats a single training example by applying the tokenizer's chat template
    to create a conversation between system, user, and assistant.
    
    Args:
        example (Dict[str, str]): Training example containing 'instruction' and 'response' keys.
        tokenizer: HuggingFace tokenizer with chat template support.
        system_prompt (str): System message to include at the start of each conversation.
        
    Returns:
        Dict[str, str]: Original example with added 'text' field containing formatted conversation.
        
    Note:
        The formatted text follows the pattern: system_prompt -> user_instruction -> assistant_response
    """
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
    """Tokenize a batch of text examples for model training.
    
    Processes a batch of text examples by tokenizing them with padding and truncation.
    Creates both input_ids and labels tensors suitable for language model training.
    
    Args:
        batch (Dict[str, List[str]]): Batch containing 'text' field with list of text examples.
        tokenizer: HuggingFace tokenizer for text tokenization.
        max_length (int): Maximum sequence length for tokenization.
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'input_ids': Token IDs for model input
            - 'attention_mask': Attention mask for padded tokens  
            - 'labels': Copy of input_ids used for loss calculation
            
    Note:
        Uses max_length padding and truncation to ensure consistent tensor shapes.
    """
    tokenized = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def prepare_datasets(raw_dataset: DatasetDict, tokenizer, data_config):
    """Prepare datasets for training by formatting and tokenizing all examples.
    
    Complete preprocessing pipeline that formats raw dataset examples using chat templates,
    tokenizes them for model training, and prepares train/validation/test splits.
    
    Args:
        raw_dataset (DatasetDict): Raw dataset containing train/validation/test splits.
        tokenizer: HuggingFace tokenizer with chat template support.
        data_config: Configuration object containing preprocessing parameters.
        
    Returns:
        tuple: Three PyTorch datasets (train_dataset, eval_dataset, test_dataset)
            ready for use with HuggingFace Trainer.
            
    Side Effects:
        - Prints dataset statistics and sample tokenized output
        - Removes original columns and adds tokenized fields
        - Converts datasets to PyTorch format
        
    Example:
        >>> train_ds, eval_ds, test_ds = prepare_datasets(
        ...     raw_dataset, tokenizer, cfg.data
        ... )
        >>> train_ds[0]['input_ids'].shape
        torch.Size([512])  # Based on config max_length
    """
    # Apply chat template to all examples
    formatted = raw_dataset.map(
        lambda x: format_example(x, tokenizer, data_config.preprocessing.system_prompt),
        remove_columns=raw_dataset["train"].column_names
    )
    
    # Tokenize all examples
    tokenized = formatted.map(
        lambda x: tokenize_batch(x, tokenizer, data_config.preprocessing.max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    train_dataset = tokenized["train"].with_format("torch")
    eval_dataset = tokenized["validation"].with_format("torch")
    test_dataset = tokenized["test"].with_format("torch")
    
    logger.info(f"Training examples  : {len(train_dataset)}")
    logger.info(f"Validation examples: {len(eval_dataset)}")
    logger.info(f"Test examples      : {len(test_dataset)}")
    
    preview_tokens = getattr(data_config.display, 'token_preview_length', 160)
    logger.info(f"First {preview_tokens} tokens of training example:")
    logger.info(tokenizer.decode(train_dataset[0]["input_ids"][:preview_tokens]))
    
    return train_dataset, eval_dataset, test_dataset