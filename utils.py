"""Simplified utilities - combines model_utils, data_utils, and inference_utils."""

import gc
import os
import torch
import json
from typing import Dict, List, Tuple, Any
from loguru import logger
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    EarlyStoppingCallback, set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


class SimpleTrainingConfig:
    """Simplified training configuration."""
    def __init__(self, cfg):
        self.batch_size = getattr(cfg, 'batch_size', 4)
        self.learning_rate = float(getattr(cfg, 'learning_rate', 2e-4))  
        self.num_epochs = getattr(cfg, 'num_epochs', 3)
        self.gradient_accumulation_steps = getattr(cfg, 'gradient_accumulation_steps', 8)
        self.logging_steps = getattr(cfg, 'logging_steps', 10)
        self.early_stopping_patience = getattr(cfg, 'early_stopping_patience', 3)


def load_and_prepare_data(file_path: str, cfg, seed: int = 42):
    """Load dataset and prepare for training."""
    logger.info("📚 Loading dataset...")
    
    # Load raw data
    raw_dataset = load_dataset("json", data_files=file_path)["train"]
    
    # Split data
    test_size = cfg.test_split + cfg.validation_split
    val_ratio = cfg.validation_split / test_size
    train_val = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    val_test = train_val["test"].train_test_split(test_size=val_ratio, seed=seed)
    
    dataset = DatasetDict({
        "train": train_val["train"],
        "validation": val_test["train"],  
        "test": val_test["test"]
    })
    
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
    return dataset


def setup_model(model_name: str, seed: int):
    """Setup model and tokenizer with quantization."""
    logger.info("⚙️ Setting up model...")
    set_seed(seed)
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        quantization_config=bnb_config,
        dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def setup_lora(model, cfg):
    """Apply LoRA configuration to model."""
    logger.info("🔧 Configuring LoRA...")
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
    peft_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def prepare_datasets(raw_dataset: DatasetDict, tokenizer, cfg):
    """Format and tokenize datasets."""
    logger.info("📊 Preparing datasets...")
    
    def format_example(example):
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}
    
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=cfg.max_length if hasattr(cfg, 'max_length') else 512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Process datasets
    formatted = raw_dataset.map(format_example, remove_columns=raw_dataset["train"].column_names)
    tokenized = formatted.map(tokenize_batch, batched=True, remove_columns=["text"])
    
    return (tokenized["train"].with_format("torch"),
            tokenized["validation"].with_format("torch"), 
            tokenized["test"].with_format("torch"))


def create_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir: str, cfg):
    """Create trainer with simplified configuration."""
    logger.info("🏃 Creating trainer...")
    
    training_cfg = SimpleTrainingConfig(cfg)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.num_epochs,
        per_device_train_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        logging_steps=training_cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=training_cfg.logging_steps,
        save_steps=training_cfg.logging_steps * 2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=None,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_cfg.early_stopping_patience)]
    )
    
    return trainer


def save_model(model, tokenizer, output_dir: str) -> str:
    """Save model adapter."""
    logger.info("💾 Saving model...")
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir


def load_inference_model(base_model: str, adapter_dir: str, cfg):
    """Load model for inference."""
    logger.info("🔄 Loading model for inference...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def run_inference(model, tokenizer, user_query: str, system_prompt: str, cfg):
    """Run inference with formatted output."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    # Format input
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=getattr(cfg, 'max_new_tokens', 512),
            temperature=getattr(cfg, 'temperature', 0.6),
            top_p=getattr(cfg, 'top_p', 0.9),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        response = full_response[len(input_text):].strip()
    
    logger.info(f"🎯 Response: {response}")
    return response


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()