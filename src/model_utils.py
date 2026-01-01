"""Model setup and training utilities."""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def setup_model_and_tokenizer(base_model: str, seed: int):
    """Set up model and tokenizer with quantization."""
    set_seed(seed)
    
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.float16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_lora_model(model):
    """Configure model with LoRA adapters."""
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Disable cache for gradient checkpointing compatibility
    model.config.use_cache = False
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # Rank - higher = more parameters, better performance, slower training
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def create_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir: str, 
                   batch_size: int, learning_rate: float, max_steps: int, 
                   validate_steps: int):
    """Create and configure trainer."""
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        optim="adamw_torch",
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=validate_steps,
        save_strategy="steps",
        save_steps=validate_steps,
        logging_steps=validate_steps,
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False,
        seed=42,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping], 
    )
    
    return trainer


def save_model(model, tokenizer, output_dir: str):
    """Save LoRA adapter and tokenizer."""
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"✅ Adapter saved to {adapter_dir}")
    return adapter_dir


def evaluate_model(trainer, test_dataset):
    """Evaluate model on test set."""
    print(f"Test dataset size: {len(test_dataset)}")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*50)
    print("📊 FINAL TEST METRICS")
    print("="*50)
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    print(f"\nAll metrics:")
    print(test_results)
    
    return test_results