"""Model setup and training utilities."""

import os
import torch
from loguru import logger
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
    """Set up base model and tokenizer with 4-bit quantization.
    
    Initializes a language model with BitsAndBytesConfig for memory-efficient
    4-bit quantization and sets up the corresponding tokenizer with proper
    padding token configuration.
    
    Args:
        base_model (str): HuggingFace model identifier or path to local model.
        seed (int): Random seed for reproducible model initialization.
        
    Returns:
        tuple: A tuple containing:
            - model: Quantized AutoModelForCausalLM ready for LoRA training
            - tokenizer: Configured AutoTokenizer with proper pad token
            
    Note:
        - Uses NF4 quantization with double quantization for optimal memory usage
        - Automatically detects and uses CUDA if available
        - Sets pad_token to eos_token if not already configured
        - Uses device_map="auto" for multi-GPU compatibility
        
    Example:
        >>> model, tokenizer = setup_model_and_tokenizer("microsoft/phi-3.5-mini", 42)
        >>> print(model.device)
        cuda:0
    """
    set_seed(seed)
    
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
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


def setup_lora_model(model, lora_config):
    """Configure quantized model with LoRA adapters for efficient fine-tuning.
    
    Prepares a quantized base model for LoRA (Low-Rank Adaptation) training by
    applying k-bit training preparation and adding trainable LoRA adapters to
    specific attention and feed-forward layer modules.
    
    Args:
        model: Base language model with quantization config (typically from setup_model_and_tokenizer).
        lora_config: Configuration object containing LoRA parameters (r, alpha, dropout, target_modules).
        
    Returns:
        PeftModel: Model with LoRA adapters applied, ready for efficient fine-tuning.
        
    LoRA Configuration (from config):
        - r: Low-rank dimension (higher = more parameters but better expressivity)
        - alpha: Scaling factor for LoRA updates  
        - target_modules: List of module names to apply LoRA to
        - dropout: Dropout rate for regularization
        - bias: Bias configuration for LoRA layers
        
    Note:
        - Disables model cache for gradient checkpointing compatibility
        - LoRA adapters applied to modules specified in config
        - Prints trainable parameter count after setup
        
    Example:
        >>> base_model, _ = setup_model_and_tokenizer("phi-3.5-mini", 42)
        >>> lora_model = setup_lora_model(base_model, cfg.lora)
        >>> # Outputs: trainable params: XXX || all params: YYY || trainable%: Z.ZZ%
    """
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Disable cache for gradient checkpointing compatibility
    model.config.use_cache = False
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def create_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir: str, 
                   training_config):
    """Create and configure HuggingFace Trainer for LoRA fine-tuning.
    
    Sets up a complete training pipeline with optimized arguments for LoRA
    fine-tuning including gradient accumulation, early stopping, and evaluation.
    
    Args:
        model: PEFT model with LoRA adapters (from setup_lora_model).
        tokenizer: Configured tokenizer for the model.
        train_dataset: Tokenized training dataset.
        eval_dataset: Tokenized validation dataset for evaluation.
        output_dir (str): Directory to save checkpoints and logs.
        training_config: Configuration object containing training parameters.
        
    Returns:
        Trainer: Configured HuggingFace Trainer ready for training.
        
    Training Configuration (from config):
        - batch_size: Per-device training batch size
        - learning_rate: Learning rate for optimizer
        - max_steps: Maximum number of training steps
        - validate_steps: Steps between evaluations and checkpoints
        - gradient_accumulation_steps: Effective batch size multiplier
        - warmup_ratio: Learning rate warmup fraction
        - early_stopping: Early stopping configuration
        
    Note:
        - Uses eval_loss as metric for best model selection
        - Saves model at each validation step
        - Includes comprehensive logging to output_dir/logs
        - Removes unused columns to optimize memory usage
        
    Example:
        >>> trainer = create_trainer(
        ...     model, tokenizer, train_ds, eval_ds, "./checkpoints", cfg.training
        ... )
        >>> trainer.train()
    """
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.scheduler.warmup_ratio,
        max_steps=training_config.max_steps,
        eval_strategy="steps",
        eval_steps=training_config.eval_steps,
        save_strategy="steps",
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
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
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=training_config.early_stopping.patience
    ) if training_config.early_stopping.enabled else None
    
    # Create trainer
    callbacks = [early_stopping] if early_stopping else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks, 
    )
    
    return trainer


def save_model(model, tokenizer, output_dir: str):
    """Save trained LoRA adapter and tokenizer to disk.
    
    Saves the LoRA adapter weights and tokenizer configuration to a subdirectory
    for later use in inference or continued training.
    
    Args:
        model: Trained PEFT model with LoRA adapters.
        tokenizer: Tokenizer used during training.
        output_dir (str): Base directory where adapter will be saved.
        
    Returns:
        str: Path to the saved adapter directory.
        
    Note:
        - Creates "lora_adapter" subdirectory in output_dir
        - Saves only the adapter weights, not the full base model
        - Includes tokenizer config for consistent inference setup
        - Prints confirmation message with save location
        
    Saved Files:
        - adapter_config.json: LoRA configuration parameters
        - adapter_model.safetensors: Trained LoRA weights
        - tokenizer files: tokenizer.json, vocab files, etc.
        
    Example:
        >>> adapter_path = save_model(trained_model, tokenizer, "./checkpoints")
        >>> print(adapter_path)
        "./checkpoints/lora_adapter"
    """
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.success(f"✅ Adapter saved to {adapter_dir}")
    return adapter_dir


def evaluate_model(trainer, test_dataset):
    """Evaluate trained model performance on test dataset.
    
    Runs comprehensive evaluation on the test set and displays formatted
    results including loss metrics and dataset statistics.
    
    Args:
        trainer: Trained HuggingFace Trainer instance.
        test_dataset: Tokenized test dataset for final evaluation.
        
    Returns:
        dict: Dictionary containing evaluation metrics including 'eval_loss' and other metrics.
        
    Side Effects:
        - Prints test dataset size
        - Displays formatted evaluation results with decorative headers
        - Shows primary eval_loss metric prominently
        - Prints complete metrics dictionary for detailed analysis
        
    Note:
        - Uses same evaluation configuration as validation during training
        - Provides final performance assessment after training completion
        - Results can be used for model comparison and performance reporting
        
    Example:
        >>> results = evaluate_model(trainer, test_dataset)
        >>> logger.info(f"Final test loss: {results['eval_loss']:.4f}")
        Test dataset size: 100
        ==================================================
        📊 FINAL TEST METRICS
        ==================================================
        Test Loss: 0.1234
    """
    logger.info(f"Test dataset size: {len(test_dataset)}")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    logger.info("📊 FINAL TEST METRICS")
    logger.info(f"Test Loss: {test_results['eval_loss']:.4f}")
    logger.info(f"All metrics:")
    logger.info(test_results)
    
    return test_results