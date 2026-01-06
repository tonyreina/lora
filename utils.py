"""Simplified utilities - combines model_utils, data_utils, and inference_utils."""

import gc
import os

import torch
from datasets import DatasetDict, load_dataset
from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def setup_hf_cache():
    """Setup HuggingFace cache settings for better performance."""
    # Set cache directory (optional, HF uses default if not set)
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables for better caching
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")

    logger.info(f"📁 HuggingFace cache directory: {cache_dir}")
    return cache_dir


def check_model_cache(model_name: str):
    """Check if model is already cached."""
    import os

    from transformers.utils import TRANSFORMERS_CACHE

    # Try to find model in cache
    cache_dir = TRANSFORMERS_CACHE or os.path.expanduser("~/.cache/huggingface/transformers")

    # Look for model files in cache
    model_cache_exists = False
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if model_name.replace("/", "--") in item:
                model_cache_exists = True
                break

    status = "✅ cached" if model_cache_exists else "⬇️  will download"
    logger.info(f"🔍 Model {model_name}: {status}")
    return model_cache_exists


class SimpleTrainingConfig:
    """Simplified training configuration."""

    def __init__(self, cfg):
        self.batch_size = getattr(cfg, "batch_size", 4)
        self.learning_rate = float(getattr(cfg, "learning_rate", 2e-4))
        self.max_steps = getattr(cfg, "max_steps", 100)  # Default to 100 steps if not specified
        self.gradient_accumulation_steps = getattr(cfg, "gradient_accumulation_steps", 8)
        self.logging_steps = getattr(cfg, "logging_steps", 10)
        self.early_stopping_patience = getattr(cfg, "early_stopping_patience", 3)


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

    dataset = DatasetDict(
        {"train": train_val["train"], "validation": val_test["train"], "test": val_test["test"]}
    )

    logger.info(
        f"Train: {len(dataset['train'])}, "
        f"Val: {len(dataset['validation'])}, "
        f"Test: {len(dataset['test'])}"
    )
    return dataset


def setup_model(model_name: str, seed: int):
    """Setup model and tokenizer with quantization."""
    logger.info("⚙️ Setting up model...")
    set_seed(seed)

    # Setup cache and check if model is cached
    setup_hf_cache()
    check_model_cache(model_name)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model with caching
    try:
        # First try to load from cache without downloading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.float16,
            local_files_only=True,  # Use cache first
        )
        logger.info(f"✅ Loaded {model_name} from cache")
    except (OSError, ValueError):
        # If not in cache, download
        logger.info(f"📥 Downloading {model_name} (not in cache)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.float16,
        )

    # Load tokenizer with caching
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    except (OSError, ValueError):
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
            max_length=cfg.max_length if hasattr(cfg, "max_length") else 512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    # Process datasets
    formatted = raw_dataset.map(format_example, remove_columns=raw_dataset["train"].column_names)
    tokenized = formatted.map(tokenize_batch, batched=True, remove_columns=["text"])

    return (
        tokenized["train"].with_format("torch"),
        tokenized["validation"].with_format("torch"),
        tokenized["test"].with_format("torch"),
    )


def create_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir: str, cfg):
    """Create trainer with simplified configuration."""
    logger.info("🏃 Creating trainer...")

    training_cfg = SimpleTrainingConfig(cfg)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=training_cfg.max_steps,
        per_device_train_batch_size=training_cfg.batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        logging_steps=training_cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=training_cfg.logging_steps,
        save_steps=training_cfg.logging_steps * 2,
        save_strategy="steps",
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
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=training_cfg.early_stopping_patience)
        ],
    )

    # Log the number of steps
    logger.info(
        f"📊 Trainer will run for up to {training_cfg.max_steps} steps for fine-tuning model."
    )

    return trainer


def save_model(model, tokenizer, output_dir: str, base_model_name: str = None) -> str:
    """Save model adapter with custom directory name."""
    logger.info("💾 Saving model...")

    # Extract model name from base_model_name
    # (e.g., "microsoft/Phi-4-mini-instruct" -> "Phi-4-mini-instruct")
    if base_model_name:
        model_name_part = (
            base_model_name.split("/")[-1] if "/" in base_model_name else base_model_name
        )
        adapter_dir_name = f"my_custom_llm_{model_name_part}"
    else:
        adapter_dir_name = "lora_adapter"  # fallback to original name

    adapter_dir = os.path.join(output_dir, adapter_dir_name)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"✅ Model saved to: {adapter_dir}")
    return adapter_dir


def load_inference_model(base_model: str, adapter_dir: str, cfg):
    """Load model for inference."""
    logger.info("🔄 Loading model for inference...")

    # Setup cache and check if model is cached
    setup_hf_cache()
    check_model_cache(base_model)

    # Create offload directory if needed
    offload_dir = cfg.get("offload_dir")
    if offload_dir:
        os.makedirs(offload_dir, exist_ok=True)

    # Load base model with caching
    try:
        # First try to load from cache
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            offload_folder=offload_dir if offload_dir else None,
        )
        logger.info(f"✅ Loaded {base_model} from cache")
    except (OSError, ValueError):
        # If not in cache, download
        logger.info(f"📥 Downloading {base_model} (not in cache)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            type=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir if offload_dir else None,
        )

    # Load adapter
    model = PeftModel.from_pretrained(
        model, adapter_dir, offload_folder=offload_dir if offload_dir else None
    )

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
            max_new_tokens=getattr(cfg, "max_new_tokens", 512),
            temperature=getattr(cfg, "temperature", 0.6),
            top_p=getattr(cfg, "top_p", 0.9),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        # More robust extraction: decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    logger.info(f"🎯 Response: {response}")
    return response


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def print_gpu_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            total_memory = torch.cuda.get_device_properties(i).total_memory
            total = total_memory / (1024**3)  # GB
            logger.info(f"🖥️  GPU {i} ({torch.cuda.get_device_name(i)}):")
            logger.info(
                f"   📊 Memory: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, {total:.2f}GB total"
            )
            logger.info(f"   💾 Free: {total - reserved:.2f}GB")
    else:
        logger.info("❌ No CUDA GPU available")
