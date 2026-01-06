#!/usr/bin/env python3
"""Simplified main entry point for medical LLM pipeline."""

import os
from dataclasses import dataclass
from typing import Any

import yaml
from loguru import logger


@dataclass
class SimpleConfig:
    """Simple configuration class to replace complex Hydra setup."""

    def __init__(self, config_dict: dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(config_file: str = "config.yaml") -> SimpleConfig:
    """Load configuration from YAML file."""
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
    return SimpleConfig(config_dict)


def run_training(cfg: SimpleConfig):
    """Simplified training pipeline."""
    logger.info("🚀 Starting training...")

    # Set environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Import utilities
    from utils import (
        cleanup_memory,
        create_trainer,
        load_and_prepare_data,
        prepare_datasets,
        print_gpu_memory_usage,
        save_model,
        setup_lora,
        setup_model,
    )

    # Load and prepare data
    raw_dataset = load_and_prepare_data(cfg.data.train_file, cfg.data, cfg.seed)

    # Setup model
    model, tokenizer = setup_model(cfg.model.name, cfg.seed)
    model = setup_lora(model, cfg.lora)

    # Prepare datasets
    train_dataset, eval_dataset, test_dataset = prepare_datasets(raw_dataset, tokenizer, cfg.data)

    # Print GPU memory usage before training
    logger.info("🔍 GPU Memory Status Before Training:")
    print_gpu_memory_usage()

    # Train
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset, cfg.output_dir, cfg.training
    )
    trainer.train()

    # Evaluate on test dataset
    logger.info("📊 Evaluating on test dataset...")
    test_results = trainer.evaluate(test_dataset)
    logger.info("🎯 Test Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")

    # Save and cleanup
    adapter_dir = save_model(model, tokenizer, cfg.output_dir, cfg.model.name)
    cleanup_memory()

    logger.success(f"✅ Training complete! Saved to: {adapter_dir}")
    return adapter_dir


def run_inference(cfg: SimpleConfig):
    """Simplified inference pipeline."""
    logger.info("🤖 Starting inference...")

    from utils import load_inference_model, run_inference

    adapter_dir = cfg.inference.adapter_path.replace("${output_dir}", cfg.output_dir)
    # Extract model name from full model path
    # (e.g., "microsoft/Phi-4-mini-instruct" -> "Phi-4-mini-instruct")
    model_name = cfg.model.name.split("/")[-1] if "/" in cfg.model.name else cfg.model.name
    adapter_dir = adapter_dir.replace("${model_name}", model_name)

    if not os.path.exists(adapter_dir):
        logger.error(f"❌ Model not found: {adapter_dir}")
        logger.error("Run training first: python main.py train")
        return

    # Load model
    model, tokenizer = load_inference_model(cfg.model.name, adapter_dir, cfg.inference)

    if cfg.inference.interactive:
        logger.info("Medical AI Assistant - Type 'quit' to exit")
        while True:
            try:
                query = input("Ask me a question: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    break
                if query:
                    run_inference(model, tokenizer, query, cfg.data.system_prompt, cfg.inference)
            except KeyboardInterrupt:
                break
        logger.info("👋 Goodbye!")
    else:
        # Demo mode
        logger.info(f"Demo: {cfg.inference.demo_question}")
        run_inference(
            model, tokenizer, cfg.inference.demo_question, cfg.data.system_prompt, cfg.inference
        )


def main():
    """Main entry point with simplified argument handling."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py [train|inference] [config_file]")
        print("Examples:")
        print("  python main.py train")
        print("  python main.py inference")
        print("  python main.py train my_config.yaml")
        return

    mode = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    if mode not in ["train", "inference"]:
        print("Mode must be 'train' or 'inference'")
        return

    cfg = load_config(config_file)
    cfg.mode = mode  # Override config file mode

    logger.info(f"Mode: {mode}")
    logger.info(f"Config: {config_file}")

    if mode == "train":
        run_training(cfg)
    elif mode == "inference":
        run_inference(cfg)


if __name__ == "__main__":
    main()
