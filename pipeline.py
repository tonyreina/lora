#!/usr/bin/env python3
"""Pipeline functions for medical LLM training and inference."""

import gc
import os
from typing import Optional
import torch
from loguru import logger
from omegaconf import DictConfig

from src.data_utils import load_medical_dataset, prepare_datasets
from src.model_utils import (
    setup_model_and_tokenizer, setup_lora_model, create_trainer,
    save_model, evaluate_model
)
from src.inference_utils import load_inference_model, run_inference


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    logger.success("✅ Memory cleaned up")


def run_training(cfg: DictConfig) -> str:
    """Run the training pipeline."""
    logger.info("📚 Starting training pipeline...")
    
    # Load and prepare data
    logger.info("📚 Loading dataset...")
    raw_dataset = load_medical_dataset(cfg.data.train_file, cfg.data, cfg.seed)
    
    # Setup model and tokenizer
    logger.info("⚙️ Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(cfg.model.name, cfg.seed)
    
    # Configure LoRA
    logger.info("🔧 Configuring LoRA...")
    model = setup_lora_model(model, cfg.lora)
    
    # Prepare datasets
    logger.info("📊 Preparing datasets...")
    train_dataset, eval_dataset, test_dataset = prepare_datasets(
        raw_dataset, 
        tokenizer, 
        cfg.data
    )
    
    # Create trainer
    logger.info("🏃 Creating trainer...")
    trainer = create_trainer(
        model, 
        tokenizer, 
        train_dataset, 
        eval_dataset, 
        cfg.output_dir,
        cfg.training
    )
    
    # Train model
    if cfg.training.early_stopping.enabled:
        patience = cfg.training.early_stopping.get('patience', 3)
        logger.info(f"🚀 Starting training with early stopping (patience={patience})...")
    else:
        logger.info("🚀 Starting training (no early stopping)...")
    trainer.train()
    logger.info("Finished training.\n\n")
    
    # Save model
    logger.info("💾 Saving model...")
    adapter_dir = save_model(model, tokenizer, cfg.output_dir)
    
    # Evaluate on test set
    logger.info("🧪 Evaluating on test set...")
    test_results = evaluate_model(trainer, test_dataset)
    
    # Clean up memory
    cleanup_memory()
    
    logger.success(f"✅ Training complete! Adapter saved to: {adapter_dir}")
    return adapter_dir


def run_interactive_inference(cfg: DictConfig) -> None:
    """Run interactive inference."""
    logger.info("🤖 Starting inference pipeline...")
    
    adapter_dir = cfg.inference.adapter_path
    
    if not os.path.exists(adapter_dir):
        logger.error(f"❌ Adapter directory not found: {adapter_dir}")
        logger.error("Please run training first to create the adapter.")
        return
    
    logger.info("🔄 Loading inference model...")
    model, tokenizer = load_inference_model(cfg.model.name, adapter_dir, cfg.inference)
    
    logger.success("🤖 Model loaded successfully!")
    
    if not cfg.inference.interactive:
        logger.info("Non-interactive inference mode - exiting")
        return
    
    logger.info("Medical AI Assistant - Interactive Mode")
    logger.info("Type 'quit' or 'exit' to stop")
    
    # Interactive loop
    while True:
        try:
            user_query = input("👨‍⚕️ Ask a medical question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            run_inference(
                model, 
                tokenizer, 
                user_query, 
                cfg.data.preprocessing.system_prompt,
                cfg.inference.generation
            )
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            continue
    
    logger.info("👋 Thank you for using the Medical AI Assistant!")


def run_demo(cfg: DictConfig) -> None:
    """Run demo inference with a sample question."""
    logger.info("🤖 Starting demo inference...")
    
    adapter_dir = cfg.inference.adapter_path
    
    if not os.path.exists(adapter_dir):
        logger.error(f"❌ Adapter directory not found: {adapter_dir}")
        logger.error("Please run training first to create the adapter.")
        return
    
    logger.info("🔄 Loading inference model...")
    model, tokenizer = load_inference_model(cfg.model.name, adapter_dir, cfg.inference)
    
    # Demo question
    demo_question = cfg.inference.demo_question
    
    logger.info("🤖 Running demo inference...")
    logger.info(f"Demo Question: {demo_question}\n")
    
    run_inference(
        model, 
        tokenizer, 
        demo_question, 
        cfg.data.preprocessing.system_prompt,
        cfg.inference.generation
    )
    
    logger.success("✅ Demo complete!")


def run_full_pipeline(cfg: DictConfig) -> None:
    """Run complete training and inference pipeline."""
    logger.info("🚀 Starting complete Medical LLM Training and Inference Pipeline")
    
    # Step 1: Training
    logger.info("📚 Step 1: Training the model...")
    adapter_dir = run_training(cfg)
    
    # Step 2: Demo inference
    logger.info("🤖 Step 2: Running demo inference...")
    run_demo(cfg)
    
    logger.success("✅ Complete pipeline finished!")
    logger.info(f"Adapter saved to: {adapter_dir}")
    logger.info("To run interactive inference, use:")
    logger.info("python main.py mode=inference")