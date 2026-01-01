#!/usr/bin/env python3
"""Pipeline functions for medical LLM training and inference."""

import gc
import os
from typing import Optional
import torch
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
    print("✅ Memory cleaned up")


def run_training(cfg: DictConfig) -> str:
    """Run the training pipeline."""
    print("📚 Starting training pipeline...")
    
    # Load and prepare data
    print("📚 Loading dataset...")
    raw_dataset = load_medical_dataset(cfg.data.train_file)
    
    # Setup model and tokenizer
    print("⚙️ Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(cfg.model.name, cfg.seed)
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    model = setup_lora_model(model)
    
    # Prepare datasets
    print("📊 Preparing datasets...")
    train_dataset, eval_dataset, test_dataset = prepare_datasets(
        raw_dataset, 
        tokenizer, 
        cfg.data.preprocessing.system_prompt,
        cfg.model.max_length
    )
    
    # Create trainer
    print("🏃 Creating trainer...")
    trainer = create_trainer(
        model, 
        tokenizer, 
        train_dataset, 
        eval_dataset, 
        cfg.output_dir,
        cfg.training.batch_size, 
        cfg.training.learning_rate, 
        cfg.training.max_steps, 
        cfg.training.validate_steps
    )
    
    # Train model
    if cfg.training.early_stopping.enabled:
        patience = cfg.training.early_stopping.get('patience', 3)
        print(f"\n🚀 Starting training with early stopping (patience={patience})...")
    else:
        print("\n🚀 Starting training (no early stopping)...")
    trainer.train()
    print("\n\nFinished training.\n\n")
    
    # Save model
    print("💾 Saving model...")
    adapter_dir = save_model(model, tokenizer, cfg.output_dir)
    
    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_results = evaluate_model(trainer, test_dataset)
    
    # Clean up memory
    cleanup_memory()
    
    print(f"✅ Training complete! Adapter saved to: {adapter_dir}")
    return adapter_dir


def run_interactive_inference(cfg: DictConfig) -> None:
    """Run interactive inference."""
    print("🤖 Starting inference pipeline...")
    
    adapter_dir = cfg.inference.adapter_path
    
    if not os.path.exists(adapter_dir):
        print(f"❌ Adapter directory not found: {adapter_dir}")
        print("Please run training first to create the adapter.")
        return
    
    print("🔄 Loading inference model...")
    model, tokenizer = load_inference_model(cfg.model.name, adapter_dir)
    
    print("🤖 Model loaded successfully!")
    
    if not cfg.inference.interactive:
        print("Non-interactive inference mode - exiting")
        return
    
    print("\n" + "="*60)
    print("Medical AI Assistant - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    # Interactive loop
    while True:
        try:
            user_query = input("👨‍⚕️ Ask a medical question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue
            
            print("\n" + "="*60)
            run_inference(
                model, 
                tokenizer, 
                user_query, 
                cfg.data.preprocessing.system_prompt
            )
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("\n👋 Thank you for using the Medical AI Assistant!")


def run_demo(cfg: DictConfig) -> None:
    """Run demo inference with a sample question."""
    print("🤖 Starting demo inference...")
    
    adapter_dir = cfg.inference.adapter_path
    
    if not os.path.exists(adapter_dir):
        print(f"❌ Adapter directory not found: {adapter_dir}")
        print("Please run training first to create the adapter.")
        return
    
    print("🔄 Loading inference model...")
    model, tokenizer = load_inference_model(cfg.model.name, adapter_dir)
    
    # Demo question
    demo_question = cfg.inference.demo_question
    
    print("🤖 Running demo inference...")
    print(f"Demo Question: {demo_question}\n")
    
    run_inference(
        model, 
        tokenizer, 
        demo_question, 
        cfg.data.preprocessing.system_prompt
    )
    
    print("\n✅ Demo complete!")


def run_full_pipeline(cfg: DictConfig) -> None:
    """Run complete training and inference pipeline."""
    print("🚀 Starting complete Medical LLM Training and Inference Pipeline")
    print("="*60)
    
    # Step 1: Training
    print("\n📚 Step 1: Training the model...")
    adapter_dir = run_training(cfg)
    
    # Step 2: Demo inference
    print("\n🤖 Step 2: Running demo inference...")
    run_demo(cfg)
    
    print("\n✅ Complete pipeline finished!")
    print(f"Adapter saved to: {adapter_dir}")
    print("\nTo run interactive inference, use:")
    print("python main.py mode=inference")