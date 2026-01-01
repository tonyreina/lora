"""Inference utilities for medical LLM."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Any


def load_inference_model(base_model: str, adapter_dir: str):
    """Load model and tokenizer for inference."""
    # Create offload directory for proper model dispatching
    offload_dir = "./offload_tmp"
    os.makedirs(offload_dir, exist_ok=True)
    
    # Load base model with more conservative memory settings
    inference_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
        max_memory={0: "6GiB", "cpu": "30GiB"},  # More conservative GPU memory usage
    )
    
    # Load adapter with offload directory
    inference_model = PeftModel.from_pretrained(
        inference_model, 
        adapter_dir,
        offload_folder=offload_dir
    )
    inference_model.eval()
    
    # Load tokenizer
    inference_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if inference_tokenizer.pad_token is None:
        inference_tokenizer.pad_token = inference_tokenizer.unk_token or "<pad>"
        if inference_tokenizer.pad_token == "<pad>" and "<pad>" not in inference_tokenizer.vocab:
            inference_tokenizer.add_special_tokens({"pad_token": "<pad>"})
            inference_model.resize_token_embeddings(len(inference_tokenizer))
    
    return inference_model, inference_tokenizer


def generate_response(model, tokenizer, messages: List[Dict[str, str]], 
                     max_new_tokens: int = 300) -> str:
    """Generate response from model."""
    # Format prompt
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs_tensor = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    # Generate response
    print("🤖 Generating response...\n")
    gen = model.generate(
        inputs_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode response
    response = tokenizer.decode(gen[0], skip_special_tokens=False)
    return response


def extract_assistant_response(response: str) -> str:
    """Extract just the assistant's response from full generation."""
    if "<|assistant|>" in response:
        assistant_response = response.split("<|assistant|>")[-1]
        if "<|end|>" in assistant_response:
            assistant_response = assistant_response.split("<|end|>")[0]
        return assistant_response.strip()
    return response


def run_inference(model, tokenizer, user_query: str, system_prompt: str):
    """Run complete inference pipeline."""
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    # Generate initial response
    response = generate_response(model, tokenizer, messages)
    
    print("Full Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
    # Extract assistant response
    assistant_response = extract_assistant_response(response)
    
    print("\n🎯 Model's Response to User:")
    print("=" * 60)
    print(assistant_response)
    print("=" * 60)