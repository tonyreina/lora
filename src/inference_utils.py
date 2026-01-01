"""Inference utilities for medical LLM."""

import os
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Any


def load_inference_model(base_model: str, adapter_dir: str, inference_config):
    """Load base model with LoRA adapter for inference.
    
    Loads a base language model and applies a LoRA adapter for inference,
    with memory-efficient settings including device mapping and offloading.
    Sets up tokenizer with proper padding token configuration.
    
    Args:
        base_model (str): Path or HuggingFace model ID for the base model.
        adapter_dir (str): Path to directory containing the LoRA adapter files.
        inference_config: Configuration object containing inference parameters.
        
    Returns:
        tuple: A tuple containing:
            - inference_model: PEFT model with LoRA adapter loaded and set to eval mode
            - inference_tokenizer: Configured tokenizer with proper pad token
            
    Note:
        - Creates offload directory for model offloading (from config)
        - Uses memory allocation settings from config
        - Automatically handles pad token setup and vocabulary resizing if needed
        
    Example:
        >>> model, tokenizer = load_inference_model(
        ...     "microsoft/phi-3.5-mini", "./checkpoints/lora_adapter", cfg.inference
        ... )
    """
    # Create offload directory for proper model dispatching
    offload_dir = inference_config.memory.offload_dir
    os.makedirs(offload_dir, exist_ok=True)
    
    # Get dtype from config
    dtype = getattr(torch, inference_config.memory.dtype)
    
    # Load base model with memory settings from config
    inference_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=dtype,
        low_cpu_mem_usage=inference_config.memory.low_cpu_mem_usage,
        offload_folder=offload_dir,
        max_memory={0: inference_config.memory.max_memory.gpu, "cpu": inference_config.memory.max_memory.cpu},
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
                     generation_config) -> str:
    """Generate text response from model using conversational messages.
    
    Takes a list of conversational messages, applies chat template formatting,
    and generates a response using the model with configured sampling parameters.
    
    Args:
        model: The loaded language model (typically PEFT model with adapter).
        tokenizer: HuggingFace tokenizer with chat template support.
        messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content' keys.
            Typically includes system, user, and optionally assistant messages.
        generation_config: Configuration object containing generation parameters.
        
    Returns:
        str: Complete generated text including the original prompt and new response.
        
    Note:
        - Uses generation parameters from config (temperature, top_p, max_new_tokens, etc.)
        - Applies chat template with generation prompt
        - Handles device placement automatically
        - Includes progress indication during generation
        
    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is machine learning?"}
        ... ]
        >>> response = generate_response(model, tokenizer, messages, cfg.inference.generation)
    """
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
    logger.info("🤖 Generating response...\n")
    gen = model.generate(
        inputs_tensor,
        attention_mask=attention_mask,
        max_new_tokens=generation_config.max_new_tokens,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature,
        top_p=getattr(generation_config, 'top_p', 0.9),
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode response
    response = tokenizer.decode(gen[0], skip_special_tokens=False)
    return response


def extract_assistant_response(response: str) -> str:
    """Extract assistant's response from full model generation.
    
    Parses the complete model output to isolate only the assistant's response
    by looking for specific chat template markers and removing system/user content.
    
    Args:
        response (str): Complete generated text from the model including all conversation parts.
        
    Returns:
        str: Cleaned assistant response with template markers and extra content removed.
        
    Note:
        - Looks for '<|assistant|>' marker to identify response start
        - Removes '<|end|>' marker if present to clean response end
        - Returns original response if no markers found
        - Strips whitespace from extracted response
        
    Example:
        >>> full_response = "<|system|>Help user<|user|>Hello<|assistant|>Hi there!<|end|>"
        >>> clean_response = extract_assistant_response(full_response)
        >>> print(clean_response)
        "Hi there!"
    """
    if "<|assistant|>" in response:
        assistant_response = response.split("<|assistant|>")[-1]
        if "<|end|>" in assistant_response:
            assistant_response = assistant_response.split("<|end|>")[0]
        return assistant_response.strip()
    return response


def run_inference(model, tokenizer, user_query: str, system_prompt: str, generation_config):
    """Execute complete inference pipeline with formatted output.
    
    Orchestrates the full inference process from user query to formatted response,
    including message preparation, generation, response extraction, and display.
    Provides detailed output formatting for both raw and cleaned responses.
    
    Args:
        model: The loaded language model ready for inference.
        tokenizer: HuggingFace tokenizer compatible with the model.
        user_query (str): User's input question or prompt.
        system_prompt (str): System message defining model behavior and context.
        generation_config: Configuration object containing generation parameters.
        
    Returns:
        None: This function handles output display but doesn't return values.
        
    Side Effects:
        - Prints full raw response with decorative borders
        - Prints extracted assistant response with formatted headers
        - Includes progress indicators during generation
        
    Example:
        >>> run_inference(
        ...     model, tokenizer, 
        ...     "What is diabetes?",
        ...     "You are a medical AI assistant.",
        ...     cfg.inference.generation
        ... )
        # Outputs formatted response to console
    """
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    # Generate initial response
    response = generate_response(model, tokenizer, messages, generation_config)
    
    logger.info("Full Response:")
    logger.info(response)
    
    # Extract assistant response
    assistant_response = extract_assistant_response(response)
    
    logger.info("🎯 Model's Response to User:")
    logger.info(assistant_response)