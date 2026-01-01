"""Inference utilities for medical LLM."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict, Any

from .function_utils import (
    has_function_calls, extract_function_calls, 
    request_permission, execute_function_call
)


def load_inference_model(base_model: str, adapter_dir: str):
    """Load model and tokenizer for inference."""
    # Create offload directory for proper model dispatching
    offload_dir = "./offload_tmp"
    os.makedirs(offload_dir, exist_ok=True)
    
    # Load base model with more conservative memory settings
    inference_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
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
                     tools: List[Dict[str, Any]], max_new_tokens: int = 300) -> str:
    """Generate response from model."""
    # Format prompt
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tools=tools,
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
    print("🤖 Generating response with function calling enabled...\n")
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


def process_function_calls(assistant_response: str, messages: List[Dict[str, str]], 
                          tools: List[Dict[str, Any]], model, tokenizer) -> str:
    """Process function calls and generate final response."""
    function_calls = extract_function_calls(assistant_response)
    function_results = []
    all_source_urls = []
    
    for call in function_calls:
        function_name = call.get("name")
        arguments = call.get("arguments", {})
        
        # Request permission and get URLs
        source_urls = request_permission(function_name, arguments)
        all_source_urls.extend(source_urls)
        
        # Execute function call
        print(f"🔧 Executing: {function_name}({arguments})")
        result, additional_urls = execute_function_call(function_name, arguments)
        all_source_urls.extend(additional_urls)
        
        function_results.append(result)
        print(f"📋 Result: {result[:100]}...")
    
    if function_results:
        print("\n🤖 Generating final response with search results...\n")
        
        # Create follow-up prompt with function results
        results_text = "\n\n".join([f"Search Result {i+1}:\n{result}" 
                                   for i, result in enumerate(function_results)])
        
        follow_up_messages = messages + [
            {"role": "assistant", "content": assistant_response},
            {"role": "system", "content": f"Here are the results from your function calls:\n\n{results_text}\n\nNow provide a comprehensive answer to the user's question based on these results. Include the medical disclaimer at the end."},
            {"role": "user", "content": "Please provide your final answer based on the search results."}
        ]
        
        # Generate final response
        final_response = generate_response(model, tokenizer, follow_up_messages, tools, 400)
        final_assistant_response = extract_assistant_response(final_response)
        
        return final_assistant_response, all_source_urls
    
    return assistant_response, all_source_urls


def run_inference(model, tokenizer, tools: List[Dict[str, Any]], 
                  user_query: str, system_prompt: str):
    """Run complete inference pipeline with function calling."""
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    
    # Generate initial response
    response = generate_response(model, tokenizer, messages, tools)
    
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
    
    # Check for and process function calls
    if has_function_calls(assistant_response):
        print("\n✅ Model generated function call(s)!")
        
        final_response, source_urls = process_function_calls(
            assistant_response, messages, tools, model, tokenizer
        )
        
        print("🎯 Final Response with Search Results:")
        print("=" * 60)
        print(final_response)
        
        # Add source URLs if any were used
        if source_urls:
            print("\n📚 Sources:")
            for i, url in enumerate(set(source_urls), 1):
                print(f"{i}. {url}")
        
        print("=" * 60)
        
    else:
        print("\n📝 Model provided direct answer without tool usage")