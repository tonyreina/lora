# Memory Optimization

This document covers the memory optimization techniques used in our LoRA fine-tuning implementation, focusing on practical approaches that enable efficient training of large language models on consumer GPUs.

## 💾 Memory Usage Overview

### Actual Memory Breakdown

For LoRA fine-tuning of Phi-4-mini-instruct (4B parameters) on a consumer GPU:

```text
Base Model (4-bit quantized):     ~3.5GB
LoRA Adapters (r=16):            ~50MB
Optimizer States (AdamW):        ~100MB
Gradients:                       ~50MB
Activation Cache:                ~1-2GB
Input Batch (batch_size=4):      ~500MB-1GB
---
Total:                          ~6-7GB
```

### Our Memory Optimization Strategy

We use a layered approach that leverages existing libraries:

1. **4-bit Quantization** (BitsAndBytesConfig)
2. **LoRA Parameter Efficiency** (PEFT)
3. **Gradient Checkpointing** (Built into Trainer)

## 🔧 Implementation

### 1. 4-bit Quantization with BitsAndBytesConfig

Our primary memory saver is [quantized LoRA (QLoRA)](https://arxiv.org/abs/2305.14314) from the
[BitsAndBytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) package.
This library loads the pre-trained base/instruction model at 4-bit precision rather than the
full 16-bit (or 32-bit) precision.
This reduces model size by ~75%. The LoRA weights and gradients are at the typical bfloat-16 precision.

```python
def setup_model(model_name: str, seed: int):
    """Setup model with 4-bit quantization for memory efficiency."""

    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # Use 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16, # Compute in float16
        bnb_4bit_use_double_quant=True,       # Double quantization for extra savings
        bnb_4bit_quant_type="nf4",            # NormalFloat4 - best for neural networks
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",           # Automatic device placement
        quantization_config=bnb_config,
        dtype=torch.float16,
    )

    return model, tokenizer

# Memory savings: ~75% reduction in model memory
```

### 2. LoRA Parameter Efficiency

Only train a small fraction of parameters:

```python
def setup_lora(model, cfg):
    """Apply LoRA - only ~0.1% of parameters become trainable."""

    # Prepare quantized model for LoRA
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Disable KV cache for training

    # LoRA configuration
    peft_config = LoraConfig(
        r=cfg.r,                          # Low rank (16) - smaller = less memory
        lora_alpha=cfg.alpha,             # Scaling factor (32)
        target_modules=cfg.target_modules, # Which layers to adapt
        lora_dropout=cfg.dropout,         # Regularization
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model

# Example output:
# trainable params: 83,886,080 || all params: 14,888,534,016 || trainable%: 0.56%
```

### 3. Gradient Checkpointing

During training, the computer keeps the track of the pre- and post-layer
gradients so that calculating the backward pass is quick. However,
this takes up a lot of memory and may cause the GPU to run out of memory (OOM).

[Gradient checkpointing](https://github.com/cybertronai/gradient-checkpointing)
is a compromise where the gradients are only stored every few layers to save
memory. During the backward pass, the missing gradients are recalculated
using the checkpoints. This balances GPU memory versus training speed.

Gradient checkpointing is automatically enabled in our `transformers` Trainer configuration.

---

This comprehensive memory optimization guide ensures efficient training
even on limited hardware resources while maintaining model performance and training stability.
