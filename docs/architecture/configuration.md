# Configuration Management

The LoRA fine-tuning system uses a centralized configuration approach that allows for easy experimentation and reproducible results. This document explains the configuration structure and how to customize it for your needs.

## 📁 Configuration File Structure

The main configuration is stored in `config.yaml` at the project root. This file contains all the parameters needed to control training, model selection, and system behavior.

```yaml
# Example config.yaml structure
model:
  base_model_name: "microsoft/Phi-4-mini-instruct"
  quantization_config:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 1e-4
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500

data:
  dataset_path: "./data/my_custom_data.jsonl"
  max_seq_length: 2048

safety:
  system_prompt: "You are a helpful medical AI assistant..."
  content_filter: true
```

## 🎛️ Configuration Sections

### Model Configuration

Controls the base model selection and quantization settings:

```yaml
model:
  # Base model from Hugging Face
  base_model_name: "microsoft/Phi-4-mini-instruct"

  # Trust remote code (needed for some models)
  trust_remote_code: true

  # Device mapping for multi-GPU setups
  device_map: "auto"

  # Quantization settings for memory efficiency
  quantization_config:
    load_in_4bit: true                    # Enable 4-bit quantization
    bnb_4bit_compute_dtype: "float16"     # Computation precision
    bnb_4bit_quant_type: "nf4"           # Quantization algorithm
    bnb_4bit_use_double_quant: true       # Double quantization for better accuracy

  # Attention implementation
  attn_implementation: "flash_attention_2"  # Use FlashAttention for speed
```

### LoRA Configuration

Defines the adapter parameters that control the efficiency-performance tradeoff:

```yaml
lora:
  # LoRA rank - higher values = more parameters but better adaptation
  r: 32

  # LoRA scaling factor - typically 2x the rank
  lora_alpha: 64

  # Dropout rate for LoRA layers
  lora_dropout: 0.1

  # Which transformer modules to adapt
  target_modules:
    - "q_proj"    # Query projection
    - "k_proj"    # Key projection
    - "v_proj"    # Value projection
    - "o_proj"    # Output projection

  # Bias parameters
  bias: "none"    # Don't adapt bias terms

  # Task type
  task_type: "CAUSAL_LM"
```

### Training Configuration

Controls the training process and hyperparameters:

```yaml
training:
  # Output directory for checkpoints
  output_dir: "./checkpoints"

  # Training duration
  num_train_epochs: 3
  max_steps: -1  # -1 means use num_train_epochs

  # Batch sizes
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4

  # Learning rate settings
  learning_rate: 1e-4
  warmup_steps: 100
  lr_scheduler_type: "cosine"

  # Optimization
  optim: "adamw_torch"
  weight_decay: 0.01
  max_grad_norm: 1.0

  # Logging and saving
  logging_steps: 10
  save_steps: 500
  eval_steps: 500

  # Memory optimization
  gradient_checkpointing: true
  dataloader_pin_memory: false
  fp16: true

  # Reproducibility
  seed: 42
  data_seed: 42
```

### Data Configuration

Specifies dataset location and processing parameters:

```yaml
data:
  # Path to training data
  dataset_path: "./data/my_custom_data.jsonl"

  # Sequence length limits
  max_seq_length: 2048

  # Data preprocessing
  truncation: true
  padding: false

  # Validation split
  eval_dataset_size: 0.1  # 10% for validation

  # Data filtering
  min_length: 10
  max_length: 2048
```

### Safety Configuration

Medical AI safety and content filtering settings:

```yaml
safety:
  # System prompt for medical context
  system_prompt: |
    You are a helpful medical AI assistant. Your responses should be:
    - Accurate and evidence-based
    - Clear about limitations and when to seek professional help
    - Respectful of patient privacy and professional boundaries
    - Compliant with medical ethics and safety guidelines

  # Content filtering
  content_filter: true

  # Safety keywords to monitor
  safety_keywords:
    - "emergency"
    - "diagnosis"
    - "prescription"
    - "dosage"

  # Professional deferral triggers
  defer_keywords:
    - "chest pain"
    - "difficulty breathing"
    - "severe"
    - "emergency"
```

## 🔧 Configuration Customization

### Environment-Specific Configs

You can create environment-specific configuration files:

```bash
# Development configuration
config.dev.yaml

# Production configuration
config.prod.yaml

# Testing configuration
config.test.yaml
```

Load specific configs with:

```python
config = load_config("config.dev.yaml")
```

### Parameter Tuning Guidelines

#### LoRA Rank Selection

- **r=8**: Minimal adaptation, fastest training, lowest memory
- **r=16**: Light adaptation, good for simple tasks
- **r=32**: Balanced adaptation, recommended for most use cases
- **r=64**: Heavy adaptation, better for complex tasks
- **r=128**: Maximum adaptation, highest memory usage

#### Batch Size Optimization

```python
# Memory usage estimation
def estimate_memory_usage(batch_size, seq_length, model_size):
    """Estimate GPU memory usage in GB"""
    base_memory = model_size * 0.5  # Quantized model
    batch_memory = batch_size * seq_length * 0.000004  # Approximate
    gradient_memory = 0.1  # LoRA gradients
    return base_memory + batch_memory + gradient_memory

# For 6GB GPU with Phi-4-mini
optimal_batch_size = 2
```

#### Learning Rate Schedule

```yaml
# Cosine schedule with warmup (recommended)
learning_rate: 1e-4
lr_scheduler_type: "cosine"
warmup_steps: 100

# Linear schedule
lr_scheduler_type: "linear"
warmup_ratio: 0.1

# Constant with warmup
lr_scheduler_type: "constant_with_warmup"
warmup_steps: 50
```

## 📊 Configuration Validation

The system includes automatic configuration validation:

```python
def validate_config(config):
    """Validate configuration parameters"""
    errors = []

    # Check required fields
    required_fields = ['model.base_model_name', 'data.dataset_path']
    for field in required_fields:
        if not get_nested_value(config, field):
            errors.append(f"Missing required field: {field}")

    # Validate LoRA parameters
    if config.get('lora', {}).get('r', 0) <= 0:
        errors.append("LoRA rank must be positive")

    # Validate paths
    if not os.path.exists(config['data']['dataset_path']):
        errors.append(f"Dataset path not found: {config['data']['dataset_path']}")

    return errors
```

## 🎯 Best Practices

### 1. Version Control Configurations

- Store configurations in version control
- Use descriptive names for experiment configs
- Document parameter changes and rationale

### 2. Environment Variables

Override config values with environment variables:

```bash
export LORA_LEARNING_RATE=2e-4
export LORA_BATCH_SIZE=4
export LORA_RANK=64
```

```python
# In code
config['training']['learning_rate'] = float(os.getenv('LORA_LEARNING_RATE', config['training']['learning_rate']))
```

### 3. Configuration Templates

Create templates for common use cases:

```yaml
# medical_qa_template.yaml
base_template: "default"
overrides:
  lora:
    r: 32
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  training:
    learning_rate: 1e-4
    num_train_epochs: 3
  safety:
    system_prompt: "Medical AI assistant prompt..."
```

### 4. Experiment Tracking

Track configurations alongside results:

```python
# Save config with each experiment
experiment_config = {
    'timestamp': datetime.now().isoformat(),
    'config': config,
    'git_hash': get_git_hash(),
    'results': training_results
}

with open(f'experiments/{experiment_id}/config.json', 'w') as f:
    json.dump(experiment_config, f, indent=2)
```

## 🚨 Common Configuration Issues

### Memory Issues

```yaml
# Reduce memory usage
training:
  per_device_train_batch_size: 1  # Smaller batches
  gradient_accumulation_steps: 8   # Maintain effective batch size
  gradient_checkpointing: true     # Trade compute for memory
  fp16: true                       # Use half precision
```

### Convergence Issues

```yaml
# Improve training stability
training:
  learning_rate: 5e-5      # Lower learning rate
  warmup_steps: 200        # More warmup
  max_grad_norm: 0.5       # Gradient clipping

lora:
  lora_dropout: 0.05       # Less dropout
```

### Performance Issues

```yaml
# Optimize for speed
model:
  attn_implementation: "flash_attention_2"

training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  remove_unused_columns: true
```

This configuration system provides the flexibility needed for medical AI applications while maintaining simplicity and reproducibility.
