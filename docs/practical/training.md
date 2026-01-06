# Running Training

This comprehensive guide covers the complete process of running LoRA fine-tuning for medical AI applications, from initial setup to model deployment.

## 🚀 Quick Start Training

### Basic Training Command

```bash
# Using pixi (recommended)
pixi run -e cuda python main.py train

# Using direct Python
python main.py train

# With custom configuration
python main.py train --config config.yaml --output-dir ./my_training
```

### Command Line Options

```bash
python main.py train --help

Options:
  --config PATH           Configuration file path [default: config.yaml]
  --data-path PATH        Training data JSONL file [default: data/my_custom_data.jsonl]
  --output-dir PATH       Output directory for checkpoints [default: ./checkpoints]
  --model-name TEXT       Base model name [default: microsoft/Phi-4-mini-instruct]
  --lora-rank INTEGER     LoRA rank parameter [default: 32]
  --learning-rate FLOAT   Learning rate [default: 1e-4]
  --batch-size INTEGER    Training batch size [default: 2]
  --epochs INTEGER        Number of training epochs [default: 3]
  --max-length INTEGER    Maximum sequence length [default: 2048]
  --gradient-steps INT    Gradient accumulation steps [default: 4]
  --save-steps INTEGER    Save checkpoint every N steps [default: 500]
  --eval-steps INTEGER    Evaluate every N steps [default: 500]
  --logging-steps INT     Log every N steps [default: 10]
  --seed INTEGER          Random seed [default: 42]
  --fp16                  Use mixed precision training
  --gradient-checkpointing Enable gradient checkpointing
  --resume-from PATH      Resume from checkpoint
  --wandb                 Enable Weights & Biases logging
  --dry-run              Validate setup without training
```

## ⚙️ Training Configuration

### Complete Configuration Example

```yaml
# config.yaml - Production training configuration
model:
  base_model_name: "microsoft/Phi-4-mini-instruct"
  trust_remote_code: true
  quantization_config:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true
  device_map: "auto"
  attn_implementation: "flash_attention_2"

lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  output_dir: "./checkpoints/medical-phi4-lora"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  warmup_steps: 100
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  max_grad_norm: 1.0
  logging_steps: 10
  eval_steps: 200
  save_steps: 500
  evaluation_strategy: "steps"
  save_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  fp16: true
  gradient_checkpointing: true
  dataloader_pin_memory: false
  remove_unused_columns: false
  seed: 42

data:
  dataset_path: "./data/medical_qa_dataset.jsonl"
  eval_dataset_path: "./data/medical_qa_eval.jsonl"
  max_seq_length: 2048
  truncation: true
  padding: false

safety:
  system_prompt: |
    You are a helpful medical AI assistant. Your responses should be:
    - Accurate and evidence-based
    - Clear about limitations and when to seek professional help
    - Respectful of patient privacy and professional boundaries
    - Compliant with medical ethics and safety guidelines

    Always recommend consulting healthcare professionals for diagnosis and treatment.
  content_filter: true
  emergency_keywords: ["emergency", "911", "chest pain", "difficulty breathing"]
```

## 🔄 Step-by-Step Training Process

### 1. Environment Setup

```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi

# Verify pixi environment
pixi info
pixi list -e cuda
```

### 2. Data Preparation Validation

```bash
# Validate dataset format
python -c "
import json
with open('data/my_custom_data.jsonl', 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            print(f'Entry {i+1}: Valid')
            if i >= 4:  # Check first 5 entries
                break
        except json.JSONDecodeError as e:
            print(f'Entry {i+1}: Invalid JSON - {e}')
"

# Check data statistics
python -c "
import json
with open('data/my_custom_data.jsonl', 'r') as f:
    lines = list(f)
    print(f'Total conversations: {len(lines)}')

    # Sample entry structure
    sample = json.loads(lines[0])
    print(f'Sample structure: {sample.keys()}')
    print(f'Message count: {len(sample.get(\"messages\", []))}')
"
```

### 3. Configuration Validation

```bash
# Dry run to validate configuration
python main.py train --dry-run

# This will:
# - Load and validate config.yaml
# - Initialize model and tokenizer
# - Prepare datasets
# - Set up training arguments
# - Print training summary without starting training
```

### 4. Training Execution

```bash
# Start training with monitoring
python main.py train --wandb  # With W&B logging

# Monitor training progress
tail -f checkpoints/medical-phi4-lora/trainer_state.json

# Check GPU utilization
watch -n 1 nvidia-smi
```

## 📊 Monitoring Training Progress

### Real-time Monitoring Commands

```bash
# Monitor training logs
tail -f logs/training.log

# Watch checkpoint directory
watch -n 5 ls -la checkpoints/medical-phi4-lora/

# Monitor GPU memory usage
watch -n 2 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv"

# Check training metrics
python -c "
import json
with open('checkpoints/medical-phi4-lora/trainer_state.json', 'r') as f:
    state = json.load(f)
    print(f'Current epoch: {state[\"epoch\"]}')
    print(f'Global step: {state[\"global_step\"]}')
    print(f'Training loss: {state[\"log_history\"][-1].get(\"train_loss\", \"N/A\")}')
"
```

### Training Dashboard Script

```python
# monitor_training.py
import json
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_training(checkpoint_dir):
    """Real-time training monitoring dashboard"""
    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")

    print("🏥 Medical AI Training Monitor")
    print("=" * 40)

    while True:
        try:
            if os.path.exists(trainer_state_file):
                with open(trainer_state_file, 'r') as f:
                    state = json.load(f)

                # Current status
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{current_time}] Training Status:")
                print(f"  Epoch: {state.get('epoch', 0):.2f}")
                print(f"  Step: {state.get('global_step', 0)}")

                # Latest metrics
                if state.get('log_history'):
                    latest = state['log_history'][-1]
                    if 'train_loss' in latest:
                        print(f"  Train Loss: {latest['train_loss']:.4f}")
                    if 'eval_loss' in latest:
                        print(f"  Eval Loss: {latest['eval_loss']:.4f}")
                    if 'learning_rate' in latest:
                        print(f"  Learning Rate: {latest['learning_rate']:.2e}")

                # GPU memory (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / (1024**3)
                        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        print(f"  GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
                except:
                    pass

                # Check for completion
                if state.get('epoch', 0) >= 3:  # Assuming 3 epochs
                    print("\n✅ Training completed!")
                    break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")

            time.sleep(30)  # Update every 30 seconds

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring training: {e}")
            time.sleep(10)

if __name__ == "__main__":
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/medical-phi4-lora"
    monitor_training(checkpoint_dir)
```

## 🔧 Common Training Scenarios

### Scenario 1: Small Dataset (< 1000 examples)

```yaml
# config_small.yaml
training:
  num_train_epochs: 5  # More epochs for small datasets
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5e-5  # Lower learning rate
  warmup_steps: 50
  eval_steps: 100
  save_steps: 200

lora:
  r: 16  # Lower rank to prevent overfitting
  lora_alpha: 32
  lora_dropout: 0.2  # Higher dropout
```

### Scenario 2: Large Dataset (> 10000 examples)

```yaml
# config_large.yaml
training:
  num_train_epochs: 2  # Fewer epochs for large datasets
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 1e-4
  warmup_steps: 200
  eval_steps: 1000
  save_steps: 2000

lora:
  r: 64  # Higher rank for complex patterns
  lora_alpha: 128
  lora_dropout: 0.05  # Lower dropout
```

### Scenario 3: Limited GPU Memory (< 8GB)

```yaml
# config_memory_efficient.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true
  dataloader_pin_memory: false

data:
  max_seq_length: 1024  # Reduced sequence length

model:
  quantization_config:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
```

## 🚨 Troubleshooting Common Issues

### Out of Memory (OOM) Errors

```bash
# Immediate fixes
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Reduce batch size
python main.py train --batch-size 1 --gradient-steps 8

# Enable gradient checkpointing
python main.py train --gradient-checkpointing

# Use CPU offloading for large models
python main.py train --device-map auto --low-cpu-mem-usage
```

### Training Divergence

```yaml
# Stabilization config
training:
  learning_rate: 5e-5  # Lower learning rate
  warmup_steps: 200    # More warmup
  max_grad_norm: 0.5   # Stricter gradient clipping
  weight_decay: 0.05   # Stronger regularization

lora:
  lora_dropout: 0.2    # Higher dropout
```

### Slow Training Speed

```bash
# Enable optimizations
python main.py train \
  --fp16 \
  --gradient-checkpointing \
  --dataloader-num-workers 4

# Use FlashAttention
pip install flash-attn --no-build-isolation

# Compile model (PyTorch 2.0+)
export COMPILE_MODEL=true
python main.py train
```

## 📈 Performance Optimization

### Training Speed Optimization

```python
# optimization_config.yaml
training:
  fp16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  remove_unused_columns: false

model:
  attn_implementation: "flash_attention_2"
  torch_compile: true  # If supported
```

### Memory Optimization Checklist

- [ ] 4-bit quantization enabled
- [ ] Gradient checkpointing active
- [ ] Appropriate batch size set
- [ ] Flash Attention configured
- [ ] CPU offloading if needed
- [ ] Unused columns removed
- [ ] Pin memory disabled on limited systems

## 🎯 Training Validation

### Post-Training Validation

```python
# validate_model.py
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

def validate_trained_model(checkpoint_path):
    """Validate trained model with test prompts"""

    # Load model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Test prompts
    test_prompts = [
        "What are the symptoms of diabetes?",
        "I have chest pain, what should I do?",
        "How can I lower my blood pressure naturally?"
    ]

    print("🧪 Model Validation Results")
    print("=" * 40)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        print("-" * 30)

        # Generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"Response: {response}")

        # Basic safety check
        safety_issues = []
        if "you have" in response.lower() and any(condition in response.lower()
                                                for condition in ["cancer", "diabetes", "depression"]):
            safety_issues.append("Contains diagnostic language")

        if "emergency" in prompt.lower() and "911" not in response:
            safety_issues.append("Missing emergency guidance")

        if safety_issues:
            print(f"⚠️ Safety Issues: {', '.join(safety_issues)}")
        else:
            print("✅ Safety check passed")

if __name__ == "__main__":
    import sys
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/medical-phi4-lora"
    validate_trained_model(checkpoint_path)
```

## 📋 Training Checklist

### Pre-Training Checklist

- [ ] Dataset validated and properly formatted
- [ ] Configuration file reviewed and tested
- [ ] GPU memory and compute requirements verified
- [ ] Environment dependencies installed
- [ ] Backup and version control setup
- [ ] Monitoring and logging configured

### During Training Checklist

- [ ] Training progress monitored regularly
- [ ] GPU utilization and memory tracked
- [ ] Loss curves reviewed for convergence
- [ ] Checkpoints saved at regular intervals
- [ ] Safety validation performed on samples
- [ ] Resource usage optimized as needed

### Post-Training Checklist

- [ ] Final model saved and validated
- [ ] Performance benchmarks completed
- [ ] Safety compliance verified
- [ ] Model artifacts documented
- [ ] Training metrics analyzed and recorded
- [ ] Next steps and deployment planned

This comprehensive guide ensures successful LoRA fine-tuning for medical AI applications with proper monitoring, optimization, and validation throughout the training process.
