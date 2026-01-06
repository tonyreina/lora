# Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues encountered during LoRA fine-tuning of medical AI models, with practical solutions and preventive measures.

## 🚨 Common Issues and Solutions

### Memory-Related Issues

#### Out of Memory (OOM) Errors

**Symptoms:**

```text
CUDA out of memory. Tried to allocate X.XX GiB (GPU 0; X.XX GiB total capacity)
RuntimeError: CUDA out of memory
```

**Immediate Solutions:**

```bash
# Quick memory cleanup
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size
python main.py train --batch-size 1 --gradient-steps 16

# Enable gradient checkpointing
python main.py train --gradient-checkpointing --fp16
```

**Configuration Fixes:**

```yaml
# memory_efficient_config.yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true
  dataloader_pin_memory: false
  dataloader_num_workers: 0

data:
  max_seq_length: 1024  # Reduced from 2048

model:
  quantization_config:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
```

**Memory Optimization Script:**

```python
# memory_optimizer.py
import torch
import gc

def optimize_memory():
    """Aggressive memory optimization"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()

    # Set memory management
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    print("Memory optimization completed")

# Run before training
optimize_memory()
```

#### Memory Leak During Training

**Symptoms:**

- Memory usage increases over time
- Training slows down progressively
- Eventually runs out of memory

**Solutions:**

```python
# Add to training loop
if step % 100 == 0:  # Every 100 steps
    torch.cuda.empty_cache()
    gc.collect()

# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(**batch)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Loading Issues

#### Model Architecture Compatibility

**Symptoms:**

```text
ValueError: Target modules ['q_proj', 'k_proj'] not found in model
AttributeError: 'PhiForCausalLM' object has no attribute 'q_proj'
```

**Diagnosis Script:**

```python
# diagnose_model.py
from transformers import AutoModel, AutoConfig
import torch

def diagnose_model_architecture(model_name):
    """Diagnose model architecture for LoRA compatibility"""

    print(f"🔍 Analyzing model: {model_name}")

    # Load config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"Architecture: {config.architectures}")

    # Load model
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    print("\n📋 Available modules:")
    for name, module in model.named_modules():
        if any(target in name for target in ['proj', 'linear', 'dense']):
            print(f"  {name}: {type(module).__name__}")

    # Suggest target modules
    suggested_modules = []
    for name, _ in model.named_modules():
        if any(suffix in name for suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            base_name = name.split('.')[-1]
            if base_name not in suggested_modules:
                suggested_modules.append(base_name)

    print(f"\n💡 Suggested target_modules: {suggested_modules}")

# Usage
diagnose_model_architecture("microsoft/Phi-4-mini-instruct")
```

**Model-Specific Solutions:**

```python
# Model-specific target modules
TARGET_MODULES_BY_ARCHITECTURE = {
    "PhiForCausalLM": ["q_proj", "k_proj", "v_proj", "dense"],
    "LlamaForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "MistralForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "Qwen2ForCausalLM": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

def get_target_modules(model):
    """Automatically detect appropriate target modules"""
    arch = model.config.architectures[0] if model.config.architectures else "Unknown"
    return TARGET_MODULES_BY_ARCHITECTURE.get(arch, ["q_proj", "k_proj", "v_proj", "o_proj"])
```

#### Quantization Issues

**Symptoms:**

```text
ImportError: bitsandbytes not installed
RuntimeError: CUDA capability sm_XX is not compatible with the current PyTorch installation
```

**Solutions:**

```bash
# Install bitsandbytes
pip install bitsandbytes

# For older GPUs
pip install bitsandbytes==0.41.0

# Verify installation
python -c "import bitsandbytes as bnb; print(bnb.__version__)"

# Check CUDA compatibility
python -c "
import torch
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU compute capability: {torch.cuda.get_device_capability()}')
"
```

### Training Convergence Issues

#### Loss Not Decreasing

**Symptoms:**

- Training loss remains flat or increases
- No improvement after many steps
- Evaluation loss stagnating

**Diagnostic Script:**

```python
# diagnose_training.py
import matplotlib.pyplot as plt
import json

def analyze_training_logs(checkpoint_dir):
    """Analyze training convergence issues"""

    trainer_state_path = f"{checkpoint_dir}/trainer_state.json"

    with open(trainer_state_path, 'r') as f:
        state = json.load(f)

    log_history = state['log_history']

    # Extract losses
    train_losses = []
    eval_losses = []
    learning_rates = []
    steps = []

    for entry in log_history:
        if 'train_loss' in entry:
            train_losses.append(entry['train_loss'])
            steps.append(entry['step'])

        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])

        if 'learning_rate' in entry:
            learning_rates.append(entry['learning_rate'])

    # Plot analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss
    axes[0, 0].plot(steps, train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')

    # Learning rate
    if learning_rates:
        axes[0, 1].plot(steps[:len(learning_rates)], learning_rates)
        axes[0, 1].set_title('Learning Rate Schedule')

    # Loss derivative (rate of change)
    if len(train_losses) > 10:
        loss_derivative = [train_losses[i] - train_losses[i-1]
                          for i in range(1, len(train_losses))]
        axes[1, 0].plot(steps[1:], loss_derivative)
        axes[1, 0].set_title('Loss Rate of Change')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')

    # Moving average
    window = min(10, len(train_losses) // 4)
    if window > 1:
        moving_avg = [sum(train_losses[max(0, i-window):i+1]) / min(window, i+1)
                     for i in range(len(train_losses))]
        axes[1, 1].plot(steps, train_losses, alpha=0.3, label='Raw')
        axes[1, 1].plot(steps, moving_avg, label='Moving Average')
        axes[1, 1].set_title('Loss Smoothing')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{checkpoint_dir}/training_analysis.png')

    # Diagnose issues
    issues = []

    # Check for plateau
    recent_losses = train_losses[-20:] if len(train_losses) >= 20 else train_losses
    if len(recent_losses) > 5:
        loss_variance = sum((x - sum(recent_losses)/len(recent_losses))**2
                          for x in recent_losses) / len(recent_losses)
        if loss_variance < 0.001:
            issues.append("Training has plateaued - consider adjusting learning rate")

    # Check for divergence
    if len(train_losses) > 10 and train_losses[-1] > train_losses[5] * 1.5:
        issues.append("Training may be diverging - reduce learning rate")

    # Check learning rate
    if learning_rates and learning_rates[-1] < 1e-7:
        issues.append("Learning rate too low - training may have stopped learning")

    return issues

# Usage
issues = analyze_training_logs("./checkpoints/medical-phi4-lora")
for issue in issues:
    print(f"⚠️ {issue}")
```

**Solutions for Convergence Issues:**

1. **Learning Rate Too High:**

```yaml
training:
  learning_rate: 5e-5  # Reduced from 1e-4
  warmup_steps: 200    # More warmup
  lr_scheduler_type: "cosine"
```

1. **Learning Rate Too Low:**

```yaml
training:
  learning_rate: 2e-4  # Increased
  warmup_steps: 50     # Less warmup
```

1. **LoRA Rank Issues:**

```yaml
lora:
  r: 64        # Increase for complex datasets
  lora_alpha: 128  # Maintain 2:1 ratio
  lora_dropout: 0.05  # Reduce dropout
```

### Data-Related Issues

#### Dataset Loading Errors

**Symptoms:**

```text
json.JSONDecodeError: Expecting ',' delimiter
ValueError: All examples must have the same number of keys
```

**Validation Script:**

```python
# validate_dataset.py
import json
import sys

def validate_jsonl_dataset(file_path):
    """Comprehensive dataset validation"""

    issues = []
    valid_entries = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON
                data = json.loads(line.strip())

                # Validate structure
                if not isinstance(data, dict):
                    issues.append(f"Line {line_num}: Not a JSON object")
                    continue

                if 'messages' not in data:
                    issues.append(f"Line {line_num}: Missing 'messages' field")
                    continue

                if not isinstance(data['messages'], list):
                    issues.append(f"Line {line_num}: 'messages' must be a list")
                    continue

                # Validate messages
                for msg_idx, msg in enumerate(data['messages']):
                    if not isinstance(msg, dict):
                        issues.append(f"Line {line_num}, Message {msg_idx}: Not an object")
                        continue

                    if 'role' not in msg or 'content' not in msg:
                        issues.append(f"Line {line_num}, Message {msg_idx}: Missing role or content")
                        continue

                    if msg['role'] not in ['system', 'user', 'assistant']:
                        issues.append(f"Line {line_num}, Message {msg_idx}: Invalid role '{msg['role']}'")

                    if not isinstance(msg['content'], str) or not msg['content'].strip():
                        issues.append(f"Line {line_num}, Message {msg_idx}: Empty or invalid content")

                valid_entries += 1

            except json.JSONDecodeError as e:
                issues.append(f"Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                issues.append(f"Line {line_num}: Unexpected error - {e}")

    print(f"Dataset Validation Results:")
    print(f"  Valid entries: {valid_entries}")
    print(f"  Issues found: {len(issues)}")

    if issues:
        print("\nIssues:")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  {issue}")

        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")

    return len(issues) == 0

# Usage
if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/my_custom_data.jsonl"
    is_valid = validate_jsonl_dataset(file_path)
    sys.exit(0 if is_valid else 1)
```

#### Tokenization Issues

**Symptoms:**

- Extremely long or short sequences
- Special tokens not handled correctly
- Chat template errors

**Debug Script:**

```python
# debug_tokenization.py
from transformers import AutoTokenizer

def debug_tokenization(model_name, sample_text):
    """Debug tokenization issues"""

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"🔍 Tokenization Debug for: {model_name}")
    print(f"Sample text: {sample_text[:100]}...")
    print()

    # Basic tokenization
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)

    print(f"Token count: {len(tokens)}")
    print(f"Token ID count: {len(token_ids)}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"Last 10 tokens: {tokens[-10:]}")
    print()

    # Special tokens
    print("Special tokens:")
    print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print()

    # Chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("Chat template available")

        sample_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        try:
            formatted = tokenizer.apply_chat_template(
                sample_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print("Chat template formatting successful")
            print(f"Formatted length: {len(formatted)}")
        except Exception as e:
            print(f"Chat template error: {e}")
    else:
        print("No chat template available")

# Usage
debug_tokenization(
    "microsoft/Phi-4-mini-instruct",
    "What are the symptoms of diabetes? Please provide a comprehensive answer."
)
```

### Environment and Dependency Issues

#### CUDA/GPU Issues

**Symptoms:**

```text
RuntimeError: No CUDA GPUs are available
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

**Diagnostic Commands:**

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Check compatibility
python -c "
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f'GPU compute capability: {major}.{minor}')
"
```

**Solutions:**

```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only training (not recommended for large models)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Package Version Conflicts

**Symptoms:**

```text
ImportError: cannot import name 'AutoPeftModelForCausalLM'
TypeError: 'NoneType' object has no attribute 'split'
```

**Version Check Script:**

```python
# check_versions.py
import subprocess
import sys

def check_package_versions():
    """Check critical package versions"""

    critical_packages = [
        'torch',
        'transformers',
        'peft',
        'datasets',
        'accelerate',
        'bitsandbytes'
    ]

    print("📦 Package Version Check")
    print("=" * 30)

    for package in critical_packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        version = line.split(': ')[1]
                        print(f"{package}: {version}")
                        break
            else:
                print(f"{package}: Not installed")
        except Exception as e:
            print(f"{package}: Error checking - {e}")

check_package_versions()
```

**Fix Version Conflicts:**

```bash
# Update to compatible versions
pip install --upgrade transformers peft accelerate datasets

# Known working combination (example)
pip install transformers==4.36.0 peft==0.6.0 torch==2.1.0

# Use requirements file
pip install -r requirements_known_good.txt
```

### Performance Issues

#### Slow Training Speed

**Diagnostic Script:**

```python
# benchmark_training.py
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark_training_speed():
    """Benchmark training components"""

    model_name = "microsoft/Phi-4-mini-instruct"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare test data
    test_text = "What are the symptoms of diabetes?" * 50  # Longer text
    inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)

    # Move to GPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Benchmark forward pass
    model.eval()
    forward_times = []

    for _ in range(10):
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)

    avg_forward_time = sum(forward_times[2:]) / len(forward_times[2:])  # Skip warmup

    print(f"Average forward pass time: {avg_forward_time:.3f}s")
    print(f"Tokens per second: {inputs['input_ids'].shape[1] / avg_forward_time:.1f}")

    # Memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"GPU memory usage: {memory_mb:.1f} MB")

benchmark_training_speed()
```

**Speed Optimization:**

```yaml
# fast_training_config.yaml
model:
  attn_implementation: "flash_attention_2"  # Requires flash-attn
  torch_compile: true

training:
  fp16: true
  gradient_checkpointing: true
  dataloader_num_workers: 4
  dataloader_pin_memory: true

# Install FlashAttention
# pip install flash-attn --no-build-isolation
```

## 🛠️ Emergency Recovery Procedures

### Training Crash Recovery

```python
# recover_training.py
import os
import json
import torch

def recover_from_crash(checkpoint_dir):
    """Recover training from crash"""

    print("🚨 Attempting training recovery...")

    # Find latest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_dir)
                  if d.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir, d))]

    if not checkpoints:
        print("❌ No checkpoints found")
        return None

    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    print(f"📁 Latest checkpoint: {latest_checkpoint}")

    # Verify checkpoint integrity
    required_files = ['adapter_config.json', 'adapter_model.bin', 'trainer_state.json']

    for file in required_files:
        file_path = os.path.join(latest_checkpoint, file)
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file}")
            return None

    # Check trainer state
    with open(os.path.join(latest_checkpoint, 'trainer_state.json'), 'r') as f:
        trainer_state = json.load(f)

    print(f"✅ Recovery possible from step {trainer_state['global_step']}")
    print(f"   Epoch: {trainer_state['epoch']}")
    print(f"   Best metric: {trainer_state.get('best_metric', 'N/A')}")

    return latest_checkpoint

# Usage
recovery_path = recover_from_crash("./checkpoints/medical-phi4-lora")
if recovery_path:
    print(f"Resume training with: --resume-from {recovery_path}")
```

### Model Corruption Fix

```python
# fix_corrupted_model.py
import torch
import os

def fix_corrupted_checkpoint(checkpoint_path):
    """Fix corrupted model checkpoints"""

    model_file = os.path.join(checkpoint_path, 'adapter_model.bin')

    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False

    try:
        # Try to load the model
        state_dict = torch.load(model_file, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {len(state_dict)}")

        # Check for NaN or inf values
        corrupted_params = []
        for name, param in state_dict.items():
            if torch.isnan(param).any() or torch.isinf(param).any():
                corrupted_params.append(name)

        if corrupted_params:
            print(f"❌ Corrupted parameters found: {corrupted_params}")

            # Attempt to fix by zeroing corrupted parameters
            for name in corrupted_params:
                print(f"   Zeroing parameter: {name}")
                state_dict[name] = torch.zeros_like(state_dict[name])

            # Save fixed model
            backup_file = model_file + '.backup'
            os.rename(model_file, backup_file)
            torch.save(state_dict, model_file)

            print(f"✅ Model repaired and saved")
            print(f"   Backup saved as: {backup_file}")
        else:
            print("✅ No corruption detected")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
```

## 📞 Getting Help

### Information to Include in Bug Reports

1. **System Information:**
   - GPU model and memory
   - CUDA version
   - PyTorch version
   - Python version
   - Operating system

2. **Configuration:**
   - Complete config.yaml
   - Command line arguments used
   - Dataset information

3. **Error Details:**
   - Complete error traceback
   - Steps to reproduce
   - When the error occurs
   - Any workarounds attempted

4. **Logs:**
   - Training logs
   - System resource usage
   - GPU utilization during failure

### Self-Diagnostic Script

```python
# system_diagnostic.py
import torch
import sys
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_full_diagnostic():
    """Run comprehensive system diagnostic"""

    print("🔧 LoRA Medical AI System Diagnostic")
    print("=" * 50)

    # System info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("CUDA available: No")

    # Package versions
    print("\n📦 Package Versions:")
    packages = ['transformers', 'peft', 'datasets', 'accelerate', 'bitsandbytes']
    for pkg in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'Unknown')
            print(f"  {pkg}: {version}")
        except ImportError:
            print(f"  {pkg}: Not installed")

    # Test model loading
    print("\n🤖 Model Loading Test:")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            trust_remote_code=True
        )
        print("  Tokenizer: ✅")

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("  Model: ✅")

        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"  GPU Memory: {memory_mb:.1f} MB")

    except Exception as e:
        print(f"  Model loading failed: {e}")

    print("\n✅ Diagnostic complete")

if __name__ == "__main__":
    run_full_diagnostic()
```

This comprehensive troubleshooting guide provides solutions for the most common issues encountered during LoRA fine-tuning, along with diagnostic tools and recovery procedures to ensure successful training of medical AI models.
