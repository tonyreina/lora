# Setup Instructions

Get your LoRA fine-tuning environment ready with this comprehensive setup guide. We'll cover everything from initial installation to verification testing.

## Quick Start

For the impatient, here's the minimal setup:

```bash
# Clone the repository
git clone https://github.com/tonyreina/lora.git
cd lora

# Install Pixi package manager
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# Install dependencies and run
pixi run --environment cuda python main.py train
```

## Detailed Setup Process

### Step 1: System Prerequisites

Ensure your system meets the requirements:

```bash
# Check GPU
nvidia-smi

# Check Python version
python --version  # Should be 3.10+

# Check available disk space (need ~50GB)
df -h
```

### Step 2: Clone Repository

```bash
# Clone your fork/copy of the repository
git clone https://github.com/tonyreina/lora.git
cd lora

# Verify structure
ls -la
# Should show: main.py, utils.py, config.yaml, data/, etc.
```

### Step 3: Package Manager Installation

We recommend **Pixi** for dependency management:

=== "Pixi (Recommended)"
    ```bash
    # Install Pixi
    curl -fsSL https://pixi.sh/install.sh | bash

    # Restart shell or source bashrc
    source ~/.bashrc

    # Verify installation
    pixi --version

    # Install project dependencies
    pixi install
    ```

=== "Pip Alternative"
    ```bash
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows

    # Install dependencies
    pip install -r requirements.txt
    ```

=== "Conda Alternative"
    ```bash
    # Create conda environment
    conda create -n lora python=3.10
    conda activate lora

    # Install PyTorch with CUDA
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # Install other dependencies
    pip install transformers datasets peft accelerate bitsandbytes loguru
    ```

### Step 4: CUDA Setup

Verify CUDA installation and compatibility:

```bash
# Check CUDA version
nvcc --version

# Check PyTorch CUDA support
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

If CUDA isn't available:

=== "Ubuntu/Debian"
    ```bash
    # Add NVIDIA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update

    # Install CUDA toolkit
    sudo apt-get -y install cuda-toolkit-12-3

    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "CentOS/RHEL"
    ```bash
    # Install CUDA repository
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

    # Install CUDA
    sudo dnf install cuda-toolkit-12-3
    ```

=== "Windows (WSL2)"
    ```bash
    # Install Windows CUDA drivers on Windows host
    # Then in WSL2:
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-3
    ```

### Step 5: Configuration

Customize your training setup:

```bash
# Copy example configuration
cp config.yaml my_config.yaml

# Edit configuration
nano my_config.yaml  # or use your favorite editor
```

Key settings to review:

```yaml
# Model selection
model:
  name: microsoft/Phi-4-mini-instruct  # or HuggingFaceTB/SmolLM-135M-Instruct

# LoRA parameters (adjust based on your GPU memory)
lora:
  r: 16          # Lower if memory constrained (8, 16, 32)
  alpha: 32      # Typically 2x the rank
  dropout: 0.1   # Regularization

# Training settings (adjust for your dataset size)
training:
  batch_size: 4              # Lower if out of memory (1, 2, 4)
  max_steps: 36              # Increase for larger datasets
  gradient_accumulation_steps: 8  # Effective batch = batch_size × this
  learning_rate: 2e-4        # Conservative starting point
```

### Step 6: Data Preparation

Create your training dataset:

```bash
# Create data directory
mkdir -p data

# Create example dataset
cat > data/my_custom_data.jsonl << 'EOF'
{"instruction": "What is hypertension?", "response": "Hypertension, commonly known as high blood pressure, is a medical condition where the blood pressure in the arteries is persistently elevated. It's defined as having a systolic pressure of 140 mmHg or higher, or a diastolic pressure of 90 mmHg or higher on multiple readings. This condition can lead to serious health complications if left untreated.", "source_urls": ["https://www.who.int/news-room/fact-sheets/detail/hypertension"]}
{"instruction": "How is diabetes diagnosed?", "response": "Diabetes is diagnosed using several blood tests. The most common methods include: 1) Fasting blood glucose test (≥126 mg/dL indicates diabetes), 2) Oral glucose tolerance test (≥200 mg/dL after 2 hours), 3) HbA1c test (≥6.5% indicates diabetes), or 4) Random blood glucose (≥200 mg/dL with symptoms). Multiple tests or repeat testing may be needed to confirm the diagnosis.", "source_urls": ["https://www.diabetes.org/diabetes/a1c/diagnosis"]}
EOF

# Validate your dataset
python -c "
import json
with open('data/my_custom_data.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line.strip())
        except Exception as e:
            print(f'Error line {i}: {e}')
        else:
            print(f'✅ Line {i} valid')
"
```

## Verification Tests

### Test 1: Environment Check

```bash
# Create test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset

def test_environment():
    print("🔍 Testing Environment Setup")
    print("=" * 40)

    # Python version
    print(f"Python version: {sys.version}")

    # PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

        # Memory test
        try:
            device = torch.device("cuda")
            x = torch.randn(1000, 1000, device=device)
            print(f"GPU memory test: ✅ Passed")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU memory test: ❌ Failed - {e}")

    # Test model loading
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        print("Model tokenizer: ✅ Loaded successfully")
    except Exception as e:
        print(f"Model tokenizer: ❌ Failed - {e}")

    # Test dataset loading
    try:
        dataset = load_dataset("json", data_files="data/my_custom_data.jsonl")
        print(f"Dataset loading: ✅ Loaded {len(dataset['train'])} examples")
    except Exception as e:
        print(f"Dataset loading: ❌ Failed - {e}")

    print("=" * 40)
    print("✅ Environment test complete!")

if __name__ == "__main__":
    test_environment()
EOF

# Run test
python test_setup.py
```

### Test 2: Memory Estimation

```bash
# Create memory test
cat > test_memory.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def estimate_memory():
    print("🧠 GPU Memory Estimation")
    print("=" * 30)

    if not torch.cuda.is_available():
        print("❌ No CUDA GPU available")
        return

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"Total GPU memory: {total_memory:.1f} GB")

    # Test quantized model loading
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print("Loading quantized model for memory test...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9

        print(f"Model memory usage:")
        print(f"  Allocated: {allocated:.1f} GB")
        print(f"  Reserved: {reserved:.1f} GB")
        print(f"  Free: {total_memory - reserved:.1f} GB")

        if total_memory - reserved > 2.0:
            print("✅ Sufficient memory for training")
        else:
            print("⚠️  Low memory - consider reducing batch size")

    except Exception as e:
        print(f"❌ Memory test failed: {e}")

if __name__ == "__main__":
    estimate_memory()
EOF

# Run memory test
python test_memory.py
```

### Test 3: Quick Training Test

```bash
# Test minimal training run
cat > test_training.py << 'EOF'
import yaml
from main import load_config, run_training

# Create minimal test config
test_config = {
    'seed': 42,
    'output_dir': './test_checkpoints',
    'model': {'name': 'microsoft/Phi-4-mini-instruct', 'max_length': 256},
    'lora': {'r': 8, 'alpha': 16, 'dropout': 0.1, 'target_modules': ['q_proj', 'v_proj']},
    'data': {
        'train_file': './data/my_custom_data.jsonl',
        'test_split': 0.2,
        'validation_split': 0.1,
        'system_prompt': 'You are a helpful assistant.'
    },
    'training': {
        'batch_size': 1,
        'max_steps': 3,  # Very short test
        'learning_rate': 2e-4,
        'gradient_accumulation_steps': 2,
        'logging_steps': 1,
        'early_stopping_patience': 10
    }
}

# Save test config
with open('test_config.yaml', 'w') as f:
    yaml.dump(test_config, f)

print("🧪 Running quick training test...")
try:
    from main import SimpleConfig
    cfg = SimpleConfig(test_config)
    adapter_dir = run_training(cfg)
    print(f"✅ Training test passed! Adapter saved to: {adapter_dir}")
except Exception as e:
    print(f"❌ Training test failed: {e}")
EOF

# Run training test
python test_training.py
```

## Troubleshooting Common Issues

### Out of Memory Errors

```yaml
# Reduce memory usage in config.yaml
training:
  batch_size: 1                    # Reduce from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size

lora:
  r: 8                            # Reduce from 16
  target_modules: [q_proj, v_proj] # Fewer modules

model:
  max_length: 256                 # Reduce from 512
```

### Model Download Issues

```bash
# Pre-download models
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-4-mini-instruct')
print('✅ Model downloaded and cached')
"
```

### Permission Issues

```bash
# Fix permissions
chmod +x main.py
sudo chown -R $USER:$USER ~/.cache/huggingface/

# If using conda/pip, ensure correct environment
which python
pip list | grep torch
```

### WSL2 Specific Issues

```bash
# Update WSL2
wsl --update

# Install Windows CUDA drivers (on Windows host)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify GPU access in WSL2
nvidia-smi
```

## Performance Optimization

### Speed Optimizations

```yaml
# Optimize for speed
training:
  gradient_checkpointing: true     # Save memory, slight speed cost
  dataloader_num_workers: 4       # Parallel data loading
  group_by_length: true           # Reduce padding

model:
  use_cache: false               # Required for training
```

### Memory Optimizations

```yaml
# Optimize for memory
training:
  gradient_accumulation_steps: 16  # Higher accumulation
  per_device_train_batch_size: 1   # Smaller batches
  gradient_checkpointing: true     # Trade compute for memory

lora:
  r: 8                            # Lower rank
  target_modules: [q_proj, v_proj] # Fewer target modules
```

## Next Steps

Once setup is complete:

1. **[Create your dataset](../dataset/creating-dataset.md)** with domain-specific examples
2. **[Run your first training](training.md)** with the default configuration
3. **[Monitor and tune](troubleshooting.md)** for optimal performance
4. **[Deploy your model](../architecture/overview.md)** for inference

---

Your environment should now be ready for LoRA fine-tuning! If you encounter issues, check the [troubleshooting guide](troubleshooting.md) or refer to the detailed error solutions above.
