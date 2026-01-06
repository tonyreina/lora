# Prerequisites

Before starting with LoRA fine-tuning, ensure you have the necessary hardware, software, and knowledge prerequisites.

## Hardware Requirements

### GPU Requirements

- **Minimum**: NVIDIA GPU with 6GB VRAM (RTX 3060, RTX 4060)
- **Recommended**: 8-12GB VRAM (RTX 3070, RTX 4070, RTX 3080)
- **Optimal**: 16GB+ VRAM (RTX 4080, RTX 4090, Tesla V100)

!!! warning "AMD GPU Support"
    While PyTorch supports AMD GPUs through ROCm, this guide focuses on NVIDIA CUDA. AMD users may need to adapt installation instructions.

### System Requirements

- **RAM**: 16GB+ system RAM (32GB recommended)
- **Storage**: 50GB+ free space for models and datasets
- **CPU**: Modern multi-core processor (for data processing)

### Tested Configurations

- **Surface Laptop Studio 2**: RTX 2000 Ada, 32GB RAM, WSL2
- **Desktop**: RTX 3080, 32GB RAM, Ubuntu 22.04
- **Cloud**: AWS p3.2xlarge (V100), Google Colab Pro+

## Software Prerequisites

### Operating System

- **Linux** (Ubuntu 20.04+, CentOS 8+) - Primary support
- **Windows** with WSL2 - Well tested
- **macOS** - Limited GPU support (CPU only)

### Python Environment

```bash
# Check Python version
python --version  # Should be 3.10+

# Recommended: Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### CUDA Installation

CUDA 11.8+ or 12.x required for GPU acceleration:

```bash
# Check CUDA version
nvidia-smi

# Install CUDA toolkit (if needed)
# Ubuntu example:
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

### Package Manager

We recommend **Pixi** for reproducible environments:

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Alternative: Use pip with requirements.txt
pip install -r requirements.txt
```

## Knowledge Prerequisites

### Essential Concepts

- **Basic machine learning**: Understanding of training, validation, overfitting
- **Neural networks**: Familiarity with layers, parameters, backpropagation
- **Transformers**: Basic understanding of attention mechanisms
- **Python programming**: Comfortable with Python syntax and libraries

### Helpful Background

- **PyTorch basics**: Tensor operations, automatic differentiation
- **Hugging Face ecosystem**: Transformers library, model hub
- **Command line proficiency**: File navigation, environment management
- **Git version control**: For tracking experiments and code changes

### Medical AI (If Applicable)

- **Medical ethics**: Understanding of healthcare AI responsibilities
- **Regulatory awareness**: FDA guidelines, HIPAA compliance
- **Domain knowledge**: Relevant medical background for your use case

## Development Environment

### Recommended IDE/Editor

- **VS Code**: Excellent Python support, Jupyter integration
- **PyCharm**: Full-featured Python IDE
- **Jupyter**: Interactive development and experimentation
- **Vim/Neovim**: For terminal-based development

### Essential Extensions (VS Code)

- Python extension pack
- Jupyter notebook support
- GitLens for version control
- Thunder Client for API testing

### Terminal Setup

```bash
# Install useful tools
sudo apt update && sudo apt install -y \
    htop \
    ncdu \
    tree \
    curl \
    wget \
    git \
    tmux

# Configure git (if not done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Verification Checklist

Before proceeding, verify your setup:

### GPU Check

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Python Packages

```python
# Test key imports
try:
    import transformers
    import torch
    import datasets
    import peft
    print("✅ All packages imported successfully")
except ImportError as e:
    print(f"❌ Missing package: {e}")
```

### Memory Test

```python
import torch

# Test GPU memory allocation
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Try to allocate 1GB tensor
    try:
        x = torch.randn(1024, 1024, 256, device=device)
        print("✅ GPU memory allocation successful")
        del x
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"❌ GPU memory issue: {e}")
```

## Common Issues and Solutions

### CUDA Out of Memory

- Reduce batch size in config.yaml
- Enable gradient checkpointing
- Use smaller models for testing

### Package Conflicts

- Use fresh virtual environment
- Pin package versions in requirements.txt
- Consider using conda instead of pip

### Model Download Issues

- Check internet connection and firewall
- Verify Hugging Face Hub access
- Use `huggingface-cli login` if needed

### WSL2 GPU Access

- Install CUDA inside WSL2
- Update Windows GPU drivers
- Enable GPU passthrough

---

Next: Learn how to [create your dataset](../dataset/creating-dataset.md) in the
proper JSONL format.
