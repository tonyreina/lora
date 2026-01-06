# Prerequisites

Before starting with LoRA fine-tuning, ensure you have the necessary hardware, software, and knowledge prerequisites.

## Hardware Requirements

### GPU Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM (*e.g.,* RTX 2000 Ada Generation)
- **Recommended**: 16GB+ VRAM (*e.g.,* RTX 4080, RTX 4090, V100)

!!! warning "AMD GPU Support"
    While PyTorch supports AMD GPUs through ROCm, this guide focuses on NVIDIA CUDA. AMD users may need to adapt installation instructions.

### System Requirements

- **RAM**: 16GB+ system RAM (32GB recommended)
- **Storage**: 50GB+ free space for models and datasets
- **CPU**: Modern multi-core processor (for data processing)

### Tested Configurations

- **Surface Laptop Studio 2**: RTX 2000 Ada, 32GB RAM, WSL2

## Software Prerequisites

### Operating System

- **Linux** (Ubuntu 22.04) - Supported.
- **Windows** with WSL2 - Well tested.
- **macOS** - I don't know. Did not test.

### Package Manager

We recommend **Pixi** for reproducible Python virtual environments:

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

```

## HuggingFace Authentication

### Creating a HuggingFace Account

1. Visit [huggingface.co](https://huggingface.co) and create a free account
2. Verify your email address
3. Go to your profile settings and create an access token:

   - Navigate to Settings → Access Tokens
   - Create a new token with **Read** permissions (sufficient for downloading models)
   - Copy the token - you'll need it for authentication

### Authentication with Pixi

Store your HuggingFace token securely using the pixi environment:

```bash
pixi run hf auth login
```

When prompted, paste your HuggingFace access token. This will store the token in your local
HuggingFace cache for authenticated model downloads.

### Requesting Access to Phi Models

Many HuggingFace models require explicit access approval.
Without such approval the training script will return an error
about a valid HuggingFace (HF) authorization token.

1. Visit the model page: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
2. Click **"Request access to this model"**
3. Fill out the access request form with:
   - **Intended use case**: Educational fine-tuning
   - **Organization**: Your organization or "Personal Research"
   - Accept the Microsoft license terms
4. Submit the request

!!! info "Access Approval"
    Access is typically granted within a few hours to a day. You'll receive an email notification when approved.

!!! warning "License Compliance"
    Ensure your use case complies with Microsoft's Phi model license terms, especially for commercial applications.

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

### Essential Extensions (VS Code)

- Python extension pack
- Jupyter notebook support

Next: Learn how to [create your dataset](../dataset/creating-dataset.md) in the
proper JSONL format.
