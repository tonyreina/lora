# Medical LLM Fine-Tuning Pipeline

A medical language model fine-tuning system using LoRA (Low-Rank Adaptation) for efficient training on consumer GPUs. Fine-tune large language models like Microsoft Phi-4 for medical question-answering tasks.

## What This Does

- **Fine-tunes medical AI models** using your own medical Q&A dataset
- **Memory efficient** training with 4-bit quantization and LoRA adapters
- **Interactive inference** to chat with your trained medical assistant
- **Automatic model saving** for later use and deployment

## Prerequisites

- NVIDIA GPU with at least 6GB RAM
- [Pixi](https://pixi.sh) package manager
- Medical training data in JSONL format

## Installation

1. Clone this repository
2. Install Pixi from [pixi.sh](https://pixi.sh)
3. Navigate to the project directory

## Usage

### Training a Model

```bash
pixi run python main.py train
```

This will:
- Load your training data from `data/my_custom_data.jsonl`
- Fine-tune the Phi-4-mini model using LoRA
- Save the trained adapter to `./checkpoints/model/lora_adapter/`

### Using the Trained Model

```bash
pixi run python main.py inference
```

This starts an interactive medical AI assistant where you can ask questions.

### Custom Configuration

Use a different config file:
```bash
pixi run python main.py train my_config.yaml
```

## Data Format

Your training data should be in JSONL format with medical instruction-response pairs:

```json
{"instruction": "What are the symptoms of diabetes?", "response": "Common symptoms include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision."}
{"instruction": "How do you treat hypertension?", "response": "Treatment typically involves lifestyle changes (diet, exercise) and may include medications like ACE inhibitors or diuretics."}
```

Place your data file at `data/my_custom_data.jsonl`.

## Configuration Options

Edit `config.yaml` to customize training:

```yaml
# Training duration and batch size
training:
  num_epochs: 3          # Number of training epochs
  batch_size: 4          # Batch size per GPU
  learning_rate: 2e-4    # Learning rate

# Model settings
model:
  name: microsoft/Phi-4-mini-instruct
  max_length: 512        # Maximum sequence length

# LoRA adapter settings
lora:
  r: 16                  # LoRA rank (higher = more parameters)
  alpha: 32              # LoRA scaling factor
  dropout: 0.1           # Dropout rate

# Output location
output_dir: ./checkpoints/model
```

### Quick Setup Presets

For **fast testing** (1 epoch, small batch):
```yaml
training:
  num_epochs: 1
  batch_size: 1
```

For **production training** (more epochs, better results):
```yaml
training:
  num_epochs: 10
  learning_rate: 1e-4
```

## Technical Details

- **LoRA (Low-Rank Adaptation)**: Efficiently fine-tunes only a small number of parameters
- **4-bit Quantization**: Reduces memory usage for training on consumer GPUs  
- **Early Stopping**: Automatically stops training when the model stops improving
- **Gradient Checkpointing**: Further reduces memory usage during training
- **Chat Template**: Formats conversations for instruction-following models

## System Requirements

- **GPU**: NVIDIA GPU with 6GB+ RAM (RTX 3060, RTX 4060, or better)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10GB+ free space for model weights and checkpoints
- **OS**: Linux, macOS, or Windows with WSL2

## File Structure

```
medical-llm-pipeline/
├── main.py                    # Main training and inference script
├── config.yaml                # Configuration settings
├── src/utils.py              # Core training and inference utilities
├── data/my_custom_data.jsonl # Your medical training data
└── checkpoints/              # Saved model adapters
    └── model/lora_adapter/   # LoRA weights after training
```

## Troubleshooting

**Out of Memory Error**: Reduce `batch_size` to 1 or `max_length` to 256 in config.yaml

**Training Too Slow**: Increase `batch_size` if you have more GPU memory

**Poor Results**: Increase `num_epochs` or check your training data quality
