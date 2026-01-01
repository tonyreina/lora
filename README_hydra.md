# Medical LLM Pipeline with Hydra Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management.

## Quick Start

The main entry point is `main.py` which uses Hydra configuration:

```bash
# Training only
pixi run python main.py mode=train

# Inference only (requires trained model)
pixi run python main.py mode=inference

# Demo inference (requires trained model)
pixi run python main.py mode=demo

# Full pipeline (train then demo) - requires sufficient GPU memory
pixi run python main.py mode=pipeline

# Quick demo (super fast training only)
pixi run python main.py --config-name=quick_demo

# Memory-optimized training for limited GPU systems
pixi run python main.py --config-name=memory_optimized

# Production training
pixi run python main.py --config-name=production
```

## Configuration Structure

All configuration is organized in the `conf/` directory:

```
conf/
├── config.yaml              # Main configuration file
├── model/
│   └── phi4_mini.yaml       # Model-specific settings
├── training/
│   ├── default.yaml         # Default training config
│   ├── production.yaml      # Production training config
│   └── debug.yaml           # Debug/fast training config
├── data/
│   └── medical.yaml         # Data-specific settings
└── inference/
    └── default.yaml         # Inference-specific settings
```

## Configuration Overrides

You can override any configuration parameter from the command line:

```bash
# Change training parameters
pixi run python main.py mode=train training.max_steps=100 training.batch_size=2

# Use different training config
pixi run python main.py mode=train training=production

# Change model
pixi run python main.py model.name=microsoft/Phi-3.5-mini-instruct

# Change output directory  
pixi run python main.py output_dir=./checkpoints/my-experiment

# Multiple overrides
pixi run python main.py mode=train training=debug model.max_length=1024 training.learning_rate=1e-3
```

## Available Modes

- `train`: Run training only
- `inference`: Run interactive inference (requires trained model)
- `demo`: Run demo inference with a sample question
- `pipeline`: Run complete training followed by demo inference

## Configuration Groups

### Model Configurations (`model=`)
- `phi4_mini`: Microsoft Phi-4-mini-instruct (default)

### Training Configurations (`training=`)
- `default`: Quick training with 10 steps (default)
- `production`: Full training with 100 steps
- `debug`: Ultra-fast training with 5 steps

### Data Configurations (`data=`)
- `medical`: Medical dataset configuration (default)

### Inference Configurations (`inference=`)
- `default`: Standard inference settings (default)

## Examples

### Quick Training and Demo
```bash
pixi run python main.py mode=pipeline
```

### Production Training
```bash
pixi run python main.py mode=train training=production
```

### Debug Training (Fast)
```bash
pixi run python main.py mode=train training=debug
```

### Interactive Inference
```bash
pixi run python main.py mode=inference
```

### Custom Configuration
```bash
pixi run python main.py mode=train \
  training.max_steps=50 \
  training.learning_rate=3e-4 \
  model.max_length=1024 \
  output_dir=./checkpoints/custom-experiment
```
