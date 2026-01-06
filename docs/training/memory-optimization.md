# Memory Optimization

This document covers advanced memory optimization techniques for LoRA fine-tuning, enabling efficient training of large language models on consumer hardware with limited VRAM.

## 💾 Memory Usage Overview

### Typical Memory Breakdown

For LoRA fine-tuning of Phi-4-mini on a 6GB GPU:

```text
Base Model (4-bit quantized):     ~3.5GB
LoRA Adapters:                    ~50MB
Optimizer States:                 ~100MB
Gradients:                        ~50MB
Activation Cache:                 ~1-2GB
Input Batch:                      ~500MB-1GB
---
Total:                           ~6-7GB
```

### Memory Optimization Hierarchy

1. **Model-level**: Quantization, parameter sharing
2. **Training-level**: Gradient checkpointing, mixed precision
3. **Data-level**: Dynamic batching, sequence packing
4. **System-level**: CPU offloading, memory mapping

## 🔧 Implementation Strategies

### 1. Gradient Checkpointing

Trade computation for memory by recomputing activations:

```python
def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Configure for transformer models
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    return model

# Memory savings: ~30-50% reduction in activation memory
```

### 2. Dynamic Batch Size Adjustment

```python
class AdaptiveBatchSizer:
    """Automatically adjust batch size based on available memory"""

    def __init__(self, initial_batch_size=2, max_batch_size=8, min_batch_size=1):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.oom_count = 0

    def adjust_batch_size(self, oom_occurred=False):
        """Adjust batch size based on memory availability"""
        if oom_occurred:
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            self.oom_count += 1
            print(f"OOM detected. Reduced batch size to {self.current_batch_size}")
        else:
            # Gradually increase if stable
            if self.oom_count == 0 and self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(
                    self.max_batch_size,
                    self.current_batch_size + 1
                )

        return self.current_batch_size

def train_with_adaptive_batching(model, dataloader, trainer_config):
    """Training loop with adaptive batch sizing"""
    batch_sizer = AdaptiveBatchSizer()

    for epoch in range(trainer_config['num_epochs']):
        try:
            # Train with current batch size
            train_epoch(model, dataloader, batch_sizer.current_batch_size)
            batch_sizer.adjust_batch_size(oom_occurred=False)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_sizer.adjust_batch_size(oom_occurred=True)

                if batch_sizer.current_batch_size < batch_sizer.min_batch_size:
                    raise RuntimeError("Cannot reduce batch size further")

                # Retry with smaller batch
                continue
            else:
                raise e
```

### 3. CPU Offloading

```python
def setup_cpu_offloading(model, offload_ratio=0.5):
    """Offload portions of model to CPU memory"""

    # Calculate layers to offload
    num_layers = len(model.model.layers)
    offload_layers = int(num_layers * offload_ratio)

    device_map = {}

    # Keep embedding and first layers on GPU
    device_map["model.embed_tokens"] = 0

    # Distribute layers between GPU and CPU
    for i in range(num_layers):
        if i < num_layers - offload_layers:
            device_map[f"model.layers.{i}"] = 0  # GPU
        else:
            device_map[f"model.layers.{i}"] = "cpu"  # CPU

    # Keep output layers on GPU for efficiency
    device_map["lm_head"] = 0

    print(f"Offloading {offload_layers} layers to CPU")
    return device_map
```

### 4. Memory-Efficient Data Loading

```python
class MemoryEfficientDataLoader:
    """Data loader optimized for memory usage"""

    def __init__(self, dataset, batch_size, max_length=2048):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        """Yield memory-efficient batches"""
        batch = []
        current_max_len = 0

        for item in self.dataset:
            item_length = len(item['input_ids'])

            # Check if adding this item would exceed memory limits
            projected_max_len = max(current_max_len, item_length)
            projected_memory = len(batch) * projected_max_len * 4  # 4 bytes per token

            if projected_memory > self.max_length * self.batch_size * 4:
                if batch:
                    yield self._create_batch(batch)
                    batch = []
                    current_max_len = 0

            batch.append(item)
            current_max_len = max(current_max_len, item_length)

            if len(batch) >= self.batch_size:
                yield self._create_batch(batch)
                batch = []
                current_max_len = 0

        if batch:
            yield self._create_batch(batch)

    def _create_batch(self, items):
        """Create optimally padded batch"""
        max_len = max(len(item['input_ids']) for item in items)

        # Pad to nearest multiple of 8 for efficiency
        max_len = ((max_len + 7) // 8) * 8

        return {
            'input_ids': self._pad_sequences([item['input_ids'] for item in items], max_len),
            'attention_mask': self._pad_sequences([item['attention_mask'] for item in items], max_len),
            'labels': self._pad_sequences([item['labels'] for item in items], max_len, pad_value=-100)
        }
```

## 📊 Memory Monitoring

### Real-time Memory Tracking

```python
import psutil
import time

class MemoryMonitor:
    """Monitor GPU and system memory usage during training"""

    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.step_count = 0
        self.memory_history = []

    def log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / (1024**3),
                'cached': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated': torch.cuda.max_memory_allocated() / (1024**3)
            }
        else:
            gpu_memory = {'allocated': 0, 'cached': 0, 'max_allocated': 0}

        # System memory
        system_memory = psutil.virtual_memory()

        memory_info = {
            'step': self.step_count,
            'gpu_allocated_gb': gpu_memory['allocated'],
            'gpu_cached_gb': gpu_memory['cached'],
            'gpu_max_allocated_gb': gpu_memory['max_allocated'],
            'system_used_percent': system_memory.percent,
            'system_available_gb': system_memory.available / (1024**3)
        }

        self.memory_history.append(memory_info)

        if self.step_count % self.log_interval == 0:
            print(f"Step {self.step_count} Memory Usage:")
            print(f"  GPU: {gpu_memory['allocated']:.2f}GB allocated, {gpu_memory['cached']:.2f}GB cached")
            print(f"  System: {system_memory.percent:.1f}% used, {system_memory.available/(1024**3):.2f}GB available")

        self.step_count += 1

        return memory_info

    def get_memory_summary(self):
        """Get summary of memory usage during training"""
        if not self.memory_history:
            return {}

        gpu_allocated = [entry['gpu_allocated_gb'] for entry in self.memory_history]

        return {
            'peak_gpu_memory_gb': max(gpu_allocated),
            'avg_gpu_memory_gb': sum(gpu_allocated) / len(gpu_allocated),
            'memory_efficiency': min(gpu_allocated) / max(gpu_allocated) if max(gpu_allocated) > 0 else 0
        }
```

## ⚡ Advanced Optimization Techniques

### 1. Sequence Packing

```python
def pack_sequences(dataset, max_length=2048):
    """Pack multiple short sequences into single training examples"""
    packed_examples = []
    current_packed = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    current_length = 0

    for example in dataset:
        example_length = len(example['input_ids'])

        if current_length + example_length + 1 <= max_length:  # +1 for separator
            # Add separator if not first sequence
            if current_length > 0:
                current_packed['input_ids'].append(tokenizer.sep_token_id)
                current_packed['attention_mask'].append(1)
                current_packed['labels'].append(-100)
                current_length += 1

            # Add example
            current_packed['input_ids'].extend(example['input_ids'])
            current_packed['attention_mask'].extend(example['attention_mask'])
            current_packed['labels'].extend(example['labels'])
            current_length += example_length

        else:
            # Save current packed example and start new one
            if current_length > 0:
                packed_examples.append({
                    'input_ids': torch.tensor(current_packed['input_ids']),
                    'attention_mask': torch.tensor(current_packed['attention_mask']),
                    'labels': torch.tensor(current_packed['labels'])
                })

            # Start new packed example
            current_packed = {
                'input_ids': example['input_ids'].tolist(),
                'attention_mask': example['attention_mask'].tolist(),
                'labels': example['labels'].tolist()
            }
            current_length = example_length

    # Add final packed example
    if current_length > 0:
        packed_examples.append({
            'input_ids': torch.tensor(current_packed['input_ids']),
            'attention_mask': torch.tensor(current_packed['attention_mask']),
            'labels': torch.tensor(current_packed['labels'])
        })

    print(f"Packed {len(dataset)} sequences into {len(packed_examples)} examples")
    return packed_examples
```

### 2. Activation Checkpointing with Selective Recomputation

```python
def setup_selective_checkpointing(model, checkpoint_layers=None):
    """Setup selective activation checkpointing"""

    if checkpoint_layers is None:
        # Checkpoint every other layer by default
        total_layers = len(model.model.layers)
        checkpoint_layers = list(range(1, total_layers, 2))

    def checkpoint_forward_hook(module, input, output):
        """Hook to checkpoint specific layers"""
        return torch.utils.checkpoint.checkpoint(
            lambda x: module._original_forward(x), input[0], use_reentrant=False
        )

    # Apply checkpointing to selected layers
    for i, layer in enumerate(model.model.layers):
        if i in checkpoint_layers:
            layer._original_forward = layer.forward
            layer.register_forward_hook(checkpoint_forward_hook)

    print(f"Applied selective checkpointing to layers: {checkpoint_layers}")
```

## 🎯 Best Practices

### Memory-Conscious Training Configuration

```python
MEMORY_OPTIMIZED_CONFIG = {
    "model": {
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        },
        "device_map": "auto",
        "low_cpu_mem_usage": True
    },
    "training": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,  # Maintain effective batch size
        "gradient_checkpointing": True,
        "fp16": True,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False
    },
    "data": {
        "max_seq_length": 1024,  # Reduced for memory efficiency
        "pack_sequences": True
    }
}
```

### Memory Emergency Procedures

```python
def handle_memory_emergency():
    """Emergency memory cleanup procedures"""

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    import gc
    gc.collect()

    # Clear any cached datasets
    if hasattr(torch.utils.data, '_utils'):
        if hasattr(torch.utils.data._utils, 'collate'):
            torch.utils.data._utils.collate._cached_functions.clear()

    print("Emergency memory cleanup completed")
```

This comprehensive memory optimization guide ensures efficient training even on limited hardware resources while maintaining model performance and training stability.
