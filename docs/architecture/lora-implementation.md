# LoRA Implementation

This document provides a detailed explanation of the Low-Rank Adaptation (LoRA) implementation, covering the mathematical foundations, technical implementation, and practical considerations for medical AI fine-tuning.

## 🧮 Mathematical Foundation

### Low-Rank Matrix Decomposition

LoRA works by decomposing weight updates into low-rank matrices, significantly reducing the number of trainable parameters while maintaining model expressiveness.

For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$, instead of learning the full update $\Delta W \in \mathbb{R}^{d \times k}$, LoRA constrains the update to a low-rank representation:

$$\Delta W = B \cdot A$$

where:

- $B \in \mathbb{R}^{d \times r}$ (down-projection matrix)
- $A \in \mathbb{R}^{r \times k}$ (up-projection matrix)
- $r \ll \min(d, k)$ (rank constraint)

### Forward Pass Modification

The modified forward pass becomes:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

With scaling factor $\alpha$:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

The scaling factor $\alpha$ allows control over the magnitude of adapter contributions, typically set to $\alpha = 2r$.

### Parameter Efficiency

For a transformer layer with weight matrix dimensions:

- Query: $W_q \in \mathbb{R}^{d_{model} \times d_{model}}$
- Key: $W_k \in \mathbb{R}^{d_{model} \times d_{model}}$
- Value: $W_v \in \mathbb{R}^{d_{model} \times d_{model}}$
- Output: $W_o \in \mathbb{R}^{d_{model} \times d_{model}}$

**Parameter Comparison:**

- Original parameters: $4 \times d_{model}^2$
- LoRA parameters: $4 \times 2 \times r \times d_{model} = 8 \times r \times d_{model}$

**Reduction ratio:** $\frac{8 \times r \times d_{model}}{4 \times d_{model}^2} = \frac{2r}{d_{model}}$

For $d_{model} = 2048$ and $r = 32$: reduction ratio = $\frac{2 \times 32}{2048} = 3.125\%$

## 🔧 Technical Implementation

### LoRA Layer Implementation

```python
import torch
import torch.nn as nn
from typing import Optional, List

class LoRALayer(nn.Module):
    """Base LoRA layer implementation"""

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool = True
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights

        # Scaling factor
        self.scaling = self.lora_alpha / self.r

        # Dropout layer
        if lora_dropout > 0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()

    def reset_parameters(self):
        """Initialize LoRA parameters"""
        # Initialize A with Gaussian distribution
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        # Initialize B to zero for stability
        nn.init.zeros_(self.lora_B.weight)


class LoRALinear(LoRALayer):
    """LoRA adaptation for Linear layers"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(r, lora_alpha, lora_dropout)

        # LoRA matrices
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        if hasattr(self, 'lora_A'):
            # Apply LoRA: B(A(x)) * scaling
            lora_output = self.lora_B(self.lora_A(self.lora_dropout_layer(x)))
            return lora_output * self.scaling
        return torch.zeros_like(x)
```

### PEFT Integration

The implementation uses Hugging Face PEFT library for seamless integration:

```python
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

def setup_lora_model(base_model, config):
    """Setup LoRA adapters on base model"""

    # Prepare model for quantized training
    base_model = prepare_model_for_kbit_training(base_model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora'].get('bias', 'none'),
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    # Apply LoRA to model
    peft_model = get_peft_model(base_model, lora_config)

    return peft_model

def print_trainable_parameters(model):
    """Print trainable parameter statistics"""
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {100 * trainable_params / all_param:.4f}%")
```

## 🎯 Target Module Selection

### Attention Modules

The most effective modules to adapt in transformer architectures:

```python
# Common target modules for different architectures
TARGET_MODULES = {
    'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'mistral': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'phi': ['q_proj', 'k_proj', 'v_proj', 'dense'],
    'qwen': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'all_linear': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
}

def get_target_modules(model_name: str, strategy: str = 'attention_only'):
    """Get appropriate target modules for model architecture"""

    model_type = None
    for key in TARGET_MODULES:
        if key.lower() in model_name.lower():
            model_type = key
            break

    if model_type is None:
        # Fallback to common attention modules
        return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    if strategy == 'attention_only':
        return TARGET_MODULES[model_type][:4]  # Q, K, V, O
    elif strategy == 'all_linear':
        return TARGET_MODULES.get('all_linear', TARGET_MODULES[model_type])

    return TARGET_MODULES[model_type]
```

### Module Impact Analysis

```python
def analyze_module_impact(model, modules_to_test):
    """Analyze the impact of adapting different modules"""

    results = {}

    for module_set in modules_to_test:
        # Create temporary LoRA config
        temp_config = LoraConfig(
            r=16,
            target_modules=module_set,
            task_type=TaskType.CAUSAL_LM
        )

        # Count parameters that would be adapted
        temp_model = get_peft_model(deepcopy(model), temp_config)
        trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)

        results[str(module_set)] = {
            'trainable_params': trainable_params,
            'percentage': trainable_params / sum(p.numel() for p in model.parameters()) * 100
        }

    return results
```

## ⚙️ Hyperparameter Optimization

### Rank Selection Strategy

```python
def select_optimal_rank(model, dataset, rank_candidates=[8, 16, 32, 64]):
    """Empirically determine optimal rank for dataset"""

    results = {}

    for r in rank_candidates:
        # Setup model with current rank
        lora_config = LoraConfig(r=r, lora_alpha=r*2, task_type=TaskType.CAUSAL_LM)
        test_model = get_peft_model(deepcopy(model), lora_config)

        # Quick evaluation on subset
        subset_results = quick_evaluate(test_model, dataset[:100])

        results[r] = {
            'loss': subset_results['loss'],
            'parameters': sum(p.numel() for p in test_model.parameters() if p.requires_grad),
            'memory_gb': torch.cuda.max_memory_allocated() / (1024**3)
        }

    # Select rank with best loss-to-parameter ratio
    best_rank = min(results.keys(),
                   key=lambda r: results[r]['loss'] / (results[r]['parameters'] / 1e6))

    return best_rank, results
```

### Alpha Scaling Analysis

The relationship between rank $r$ and alpha $\alpha$:

```python
def analyze_alpha_scaling(base_model, dataset, r=32):
    """Analyze different alpha scaling strategies"""

    scaling_strategies = {
        'alpha_equals_r': r,           # α = r
        'alpha_double_r': r * 2,       # α = 2r (common)
        'alpha_sqrt_r': int(r ** 0.5), # α = √r
        'alpha_fixed': 16              # Fixed value
    }

    results = {}

    for strategy, alpha in scaling_strategies.items():
        # Test configuration
        config = LoraConfig(r=r, lora_alpha=alpha, task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(deepcopy(base_model), config)

        # Evaluate performance
        performance = evaluate_model(model, dataset[:200])

        results[strategy] = {
            'alpha': alpha,
            'scaling_factor': alpha / r,
            'loss': performance['loss'],
            'convergence_steps': performance['convergence_steps']
        }

    return results
```

## 🔄 Training Dynamics

### Gradient Flow Analysis

```python
def analyze_gradient_flow(model, data_loader):
    """Analyze gradient flow through LoRA adapters"""

    gradient_norms = {}

    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name:
            gradient_norms[name] = []

    # Track gradients during training
    for batch in data_loader[:10]:  # Sample batches
        loss = compute_loss(model, batch)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[name].append(grad_norm)

        model.zero_grad()

    # Analyze gradient statistics
    gradient_stats = {}
    for name, norms in gradient_norms.items():
        gradient_stats[name] = {
            'mean': np.mean(norms),
            'std': np.std(norms),
            'max': np.max(norms),
            'min': np.min(norms)
        }

    return gradient_stats
```

### Adapter Contribution Analysis

```python
def analyze_adapter_contributions(model, test_inputs):
    """Analyze relative contributions of different adapters"""

    contributions = {}

    # Hook to capture intermediate outputs
    def hook_fn(name):
        def hook(module, input, output):
            contributions[name] = output.detach().abs().mean().item()
        return hook

    # Register hooks on LoRA modules
    hooks = []
    for name, module in model.named_modules():
        if 'lora' in name and hasattr(module, 'weight'):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Forward pass to collect contributions
    with torch.no_grad():
        model(test_inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return contributions
```

## 🚀 Advanced LoRA Techniques

### Dynamic Rank Adaptation

```python
class DynamicRankLoRA(LoRALayer):
    """LoRA with dynamic rank adaptation during training"""

    def __init__(self, max_rank=64, initial_rank=8, **kwargs):
        super().__init__(initial_rank, **kwargs)
        self.max_rank = max_rank
        self.current_rank = initial_rank

        # Initialize maximum rank matrices
        self.lora_A_full = nn.Parameter(torch.zeros(kwargs['in_features'], max_rank))
        self.lora_B_full = nn.Parameter(torch.zeros(max_rank, kwargs['out_features']))

    def expand_rank(self, new_rank):
        """Expand the effective rank of adapters"""
        if new_rank <= self.current_rank or new_rank > self.max_rank:
            return

        # Expand active submatrices
        self.current_rank = new_rank
        print(f"Expanded LoRA rank to {new_rank}")

    def forward(self, x):
        # Use only active portion of matrices
        A_active = self.lora_A_full[:, :self.current_rank]
        B_active = self.lora_B_full[:self.current_rank, :]

        return torch.mm(torch.mm(x, A_active), B_active) * self.scaling
```

### Multi-Adapter Composition

```python
def setup_multi_adapter_model(base_model, adapter_configs):
    """Setup model with multiple specialized adapters"""

    model = base_model

    for adapter_name, config in adapter_configs.items():
        # Create LoRA config for this adapter
        lora_config = LoraConfig(
            r=config['r'],
            lora_alpha=config['alpha'],
            target_modules=config['modules'],
            task_type=TaskType.CAUSAL_LM
        )

        # Add adapter to model
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)

    return model

# Example: Medical specialization adapters
MEDICAL_ADAPTERS = {
    'cardiology': {
        'r': 32,
        'alpha': 64,
        'modules': ['q_proj', 'v_proj']
    },
    'general_medicine': {
        'r': 16,
        'alpha': 32,
        'modules': ['k_proj', 'o_proj']
    }
}
```

## 📊 Performance Analysis

### Memory and Speed Benchmarks

```python
def benchmark_lora_efficiency(model_configs):
    """Benchmark different LoRA configurations"""

    results = []

    for config in model_configs:
        # Setup model
        model = setup_lora_model(base_model, config)

        # Memory benchmark
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()

        # Speed benchmark
        dummy_input = torch.randint(0, 1000, (2, 512)).cuda()

        # Warmup
        for _ in range(10):
            _ = model(dummy_input)

        # Timing
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(100):
            _ = model(dummy_input)

        torch.cuda.synchronize()
        end_time = time.time()

        # Results
        memory_used = torch.cuda.memory_allocated() - start_memory
        avg_time = (end_time - start_time) / 100
        tokens_per_second = (2 * 512) / avg_time

        results.append({
            'config': config,
            'memory_mb': memory_used / (1024**2),
            'tokens_per_second': tokens_per_second,
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        })

    return results
```

This comprehensive LoRA implementation provides the foundation for efficient and effective fine-tuning of large language models for medical AI applications, balancing parameter efficiency with adaptation capability.
