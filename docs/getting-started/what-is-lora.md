# What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning
technique that allows you to adapt [large language models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model)
without modifying all the model parameters. Instead of updating billions of
parameters, LoRA introduces small "adapter" matrices that can be trained
efficiently.

!!! info "This one weird trick ..."
    LoRA's key *trick* is that a large matrix can be created
    by multiplying two smaller matricies. This reduces
    the number of trainable parameters from N x M
    down to about N + M.

## Key Benefits

- **Memory Efficient**: Only ~1% of parameters need training
- **Fast Training**: Significantly reduced training time (minutes to hours, not days to weeks)
- **Modular**: Adapters can be swapped without retraining the base model
- **Consumer Hardware Friendly**: Can run on GPUs with as little as 8GB RAM

!!! important "Modular adapters"
    Modular adapters are essential for fast inference. The large, base LLM
    can be loaded into memory once.
    The smaller LoRA adapter weights can be quickly loaded and
    unloaded based on the inference task.

## How LoRA Works

Instead of updating the full weight matrix $W$, LoRA decomposes the weight update into two smaller matrices:

$$W' = W + \Delta W = W + \frac{\alpha}{r} \cdot A \cdot B$$

Where:

- $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ are low-rank matrices
- $r$ is the rank (typically 8, 16, 32, or 64)
- $\alpha$ is a scaling parameter
- Only $A$ and $B$ are trained, keeping the original weights $W$ frozen

![trick](../lora_trick.png)

## Mathematical Intuition

The core insight is that weight updates during fine-tuning often have low
"intrinsic rank". Instead of learning a full $d \times k$ matrix update, we
can approximate it as the product of two much smaller matrices:

- Matrix $A$: $d \times r$ (where $r \ll d$)
- Matrix $B$: $r \times k$ (where $r \ll k$)

This reduces parameters from $d \times k$ to $r \times (d + k)$.

!!! tip "Analogs to LoRA?"
    This is similar (though not mathematically equivalent) to using
    SVD or PCA to reduce a large matrix’s dimensionality by
    projecting it into a lower-dimensional subspace.

!!! example

    For a typical attention layer with $d = k = 4096$ and rank $r = 16$:

    - **Full fine-tuning**: $4096 \times 4096 = 16.7M$ parameters
    - **LoRA**: $16 \times (4096 + 4096) = 131K$ parameters
    - **Reduction**: 99.2% fewer parameters!

## LoRA Configuration Parameters

### Rank ($r$)

The intrinsic dimensionality of the weight updates:

- **Lower values (8-16)**: Fewer parameters, faster training, less expressive
- **Higher values (32-64)**: More parameters, slower training, more expressive
- **Sweet spot**: Usually 16-32 for most tasks

### Alpha ($\alpha$)

Scaling factor controlling the magnitude of LoRA updates:

- **Typical values**: 16, 32, 64
- **Common pattern**: $\alpha = 2 \times r$
- **Higher values**: Stronger adaptation, risk of overfitting

### Target Modules

Which parts of the model to adapt:

- **Attention layers**:
  - `q_proj` = Query projection
  - `k_proj` = Key projection
  - `v_proj` = Value projection
  - `o_proj` = Output projection

- **Feed-forward layers**: `gate_proj`, `up_proj`, `down_proj`
- **Common choice**: Only attention layers for efficiency

## Comparison with Other Methods

| Method | Parameters | Memory | Speed | Quality |
|--------|------------|---------|-------|---------|
| Full Fine-tuning | 100% | High | Slow | Excellent |
| LoRA | 0.1-1% | Low | Fast | Very Good |
| Adapters | 2-4% | Medium | Medium | Good |
| Prompt Tuning | 0.01% | Very Low | Very Fast | Limited |

## When to Use LoRA

**Ideal for:**

- Limited computational resources
- Quick experimentation
- Domain adaptation
- Multiple task variants
- Rapid prototyping

**Consider alternatives when:**

- Maximum performance is critical
- Unlimited computational resources
- Completely new capabilities needed
- Large-scale architectural changes required

---

Next: Learn about the [prerequisites](prerequisites.md) for getting started with LoRA fine-tuning.
