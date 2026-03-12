# MindSpore API Reference

## API Layer Hierarchy

MindSpore provides multiple API layers. `mindspore.mint` is the recommended high-level interface; `mindspore.ops` is the lower-level operator layer; `mindspore.nn` provides traditional network modules.

```
mindspore.mint          ← Recommended, PyTorch-compatible, highest priority
  ├── mint.nn            ← NN layers (Cell subclasses)
  ├── mint.nn.functional ← Functional NN operations
  ├── mint.optim         ← Optimizers (AdamW, Adam, SGD, FusedAdamW)
  ├── mint.linalg        ← Linear algebra (inv, norm, qr)
  ├── mint.special        ← Special math functions
  └── mint.distributed   ← Collective communication

mindspore.ops           ← Lower-level operators
  ├── ops.function.*     ← Functional API wrappers
  └── ops.operations.*   ← Primitive classes (P.Add, P.MatMul, etc.)

mindspore.nn            ← Traditional NN modules
  └── nn.Cell subclasses ← Dense, Conv2d, BatchNorm2d, etc.
```

## mindspore.mint (Primary API)

### Overview

`mint` is MindSpore's curated, PyTorch-compatible high-level interface. It wraps `ops.*_ext` (newer operator implementations) and adds view semantics, NN layers, optimizers, and distributed APIs. Users migrating from PyTorch should prefer `mint` for the most familiar experience.

```python
from mindspore import mint

result = mint.add(x, y)
result = mint.matmul(x, y)
result = mint.softmax(x, dim=-1)
```

### mint Top-Level Functions

**Tensor Creation:**
`zeros`, `ones`, `full`, `empty`, `empty_like`, `zeros_like`, `ones_like`, `full_like`, `eye`, `arange`, `rand`, `rand_like`, `randn`, `randn_like`, `randint`, `randint_like`, `randperm`

**Arithmetic:**
`add`, `sub`, `mul`, `div`, `divide`, `pow`, `fmod`, `remainder`, `addcmul`, `addcdiv`, `reciprocal`, `negative`, `neg`, `square`, `abs`, `ceil`, `floor`, `round`, `clamp`, `frac`, `sign`

**Math / Trigonometric:**
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `arcsin`, `arccos`, `arctan`, `arctan2`, `arcsinh`, `arccosh`, `arctanh`, `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `rsqrt`, `erf`, `erfinv`, `erfc`, `expm1`, `xlogy`

**Reduction:**
`sum`, `mean`, `prod`, `max`, `min`, `amax`, `amin`, `all`, `any`, `cumsum`, `cumprod`, `cummax`, `cummin`, `var`, `std`, `std_mean`, `var_mean`, `nansum`, `count_nonzero`, `logsumexp`

**Matrix / Linear Algebra:**
`matmul`, `mm`, `bmm`, `mv`, `dot`, `baddbmm`, `addbmm`, `addmm`, `addmv`, `einsum`, `outer`, `cross`

**Tensor Manipulation:**
`cat`, `concat`, `stack`, `split`, `chunk`, `squeeze`, `unsqueeze`, `flatten`, `reshape`, `permute`, `transpose`, `swapaxes`, `tile`, `repeat_interleave`, `flip`, `roll`, `narrow`, `index_select`, `masked_select`, `select`, `gather`, `scatter_add`, `broadcast_to`, `unbind`, `dstack`, `hstack`, `ravel`, `t`, `diff`

**Comparison / Logic:**
`eq`, `ne`, `not_equal`, `gt`, `greater`, `greater_equal`, `lt`, `less`, `less_equal`, `le`, `isclose`, `allclose`, `isfinite`, `isinf`, `isnan`, `isneginf`, `logical_and`, `logical_or`, `logical_not`, `logical_xor`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`

**Sorting / Searching:**
`sort`, `argsort`, `topk`, `argmax`, `argmin`, `searchsorted`, `nonzero`, `where`, `kthvalue`, `unique_consecutive`

**Other:**
`clone`, `equal`, `item`, `softmax`, `norm`, `bernoulli`, `normal`, `multinomial`, `diag`, `trace`, `tril`, `triu`, `polar`, `meshgrid`, `bincount`, `histc`, `take`, `nan_to_num`, `float_power`, `logaddexp`, `logaddexp2`, `triangular_solve`, `index_add`, `real`, `imag`, `gcd`, `median`, `lerp`, `inverse`, `fix`, `scatter`, `cdist`

### mint.nn (Neural Network Layers)

All layers are `nn.Cell` subclasses compatible with MindSpore's graph compilation.

**Convolution:**
`Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose2d`

**Normalization:**
`BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `LayerNorm`, `GroupNorm`, `SyncBatchNorm`

**Activation:**
`ReLU`, `ReLU6`, `GELU`, `SELU`, `SiLU`, `Sigmoid`, `LogSigmoid`, `ELU`, `Tanh`, `Hardtanh`, `PReLU`, `Mish`, `Threshold`, `GLU`, `Softmax`, `LogSoftmax`

**Pooling:**
`MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d`, `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `MaxUnpool2d`

**Padding:**
`ConstantPad1d/2d/3d`, `ZeroPad1d/2d/3d`, `ReflectionPad1d/2d/3d`, `ReplicationPad1d/2d/3d`

**Loss:**
`L1Loss`, `MSELoss`, `SmoothL1Loss`, `CrossEntropyLoss`, `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `KLDivLoss`, `SoftMarginLoss`, `CosineEmbeddingLoss`

**Other:**
`Identity`, `Flatten`, `Unfold`, `Fold`, `Upsample`, `Dropout`, `Dropout2d`, `UpsamplingNearest2d`, `UpsamplingBilinear2d`, `PixelShuffle`

### mint.nn.functional

Functional versions of all `mint.nn` layers:

**Conv:** `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d`
**Pooling:** `max_pool2d`, `avg_pool1d/2d/3d`, `adaptive_avg_pool2d/3d`, `adaptive_max_pool1d/2d`, `max_unpool2d`
**Activation:** `relu`, `relu_`, `gelu`, `silu`, `elu`, `elu_`, `sigmoid`, `sigmoid_`, `tanh`, `hardtanh`, `hardtanh_`, `relu6`, `mish`, `prelu`, `leaky_relu`, `softshrink`, `hardshrink`, `hardsigmoid`, `hardswish`, `glu`, `logsigmoid`, `softmax`, `log_softmax`, `threshold`, `threshold_`
**Normalization:** `layer_norm`, `batch_norm`, `group_norm`, `normalize`
**Dropout:** `dropout`, `dropout2d`
**Loss:** `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `cross_entropy`, `nll_loss`, `mse_loss`, `smooth_l1_loss`, `soft_margin_loss`, `kl_div`, `cosine_embedding_loss`, `l1_loss`
**Other:** `pad`, `interpolate`, `upsample`, `unfold`, `fold`, `flatten`, `pixel_shuffle`, `grid_sample`, `embedding`, `one_hot`, `linear`

### mint.optim (Optimizers)

| Class | Description |
|-------|-------------|
| `AdamW` | AdamW optimizer with weight decay |
| `Adam` | Adam optimizer |
| `SGD` | Stochastic Gradient Descent |
| `FusedAdamW` | Fused (performance-optimized) AdamW |

### mint.linalg

`inv` — Matrix inverse
`norm` — Matrix/vector norm
`vector_norm` — Vector norm
`matrix_norm` — Matrix norm
`qr` — QR decomposition

### mint.special

`erfc`, `expm1`, `exp2`, `round`, `sinc`, `log1p`, `log_softmax`

### mint.distributed

`init_process_group`, `all_reduce`, `barrier`, and other collective communication APIs.
Uses `TCPStore` for process coordination.

## mint-Specific Behaviors and Pitfalls

These behaviors distinguish `mint` from `ops` and are common sources of errors:

### 1. View vs Copy Semantics

Several mint functions use **view semantics** (return a view of the original tensor rather than a copy). These are decorated with `@jit_view_unsupported`:

- `squeeze`, `unsqueeze`, `flatten`, `reshape`, `t`, `narrow`, `split`, `broadcast_to`, `permute`, `transpose`

**Pitfall:** View ops may not work in `GRAPH_MODE` / JIT compilation. If you encounter errors with these ops under JIT, switch to their non-view equivalents from `ops` or use `PYNATIVE_MODE`.

### 2. Return Type Differences

| Function | `mint` behavior | `ops` behavior |
|----------|----------------|----------------|
| `equal()` | Returns Python `bool` | Returns Tensor |
| `item()` | Returns Python scalar (validates single-element) | N/A |
| `allclose()` | Returns Python `bool` | Returns Tensor |

### 3. Experimental APIs

Many mint APIs are marked "experimental" and subject to change or deletion across MindSpore versions. When an API disappears or changes signature after a version upgrade, check MindSpore release notes.

### 4. `_ext` Operator Variants

mint wraps `ops.*_ext` variants (e.g., `mean_ext`, `sum_ext`, `topk_ext`, `softmax_ext`), which are newer, optimized implementations. These may have slightly different behavior or constraints compared to the original `ops.*` versions.

## mindspore.ops (Lower-Level API)

### Functional API (`ops.function`)

```python
import mindspore.ops as ops
result = ops.add(x, y)
result = ops.relu(x)
```

Modules: `array_func`, `math_func`, `nn_func`, `linalg_func`, `sparse_func`, etc.

### Primitive API (`ops.operations`)

```python
from mindspore.ops import operations as P
add_op = P.Add()
result = add_op(x, y)
```

Modules: `nn_ops`, `math_ops`, `array_ops`, `image_ops`, `linalg_ops`, etc.

## mindspore.nn (Traditional NN Modules)

```python
import mindspore.nn as nn
layer = nn.Dense(in_channels=256, out_channels=10)
layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
layer = nn.BatchNorm2d(num_features=64)
```

## Backend Registration

### Operator Registration Classes

| Class | Backend | Processor | Description |
|-------|---------|-----------|-------------|
| `TBERegOp` | Ascend | AiCore | TBE (Tensor Boost Engine) operators |
| `AiCPURegOp` | Ascend | AiCPU | Ascend CPU operators |
| `AkgAscendRegOp` | Ascend | AiCore | AKG auto-generated operators |
| `AkgGpuRegOp` | GPU | CUDA | AKG CUDA operators |
| `AkgCpuRegOp` | CPU | CPU | AKG CPU operators |
| `CpuRegOp` | CPU | CPU | Native CPU operators |
| `CustomRegOp` | All | Configurable | Custom operators (`.target("Ascend"/"GPU"/"CPU")`) |

### Checking Operator Backend Support

Operator docstrings declare supported platforms:
```
Supported Platforms: ``Ascend`` ``GPU`` ``CPU``
```

If called on an unsupported backend, MindSpore raises `RuntimeError` or falls back.

## Execution Modes

### GRAPH_MODE (Static Graph, mode=0)

- Compiles entire `construct()` into optimized graph before execution
- Better performance, more restrictions on Python syntax
- mint view ops (`@jit_view_unsupported`) may not work
- Errors may appear during graph compilation (type inference, shape inference)

### PYNATIVE_MODE (Dynamic Graph, mode=1)

- Executes operations eagerly, line by line
- Full Python syntax support, easier debugging
- All mint APIs work correctly
- Errors appear at the exact line of execution

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)   # Static graph
ms.set_context(mode=ms.PYNATIVE_MODE) # Dynamic graph (default)
```

## Context and Device Management

### Setting Device Target

```python
import mindspore as ms
ms.set_context(device_target="Ascend")  # or "GPU" or "CPU"
ms.set_context(device_id=0)
```

### Backend-Specific Configuration

**Ascend:**
```python
ms.set_context(ascend_config={
    "precision_mode": "allow_fp32_to_fp16",
    "jit_compile": True,
    "op_debug_option": "oom"
})
```

**GPU:**
```python
ms.set_context(gpu_config={
    "conv_fprop_algo": "normal",
    "conv_allow_tf32": True,
    "matmul_allow_tf32": True
})
```

## Common Differences Across Backends

| Aspect | Ascend | GPU | CPU |
|--------|--------|-----|-----|
| Precision | FP16 preferred, FP32 via config | FP32 default, TF32 optional | FP32 default |
| Execution | Async (task-based) | Async (CUDA stream) | Sync |
| Operator set | TBE + AKG + AiCPU | CUDA kernels + AKG | Native CPU kernels |
| Graph mode | Full graph offload (SuperKernel) | Kernel-by-kernel | Kernel-by-kernel |
| Distributed | HCCL | NCCL | MPI/Gloo |
| Memory | HBM (device) | VRAM (device) | RAM (host) |

## Numerical Accuracy Considerations

- Ascend may use FP16 internally even when FP32 is specified (controlled by `precision_mode`)
- GPU TF32 mode reduces precision for matmul/conv (controlled by `matmul_allow_tf32`)
- Cross-backend numerical differences are expected within tolerance
- Recommended tolerances for cross-backend comparison:

```python
atol, rtol = 1e-3, 1e-3   # Ascend FP16
atol, rtol = 1e-4, 1e-4   # Ascend FP32
atol, rtol = 1e-5, 1e-5   # GPU FP32
atol, rtol = 1e-7, 1e-7   # CPU FP32
```

## Operator Debugging

### 1. Check operator availability

```python
import mindspore as ms
from mindspore import mint

ms.set_context(device_target="Ascend")
try:
    result = mint.abs(ms.Tensor([1.0, -2.0]))
    print("Supported!")
except Exception as e:
    print(f"Error: {e}")
```

### 2. Enable verbose logging

```bash
export GLOG_v=1                    # MindSpore INFO level
export ASCEND_GLOBAL_LOG_LEVEL=1   # CANN INFO level
```

### 3. Enable synchronous execution (Ascend debugging)

```bash
export MS_DEV_FORCE_ACL=1          # Force ACL execution
```

### 4. Graph dump for debugging

```python
ms.set_context(save_graphs=True, save_graphs_path="./graphs")
```

## Source Code Search Guide

Directory structures may change across MindSpore versions. Use keyword search instead of hardcoded paths:

| Component | Search Keywords / Patterns |
|-----------|--------------------------|
| mint top-level functions | `mindspore/mint/__init__.py` or search `def <func_name>` under `mint/` |
| mint.nn layers | search `class <LayerName>` under `mint/nn/layer/` |
| mint.nn.functional | search `def <func_name>` in `mint/nn/functional.py` |
| mint.optim | search `class <OptimizerName>` under `mint/optim/` |
| mint.linalg / special / distributed | search under `mint/linalg/`, `mint/special/`, `mint/distributed/` |
| ops functional API | search `def <func_name>` under `ops/function/` |
| ops Primitive classes | search `class <OpName>` under `ops/operations/` |
| Auto-generated ops | search under `ops/auto_generate/` |
| Backend registration | search `op_info_register` or `RegOp` |
| nn traditional modules | search `class <LayerName>` under `nn/layer/` |
| Context management | search `set_context` or `get_context` in `context.py` |
| YAML op definitions | search `<op_name>` under `ops/op_def/yaml/` |
| ACLNN bindings | search `LAUNCH_ACLNN` or `MS_ACLNN_KERNEL_FACTORY_REG` |

## See Also

- [Error Codes](error-codes.md) — Error code mappings for MindSpore exceptions, CANN, ACLNN, CUDA
- [CANN API Reference](cann-api-reference.md) — ACLNN two-phase interface, adaptation flow, operator diagnostics
- [Backend Diagnosis](backend-diagnosis.md) — Per-backend diagnosis steps (Ascend/GPU/CPU)
- [Failure Showcase](failure-showcase.md) — Historical failures and solutions
