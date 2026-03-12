# CANN API Reference & ACLNN Adaptation Guide

## Table of Contents

- [1. ACLNN Two-Phase Interface Pattern](#aclnn-two-phase-interface-pattern)
- [2. ACLNN API Index](#aclnn-api-index)
- [3. ACLNN Error Handling](#aclnn-error-handling)
- [4. MindSpore ACLNN Adaptation Flow](#mindspore-aclnn-adaptation-flow)
- [5. Common ACLNN Adaptation Issues](#common-aclnn-adaptation-issues)
- [6. View Operator Specifics](#view-operator-specifics)
- [7. ACLNN Call Chain Analysis (Composite Ops)](#aclnn-call-chain-analysis)
- [8. Diagnostic Checklist](#diagnostic-checklist)

For the complete ACLNN adaptation development reference, see [docs/mindspore/mindspore_aclnn_api_adaptation.md](../../../docs/mindspore/mindspore_aclnn_api_adaptation.md).

## ACLNN Two-Phase Interface Pattern

All aclnn operators follow a two-phase execution pattern:

**Phase 1: GetWorkspaceSize** — Computes required workspace memory and creates an executor.
```c
aclnnStatus aclnnXxxGetWorkspaceSize(/* inputs, params, outputs */, uint64_t *workspaceSize, aclOpExecutor **executor);
```

**Phase 2: Execute** — Runs the operator on the specified stream.
```c
aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

## ACLNN API Index

The full aclnn API documentation is available in the project's docs directory:
- **Index:** [docs/cann/aclnn_api_index.md](../../../docs/cann/aclnn_api_index.md) — 995 aclnn APIs
- **Individual API docs:** [docs/cann/aclnn_api_docs/](../../../docs/cann/aclnn_api_docs/) — Per-API documentation

### `aclnn_api_docs` Content Summary (Read-on-Demand)

`docs/cann/aclnn_api_docs/` stores per-API specification files (one or a few closely related APIs per document), typically covering:
- API prototype(s) and two-phase interfaces (`GetWorkspaceSize` + execute)
- Input/output constraints (shape, dtype, layout, device support, optional params)
- Attribute semantics and parameter rules (default behavior, valid ranges)
- Return status and common failure conditions

To keep diagnosis efficient, **do not read all API docs upfront**. Only when a specific API detail is needed, first look it up in [docs/cann/aclnn_api_index.md](../../../docs/cann/aclnn_api_index.md), then open the corresponding file under `aclnn_api_docs/`.

### API Categories

| Source | Category | Example APIs |
|--------|----------|-------------|
| ops-math | Basic math | aclnnAbs, aclnnAdd, aclnnSub, aclnnMul, aclnnDiv, aclnnMatmul |
| ops-nn | Neural network | aclnnSoftmax, aclnnBatchNorm, aclnnLayerNorm, aclnnDropout |
| ops-transformer | Attention/MoE | aclnnFlashAttentionScore, aclnnMoeDistributeDispatch, aclnnFFNV2 |
| ops-cv | Computer vision | aclnnRoiAlign, aclnnNms, aclnnDeformableConv2d |

## ACLNN Error Handling

When an aclnn API fails, use `aclGetRecentErrMsg()` to get detailed error information.

### Common ACLNN Failure Patterns

| Symptom | Likely Error Code | Cause | Solution |
|---------|------------------|-------|----------|
| nullptr parameter | 161001 | Uninitialized tensor passed | Check all tensor allocations |
| dtype mismatch | 161002 | Input types don't satisfy promotion rules | Cast tensors to compatible types |
| Runtime exception | 361001 | NPU runtime call failed internally | Check CANN logs for details |
| Shape inference error | 561001 | Output shape cannot be computed | Check input shapes are valid |
| Tiling error | 561002 | Kernel tiling computation failed | Check tensor dimensions, may exceed hardware limits |
| Kernel not found | 561003 | Operator binary package not installed | Install correct CANN operator packages |
| OPP path missing | 561107 | ASCEND_OPP_PATH not set | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Kernel package missing | 561112 | Operator binary kernel not loaded | Check CANN installation completeness |

## MindSpore ACLNN Adaptation Flow

### Two Adaptation Paths (Core Decision)

The critical decision in ACLNN adaptation is whether MindSpore API parameters can be passed directly to the ACLNN interface. This determines the entire development workflow.

#### Path 1: Auto-Generated (Parameters Pass Through)

**When it applies:** MindSpore API parameters are identical to ACLNN interface — same count, order, types, defaults.

**YAML signature:**
```yaml
dispatch:
  enable: True
  # No Ascend field → auto-generated
```

**What gets auto-generated:** PyBoost call code, KBK registration, Python interface wrappers.

**Key file:** `aclnn_config.yaml` must contain the operator name mapping.

**Diagnosis:** If an auto-generated operator fails, check:
- `aclnn_config.yaml` mapping exists and is correct
- YAML `op_def` field structure matches framework expectations
- Auto-generated code in build output directory

#### Path 2: Manual Customize (Parameters Need Preprocessing)

**When it applies:** Parameter conversion needed before calling ACLNN:
- `tuple[int]` → `std::vector<int64_t>`
- `Optional[Tensor]` None handling
- `str` → enum/int conversion
- Scalar parameter extraction
- Input reordering/merging
- Manual output tensor allocation

**YAML signature:**
```yaml
dispatch:
  enable: True
  Ascend: XxxAscend  # Explicit Customize class name
```

**Additional files required:** PyBoost customize `.h`/`.cc` + KBK kernel `.h`/`.cc`

**Diagnosis:** If a customize operator fails, check:
- Customize class implementation matches YAML declaration
- Parameter conversion in PyBoost/KBK is correct
- Output tensor shapes are allocated correctly

#### Path Decision: "Type Classification"

| Type | Description | Path |
|------|-------------|------|
| Type 1 | API definition identical to ACLNN | Path 1 (auto-generated) |
| Type 2 | Names differ but functionality matches | Usually Path 1 (YAML `class` field for name mapping) |
| Type 3 | Prototype/semantics don't match | Path 2 (must customize) |

### Operator Implementation Layers

Each ACLNN-adapted operator involves multiple layers, each a potential failure point:

```
YAML Definition (op_def + api_def + function_doc)
    ↓
gen_ops.py (code generation)
    ↓
GeneralInfer (C++ shape/type inference)
    ↓
PyBoost (Pynative execution)  ←→  KBK (Graph kernel execution)
    ↓                                   ↓
LAUNCH_ACLNN / RunOp          MS_ACLNN_KERNEL_FACTORY_REG
    ↓
BPROP (backward graph builder)
```

### Key File Locations (Search, Don't Hardcode)

| Component | Search Keywords |
|-----------|----------------|
| YAML definitions | `mindspore/ops/op_def/yaml/` |
| Inference/meta | `ops_func_impl`, `infer` directories |
| PyBoost/KBK | `LAUNCH_ACLNN`, `MS_ACLNN_KERNEL_FACTORY_REG` |
| BPROP | `REG_BPROP_BUILDER`, `grad_*ops.cc` |
| Tests | `tests/ut/`, `tests/st/ops/share/` |

## Common ACLNN Adaptation Issues

### gen_ops.py Errors

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Keys structure mismatch | YAML field hierarchy incorrect | Compare with working operator YAML (e.g., `add`) |
| Missing `py_method` | Python exposure field not set | Add `py_method` field in api_def |
| Missing function_doc entry | Doc YAML incomplete | Add corresponding doc node with consistent parameters |

### GeneralInfer (C++ Shape/Type Inference) Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Output shape incorrect | Shape inference logic wrong | Check against PTA/PyTorch reference implementation |
| Dynamic shape crash | No fallback for unknown dimensions | Use `kShapeDimAny`/`kShapeRankAny` for unknown dims/ranks |
| Unknown scalar value crash | `GetScalarValue` used without checking `has_value()` | Check `has_value()` before accessing; fallback for unknown |
| Array value handling | `GetArrayValue` not handling element-level unknowns | Check `IsValueUnknown(i)` per element |

**Dynamic Shape Three-Type Model:**

| Type | Meaning | InferShape Strategy |
|------|---------|-------------------|
| InputDynamic | Input shape unknown at compile time | Set corresponding output dims to -1 |
| OutputDynamic (Value Depend) | Output shape depends on input values | `GetScalarValue`/`GetArrayValue`; unknown → fallback |
| OutputDynamic (Compute Depend) | Output shape needs runtime computation | Allocate max possible + `SyncOutputShape` after run |

### PyBoost (Pynative) Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Parameter type mismatch | tuple/list not converted to `std::vector<int64_t>` | Add proper conversion in customize |
| None handling crash | Optional None not handled | Define "None semantics" and handle in PyBoost/Infer/KBK |
| String parameter error | String not converted to enum/int | Add str→enum mapping |

### KBK (Graph Kernel) Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| "Undeclared/undefined" errors | Namespace mismatch between header and implementation | Align namespaces |
| Resize/Launch confusion | Logic in wrong phase (Init vs Resize vs Launch) | Move shape-dependent logic to Resize; Init for static; Launch for execution only |
| Device memory in runtime | cudaMalloc/cudaFree in Launch | Use workspace mechanism, let framework manage device memory |
| InferShape modifies attributes | Attribute set in InferShape | Never modify attributes in InferShape (causes Pynative issues) |

### BPROP (Backward) Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Input/output count mismatch | Backward inputs ≠ forward inputs + 2 | Backward inputs = forward inputs + out + dout |
| Missing zero gradient | Non-differentiable input without zero placeholder | Use `ib->OutZeros(x)` for non-differentiable inputs |
| Dynamic value in bprop crash | C++ `if/else` on unknown scalar in graph mode | Use `ib->Conditional()` for runtime branching |
| Dynamic shape in bprop | `ib->GetShape()` on dynamic tensor | Use `DEF_PURE_SHAPE_CALC` + `ib->ShapeCalc()` |
| Dynamic rank in bprop | Rank unknown at compile time | Add `IsDynamicRank` check with dedicated dynamic path |
| Inplace bprop error | Updated self needed but overwritten | Register `CloneInplaceInput()` to preserve old value |

## View Operator Specifics

View operators perform zero-copy shape/strides transformations (e.g., `transpose`, `reshape`, `squeeze`, `narrow`, `chunk`).

### YAML Markers

| Field | Meaning |
|-------|---------|
| `view: True` | Enable PyNative View path (strides calculation) |
| `graph_view: True` | Enable KBK graph-mode View path (host kernel) |
| `labels: side_effect_mem: True` | Memory side-effect annotation |

### View vs Regular ACLNN

| Aspect | Regular ACLNN | View Operator |
|--------|--------------|---------------|
| Data movement | Yes (kernel computation) | No (zero-copy) |
| Output memory | Independently allocated | Shares input device memory |
| Core implementation | `LAUNCH_ACLNN`/`RunOp` | Strides calculation function |
| InferShape | Must hand-write | View YAML: framework handles |

### Common View Issues

- View ops in GRAPH_MODE may fall back to ACLNN if `graph_view` not set
- Strides calculation errors cause silent data corruption
- `@jit_view_unsupported` in mint = Python-level marker for view ops that may not work in JIT

## ACLNN Call Chain Analysis

When a single MindSpore API maps to **multiple ACLNN sub-operators** (composite operator):

### Diagnosis Approach

1. Check PTA C++ implementation for multiple `EXEC_NPU_CMD`/`LAUNCH_ACLNN` calls
2. Map each sub-operator to its MindSpore status (YAML + Infer + PyBoost + KBK)
3. Missing sub-operators are the most common cause of composite op failures

### Composite Op Patterns

**PyBoost (Pynative):** C++ small-op API chaining — calls wrapped C++ functions from `functions/auto_generate/functions.h`

**KBK (Graph):** Meta DSL — `REGISTER_FUNCTION_OP` + `BeginFunction/EndFunction` + `Call(Prim(OpName), ...)` constructs graph

**Key YAML setting for composite ops:**
```yaml
bprop_expander: False  # Sub-ops handle their own autodiff
```

### Composite Op Verification Strategy

| Phase | What to Verify | Method |
|-------|---------------|--------|
| Sub-operator level | Each sub-op works independently | Sub-op UT/ST |
| Composite - intermediate | Intermediate tensors match PTA | Dump intermediate tensors, compare |
| Composite - final output | Final output matches PTA | Standard ST alignment |
| Backward | Gradient correctness | Backward ST + numerical gradient check |

## Diagnostic Checklist

When encountering an ACLNN-related error in MindSpore on Ascend:

### Quick Triage

1. **Is it an ACLNN error code?** Check error-codes.md for 161xxx/361xxx/561xxx
2. **Is it a CANN runtime error?** Check for 107xxx/207xxx/507xxx codes
3. **Is it a gen_ops.py build error?** Look for YAML structure issues
4. **Is it a shape inference error?** Check GeneralInfer dynamic handling
5. **Is it a PyBoost/KBK runtime error?** Check parameter conversion and ACLNN call
6. **Is it a backward error?** Check BPROP input count, dynamic handling, zero gradients
7. **Is it a View op issue?** Check YAML `view`/`graph_view` flags and strides calc

### Collecting Evidence

```bash
# CANN version
cat /usr/local/Ascend/ascend-toolkit/version

# CANN device logs (operator-level errors)
tail -f /var/log/npu/slog/*/device-*/plog/*.log

# Recent error message (in code)
# C: aclGetRecentErrMsg()
# Python: via MindSpore exception propagation

# Check operator binary packages
ls /usr/local/Ascend/ascend-toolkit/latest/opp/

# Search for operator YAML in MindSpore source (replace <op_name> with actual name)
grep -r "<op_name>" mindspore/ops/op_def/yaml/ --include="*.yaml"

# Search for operator ACLNN binding (replace <OpName> with actual name)
grep -r "LAUNCH_ACLNN.*aclnn<OpName>" mindspore/
grep -r "MS_ACLNN_KERNEL_FACTORY_REG.*<OpName>" mindspore/
```

### PTA Source Audit (For Alignment Issues)

When MindSpore operator behavior differs from PTA/PyTorch, check three key PTA files:

| File | Path Pattern | Extract |
|------|-------------|---------|
| Function signature YAML | `op_plugin/config/op_plugin_functions.yaml` | Parameter names, types, defaults |
| Backward registration | `op_plugin/config/derivatives.yaml` | Differentiable inputs, grad function |
| C++ implementation | `op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp` | Actual aclnn calls, parameter preprocessing, hidden defaults |

Common PTA-vs-MindSpore discrepancies:
- Forward/backward parameter name differences
- Hidden hardcoded parameters in backward (e.g., `deterministic=true`)
- Optional None handling differences (empty tensor vs null)
- Output tensor count/shape differences

## See Also

- [Error Codes](error-codes.md) — ACLNN error code table (161xxx/361xxx/561xxx) and full CANN/CUDA code mappings
- [Backend Diagnosis](backend-diagnosis.md) — Step-by-step Ascend/GPU/CPU diagnosis and further location techniques
- [MindSpore API](mindspore-api.md) — API layers (mint/ops/nn), execution modes, backend registration
- [Failure Showcase](failure-showcase.md) — Historical failures indexed by error keywords
- Only read the full adaptation development guide [docs/mindspore/mindspore_aclnn_api_adaptation.md](../../../docs/mindspore/mindspore_aclnn_api_adaptation.md) when helping users write or review new operator adaptations — not needed for routine error diagnosis
