# Failure Showcase

Historical MindSpore failures and their solutions. Format:
```yaml
- failure_info: "[error keywords/context]"
  observed_at: "[file:function or test location where observed]"
  backend: "ascend|gpu|cpu|all"
  failure_type: "platform|scripts|framework|backend"
  root_cause: "[specific cause]"
  solution: "[actionable steps]"
  last_seen: "[timestamp]"
  occurrences: [count]
```

## Common Failure Patterns

### Out of Memory (Ascend HBM)
- failure_info: "EL0004, 200000, 207018, RuntimeError, out of memory, device memory exhausted"
- observed_at: ""
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Ascend HBM memory exhausted due to large tensors, batch size, or graph memory fragmentation"
- solution: "Reduce batch size; use gradient checkpointing (ms.nn.CellRecompute); call ms.hal.memory.empty_cache(); set max_device_memory in context"
- last_seen: "2026-03-12"
- occurrences: 1

### Out of Memory (GPU VRAM)
- failure_info: "CUDA out of memory, RuntimeError, CUDA error"
- observed_at: ""
- backend: "gpu"
- failure_type: "platform"
- root_cause: "GPU VRAM exhausted due to large model or batch size"
- solution: "Reduce batch size; use gradient checkpointing; set CUDA_VISIBLE_DEVICES to correct GPU; check nvidia-smi for other processes using VRAM"
- last_seen: "2026-03-12"
- occurrences: 1

### Missing CANN Environment
- failure_info: "libascendcl.so not found, ImportError, cannot find CANN, ASCEND_OPP_PATH"
- observed_at: ""
- backend: "ascend"
- failure_type: "scripts"
- root_cause: "CANN environment variables not set or CANN not installed"
- solution: "Source CANN environment: source /usr/local/Ascend/ascend-toolkit/set_env.sh; verify CANN installation with cat /usr/local/Ascend/ascend-toolkit/version"
- last_seen: "2026-03-12"
- occurrences: 1

### Device Target Mismatch
- failure_info: "RuntimeError, device_target, should be one of, Ascend GPU CPU, invalid device"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "set_context device_target does not match available hardware or is misspelled"
- solution: "Check available hardware; use correct device_target string ('Ascend', 'GPU', 'CPU'); verify device availability with npu-smi info or nvidia-smi"
- last_seen: "2026-03-12"
- occurrences: 1

### Graph Compilation Error (Static Graph)
- failure_info: "RuntimeError, graph compile, type inference, abstract type, infer failed"
- observed_at: ""
- backend: "all"
- failure_type: "framework"
- root_cause: "MindSpore GRAPH_MODE cannot compile construct() due to unsupported Python syntax, dynamic control flow, or type inference failure"
- solution: "Switch to PYNATIVE_MODE for debugging; remove dynamic control flow from construct(); use @ms.jit_class for custom classes; check Supported Platforms for operators"
- last_seen: "2026-03-12"
- occurrences: 1

### Operator Not Supported on Backend
- failure_info: "RuntimeError, not supported, operator, backend, fallback"
- observed_at: ""
- backend: "all"
- failure_type: "framework"
- root_cause: "Operator/feature not implemented for target backend in current MindSpore version"
- solution: "Check operator's Supported Platforms; upgrade MindSpore/CANN version; use alternative operator; or use CPU fallback for specific ops"
- last_seen: "2026-03-12"
- occurrences: 1

### Context Empty (Ascend)
- failure_info: "107002, context is empty, aclrtSetContext, aclrtSetDevice, RuntimeError"
- observed_at: ""
- backend: "ascend"
- failure_type: "framework"
- root_cause: "NPU context not initialized before calling NPU operations"
- solution: "Ensure mindspore.set_context(device_target='Ascend', device_id=N) is called before any tensor operations; check initialization sequence"
- last_seen: "2026-03-12"
- occurrences: 1

### Device Heartbeat Lost (Ascend)
- failure_info: "507010, lost heartbeat, RuntimeError, task scheduler, device hang"
- observed_at: ""
- backend: "ascend"
- failure_type: "platform"
- root_cause: "NPU device task scheduler heartbeat lost, device may be unresponsive"
- solution: "Check device health with npu-smi info; device may need reset; check for thermal throttling or hardware failure"
- last_seen: "2026-03-12"
- occurrences: 1

### HBM ECC Error (Ascend)
- failure_info: "507054, HBM ECC, multi-bit, hardware error, memory fault"
- observed_at: ""
- backend: "ascend"
- failure_type: "platform"
- root_cause: "HBM memory hardware ECC fault on Ascend device"
- solution: "Hardware fault — contact hardware support; try different device; check npu-smi info for device health status"
- last_seen: "2026-03-12"
- occurrences: 1

### HCCL Communication Error (Ascend Distributed)
- failure_info: "HCCL, timeout, EI0002, EI0006, notify wait, socket build, distributed"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "HCCL distributed communication failure — network timeout, rank configuration error, or HCCS link issue"
- solution: "Check network connectivity between devices; verify ranktable configuration; check HCCL_WHITELIST_DISABLE setting; ensure all ranks start correctly"
- last_seen: "2026-03-12"
- occurrences: 1

### NCCL Communication Error (GPU Distributed)
- failure_info: "NCCL error, timeout, unhandled system error, distributed, GPU"
- observed_at: ""
- backend: "gpu"
- failure_type: "backend"
- root_cause: "NCCL distributed communication failure — network issue, GPU topology problem, or version mismatch"
- solution: "Check GPU connectivity with nvidia-smi topo; set NCCL_DEBUG=INFO for detailed logs; verify NCCL version compatibility; check firewall settings"
- last_seen: "2026-03-12"
- occurrences: 1

### TBE Operator Compilation Error (Ascend)
- failure_info: "TBE, compile failed, E9xxxx, EBxxxx, operator compilation, UB overflow"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "TBE operator compilation failed due to unsupported shape/dtype, UB memory overflow, or CANN version incompatibility"
- solution: "Check input shapes/dtypes match TBE operator constraints; check CANN version compatibility; try jit_compile=False in ascend_config; check CANN logs for detailed compilation error"
- last_seen: "2026-03-12"
- occurrences: 1

### Shape Inference Failure
- failure_info: "ValueError, shape, infer, expected, dimensions, rank mismatch"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "Input tensor shape does not match operator requirements (wrong dimensions, rank mismatch)"
- solution: "Check input tensor shapes; verify reshape/transpose operations; use print(tensor.shape) to debug; check operator documentation for expected input shapes"
- last_seen: "2026-03-12"
- occurrences: 1

### AI Core Execution Timeout (Ascend)
- failure_info: "507014, AI Core timeout, AICORE_TIMEOUT, execution timeout"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "AI Core operator execution timed out, possibly due to infinite loop in operator or hardware issue"
- solution: "Check CANN logs for specific operator; try reducing input size; check if operator has known timeout issues in current CANN version; contact support if persistent"
- last_seen: "2026-03-12"
- occurrences: 1

### Dtype Mismatch
- failure_info: "TypeError, dtype, expected, Float32, Float16, type not match, cast"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "Tensor data types don't match operator requirements or between operands"
- solution: "Use ops.Cast() to convert types; check precision_mode in ascend_config for auto-casting behavior; verify model parameter dtypes match input dtypes"
- last_seen: "2026-03-12"
- occurrences: 1

### mint View Op in GRAPH_MODE / JIT
- failure_info: "RuntimeError, jit_view_unsupported, view, squeeze, flatten, reshape, GRAPH_MODE, graph compile"
- observed_at: ""
- backend: "all"
- failure_type: "framework"
- root_cause: "mint view operations (squeeze, unsqueeze, flatten, reshape, t, narrow, split, broadcast_to, permute, transpose) are decorated with @jit_view_unsupported and may fail or produce incorrect results in GRAPH_MODE / JIT compilation"
- solution: "Switch to PYNATIVE_MODE for debugging; use ops.* equivalents instead of mint view ops in GRAPH_MODE; or use .copy() after view ops to materialize the tensor"
- last_seen: "2026-03-12"
- occurrences: 1

### mint.equal() Return Type Confusion
- failure_info: "TypeError, AttributeError, mint.equal, bool, Tensor expected, item"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.equal() returns a Python bool (not a Tensor), unlike ops.equal() which returns a Tensor; code expecting a Tensor from equal() breaks"
- solution: "Use ops.equal() if you need a Tensor result; use mint.equal() only when a scalar bool comparison is intended; check return type docs"
- last_seen: "2026-03-12"
- occurrences: 1

### mint.item() on Multi-Element Tensor
- failure_info: "RuntimeError, cannot be converted to Scalar, mint.item, elements"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.item() requires a single-element Tensor; calling it on a Tensor with multiple elements raises RuntimeError"
- solution: "Ensure the Tensor has exactly 1 element before calling item(); use indexing (tensor[0]) or reduction (tensor.sum()) to get a single element first"
- last_seen: "2026-03-12"
- occurrences: 1

### mint Experimental API Removed After Upgrade
- failure_info: "AttributeError, ImportError, module mint has no attribute, API removed, experimental"
- observed_at: ""
- backend: "all"
- failure_type: "framework"
- root_cause: "Many mint APIs are marked 'experimental' and may be removed or renamed across MindSpore versions; code using these APIs breaks after version upgrade"
- solution: "Check MindSpore release notes for API changes; use stable ops.* equivalents as fallback; pin MindSpore version if API stability is required"
- last_seen: "2026-03-12"
- occurrences: 1

### mint.nn Layer Parameter Validation
- failure_info: "ValueError, TypeError, in_channels, out_channels, groups, Validator, mint.nn.Conv, mint.nn.BatchNorm"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.nn layers (Conv, BatchNorm, etc.) use strict parameter validation via Validator; invalid params like in_channels not divisible by groups raise ValueError"
- solution: "Check parameter constraints in mint.nn layer documentation; ensure in_channels % groups == 0 and out_channels % groups == 0 for Conv layers"
- last_seen: "2026-03-12"
- occurrences: 1

### mint.distributed init_process_group Failure
- failure_info: "RuntimeError, init_process_group, TCPStore, connection refused, distributed, mint.distributed"
- observed_at: ""
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.distributed.init_process_group failed due to TCPStore connection issues — master address/port unreachable, firewall blocking, or incorrect rank configuration"
- solution: "Verify MASTER_ADDR and MASTER_PORT environment variables; check firewall rules; ensure all ranks use consistent world_size; check if master process started before workers"
- last_seen: "2026-03-12"
- occurrences: 1

### mint.optim FusedAdamW Not Available
- failure_info: "RuntimeError, NotImplementedError, FusedAdamW, not supported, backend"
- observed_at: ""
- backend: "cpu"
- failure_type: "framework"
- root_cause: "mint.optim.FusedAdamW is a performance-optimized optimizer that may not be available on all backends (especially CPU)"
- solution: "Use mint.optim.AdamW as fallback on unsupported backends; FusedAdamW is primarily optimized for Ascend/GPU"
- last_seen: "2026-03-12"
- occurrences: 1

### gen_ops.py YAML Build Error
- failure_info: "gen_ops.py, KeyError, YAML, keys structure, py_method missing, function_doc"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "ACLNN operator YAML definition has incorrect field hierarchy, missing py_method, or missing function_doc entry; gen_ops.py fails during code generation"
- solution: "Compare YAML structure with a working operator (e.g., add); ensure op_def + api_def + function_doc are consistent; check for Chinese characters in English YAML files (encoding issues)"
- last_seen: "2026-03-12"
- occurrences: 1

### GeneralInfer Dynamic Shape Crash
- failure_info: "RuntimeError, InferShape, dynamic shape, unknown, kShapeDimAny, GetScalarValue, has_value"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "C++ GeneralInfer does not handle dynamic shape/rank fallback — GetScalarValue used without checking has_value(), or no fallback for unknown dimensions"
- solution: "Check has_value() before accessing scalar values; return kShapeDimAny for unknown dims; return kShapeRankAny for dynamic rank; use GetArrayValue with per-element IsValueUnknown check"
- last_seen: "2026-03-12"
- occurrences: 1

### PyBoost Parameter Conversion Failure
- failure_info: "RuntimeError, TypeError, PyBoost, tuple, vector, Optional, None, LAUNCH_ACLNN, parameter"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "PyBoost customize fails to convert MindSpore parameters to ACLNN-expected types: tuple[int] not converted to std::vector<int64_t>, Optional None not handled, or string not converted to enum"
- solution: "Add proper parameter conversion in PyBoost customize: tuple→vector, None→empty tensor or default, str→enum mapping; compare with PTA source for exact conversion logic"
- last_seen: "2026-03-12"
- occurrences: 1

### KBK Kernel Registration / Namespace Error
- failure_info: "RuntimeError, undeclared, undefined, MS_ACLNN_KERNEL_FACTORY_REG, namespace, KBK, Graph kernel"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "KBK kernel has namespace mismatch between header and implementation, or registration macro uses wrong class name"
- solution: "Align namespace in header and .cc files; verify MS_ACLNN_KERNEL_FACTORY_REG class name matches; check forward/backward are in separate files with separate registration"
- last_seen: "2026-03-12"
- occurrences: 1

### BPROP Input/Output Count Mismatch
- failure_info: "RuntimeError, bprop, gradient, input count, output count, REG_BPROP_BUILDER, backward"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "BPROP builder has wrong input/output count — backward inputs should equal forward inputs + 2 (out + dout), backward outputs should equal forward input count"
- solution: "Backward inputs = forward inputs + out + dout; each forward input gets one gradient output; non-differentiable inputs return ib->OutZeros(x); multi-output forward: use TupleGetItem for out"
- last_seen: "2026-03-12"
- occurrences: 1

### BPROP Dynamic Value in Graph Mode
- failure_info: "RuntimeError, bprop, Conditional, ShapeCalc, dynamic, ValueAny, graph mode, backward compile"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "BPROP builder uses C++ if/else on scalar values that may be unknown at compile time in graph mode; needs runtime branching via Conditional/ShapeCalc"
- solution: "Check BuildValue()->ContainsValueAny() for each scalar input in bprop; use ib->Conditional() for unknown scalars instead of C++ if; use DEF_PURE_SHAPE_CALC + ib->ShapeCalc() for dynamic shape dependencies; use IsDynamicRank check with dedicated dynamic path function"
- last_seen: "2026-03-12"
- occurrences: 1

### ACLNN Composite Op Missing Sub-Operator
- failure_info: "RuntimeError, aclnn, composite, sub-operator, missing, LAUNCH_ACLNN, call chain"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Composite operator (maps to multiple ACLNN sub-operators) fails because one or more sub-operators are not yet adapted in MindSpore — missing YAML/Infer/PyBoost/KBK for a sub-op"
- solution: "Analyze PTA C++ implementation for all EXEC_NPU_CMD/LAUNCH_ACLNN calls; check each sub-operator status in MindSpore (YAML + Infer + PyBoost + KBK); implement missing sub-ops first (leaf before composite)"
- last_seen: "2026-03-12"
- occurrences: 1

### View Op Strides Calculation Error
- failure_info: "RuntimeError, view, strides, transpose, reshape, zero-copy, silent corruption, incorrect output"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "View operator's strides calculation function produces incorrect output shape/strides/offset; causes silent data corruption or incorrect results"
- solution: "Verify strides calculation against PyTorch TensorShape.cpp reference; check YAML has correct view: True / graph_view: True flags; verify REG_VIEW_STRIDES_CALC_FUN registration; test with non-contiguous inputs"
- last_seen: "2026-03-12"
- occurrences: 1

### PTA-MindSpore Parameter Mismatch
- failure_info: "RuntimeError, ValueError, parameter mismatch, PTA, forward backward, different names, hidden parameter"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "MindSpore ACLNN adaptation has parameter alignment issues with PTA: forward/backward parameter name differences, hidden hardcoded parameters in backward, Optional None handling divergence, or output tensor count mismatch"
- solution: "Audit PTA source: check op_plugin_functions.yaml for exact signatures, derivatives.yaml for backward registration, C++ implementation for hidden defaults and None handling; record all discrepancies and align MindSpore implementation"
- last_seen: "2026-03-12"
- occurrences: 1

### aclnn_config.yaml Missing Mapping (Auto-Gen Path)
- failure_info: "RuntimeError, aclnn_config, mapping, auto-generated, dispatch, operator not found"
- observed_at: ""
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Auto-generated ACLNN operator (Path 1) fails because aclnn_config.yaml is missing the operator name mapping entry"
- solution: "Add operator mapping to aclnn_config.yaml; verify YAML op_def has dispatch.enable: True without explicit Ascend field; rebuild to trigger auto-generation"
- last_seen: "2026-03-12"
- occurrences: 1

## Searchable Keywords

- Memory: OOM, EL0004, 200000, 207018, memory exhausted, CUDA out of memory, bad_alloc
- Hardware: 507010, 507053, 507054, 507056, heartbeat, ECC, link error, device hang
- Distributed: HCCL, NCCL, timeout, all_reduce, broadcast, EI0002, EI0006, mint.distributed, init_process_group, TCPStore
- Framework: context empty, 107002, graph compile, type inference, infer failed
- Operator: TBE, AKG, compile failed, not supported, fallback, E9xxxx, EBxxxx
- Environment: CANN, libascendcl.so, ASCEND_OPP_PATH, CUDA, nvidia-smi
- Mode: GRAPH_MODE, PYNATIVE_MODE, construct, ms_function, jit
- Config: set_context, device_target, precision_mode, jit_compile
- Dtype: Float16, Float32, BFloat16, cast, dtype mismatch
- Shape: shape mismatch, rank, dimensions, reshape, unsqueeze
- Mint: mint, mint.nn, mint.nn.functional, mint.optim, mint.linalg, mint.distributed, mint.special
- Mint view: jit_view_unsupported, view, squeeze, flatten, reshape, t, narrow, split, broadcast_to
- Mint API: mint.equal, mint.item, experimental, FusedAdamW, Validator
- ACLNN adaptation: gen_ops.py, YAML, op_def, api_def, function_doc, aclnn_config
- ACLNN inference: GeneralInfer, InferShape, InferType, dynamic shape, kShapeDimAny, kShapeRankAny
- ACLNN execution: PyBoost, LAUNCH_ACLNN, KBK, MS_ACLNN_KERNEL_FACTORY_REG, RunOp, customize
- ACLNN backward: BPROP, REG_BPROP_BUILDER, Conditional, ShapeCalc, OutZeros, backward
- ACLNN composite: call chain, sub-operator, bprop_expander, Meta DSL, REGISTER_FUNCTION_OP
- ACLNN view: strides, view, graph_view, REG_VIEW_STRIDES_CALC_FUN, host kernel, zero-copy
- PTA alignment: PTA, parameter mismatch, derivatives.yaml, op_plugin_functions.yaml, hidden parameter

## Adding New Failures

When analyzing a new failure:

1. Extract key information: error code, keywords, context, backend
2. Identify failure type (platform/scripts/framework/backend)
3. Determine root cause through orientation analysis
4. Document solution (verified or proposed)
5. Update this file with YAML format above
6. Include approximate timestamp and occurrence count

### failure_info Guidelines

**Include:** error codes, error keywords, operators, operation patterns, exception types
**Exclude:** test/function names, file names, variable names, specific parameter values

**Examples:**
- OK: "507010, lost heartbeat, RuntimeError, device hang"
- BAD: "test_matmul_npu_fp16_case3 failed"
- OK: "HCCL timeout, EI0002, distributed"
- BAD: "rank_table_file=/path/to/config.json"

**Purpose:** Enable semantic pattern matching; use `observed_at` for specific instances.
