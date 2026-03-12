---
name: ms-failure-analyze
description: "MindSpore failure analyzer for Ascend/GPU/CPU backends. Use this skill whenever users report ANY MindSpore error, crash, or unexpected behavior — even if they don't know the root cause. Triggers for: (1) Error codes: CANN 1xxxxx-5xxxxx, ACLNN 161xxx/361xxx/561xxx, CANN inner E[x]xxxx, CUDA errors, NCCL errors; (2) Python exceptions: RuntimeError, ValueError, TypeError, MemoryError from MindSpore; (3) Device issues: OOM, heartbeat lost, ECC errors, device hang on Ascend/GPU; (4) Framework issues: graph compilation, type inference, operator not supported, set_context problems, GRAPH_MODE/PYNATIVE_MODE errors; (5) API issues: mindspore.mint (mint.nn/optim/linalg/distributed), mindspore.ops, view-vs-copy, jit_view_unsupported; (6) ACLNN adaptation: gen_ops.py YAML errors, GeneralInfer failures, PyBoost/KBK issues, BPROP wiring, composite op chains, View strides errors, PTA alignment; (7) Test failures: MindSporeTest repo ST/UT failures with CONTEXT_DEVICE_TARGET/CONTEXT_MODE/CONTEXT_JIT_LEVEL. Also use when users paste error logs, stack traces, or CANN debug output and need help interpreting them."
---

# MS (MindSpore) Failure Analyzer

## Stage 1: Find Similar Problem

### 1. Identify Error Pattern

Parse error message to determine error source. MindSpore errors come in several forms:

**Python Exceptions (MindSpore Framework):**
- `RuntimeError` — device errors, operator execution failures, graph compilation errors
- `ValueError` — parameter validation, shape mismatches, invalid configurations
- `TypeError` — wrong data types, incompatible tensor types
- `MemoryError` — host or device memory exhaustion
- `NotImplementedError` — unsupported features on specific backend

**CANN Error Codes (Ascend backend, numeric):**
- `1xxxxx` — environment/logic errors (user-fixable)
- `2xxxxx` — resource exhaustion (memory, streams, devices)
- `3xxxxx` — business exceptions (queue full/empty, storage limits)
- `5xxxxx` — internal software/hardware errors (need support)

**CANN Inner Error Codes (Ascend backend, alphanumeric):**
- `E[5-B]xxxx` — AICORE errors (TBE compilation, operator spec issues)
- `E3xxxx` — AICPU kernel execution failures
- `EExxxx` — Runtime errors (HBM OOM, task failure)
- `EI/EJxxxx` — HCCL/HCCP distributed errors
- `EKxxxx` — Profiling errors
- `EZxxxx` — AICORE execution failures (rtStreamSynchronize, model execution)

**ACLNN Error Codes (Ascend operator API):**
- `161xxx` — parameter errors (nullptr, invalid params)
- `361xxx` — runtime call errors
- `561xxx` — internal errors (shape inference, tiling, kernel finding)

If error code found:
1. Extract error code/exception type and context
2. Check [Error Codes reference](references/error-codes.md) for known solutions
3. If direct match → provide solution
4. If partial match → proceed to Stage 2

### 2. Check Failure Showcase

Search failure showcase for matching patterns:
- Error keywords (OOM, timeout, ECC, link error, compile failed, etc.)
- Call stack patterns (operator names, functions, graph nodes)
- Backend + module combinations (Ascend+HCCL, GPU+NCCL, graph compile, etc.)
- Execution mode issues (GRAPH_MODE vs PYNATIVE_MODE)

If matching failure found:
1. Show historical failure info
2. Provide previously successful solution
3. Ask user: "Does this solution work for your issue?"
4. If yes → END Stage 1
5. If no → proceed to Stage 2

## Stage 2: Analyze Failure

**Failure Orientation Strategy:** Platform → Scripts → MindSpore Framework → Backend (CANN/CUDA/CPU)

### Quick Route (skip levels when evidence is clear)

Based on Stage 1 error pattern, jump directly to the most likely level:

| Stage 1 Finding | Start At | Rationale |
|-----------------|----------|-----------|
| CANN error code (1xxxxx-5xxxxx) | Backend (Ascend) | Error originates from CANN runtime |
| ACLNN error code (161xxx/361xxx/561xxx) | Backend (Ascend) | Operator-level ACLNN failure |
| CANN inner code (E[x]xxxx) | Backend (Ascend) | TBE/AICPU/HCCL internal error |
| Hardware keywords (heartbeat, ECC, link error) | Platform | Hardware fault signals |
| CUDA error / GPU OOM | Backend (GPU) | GPU runtime error |
| NCCL error | Backend (GPU) | GPU distributed communication |
| Python exception only (ValueError, TypeError) | Scripts | Likely user code or API misuse |
| Graph compilation error / type inference | Framework | MindSpore graph compiler issue |
| gen_ops.py / YAML error | Backend (Ascend) | ACLNN adaptation build error |
| `set_context` / mode / device error | Scripts | Configuration issue |
| No clear signal | Platform (top-down) | Default: scan from top |

When jumping to a specific level, still collect basic evidence from upper levels (version info, device status) to rule out environmental factors.

### Step 1: Collect Evidence

For each orientation layer, collect:
- Platform: HW type, driver version, CANN version (Ascend), CUDA version (GPU), OS info
- Scripts: User code patterns, context configuration, execution mode, library calls
- MindSpore Framework: Operator calls, graph structure, parameter validation, debug logs
- Backend: Error codes from ACL/HCCL/GE (Ascend), CUDA errors (GPU), system errors (CPU)

### Step 2: Orient and Diagnose

Apply orientation strategy (start at the level identified by Quick Route, then expand if needed):

**Platform Level:**
- Ascend: Check NPU device health with `npu-smi info`, verify CANN version compatibility
- GPU: Check GPU status with `nvidia-smi`, verify CUDA/cuDNN version compatibility
- CPU: Check system resources (memory, CPU usage)
- **Version compatibility check** (very common failure source):
  - MindSpore ↔ CANN: each MindSpore version requires a specific CANN version range. Check:
    ```bash
    python -c "import mindspore; print(mindspore.__version__)"
    cat /usr/local/Ascend/ascend-toolkit/version
    ```
    Cross-reference with MindSpore release notes for supported CANN versions.
  - MindSpore ↔ Python: verify Python version matches MindSpore requirements
  - MindSpore ↔ CUDA/cuDNN: for GPU, check CUDA toolkit and cuDNN version match
  - Driver version: Ascend driver (`npu-smi info`) or NVIDIA driver (`nvidia-smi`) must meet minimum requirements
  - If version mismatch detected → this is likely the root cause; advise version alignment before further diagnosis
- Check for hardware errors:
  - Ascend: 507010 heartbeat lost, 507053 memory UCE, 507054 HBM ECC, 507056 link error
  - GPU: CUDA device errors, ECC errors
- If hardware issue → Provide hardware-specific fix → Validate with user

**Script Level:**
- Check `mindspore.set_context()` configuration:
  - `device_target` matches available hardware ("Ascend"/"GPU"/"CPU")
  - `mode` is appropriate (GRAPH_MODE=0, PYNATIVE_MODE=1)
  - `ascend_config` / `gpu_config` settings are valid
- Check environment variables (ASCEND_OPP_PATH, LD_LIBRARY_PATH, CUDA paths)
- **MindSporeTest repo cases** (the following env vars are only effective when test cases originate from the MindSporeTest repo):
  - Global device, mode and jit_level are configured via environment variables, not in-script `set_context()`:
    - `CONTEXT_DEVICE_TARGET` — determines test device ("Ascend"/"GPU"/"CPU")
    - `CONTEXT_MODE` — determines execution mode ("0"=GRAPH_MODE, "1"=PYNATIVE_MODE)
    - `CONTEXT_JIT_LEVEL` — determines global jit_level ("O0"/"O1"/"O2")
  - When reproducing test failures from MindSporeTest, must set these env vars to match the CI/test environment
  - Test scripts may also use `set_context_mode(mode='pynative'|'kbk'|'ge')` from the ST framework:
    - `pynative` = PYNATIVE_MODE (eager execution)
    - `kbk` = KernelByKernel (GRAPH_MODE + jit_level O0, operators execute one-by-one in graph)
    - `ge` = GraphEngine (GRAPH_MODE + jit_level O1/O2, full graph optimization and offload)
- Analyze user code for misuse:
  - Wrong device placement, cross-device tensor operations
  - Shape/dtype mismatches in network definition
  - Improper `construct()` / `@ms_function` usage in GRAPH_MODE
  - Dynamic control flow in static graph mode
- Review script patterns (repeated context init, improper resource cleanup)
- If script issue → Provide code fix → Validate with user

**MindSpore Framework Level:**
- Check operator availability on target backend (`Supported Platforms: Ascend GPU CPU`)
- Verify operator parameter constraints (types, shapes, value ranges)
- Review graph compilation errors (IR optimization, type inference failures)
- Identify which API layer the user is using and check accordingly:
  - **`mindspore.mint`** (preferred high-level API, PyTorch-compatible):
    - `mint.*` functions — check view-vs-copy semantics (`@jit_view_unsupported` on squeeze/flatten/reshape/t/narrow/split/broadcast_to)
    - `mint.nn.*` layers — Conv1d/2d/3d, BatchNorm, LayerNorm, GroupNorm, Dropout, loss functions, activations
    - `mint.nn.functional.*` — functional versions of all nn layers
    - `mint.optim.*` — AdamW, Adam, SGD, FusedAdamW optimizers
    - `mint.linalg.*` — inv, norm, vector_norm, matrix_norm, qr
    - `mint.distributed.*` — init_process_group, all_reduce, barrier, etc.
    - `mint.special.*` — erfc, expm1, exp2, log1p, log_softmax
    - Note: many mint APIs are marked "experimental" and may change between versions
  - **`mindspore.ops`** (lower-level operator API):
    - `ops.function.*` — functional API wrappers
    - `ops.operations.*` (Primitive) — low-level operator primitives
  - **`mindspore.nn`** (traditional high-level layers)
  - Check for mint-specific pitfalls:
    - `mint.equal()` returns Python `bool` (not Tensor) — differs from `ops.equal()`
    - `mint.item()` requires single-element Tensor, raises RuntimeError otherwise
    - View ops in mint may not work in JIT/GRAPH_MODE (decorated with `@jit_view_unsupported`)
    - mint wraps `ops.*_ext` variants which are the newer implementations
- Read [MindSpore API reference](references/mindspore-api.md) for API details (includes mint, ops, nn)
- If framework issue → Provide framework fix → Validate with user

**Backend Level (Ascend):**
- Parse CANN error codes using [Error Codes reference](references/error-codes.md)
- Parse ACLNN error codes for operator-level failures
- Check CANN logs: `/var/log/npu/slog/*/device-*/plog/`
- Check TBE/AKG compilation errors
- Cross-reference with [CANN API Reference](references/cann-api-reference.md) for aclnn API constraints and adaptation flow
- **ACLNN adaptation-level diagnosis** (when error originates from operator development/adaptation):
  - Determine which adaptation path the operator uses (auto-generated vs Customize):
    - Auto-generated: YAML `dispatch.enable: True` without `Ascend:` field → check `aclnn_config.yaml` mapping
    - Customize: YAML has `dispatch.Ascend: XxxAscend` → check PyBoost customize `.cc` and KBK kernel `.cc`
  - Check common ACLNN integration failure points:
    - **gen_ops.py errors**: YAML field structure mismatch, missing `py_method`, missing function_doc entries
    - **GeneralInfer (C++ shape/type inference)**: dynamic shape/rank handling, incorrect output shape, missing unknown-value fallback
    - **PyBoost (Pynative)**: parameter conversion issues (tuple→vector, Optional None handling, str→enum)
    - **KBK (Graph kernel)**: Init/Resize/Launch separation issues, workspace allocation, `MS_ACLNN_KERNEL_FACTORY_REG` registration
    - **BPROP**: input/output count mismatch (backward inputs = forward inputs + 2), unused input marking, dynamic shape in bprop (`Conditional`/`ShapeCalc` missing)
    - **View ops**: strides calculation errors, `view: True`/`graph_view: True` YAML misconfiguration, fallback to ACLNN kernel
    - **Composite ops**: missing sub-operators in ACLNN call chain, `bprop_expander: False` without proper sub-op bprop
  - Read [ACLNN Adaptation Reference](references/cann-api-reference.md) for detailed adaptation flow
- If CANN issue → Provide CANN-specific fix → Validate with user

**Backend Level (GPU):**
- Parse CUDA error codes using [Error Codes reference](references/error-codes.md) — GPU/CUDA/NCCL/cuDNN sections
- Check CUDA errors: enable synchronous execution with `CUDA_LAUNCH_BLOCKING=1` to pinpoint exact failing operation
- Check GPU memory: `nvidia-smi` for VRAM usage; CUDA OOM → reduce batch size, use gradient checkpointing, or `torch.cuda.empty_cache()` equivalent
- Check NCCL errors for distributed training:
  - Enable NCCL debug: `export NCCL_DEBUG=INFO`
  - Check GPU topology: `nvidia-smi topo -m`
  - Verify all ranks use consistent arguments for collective operations
- Check cuDNN errors: dtype/format not supported, workspace allocation failure
- Verify CUDA compute capability matches compiled MindSpore (e.g., sm_70 for V100, sm_80 for A100)
- If GPU issue → Provide GPU-specific fix → Validate with user

**Backend Level (CPU):**
- Check system resource limits: `free -h` for memory, `ulimit -a` for file descriptors and stack size
- Verify CPU operator implementation availability — check `Supported Platforms` in operator docs
- Check threading configuration:
  - `OMP_NUM_THREADS` — OpenMP thread count (may conflict with MindSpore internal threads)
  - `MS_WORKER_NUM` — MindSpore data loading worker count
  - Over-subscription (total threads > CPU cores) causes severe slowdown
- Check for segfaults: often caused by version mismatch between MindSpore and system libraries, or corrupt tensor data
- CPU operators may have different numerical behavior than Ascend/GPU — use larger tolerance for cross-backend comparison
- If CPU issue → Provide CPU-specific fix → Validate with user

**MindSpore Framework Log Analysis:**
- **GLOG output** (controlled by `GLOG_v`): search for key patterns when analyzing MindSpore framework logs:
  ```bash
  grep -i "error\|exception\|fail\|abort" mindspore.log | head -30
  grep -i "infer shape\|infer type\|abstract" mindspore.log | head -20
  grep -i "select kernel\|launch kernel\|not supported" mindspore.log | head -20
  ```
- **Graph dump analysis** (when `save_graphs=True`): check IR files in `save_graphs_path`:
  - `*_validate.ir` — after graph validation (check for type/shape errors)
  - `*_optimize.ir` — after optimization passes (check for failed optimizations)
  - Look for `%para` (parameters), `%load` (weight loading), operator nodes with error annotations

**Show Fix Advice:**

When root cause is **confirmed**:
```
Analysis: [Failure type identified]
Backend: [Ascend/GPU/CPU]
Root Cause: [Specific cause]
Solution: [Actionable steps]
```

When root cause is **uncertain** — provide further location means instead of guessing:
```
Analysis: [Failure type identified]
Backend: [Ascend/GPU/CPU]
Root Cause: Unable to confirm — further debugging required
Further Location:
  [Specific debug steps with env vars, log commands, or debug patches]
```

**Further location techniques by level:**

- **CANN Level:** Enable CANN debug logs to stdout:
  ```bash
  export ASCEND_GLOBAL_LOG_LEVEL=0       # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
  export ASCEND_SLOG_PRINT_TO_STDOUT=1   # print CANN logs to stdout
  ```
  Then ask user to re-run and provide the debug log output.
  **When user provides CANN debug logs:** Do NOT read the full log (can be extremely large). Instead, search for key patterns first:
  ```bash
  grep -i "error\|fail\|exception\|abort" cann_debug.log | head -50
  grep -i "aclnn\|acl_error\|ret=" cann_debug.log | tail -30
  grep -i "EE\|EI\|EJ\|EH\|EP" cann_debug.log | head -20
  ```
  Start from error/failure lines, then read surrounding context only as needed to trace the call chain.

- **MindSpore Framework Level:** Provide a **debug patch** that adds targeted logging
  (e.g., `MS_LOG(ERROR)` / `MS_LOG(INFO)` at suspected failure points, or Python `logger.error()` / `print()` in framework code).

- **Debug patch requirements:**
  1. Generate the patch using `git diff` or `diff -u` format so user can apply it directly
  2. Verify the patch can be applied cleanly using `patch --dry-run -p1 < debug.patch` or `git apply --check debug.patch`
  3. The patch should be minimal — only add logging/debug statements at suspected failure points
  4. Include clear instructions: which repo/branch to apply against, how to apply, and what output to look for
  5. After user provides debug output, analyze it and narrow down root cause

  Debug patch output template:
  ```
  Debug Patch (apply to [repo] [branch]):

  [git diff / unified diff content]

  Apply: patch -p1 < debug.patch  (or: git apply debug.patch)
  Verify: patch --dry-run -p1 < debug.patch  (or: git apply --check debug.patch)
  Expected output to look for: [description of key log lines]
  ```

### Step 3: Validate and Iterate (MANDATORY — must NOT be skipped)

After providing fix or further location advice:
1. **MUST** ask user: "Did this advice resolve your issue?" — do NOT proceed without user confirmation
   - Yes → Proceed to Stage 3
   - No → Collect additional evidence, re-orient at current or deeper level
   - User provided debug output → Analyze the new evidence, refine diagnosis, loop back to Step 2
2. Never end Stage 2 unless user explicitly confirms the issue is resolved
3. This step is **not optional** — every fix/debug suggestion must be validated before moving on

## Stage 3: Accumulate Experience

### Step 1: Report Analysis Summary

Review analysis and extract key points:
- **Failure Info:** Error message, context, environment
- **Backend:** Ascend/GPU/CPU
- **Failure Type:** Platform/Scripts/Framework/Backend
- **Root Cause:** Specific issue identified
- **Solution:** Steps that resolved the issue

Report to user as structured summary.

### Step 2: Update Failure Showcase

First, search [Failure Showcase reference](references/failure-showcase.md) for matching existing entries.

**Matching criteria** — an existing entry "matches" if ANY of these hold:
- Same error code (e.g., both have `507010` or `561003`)
- Same exception type AND similar root cause (e.g., both are `RuntimeError` from dynamic shape in bprop)
- Overlapping error keywords in `failure_info` (≥ 2 significant keywords match)

**If matching entry exists** — update it in place (do NOT create a duplicate):
- Enrich `failure_info` / `root_cause` / `solution` with any new details learned
- Refresh `last_seen` to current date
- Increment `occurrences` count
- **Fill `observed_at`** if it was empty — always record at least one concrete observation location

**If no matching entry exists** — create a new entry (all fields required, `observed_at` must NOT be left empty):
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

## Quick References

- [Error Codes](references/error-codes.md) - Complete error code mappings (MindSpore + CANN + ACLNN)
- [Failure Showcase](references/failure-showcase.md) - Historical failures and solutions
- [MindSpore API](references/mindspore-api.md) - API system (mint/ops/nn), backend registration, operator patterns
- [CANN API Reference](references/cann-api-reference.md) - CANN aclnn API index and documentation pointers

## Diagnostic Commands

```bash
# Ascend
npu-smi info                                     # NPU device status
cat /usr/local/Ascend/ascend-toolkit/version      # CANN version
ls /usr/local/Ascend/ascend-toolkit/latest/opp/   # Operator packages
tail -f /var/log/npu/slog/*/device-*/plog/*.log   # CANN device logs

# GPU
nvidia-smi                                        # GPU device status
nvcc --version                                    # CUDA version
nvidia-smi topo -m                                # GPU topology (distributed)

# MindSpore
python -c "import mindspore; print(mindspore.__version__)"
python -c "import mindspore; print(mindspore.get_context('device_target'))"
python -c "import mindspore; print(mindspore.get_context('mode'))"
python -c "import mindspore; print(mindspore.hal.device_count())"  # Available devices
python -c "import mindspore; print(mindspore.get_context('jit_config'))"  # JIT config

# System
free -h                                           # Memory status
ulimit -a                                         # System limits
```

## Environment Variables

- `ASCEND_OPP_PATH` — Operator compiler path (Ascend)
- `ASCEND_GLOBAL_LOG_LEVEL` — CANN log level (0=debug, 1=info, 2=warning, 3=error, 4=null)
- `ASCEND_SLOG_PRINT_TO_STDOUT` — Print CANN slog to stdout (1=enable, 0=disable)
- `GLOG_v` — MindSpore log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR, 4=CRITICAL)
- `MS_EXCEPTION_DISPLAY_LEVEL` — Exception display (0=full, 1=hide framework stack)
- `CUDA_VISIBLE_DEVICES` — GPU device visibility
- `NCCL_DEBUG` — NCCL debug level (GPU distributed)
- `MS_DEV_FORCE_ACL` — Force ACL kernel execution (Ascend debugging)
- `CONTEXT_DEVICE_TARGET` — MindSporeTest repo: global device target ("Ascend"/"GPU"/"CPU")
- `CONTEXT_MODE` — MindSporeTest repo: global execution mode ("0"=GRAPH_MODE, "1"=PYNATIVE_MODE)
- `CONTEXT_JIT_LEVEL` — MindSporeTest repo: global jit_level ("O0"/"O1"/"O2")
