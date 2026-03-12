# Backend Diagnosis Reference

Detailed per-backend diagnosis steps. Read the section matching the target backend identified by the Quick Route in SKILL.md.

## Table of Contents

- [Ascend Backend (CANN)](#ascend-backend-cann)
- [GPU Backend (CUDA)](#gpu-backend-cuda)
- [CPU Backend](#cpu-backend)
- [MindSpore Framework Log Analysis](#mindspore-framework-log-analysis)
- [Further Location Techniques](#further-location-techniques)

## Ascend Backend (CANN)

- Parse CANN error codes using [Error Codes reference](error-codes.md)
- Parse ACLNN error codes for operator-level failures
- Check CANN logs: `/var/log/npu/slog/*/device-*/plog/`
- Check TBE/AKG compilation errors
- Cross-reference with [CANN API Reference](cann-api-reference.md) for aclnn API constraints and adaptation flow
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
  - Read [ACLNN Adaptation Reference](cann-api-reference.md) for detailed adaptation flow
- If CANN issue → Provide CANN-specific fix → Validate with user

## GPU Backend (CUDA)

- Parse CUDA error codes using [Error Codes reference](error-codes.md) — GPU/CUDA/NCCL/cuDNN sections
- Check CUDA errors: enable synchronous execution with `CUDA_LAUNCH_BLOCKING=1` to pinpoint exact failing operation
- Check GPU memory: `nvidia-smi` for VRAM usage; CUDA OOM → reduce batch size, use gradient checkpointing, or call `ms.hal.memory.empty_cache()` to release cached GPU memory
- Check NCCL errors for distributed training:
  - Enable NCCL debug: `export NCCL_DEBUG=INFO`
  - Check GPU topology: `nvidia-smi topo -m`
  - Verify all ranks use consistent arguments for collective operations
- Check cuDNN errors: dtype/format not supported, workspace allocation failure
- Verify CUDA compute capability matches compiled MindSpore (e.g., sm_70 for V100, sm_80 for A100)
- If GPU issue → Provide GPU-specific fix → Validate with user

## CPU Backend

- Check system resource limits: `free -h` for memory, `ulimit -a` for file descriptors and stack size
- Verify CPU operator implementation availability — check `Supported Platforms` in operator docs
- Check threading configuration:
  - `OMP_NUM_THREADS` — OpenMP thread count (may conflict with MindSpore internal threads)
  - `MS_WORKER_NUM` — MindSpore data loading worker count
  - Over-subscription (total threads > CPU cores) causes severe slowdown
- Check for segfaults: often caused by version mismatch between MindSpore and system libraries, or corrupt tensor data
- CPU operators may have different numerical behavior than Ascend/GPU — use larger tolerance for cross-backend comparison
- If CPU issue → Provide CPU-specific fix → Validate with user

## MindSpore Framework Log Analysis

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

## Further Location Techniques

When the root cause cannot be confirmed from available evidence, use these techniques to narrow down the issue. The goal is to give users targeted debug steps rather than guessing — an incorrect diagnosis wastes more time than asking for more data.

### CANN Level

Enable CANN debug logs to stdout:
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

### MindSpore Framework Level

Provide a **debug patch** that adds targeted logging (e.g., `MS_LOG(ERROR)` / `MS_LOG(INFO)` at suspected failure points, or Python `logger.error()` / `print()` in framework code).

**Debug patch requirements:**
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

## See Also

- [Error Codes](error-codes.md) — Error code lookup tables (CANN, ACLNN, CUDA, CPU)
- [CANN API Reference](cann-api-reference.md) — ACLNN API constraints, adaptation flow, per-API docs
- [MindSpore API](mindspore-api.md) — API layers, execution modes, operator debugging
- [Failure Showcase](failure-showcase.md) — Historical failures and solutions
