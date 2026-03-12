# Error Codes Reference

## MindSpore Python Exception Types

MindSpore raises standard Python exceptions with structured error messages.

### Exception Categories
| Exception | Common Causes |
|-----------|--------------|
| `RuntimeError` | Device errors, operator execution failures, graph compilation errors, Ascend task abort |
| `ValueError` | Invalid parameter values, shape mismatches, invalid configurations |
| `TypeError` | Wrong data types, incompatible tensor types, invalid argument types |
| `MemoryError` | Host or device memory exhaustion |
| `NotImplementedError` | Feature/operator not supported on current backend |
| `IndexError` | Tensor index out of range |
| `KeyError` | Invalid configuration key, missing attribute |
| `AttributeError` | Invalid API call, missing method/property |

### Error Message Structure

MindSpore error messages follow this pattern (controllable via `MS_EXCEPTION_DISPLAY_LEVEL`):

1. Python stack trace (user code → framework)
2. Error type and description
3. "Traceback of Net Construct Code" (maps C++ execution back to `construct()` / `@ms_function`)
4. Framework developer info (C++ stacks, optional when `MS_EXCEPTION_DISPLAY_LEVEL=1`)

## CANN/ACL Error Codes (Ascend Backend)

Applies to MindSpore running on Ascend NPU. Error code rules:
- `1xxxxx` — Environment/logic errors, user-fixable
- `2xxxxx` — Resource exhaustion (memory, streams, devices)
- `3xxxxx` — Business exceptions (queue full/empty, storage)
- `5xxxxx` — Internal software/hardware errors, need technical support

### ACL General Errors (100000-148053)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 100000 | ACL_ERROR_INVALID_PARAM | Parameter verification failed | Check input parameters |
| 100001 | ACL_ERROR_UNINITIALIZE | Not initialized | Ensure aclInit called before other APIs |
| 100002 | ACL_ERROR_REPEAT_INITIALIZE | Repeated initialization | Avoid duplicate init calls |
| 100003-100006 | File errors | Invalid/failed/parse file errors | Check file exists, permissions, content |
| 100020 | ACL_ERROR_OP_TYPE_NOT_MATCH | Operator type mismatch | Check operator type |
| 100021 | ACL_ERROR_OP_INPUT_NOT_MATCH | Operator input mismatch | Check operator inputs |
| 100024 | ACL_ERROR_OP_NOT_FOUND | Operator not found | Check if operator type is supported |
| 100026 | ACL_ERROR_UNSUPPORTED_DATA_TYPE | Unsupported data type | Check dtype support |
| 100027 | ACL_ERROR_FORMAT_NOT_MATCH | Format mismatch | Check tensor format |
| 148049 | ACL_ERROR_INVALID_OPP_PATH | ASCEND_OPP_PATH not set or invalid | Set ASCEND_OPP_PATH to opp install path |

### ACL Resource Errors (200000-200007)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 200000 | ACL_ERROR_BAD_ALLOC | Memory allocation failed (OOM) | Check available memory, reduce batch size |
| 200001 | ACL_ERROR_API_NOT_SUPPORT | API not supported | Check API compatibility |
| 200002 | ACL_ERROR_INVALID_DEVICE | Invalid device | Check device exists |
| 200006 | ACL_ERROR_FEATURE_UNSUPPORTED | Feature not supported | Check CANN version compatibility |

### ACL Internal Errors (500000-500005)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 500000 | ACL_ERROR_INTERNAL_ERROR | Unknown internal error | Collect logs, contact support |
| 500001 | ACL_ERROR_FAILURE | Internal error | Collect logs, contact support |
| 500002 | ACL_ERROR_GE_FAILURE | Graph Engine error | Collect logs, contact support |
| 500003 | ACL_ERROR_RT_FAILURE | Runtime error | Collect logs, contact support |
| 500004 | ACL_ERROR_DRV_FAILURE | Driver error | Collect logs, contact support |

### RUNTIME Passthrough Errors (107000-507911)

**Parameter/Context Errors (107xxx):**
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 107000 | ACL_ERROR_RT_PARAM_INVALID | Parameter verification failed | Check input parameters |
| 107001 | ACL_ERROR_RT_INVALID_DEVICEID | Invalid device ID | Check device ID is valid |
| 107002 | ACL_ERROR_RT_CONTEXT_NULL | Context is empty | Call aclrtSetCurrentContext or aclrtSetDevice |
| 107003 | ACL_ERROR_RT_STREAM_CONTEXT | Stream not in current context | Check stream-context association |
| 107019 | ACL_ERROR_RT_WAIT_TIMEOUT | Wait timeout | Retry the operation |
| 107020 | ACL_ERROR_RT_TASK_TIMEOUT | Task execution timeout | Check business logic or adjust timeout |
| 107022 | ACL_ERROR_RT_DEVICE_TASK_ABORT | Device task abort conflict | Wait for abort to finish before other ops |

**Resource Errors (207xxx):**
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 207000 | ACL_ERROR_RT_FEATURE_NOT_SUPPORT | Feature not supported | Check CANN logs |
| 207001 | ACL_ERROR_RT_MEMORY_ALLOCATION | Memory allocation failed | Check storage space |
| 207003 | ACL_ERROR_RT_AICORE_OVER_FLOW | AI Core overflow | Check operator for overflow, use dump data |
| 207004 | ACL_ERROR_RT_NO_DEVICE | Device unavailable | Check device is running |
| 207018 | ACL_ERROR_RT_DEVIDE_OOM | Device memory exhausted | Check device memory, optimize usage |

**Internal/Hardware Errors (507xxx):**
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 507010 | ACL_ERROR_RT_LOST_HEARTBEAT | Task scheduler heartbeat lost | Contact support |
| 507011 | ACL_ERROR_RT_MODEL_EXECUTE | Model execution failed | Contact support |
| 507014 | ACL_ERROR_RT_AICORE_TIMEOUT | AI Core execution timeout | Contact support |
| 507015 | ACL_ERROR_RT_AICORE_EXCEPTION | AI Core execution exception | Contact support |
| 507017 | ACL_ERROR_RT_AICPU_TIMEOUT | AI CPU execution timeout | Contact support |
| 507034 | ACL_ERROR_RT_VECTOR_CORE_TIMEOUT | Vector core execution timeout | Contact support |
| 507053 | ACL_ERROR_RT_DEVICE_MEM_ERROR | Memory UCE error | Use aclrtGetMemUceInfo to fix |
| 507054 | ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR | HBM ECC error | Hardware fault, contact support |
| 507056 | ACL_ERROR_RT_LINK_ERROR | Inter-device communication link error | Retry, check communication link |

### GE Passthrough Errors (145000-545602)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 145000 | ACL_ERROR_GE_PARAM_INVALID | Parameter verification failed | Check input parameters |
| 145019 | ACL_ERROR_GE_PLGMGR_PATH_INVALID | Invalid so file/path | Check LD_LIBRARY_PATH |
| 145020 | ACL_ERROR_GE_FORMAT_INVALID | Invalid format | Check tensor format |
| 145021 | ACL_ERROR_GE_SHAPE_INVALID | Invalid shape | Check tensor shape |
| 145022 | ACL_ERROR_GE_DATATYPE_INVALID | Invalid data type | Check tensor dtype |
| 245000 | ACL_ERROR_GE_MEMORY_ALLOCATION | Memory allocation failed | Check available memory |
| 545000 | ACL_ERROR_GE_INTERNAL_ERROR | Unknown internal error | Contact support |
| 545601 | ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT | Model execution timeout | Contact support |

## ACLNN Error Codes (Ascend Operator API)

ACLNN errors use a two-phase interface pattern: `aclnnXxxGetWorkspaceSize` → `aclnnXxx`.

### General Status Codes
| Code | Name | Description |
|------|------|-------------|
| 0 | ACLNN_SUCCESS | Success |
| 161001 | ACLNN_ERR_PARAM_NULLPTR | Illegal nullptr in parameter |
| 161002 | ACLNN_ERR_PARAM_INVALID | Parameter validation error (dtype mismatch, etc.) |
| 361001 | ACLNN_ERR_RUNTIME_ERROR | Internal NPU runtime call exception |

### Internal Error Codes (561xxx)
| Code | Name | Description |
|------|------|-------------|
| 561000 | ACLNN_ERR_INNER | General internal exception |
| 561001 | ACLNN_ERR_INNER_INFERSHAPE_ERROR | Output shape inference error |
| 561002 | ACLNN_ERR_INNER_TILING_ERROR | NPU kernel tiling error |
| 561003 | ACLNN_ERR_INNER_FIND_KERNEL_ERROR | Kernel not found (binary package may not be installed) |
| 561101 | ACLNN_ERR_INNER_CREATE_EXECUTOR | Failed to create aclOpExecutor |
| 561107 | ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND | ASCEND_OPP_PATH not configured |
| 561108 | ACLNN_ERR_INNER_LOAD_JSON_FAILED | Failed to load operator kernel JSON |
| 561112 | ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND | Operator binary kernel package not found |

## CANN Inner Error Codes (Alphanumeric)

Format: `E[module-prefix][error-code]`

| Prefix | Module | Description |
|--------|--------|-------------|
| E5-E8 | AICORE checksum | Checksum errors during AICORE compilation |
| E9-EB | AICORE TBE | TBE compilation, operator spec issues |
| E80xxx | AICORE ops | Specific operator errors (StridedSliceGradD, ReduceSum, etc.) |
| EB0xxx | AICORE | Transpose spec unsupported, UB memory overflow |
| EZ9999 | AICORE execution | rtStreamSynchronize failure, model execution failed |
| E39999 | AICPU | AICPU kernel execution failure |
| EE9999 | Runtime | HBM OOM, task failure, halMemAlloc failure |
| EE1001 | Runtime | Invalid device ID |
| EI0002 | HCCL | Notify wait timeout |
| EI0004 | HCCL | Invalid ranktable |
| EI0005 | HCCL | Inconsistent comm parameters (AllReduce count mismatch, etc.) |
| EI0006 | HCCL | Socket build timeout |
| EJ0001 | HCCP | HCCP init failure |
| EK0001 | Profiling | Invalid profiling path/parameters |
| EL0004 | CANN | OOM error |

## Hardware Error Keywords

Keywords indicating hardware errors in MindSpore on Ascend:
- `DEVICE_TASK_ABORT` or `device task abort` — device hardware failure
- `HBM_MULTIBIT_ECC_ERROR` or `hbm Multi-bit ECC error` — HBM memory ECC error
- `DEVICE_MEM_ERROR` or `UCE ERROR` — uncorrectable memory error
- `LINK_ERROR` or `network link error` — network/hardware link issue
- `LOST_HEARTBEAT` — task scheduler heartbeat lost (507010)
- `AICORE_EXCEPTION` — AI Core execution exception (507015)

## CUDA Error Codes (GPU Backend)

Applies to MindSpore running on NVIDIA GPUs. CUDA errors are propagated as `RuntimeError` with the CUDA error name.

### CUDA Runtime Errors
| Error | Name | Description | Solution |
|-------|------|-------------|----------|
| 2 | cudaErrorMemoryAllocation | GPU memory allocation failed (OOM) | Reduce batch size, use gradient checkpointing, check `nvidia-smi` for VRAM usage |
| 4 | cudaErrorLaunchFailure | Kernel launch failed | Check input shapes/dtypes, may indicate illegal memory access in kernel |
| 6 | cudaErrorLaunchTimeout | Kernel took too long | Reduce workload, check for infinite loops in custom ops |
| 7 | cudaErrorLaunchOutOfResources | Too many resources requested | Reduce block size, check shared memory usage |
| 11 | cudaErrorInvalidValue | Invalid argument to CUDA API | Check tensor shapes, strides, and memory alignment |
| 35 | cudaErrorDevicesUnavailable | All CUDA devices in use or unavailable | Check `CUDA_VISIBLE_DEVICES`, release other GPU processes |
| 46 | cudaErrorDeviceAlreadyInUse | Device already in use with incompatible flags | Check for conflicting processes or contexts |
| 59 | cudaErrorAssert | Device-side assertion triggered | Enable `CUDA_LAUNCH_BLOCKING=1` to get exact failure location |
| 71 | cudaErrorIllegalAddress | Illegal memory access on device | Out-of-bounds tensor access, check indexing operations |
| 77 | cudaErrorIllegalInstruction | Illegal instruction on device | CUDA compute capability mismatch, recompile for correct GPU arch |
| 700 | cudaErrorIllegalAddress (async) | Asynchronous illegal memory access | Set `CUDA_LAUNCH_BLOCKING=1` to find the exact operation |
| 719 | cudaErrorLaunchFailure (async) | Asynchronous kernel launch failure | Set `CUDA_LAUNCH_BLOCKING=1` for detailed trace |

### NCCL Errors (GPU Distributed)
| Error | Description | Solution |
|-------|-------------|----------|
| unhandled system error | Generic NCCL internal error | Set `NCCL_DEBUG=INFO`, check network and GPU topology |
| remote process exited | A rank process crashed | Check logs of all ranks, look for the first error |
| connection refused | TCP connection to peer failed | Check firewall, verify MASTER_ADDR/MASTER_PORT |
| timeout | Operation exceeded deadline | Increase timeout, check for imbalanced workload across ranks |
| invalid usage | Incorrect NCCL API call sequence | Check collective op arguments match across all ranks |

### cuDNN Errors
| Error | Description | Solution |
|-------|-------------|----------|
| CUDNN_STATUS_NOT_SUPPORTED | Operation not supported for given config | Check input dtype/format, try different algorithm |
| CUDNN_STATUS_BAD_PARAM | Invalid parameter to cuDNN | Check input dimensions and strides |
| CUDNN_STATUS_ALLOC_FAILED | cuDNN workspace allocation failed | Reduce model size or batch size |

## CPU Backend Errors

CPU errors typically manifest as standard Python exceptions or system-level errors.

### Common CPU Error Patterns
| Error Pattern | Description | Solution |
|--------------|-------------|----------|
| `MemoryError` / `std::bad_alloc` | Host memory exhaustion | Reduce batch size, check `free -h`, close other applications |
| `Segmentation fault` (SIGSEGV) | Illegal memory access in native code | Check for corrupt tensors, version mismatch, or custom op bugs |
| `RuntimeError: not implemented` | Operator not available on CPU | Check operator's Supported Platforms, use GPU/Ascend instead |
| `OpenMP` errors | Thread configuration issues | Check `OMP_NUM_THREADS`, may conflict with MindSpore threads |
| `RuntimeError: parallel` | Parallel execution error | Check `MS_WORKER_NUM`, reduce parallelism |

## Error Trigger Sources

1. **MindSpore Framework** — Python exceptions from parameter validation, graph compilation, type inference
2. **CANN Runtime** — 1xxxxx/2xxxxx/5xxxxx codes from ACL/HCCL/GE API returns (Ascend)
3. **CANN Operators** — ACLNN 161xxx/361xxx/561xxx codes from operator execution (Ascend)
4. **CANN Inner** — E[x]xxxx codes from TBE/AICPU/HCCL internal modules (Ascend)
5. **Hardware/Driver** — 507xxx codes from device hardware conditions (Ascend)
6. **CUDA Runtime** — CUDA error codes from GPU execution (GPU)
7. **System** — OS-level errors from CPU execution or resource limits (CPU)

## Error Location

- Application logs: stdout/stderr
- MindSpore logs: controlled by `GLOG_v` environment variable
- CANN device logs: `/var/log/npu/slog/*/device-*/plog/` (Ascend)
- GPU logs: `nvidia-smi`, CUDA error output (GPU)
