---
name: pta-performance-analyze
description: Use when users report NPU memory exceeding GPU, memory consistency issues, operator-level memory profiling, workspace overhead, aclnn memory allocation problems, or ask to "diagnose/analyze operator memory", "run memory comparison", "plog shows abnormal allocation".
---

# PTA Memory Consistency Analyzer

Diagnoses why torch_npu uses more device memory than torch_gpu and provides actionable fixes.

## When to Use

| Trigger | Starting Point |
|---------|---------------|
| "diagnose/analyze operator memory", "NPU memory higher than GPU", "memory consistency not met" | Stage 1 |
| NPU test script provided + GPU comparison requested | Stage 1 Step 2 |
| **Both** `mem_results.md` **and** `filtered_plog_*.log` already exist | Stage 3 |
| Only memory data OR only plog (not both) | Stage 1 & 2 (regenerate both) |

**Required output:** data evidence (plog values, memory tables), root cause attribution (torch_npu vs CANN), actionable fixes.

## Stage 1: Prepare Scripts

### Step 1: Identify NPU Script

- User provides script path → use directly
- No path specified → scan cwd for `torchapi_*.py`:
  - 1 match → use it
  - Multiple → ask user
  - None → ask user for path
- Extract `TARGET_API` from script

### Step 2: Generate GPU Script

```bash
python tools/convert_npu_to_gpu.py <npu_script.py>
```

> **HARD GATE**: Must have both NPU and GPU scripts before Stage 2.

## Stage 2: Remote Execution

### Step 1: Run Benchmarks

**Don't ask user first.** Check for existing artifacts:
- `mem_results.md` exists **and** `filtered_plog_*.log` exists → reuse, go to Stage 3
- Otherwise → run remote tests:

```bash
python tools/run_remote_mem_test.py <npu_script> <gpu_script>
```

**Outputs:** `mem_results.md` (comparison table), `filtered_plog_<api>.log` (NPU memory trace)

### Step 2: Check Results

Read `mem_results.md`. Key metric: **activated_GB ratio** (NPU/GPU)

| Ratio | Action |
|-------|--------|
| ≤ 1.05 | Inform user: memory is normal. Ask to verify script correctness. **EXIT** (skip Stage 3 & 4) |
| > 1.05 | Proceed to Stage 3 |

> **HARD GATE**: Must have `mem_results.md` and `filtered_plog_*.log` before Stage 3. Do NOT skip to known issue lookup by API name alone. If NPU/GPU server connection fails, inform user with error and STOP.

## Stage 3: Root Cause Analysis

### Step 1: Locate Overconsumption

**Inputs:**
1. `mem_results.md` → test scenario (input shape, dtype), memory metrics, `TARGET_API`
2. `filtered_plog_*.log` → workspace allocs (`#N: bytes | op: aclnnXXX`), malloc/free events, peak allocation

**Analysis goal:** Pinpoint which aclnn ops cause extra memory. For each, explain **why** NPU > GPU:
- Operator semantics (what it does, intermediate buffers needed, dtype conversions)
- Input/output footprint (`shape × dtype_size`)
- Delta explanation (workspace algorithm, Cast/Contiguous chain, worst-case pre-allocation)

**Important:** Check if the overconsumption originates from `TARGET_API` or other ops:
- If from `TARGET_API` → proceed with normal analysis
- If from other ops (e.g., input data preprocessing, implicit Cast) → explicitly state this in output, as the issue is NOT in the target API itself

**Common patterns:**

| Pattern | Plog Signature | Root Cause |
|---------|---------------|------------|
| Internal Cast | Cast node in workspace | aclnn does dtype conversion internally |
| Large workspace | `workspaceSize_` >> input | Algorithm needs large scratch buffer |
| Redundant Contiguous | Multiple Contiguous calls | Non-contiguous tensor triggers copy |

**Output:** List of suspected aclnn interfaces + memory impact + root cause. Ask user: "Proceed with deeper analysis?"

- No → Stage 4 Step 2
- Yes → Step 2

### Step 2: Match Known Issues

For each suspected interface, search [memory_consistency_issue_cases.md](references/memory_consistency_issue_cases.md).

**Name extraction:** From `aclnnInplaceNormal_1_CastAiCore` → extract `aclnnInplaceNormal`

| Match Found | Action |
|------------|--------|
| Known issue corroborates Step 1 | Mark "closed-loop", skip Step 3 for this interface |
| Known issue doesn't corroborate | Step 3 required |
| No known issue | Step 3 required |

### Step 3: Source Code Analysis

**Goal:** Provide confident root cause with source-level evidence.

**Prerequisite:** Ask user for `pytorch_npu` source path. If unavailable → skip Step 3, warn user analysis is limited to plog-level evidence.

**Search:**
1. `pytorch_npu/third_party/op-plugin/` → kernel implementation, dispatch path
2. `torch_npu/_inductor/` → check if compiled mode has optimization missing in eager

**Report root cause (required for all cases):**

| Issue Location | Root Cause Report | Action |
|---------------|-------------------|--------|
| torch_npu | Describe code-level defect | Fix if possible; otherwise suggest fix approach |
| CANN | Describe workspace/cast behavior | Recommend filing DTS or feature request to CANN team |

**Note:** If overconsumption is NOT from `TARGET_API` (identified in Step 1), still report root cause but clarify that the issue is in auxiliary ops, not the target API being tested.

### Step 4: Present Findings

```
## Analysis: <API> Memory Overconsumption

### Memory Timeline
- Peak: X GiB | GPU baseline: Y GiB | Ratio: X/Y

### Root Causes
1. [aclnn interface] — [torch_npu/CANN] — [N GiB impact]
   - Evidence: [plog summary]
   - Known issue: [DTS] / None
   - Fix: [description]
```

## Stage 4: Validate & Accumulate

### Step 1: Validate

Ask: **"以上分析是否解决了你的问题？"**

- Yes → Step 2
- No → return to Stage 3 Step 2/3 with new evidence

### Step 2: Record Case

Ask for DTS number. If none, generate: `INT-YYYYMMDD-NNN`

Append to [memory_consistency_issue_cases.md](references/memory_consistency_issue_cases.md):

```yaml
### <dts_number>
- dts_number: "<dts_number>"
- description: "<brief description>"
- aclnn_interface: "<interface>"
- root_cause: "<analysis>"
- solution: "<fix>"
- category: "<classification>"
```

**Categories:** `cast导致显存占用升高`, `缺少aclnn算子`, `worst-case预分配`, `workspace过大`, `torch_npu逻辑缺陷`, `其他`

## Quick Reference

| Resource | Purpose |
|----------|---------|
| [memory_consistency_issue_cases.md](references/memory_consistency_issue_cases.md) | Historical issues |
| [convert_npu_to_gpu.py](tools/convert_npu_to_gpu.py) | NPU→GPU converter |
| [run_remote_mem_test.py](tools/run_remote_mem_test.py) | Remote benchmark runner |
| [filter_plog_memory.py](tools/filter_plog_memory.py) | Plog filter |

## Environment

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0        # Enable debug plog
export ASCEND_PROCESS_LOG_PATH=<dir>    # Redirect plog output
```
