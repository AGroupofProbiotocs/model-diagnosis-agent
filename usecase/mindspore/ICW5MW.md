<!-- https://e.gitee.com/mind_spore/dashboard?issue=ICWM5W -->
### Failure log
```
 @Level2
    @SKIP_ENV_CPU(reason="功能回退，仅支持bool，id=I9IXPE")
    @SKIP_ENV_GPU(reason="功能回退，仅支持bool，id=I9IXPE")
    def test_f_all_axis_list_int():
        x = Tensor(np.random.randn(3, 3, 3), mstype.float32)
        axis = [-1, -1, -1]
        keep_dims = False
        fact = AllMock(
            attributes={'axis': axis, 'keep_dims': keep_dims},
            inputs=[x])
>       fact.forward_mindspore_impl()

E       RuntimeError: aclnnAllGetWorkspaceSize call failed, please check!
E       
E       ----------------------------------------------------
E       - Ascend Error Message:
E       ----------------------------------------------------
E       EZ1001: [PID: 2942910] 2025-09-05-01:37:17.462.248 Dim 2 appears multiple times in the list of dims.[THREAD:2942910]
E       
E       (Please search "CANN Common Error Analysis" at https://www.mindspore.cn/en for error code description)
E       
E       ----------------------------------------------------
E       - C++ Call Stack: (For framework developers)
E       ----------------------------------------------------
E       mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize/reduce_all_aclnn_kernel.h:37 operator()

/home/miniconda3/envs/ci_310/lib/python3.10/site-packages/mindspore/common/api.py:2129: RuntimeError
=============================== warnings summary ===============================
```

### Root Cause
Script error: aclnnAll does not support repeated dimensions.

### Solution
Modify test cases to eliminate duplicate dims.
