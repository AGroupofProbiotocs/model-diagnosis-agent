#!/usr/bin/env python3
"""
GPU 算子级显存抓取 - 输出与 NPU filtered_plog 同结构的 log，便于 Agent 对比分析。

通过 PyTorch Profiler (profile_memory=True) 记录每个算子的 self_cuda_memory_usage，
写入 [Summary] + Operator memory 列表，格式与 filter_plog_memory.py 的 Summary 对齐。

用法:
    python capture_gpu_mem_trace.py <gpu_script.py> [-o output.log] [--32align]

集成: run_remote_mem_test.py 在 GPU 测试成功后可选执行本脚本并下载 gpu_mem_trace_<api>.log；
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent.resolve()


def format_bytes(n):
    if n is None or n < 0:
        return "N/A"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.3f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def run_with_profiler(script_path, use_32align=False):
    """加载 gpu_script，在 profiler 下执行测试函数，返回 (sorted_events, api_name)。"""
    script_path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location("gpu_mem_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpu_mem_script"] = mod
    spec.loader.exec_module(mod)

    api_name = getattr(mod, "TARGET_API", "unknown")
    if use_32align:
        fn = getattr(mod, "calculate_nanmean_32aligned", None) or getattr(
            mod, "calculate_32aligned", None
        )
    else:
        fn = getattr(mod, "calculate_nanmean_non32aligned", None) or getattr(
            mod, "calculate_non32aligned", None
        )
    if fn is None:
        for name in (
            "run_test",
            "main_work",
            "calculate_nanmean_32aligned",
            "calculate_nanmean_non32aligned",
        ):
            fn = getattr(mod, name, None)
            if callable(fn):
                break
    if not callable(fn):
        raise RuntimeError(
            f"脚本中未找到可调用测试函数 "
            "(如 calculate_nanmean_32aligned / calculate_nanmean_non32aligned / run_test)"
        )

    if hasattr(torch.cuda, "set_device"):
        torch.cuda.set_device(0)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        out = fn()
        if isinstance(out, tuple):
            out = out[0]

    key_averages = prof.key_averages()

    def _self_mem(e):
        v = getattr(e, "self_device_memory_usage", None) or getattr(
            e, "self_cuda_memory_usage", None
        )
        return v if v is not None else 0

    try:
        sorted_events = sorted(key_averages, key=_self_mem, reverse=True)
    except Exception:
        sorted_events = list(key_averages)

    return sorted_events, api_name


def write_log_file(events, api_name, source_label, out_path=None):
    """
    输出与 filtered_plog 同结构的 log：分隔线 + [Summary] + Operator memory 列表。
    Agent 可像解析 NPU Workspace allocs 一样解析本文件。
    """
    sep = "=" * 80
    lines = [
        sep,
        "  GPU Memory Trace (operator-level, comparable to NPU filtered_plog Summary)",
        f"  Target API: {api_name}",
        f"  Source: {source_label}",
        sep,
        "",
        "[Summary]",
        "  (Operator memory = self_cuda_memory_usage, comparable to NPU Workspace allocs)",
        "",
    ]

    total_self = 0
    for i, evt in enumerate(events, 1):
        name = getattr(evt, "key", None) or getattr(evt, "name", None) or str(evt)
        self_mem = getattr(evt, "self_device_memory_usage", None) or getattr(
            evt, "self_cuda_memory_usage", None
        )
        if self_mem is None:
            self_mem = 0
        total_self += self_mem
        self_str = format_bytes(self_mem) if self_mem else "0 B"
        count = getattr(evt, "count", 1)
        # 与 filter_plog_memory 的 "#1: ... bytes (...)  |  op: xxx" 对齐
        lines.append(
            f"    #{i}: {self_mem:>15,} bytes  ({self_str})  |  op: {name}  (count={count})"
        )
    lines.append(f"    Total: {total_self:>13,} bytes  ({format_bytes(total_self)})")
    lines.append("")

    text = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(
        description="GPU 算子级显存抓取，输出与 filtered_plog 同结构的 log",
    )
    parser.add_argument(
        "gpu_script",
        type=Path,
        help="GPU 测试脚本路径（需定义 TARGET_API 和可调用测试函数）",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出 log 文件路径")
    parser.add_argument("--32align", action="store_true", help="使用 32 字节对齐测试分支")
    args = parser.parse_args()

    if not args.gpu_script.is_file():
        print(f"错误: 文件不存在 {args.gpu_script}", file=sys.stderr)
        sys.exit(1)

    try:
        events, api_name = run_with_profiler(args.gpu_script, use_32align=args.32align)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    source_label = str(args.gpu_script)
    text = write_log_file(
        events,
        api_name,
        source_label,
        out_path=args.output,
    )
    print(text, flush=True)
    if args.output:
        print(f"已写入: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
