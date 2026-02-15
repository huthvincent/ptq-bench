# -*- coding: utf-8 -*-
"""
环境信息收集模块

收集运行环境的完整信息，包括：
- GPU 型号、VRAM、CUDA 版本、driver 版本
- Python 包版本（PyTorch、transformers 等）
- Git commit hash
- 操作系统信息

每次实验运行时自动调用，输出到结果文件中，确保可复现性。
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timezone


def get_git_info(repo_path: str | Path | None = None) -> dict:
    """
    获取当前 Git 仓库的信息。

    参数:
        repo_path: Git 仓库路径，None 时使用项目根目录

    返回:
        dict: 包含 commit_hash、branch、is_dirty 的字典
    """
    if repo_path is None:
        repo_path = Path(__file__).resolve().parent.parent

    info = {"commit_hash": None, "branch": None, "is_dirty": None}
    try:
        # 获取 commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit_hash"] = result.stdout.strip()

        # 获取分支名
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # 检查是否有未提交的修改
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["is_dirty"] = len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return info


def get_gpu_info() -> list[dict]:
    """
    获取 GPU 信息。

    返回:
        list[dict]: 每张 GPU 的信息列表，包含 name、memory_total、driver_version 等
    """
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "driver_version": parts[3],
                        "compute_capability": parts[4],
                    })
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return gpus


def get_python_packages() -> dict:
    """
    获取关键 Python 包的版本信息。

    返回:
        dict: 包名 → 版本号 的映射
    """
    packages = {}
    critical_packages = [
        "torch", "transformers", "datasets", "accelerate",
        "auto-gptq", "autoawq", "smoothquant",
        "lm-eval", "vllm", "safetensors", "tokenizers",
        "bitsandbytes", "scipy", "numpy",
    ]
    for pkg_name in critical_packages:
        try:
            # 尝试用模块名导入获取版本（处理包名和模块名不一致的情况）
            module_name = pkg_name.replace("-", "_")
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "installed (version unknown)")
            packages[pkg_name] = version
        except ImportError:
            packages[pkg_name] = "not installed"
    return packages


def get_cuda_info() -> dict:
    """
    获取 CUDA 相关信息。

    返回:
        dict: 包含 cuda_version、cudnn_version 等
    """
    info = {"cuda_available": False, "cuda_version": None, "cudnn_version": None}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            info["torch_cuda_arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    except ImportError:
        pass
    return info


def collect_env_info(repo_path: str | Path | None = None) -> dict:
    """
    收集完整的运行环境信息。

    这是主要的入口函数，每次实验开始时调用一次。

    参数:
        repo_path: Git 仓库路径

    返回:
        dict: 完整的环境信息字典
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version,
        "git": get_git_info(repo_path),
        "gpus": get_gpu_info(),
        "cuda": get_cuda_info(),
        "packages": get_python_packages(),
    }


def format_env_info(env_info: dict) -> str:
    """
    将环境信息格式化为 Markdown 字符串。

    参数:
        env_info: collect_env_info() 的返回值

    返回:
        str: 格式化的 Markdown 文本
    """
    lines = ["## 运行环境", ""]

    # 基础信息
    lines.append(f"- **时间**: {env_info.get('timestamp', 'N/A')}")
    lines.append(f"- **主机**: {env_info.get('hostname', 'N/A')}")
    lines.append(f"- **OS**: {env_info.get('os', 'N/A')}")
    lines.append(f"- **Python**: {env_info.get('python_version', 'N/A')}")

    # Git 信息
    git = env_info.get("git", {})
    if git.get("commit_hash"):
        dirty_flag = " ⚠️ (有未提交修改)" if git.get("is_dirty") else ""
        lines.append(f"- **Git**: `{git['commit_hash'][:8]}` ({git.get('branch', '?')}){dirty_flag}")

    # GPU 信息
    gpus = env_info.get("gpus", [])
    if gpus:
        for gpu in gpus:
            lines.append(f"- **GPU {gpu['index']}**: {gpu['name']} ({gpu['memory_total_mb']} MB)")
        lines.append(f"- **Driver**: {gpus[0].get('driver_version', 'N/A')}")

    # CUDA 信息
    cuda = env_info.get("cuda", {})
    lines.append(f"- **CUDA**: {cuda.get('cuda_version', 'N/A')}")
    lines.append(f"- **cuDNN**: {cuda.get('cudnn_version', 'N/A')}")

    # 关键包版本
    lines.append("")
    lines.append("### 关键包版本")
    lines.append("")
    packages = env_info.get("packages", {})
    for pkg, version in packages.items():
        if version != "not installed":
            lines.append(f"- `{pkg}`: {version}")

    return "\n".join(lines)
