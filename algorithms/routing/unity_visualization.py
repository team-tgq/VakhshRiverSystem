from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITY_DIR = PROJECT_ROOT / "tjk"
DEFAULT_UNITY_EXE = DEFAULT_UNITY_DIR / "tjk.exe"


@dataclass(frozen=True)
class UnityLaunchResult:
    exe_path: str
    cwd: str
    pid: int


def resolve_unity_exe(custom_path: str | None = None) -> Path:
    candidate = Path(custom_path).expanduser() if custom_path else DEFAULT_UNITY_EXE
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"未找到 Unity 可执行文件: {candidate}")
    if candidate.suffix.lower() != ".exe":
        raise ValueError(f"Unity 可执行文件必须是 .exe: {candidate}")
    return candidate


def check_unity_build(exe_path: Path) -> None:
    data_dir = exe_path.with_name(f"{exe_path.stem}_Data")
    player_dll = exe_path.with_name("UnityPlayer.dll")
    missing = [str(path) for path in (data_dir, player_dll) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Unity 打包文件不完整，缺少: " + "；".join(missing)
        )


def launch_unity_visualization(
    exe_path: str | None = None,
    *,
    extra_args: list[str] | None = None,
) -> UnityLaunchResult:
    resolved_exe = resolve_unity_exe(exe_path)
    check_unity_build(resolved_exe)

    cmd = [str(resolved_exe)]
    if extra_args:
        cmd.extend(str(arg) for arg in extra_args)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        cmd,
        cwd=str(resolved_exe.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        env=os.environ.copy(),
    )

    return UnityLaunchResult(
        exe_path=str(resolved_exe),
        cwd=str(resolved_exe.parent),
        pid=int(process.pid),
    )
