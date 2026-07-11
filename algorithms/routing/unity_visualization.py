from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNITY_EXE_ENV = "VAKHSH_ROUTING_UNITY_EXE"
UNITY_SEARCH_DIRS = (
    PROJECT_ROOT / "algorithms" / "routing" / "tjk",
    PROJECT_ROOT / "algorithms" / "routing" / "unity_build",
    PROJECT_ROOT / "algorithms" / "routing" / "unity",
    PROJECT_ROOT / "plugins" / "routing_plugin" / "tjk",
    PROJECT_ROOT / "tjk",
)
PREFERRED_EXE_NAMES = ("tjk.exe", "TJK.exe")


@dataclass(frozen=True)
class UnityLaunchResult:
    exe_path: str
    cwd: str
    pid: int


def resolve_unity_exe(custom_path: str | None = None) -> Path:
    candidate_text = custom_path or os.environ.get(UNITY_EXE_ENV)
    if candidate_text:
        candidate = Path(candidate_text).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"未找到 Unity 可执行文件: {candidate}")
        if candidate.suffix.lower() != ".exe":
            raise ValueError(f"Unity 可执行文件必须是 .exe: {candidate}")
        return candidate

    attempted: list[Path] = []
    incomplete: list[str] = []
    for candidate in _iter_unity_exe_candidates():
        attempted.append(candidate)
        if not candidate.exists() or candidate.suffix.lower() != ".exe":
            continue
        try:
            check_unity_build(candidate)
        except FileNotFoundError as exc:
            incomplete.append(str(exc))
            continue
        return candidate.resolve()

    checked = "\n".join(f"- {path}" for path in attempted)
    hint = (
        "未找到可启动的 Unity 打包程序。\n"
        "请把完整 Unity build 放到 algorithms/routing/tjk/ 或 algorithms/routing/unity_build/，"
        "目录内需要同时包含 .exe、*_Data 文件夹和 UnityPlayer.dll。\n"
        f"也可以设置环境变量 {UNITY_EXE_ENV}=完整 exe 路径。"
    )
    if incomplete:
        hint += "\n已发现但不完整的 Unity build:\n" + "\n".join(f"- {item}" for item in incomplete)
    if checked:
        hint += "\n已检查路径:\n" + checked
    raise FileNotFoundError(hint)


def _iter_unity_exe_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    for directory in UNITY_SEARCH_DIRS:
        for exe_name in PREFERRED_EXE_NAMES:
            add(directory / exe_name)
        if directory.exists():
            for exe_path in sorted(directory.glob("*.exe")):
                add(exe_path)
            for subdir in sorted(path for path in directory.iterdir() if path.is_dir()):
                for exe_name in PREFERRED_EXE_NAMES:
                    add(subdir / exe_name)
                for exe_path in sorted(subdir.glob("*.exe")):
                    add(exe_path)
    return candidates


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
