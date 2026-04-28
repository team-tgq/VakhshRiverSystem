from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from .daily_ml_pipeline import (
    _describe_h5py_runtime,
    ensure_model as _ensure_model_local,
    load_existing_results as _load_existing_results_local,
    progress_reporter as _progress_reporter_local,
    run_backfill as _run_backfill_local,
    run_legacy_compatible_assessment as _run_legacy_compatible_assessment_local,
    run_update_latest as _run_update_latest_local,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ProgressCallback = Callable[[str], None]
_NO_RESULT = object()


def _worker_runtime_info_local() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "h5py_detail": _describe_h5py_runtime(),
    }


def _model_bundle_summary_local(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "model"}


def _run_worker_action_local(action: str, kwargs: dict[str, Any]) -> Any:
    action_map = {
        "run_legacy_compatible_assessment": lambda payload: _run_legacy_compatible_assessment_local(),
        "run_update_latest": lambda payload: _run_update_latest_local(
            force_retrain=bool(payload.get("force_retrain", False))
        ),
        "run_backfill": lambda payload: _run_backfill_local(
            days_back=int(payload.get("days_back", 7)),
            force_retrain=bool(payload.get("force_retrain", False)),
        ),
        "ensure_model": lambda payload: _model_bundle_summary_local(
            _ensure_model_local(force_retrain=bool(payload.get("force_retrain", False)))
        ),
        "load_existing_results": lambda payload: _load_existing_results_local(),
        "runtime_info": lambda payload: _worker_runtime_info_local(),
    }
    if action not in action_map:
        raise ValueError(f"Unsupported SWE worker action: {action}")
    return action_map[action](kwargs)


def _format_worker_failure(
    action: str,
    *,
    stderr_text: str,
    error_response: dict[str, Any] | None = None,
    raw_stdout_lines: list[str] | None = None,
) -> str:
    detail_lines = [
        "\u96ea\u6c34\u5f53\u91cf\u4f30\u7b97\u72ec\u7acb\u8ba1\u7b97\u8fdb\u7a0b\u6267\u884c\u5931\u8d25\u3002",
        f"\u4efb\u52a1: {action}",
    ]
    if error_response is not None and error_response.get("error"):
        detail_lines.append(str(error_response["error"]).strip())
    if stderr_text:
        detail_lines.append(f"stderr: {stderr_text}")
    if raw_stdout_lines:
        detail_lines.append(f"stdout: {' | '.join(raw_stdout_lines)}")
    return "\n".join(detail_lines)


def _notify_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is None:
        return
    text = str(message).strip()
    if not text:
        return
    try:
        progress_callback(text)
    except Exception:
        return


def _collect_stream(stream, bucket: list[str]) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            bucket.append(line)
    finally:
        stream.close()


def _run_in_clean_process(
    action: str,
    *,
    progress_callback: ProgressCallback | None = None,
    **kwargs: Any,
) -> Any:
    request_payload = {"action": action, "kwargs": kwargs}
    process = subprocess.Popen(
        [sys.executable, "-m", "algorithms.swe.swe_assessment", "--worker"],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(request_payload, ensure_ascii=False))
    process.stdin.close()

    stderr_lines: list[str] = []
    stderr_thread = threading.Thread(
        target=_collect_stream,
        args=(process.stderr, stderr_lines),
        daemon=True,
    )
    stderr_thread.start()

    raw_stdout_lines: list[str] = []
    result: Any = _NO_RESULT
    error_response: dict[str, Any] | None = None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            raw_stdout_lines.append(line)
            continue

        event_name = str(response.get("event", "")).strip().lower()
        if event_name == "progress":
            _notify_progress(progress_callback, str(response.get("message", "")))
            continue
        if event_name == "result":
            result = response.get("result")
            continue
        if event_name == "error":
            error_response = response
            continue
        raw_stdout_lines.append(line)

    process.stdout.close()
    return_code = process.wait()
    stderr_thread.join(timeout=5)
    stderr_text = "".join(stderr_lines).strip()

    if return_code != 0:
        raise RuntimeError(
            _format_worker_failure(
                action,
                stderr_text=stderr_text,
                error_response=error_response,
                raw_stdout_lines=raw_stdout_lines,
            )
        )

    if error_response is not None:
        raise RuntimeError(
            _format_worker_failure(
                action,
                stderr_text=stderr_text,
                error_response=error_response,
                raw_stdout_lines=raw_stdout_lines,
            )
        )

    if result is _NO_RESULT:
        raise RuntimeError(
            _format_worker_failure(
                action,
                stderr_text=stderr_text,
                error_response=None,
                raw_stdout_lines=raw_stdout_lines,
            )
        )

    return result


def run_swe_assessment(progress_callback: ProgressCallback | None = None):
    return _run_in_clean_process("run_legacy_compatible_assessment", progress_callback=progress_callback)


def run_update_latest_swe(
    force_retrain: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _run_in_clean_process(
        "run_update_latest",
        force_retrain=bool(force_retrain),
        progress_callback=progress_callback,
    )


def run_backfill_swe(
    days_back: int = 7,
    force_retrain: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _run_in_clean_process(
        "run_backfill",
        days_back=int(days_back),
        force_retrain=bool(force_retrain),
        progress_callback=progress_callback,
    )


def ensure_swe_model(
    force_retrain: bool = False,
    progress_callback: ProgressCallback | None = None,
):
    return _run_in_clean_process(
        "ensure_model",
        force_retrain=bool(force_retrain),
        progress_callback=progress_callback,
    )


def get_worker_runtime_info() -> dict[str, Any]:
    return _run_in_clean_process("runtime_info")


def _emit_worker_event(event: str, *, stream, **payload: Any) -> None:
    data = {"event": event, **payload}
    stream.write(json.dumps(data, ensure_ascii=False) + "\n")
    stream.flush()


def _worker_main() -> int:
    request_text = sys.stdin.read().strip()
    worker_stdout = sys.stdout
    try:
        payload = json.loads(request_text) if request_text else {}
        action = str(payload.get("action", "")).strip()
        kwargs = payload.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise TypeError("SWE worker kwargs must be a JSON object.")

        def _worker_progress(progress_payload: dict[str, Any]) -> None:
            _emit_worker_event("progress", stream=worker_stdout, **progress_payload)

        with _progress_reporter_local(_worker_progress), contextlib.redirect_stdout(sys.stderr):
            result = _run_worker_action_local(action, kwargs)
        _emit_worker_event("result", stream=worker_stdout, ok=True, result=result)
        exit_code = 0
    except Exception as exc:
        _emit_worker_event(
            "error",
            stream=worker_stdout,
            ok=False,
            error=str(exc) or repr(exc),
            traceback=traceback.format_exc(),
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        raise SystemExit(_worker_main())
