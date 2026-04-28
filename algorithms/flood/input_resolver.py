from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Dict, Iterable, Optional

import geopandas as gpd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TOKEN_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")

DYNAMIC_KEYWORDS = {
    "rain": ("rain", "precip", "tp"),
    "soil_moist": ("soil", "swvl1"),
}

LEGACY_DYNAMIC_NAMES = {
    "rain": "rain_mm_demgrid.tif",
    "soil_moist": "soil_moist_demgrid.tif",
}


def parse_target_date(value: Optional[object]) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y_%m_%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"无法解析日期: {value!r}")


def _dedupe_paths(paths: Iterable[Optional[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
    return result


def _first_existing(paths: Iterable[Optional[str]]) -> Optional[str]:
    for path in _dedupe_paths(paths):
        if os.path.exists(path):
            return path
    return None


def _extract_date_from_name(file_name: str) -> Optional[date]:
    match = DATE_TOKEN_RE.search(file_name)
    if match is None:
        return None

    year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _score_dynamic_candidate(path: str) -> tuple[int, int]:
    lower = os.path.basename(path).lower()
    score = 0
    if "demgrid" in lower:
        score += 4
    if "daily" in lower:
        score += 2
    if "processed" in path.lower():
        score += 1
    return score, -len(path)


def _discover_daily_candidates(proc_dir: str, kind: str) -> Dict[date, str]:
    if not os.path.isdir(proc_dir):
        return {}

    keywords = DYNAMIC_KEYWORDS[kind]
    candidates: Dict[date, str] = {}

    for root, _, files in os.walk(proc_dir):
        for file_name in files:
            lower = file_name.lower()
            if not lower.endswith((".tif", ".tiff")):
                continue
            if not any(keyword in lower for keyword in keywords):
                continue

            data_date = _extract_date_from_name(lower)
            if data_date is None:
                continue

            path = os.path.join(root, file_name)
            previous = candidates.get(data_date)
            if previous is None or _score_dynamic_candidate(path) > _score_dynamic_candidate(previous):
                candidates[data_date] = path

    return candidates


def _format_date_list(dates: Iterable[date], limit: int = 8) -> str:
    ordered = sorted(dates)
    if not ordered:
        return "无"
    if len(ordered) > limit:
        ordered = ordered[-limit:]
    return ", ".join(item.isoformat() for item in ordered)


def _static_candidates(cfg: dict) -> dict[str, list[str]]:
    proc_dir = cfg["proc_dir"]
    raw_dir = cfg["raw_dir"]
    data_dir = os.path.join(BASE_DIR, "data")

    return {
        "study_area_shp": _dedupe_paths(
            [
                cfg.get("study_area_shp"),
                os.path.join(BASE_DIR, "study_area.shp"),
            ]
        ),
        "dem_path": _dedupe_paths(
            [
                os.path.join(proc_dir, "dem_clip.tif"),
                os.path.join(data_dir, "processed", "dem_clip.tif"),
                os.path.join(BASE_DIR, "dem_clip.tif"),
            ]
        ),
        "landcover_path": _dedupe_paths(
            [
                os.path.join(proc_dir, "landcover_demgrid.tif"),
                os.path.join(data_dir, "processed", "landcover_demgrid.tif"),
                os.path.join(BASE_DIR, "landcover_demgrid.tif"),
            ]
        ),
        "rivers_path": _dedupe_paths(
            [
                os.path.join(raw_dir, "hydrorivers.gpkg"),
                os.path.join(data_dir, "raw", "hydrorivers.gpkg"),
                os.path.join(BASE_DIR, "hydrorivers.gpkg"),
            ]
        ),
    }


def _dem_source_candidates(cfg: dict) -> list[str]:
    raw_dir = cfg["raw_dir"]
    data_dir = os.path.join(BASE_DIR, "data")
    return _dedupe_paths(
        [
            cfg.get("dem_tif"),
            os.path.join(BASE_DIR, "dem.tif"),
            os.path.join(raw_dir, "dem.tif"),
            os.path.join(data_dir, "raw", "dem.tif"),
        ]
    )


def _prepare_missing_static_inputs(cfg: dict, study_area_path: str) -> list[str]:
    try:
        try:
            from . import download_data
        except ImportError:
            import download_data  # type: ignore
    except Exception as exc:
        raise RuntimeError("静态数据缺失，且无法加载自动准备脚本。") from exc

    proc_dir = cfg["proc_dir"]
    raw_dir = cfg["raw_dir"]
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    actions: list[str] = []
    study = gpd.read_file(study_area_path).to_crs(4326)

    dem_path = os.path.join(proc_dir, "dem_clip.tif")
    if not os.path.exists(dem_path):
        dem_source = _first_existing(_dem_source_candidates(cfg))
        if dem_source:
            download_data.clip_to_study_area(dem_source, study, dem_path)
            actions.append(f"生成 DEM 裁剪数据: {dem_path}")

    if os.path.exists(dem_path):
        landcover_path = os.path.join(proc_dir, "landcover_demgrid.tif")
        if not os.path.exists(landcover_path):
            download_data.download_worldcover_to_demgrid(dem_path, landcover_path)
            actions.append(f"生成土地覆盖数据: {landcover_path}")

    rivers_path = os.path.join(raw_dir, "hydrorivers.gpkg")
    if not os.path.exists(rivers_path):
        download_data.download_hydrorivers(study, rivers_path)
        actions.append(f"生成河网数据: {rivers_path}")

    return actions


def resolve_static_inputs(cfg: dict, auto_prepare: bool = True) -> dict:
    candidates = _static_candidates(cfg)
    study_area_path = _first_existing(candidates["study_area_shp"])
    if not study_area_path:
        raise FileNotFoundError("未找到研究区边界 study_area.shp，无法运行洪涝风险评估。")

    actions: list[str] = []
    resolved = {
        "study_area_shp": study_area_path,
        "dem_path": _first_existing(candidates["dem_path"]),
        "landcover_path": _first_existing(candidates["landcover_path"]),
        "rivers_path": _first_existing(candidates["rivers_path"]),
    }

    if auto_prepare and any(value is None for key, value in resolved.items() if key != "study_area_shp"):
        actions = _prepare_missing_static_inputs(cfg, study_area_path)
        candidates = _static_candidates(cfg)
        resolved = {
            "study_area_shp": study_area_path,
            "dem_path": _first_existing(candidates["dem_path"]),
            "landcover_path": _first_existing(candidates["landcover_path"]),
            "rivers_path": _first_existing(candidates["rivers_path"]),
        }

    missing = [key for key, value in resolved.items() if value is None]
    if missing:
        detail_map = {
            "dem_path": "DEM 裁剪栅格 dem_clip.tif",
            "landcover_path": "土地覆盖栅格 landcover_demgrid.tif",
            "rivers_path": "河网矢量 hydrorivers.gpkg",
        }
        missing_text = "、".join(detail_map[item] for item in missing)
        dem_hint = _first_existing(_dem_source_candidates(cfg))
        if "dem_path" in missing and dem_hint is None:
            missing_text += "。提示：请先放入原始 DEM 文件 dem.tif，系统才能自动准备静态数据。"
        raise FileNotFoundError(f"静态地理数据缺失: {missing_text}")

    resolved["static_actions"] = actions
    return resolved


def _prepare_missing_dynamic_inputs(cfg: dict, target_date: Optional[date]) -> list[str]:
    try:
        try:
            from . import download_data
        except ImportError:
            import download_data  # type: ignore
    except Exception as exc:
        raise RuntimeError("逐日气象数据缺失，且无法加载自动下载模块。") from exc

    try:
        result = download_data.download_gfs_daily_inputs(target_date=target_date, cfg=cfg)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError("系统尝试自动获取近实时逐日气象数据时失败，请稍后重试或检查网络连接。") from exc

    return result.get("actions", [])


def _record_dynamic_prepare_failure(actions: list[str], target_date: Optional[date], exc: Exception) -> None:
    date_text = target_date.isoformat() if target_date else "最近可用日期"
    detail = str(exc).strip() or repr(exc)
    actions.append(
        f"目标日期 {date_text} 的近实时逐日气象数据暂时没有获取成功，"
        f"系统已继续查找本地最近可用的逐日数据。原因：{detail}"
    )


def _select_fallback_date(available_dates: list[date], requested_date: date) -> Optional[date]:
    if not available_dates:
        return None
    not_later_dates = [item for item in available_dates if item <= requested_date]
    if not_later_dates:
        return not_later_dates[-1]
    return available_dates[-1]


def resolve_dynamic_inputs(
    cfg: dict,
    target_date: Optional[object] = None,
    allow_legacy_dynamic: bool = True,
    auto_prepare: bool = True,
) -> dict:
    proc_dir = cfg["proc_dir"]
    requested_date = parse_target_date(target_date)
    legacy_paths = {
        key: os.path.join(proc_dir, file_name)
        for key, file_name in LEGACY_DYNAMIC_NAMES.items()
    }
    legacy_available = all(os.path.exists(path) for path in legacy_paths.values())

    dynamic_actions: list[str] = []

    def _scan_daily_inputs():
        rain = _discover_daily_candidates(proc_dir, "rain")
        soil = _discover_daily_candidates(proc_dir, "soil_moist")
        dates = sorted(set(rain).intersection(soil))
        return rain, soil, dates

    rain_candidates, soil_candidates, available_dates = _scan_daily_inputs()

    if requested_date is not None and requested_date not in available_dates and auto_prepare:
        try:
            dynamic_actions.extend(_prepare_missing_dynamic_inputs(cfg, requested_date))
        except Exception as exc:
            _record_dynamic_prepare_failure(dynamic_actions, requested_date, exc)
        rain_candidates, soil_candidates, available_dates = _scan_daily_inputs()
    elif requested_date is None and not available_dates and auto_prepare:
        try:
            dynamic_actions.extend(_prepare_missing_dynamic_inputs(cfg, None))
        except Exception as exc:
            _record_dynamic_prepare_failure(dynamic_actions, None, exc)
        rain_candidates, soil_candidates, available_dates = _scan_daily_inputs()

    if requested_date is not None:
        if requested_date not in available_dates:
            fallback_date = _select_fallback_date(available_dates, requested_date)
            if fallback_date is not None:
                dynamic_actions.append(
                    f"目标日期 {requested_date.isoformat()} 没有完整逐日气象数据，"
                    f"已自动改用最近可用日期 {fallback_date.isoformat()} 运行。"
                )
                return {
                    "rain_path": rain_candidates[fallback_date],
                    "soil_path": soil_candidates[fallback_date],
                    "requested_target_date": requested_date.isoformat(),
                    "resolved_target_date": fallback_date.isoformat(),
                    "dynamic_scale": "daily",
                    "available_dynamic_dates": [item.isoformat() for item in available_dates],
                    "dynamic_actions": dynamic_actions,
                }
            if not available_dates and legacy_available:
                raise FileNotFoundError(
                    f"No daily dynamic inputs found for {requested_date.isoformat()}. "
                    "Only legacy monthly files are available, and monthly fallback is disabled "
                    "when a specific date is selected."
                )
            available_text = _format_date_list(available_dates)
            raise FileNotFoundError(
                f"未找到 {requested_date.isoformat()} 的逐日气象输入。"
                f"当前同时具备降雨和土壤湿度的日期有: {available_text}"
            )

        return {
            "rain_path": rain_candidates[requested_date],
            "soil_path": soil_candidates[requested_date],
            "requested_target_date": requested_date.isoformat(),
            "resolved_target_date": requested_date.isoformat(),
            "dynamic_scale": "daily",
            "available_dynamic_dates": [item.isoformat() for item in available_dates],
            "dynamic_actions": dynamic_actions,
        }

    if available_dates:
        resolved_date = available_dates[-1]
        return {
            "rain_path": rain_candidates[resolved_date],
            "soil_path": soil_candidates[resolved_date],
            "requested_target_date": None,
            "resolved_target_date": resolved_date.isoformat(),
            "dynamic_scale": "daily",
            "available_dynamic_dates": [item.isoformat() for item in available_dates],
            "dynamic_actions": dynamic_actions,
        }

    if allow_legacy_dynamic and legacy_available:
        return {
            "rain_path": legacy_paths["rain"],
            "soil_path": legacy_paths["soil_moist"],
            "requested_target_date": requested_date.isoformat() if requested_date else None,
            "resolved_target_date": None,
            "dynamic_scale": "legacy-monthly",
            "available_dynamic_dates": [],
            "dynamic_actions": dynamic_actions,
        }

    raise FileNotFoundError(
        "未找到逐日气象输入数据。请在 data/processed 目录下提供带日期的降雨和土壤湿度栅格，"
        "例如 rain_mm_demgrid_2024-07-15.tif 和 soil_moist_demgrid_2024-07-15.tif。"
    )


def resolve_flood_input_paths(
    cfg: dict,
    target_date: Optional[object] = None,
    auto_prepare_static: bool = True,
    allow_legacy_dynamic: bool = True,
    auto_prepare_dynamic: bool = True,
) -> dict:
    resolved = {}
    resolved.update(resolve_static_inputs(cfg, auto_prepare=auto_prepare_static))
    resolved.update(
        resolve_dynamic_inputs(
            cfg,
            target_date=target_date,
            allow_legacy_dynamic=allow_legacy_dynamic,
            auto_prepare=auto_prepare_dynamic,
        )
    )
    return resolved
