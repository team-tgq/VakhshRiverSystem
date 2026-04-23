from __future__ import annotations

from .daily_ml_pipeline import (
    ensure_model,
    load_existing_results,
    run_backfill,
    run_legacy_compatible_assessment,
    run_update_latest,
)


def run_swe_assessment():
    return run_legacy_compatible_assessment()


def run_update_latest_swe(force_retrain: bool = False):
    return run_update_latest(force_retrain=force_retrain)


def run_backfill_swe(days_back: int = 7, force_retrain: bool = False):
    return run_backfill(days_back=days_back, force_retrain=force_retrain)


def ensure_swe_model(force_retrain: bool = False):
    return ensure_model(force_retrain=force_retrain)

