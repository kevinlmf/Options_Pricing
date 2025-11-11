import json
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np


def _serialize(value: Any):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if value is None:
        return None
    return value


def generate_training_report(history: Iterable, summary: Dict[str, Any], output_path: Path) -> None:
    """
    Persist training history and aggregate metrics to a JSON report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_payload = []
    for step in history:
        history_payload.append({
            "iteration": _serialize(step.iteration),
            "reward": _serialize(step.reward),
            "mean_error": _serialize(step.mean_error),
            "std_error": _serialize(step.std_error),
            "p_value": _serialize(step.p_value),
            "learning_rate": _serialize(step.learning_rate),
            "is_valid": bool(step.is_valid),
            "mean_within_ci": _serialize(step.mean_within_ci),
            "volatility_error": _serialize(step.volatility_error),
            "volatility_within_ci": _serialize(step.volatility_within_ci),
            "option_error": _serialize(getattr(step, "option_error", np.nan)),
            "option_within_ci": _serialize(getattr(step, "option_within_ci", None)),
            "tail_var_diff": _serialize(getattr(step, "tail_var_diff", np.nan)),
            "tail_cvar_diff": _serialize(getattr(step, "tail_cvar_diff", np.nan)),
        })

    summary_payload = {k: _serialize(v) for k, v in summary.items()}

    report = {
        "summary": summary_payload,
        "history": history_payload
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


