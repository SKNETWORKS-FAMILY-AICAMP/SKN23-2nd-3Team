from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch
import re
from app.utils.paths import PATHS


def predict_proba_ml(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        return np.asarray(y_prob, dtype=float).reshape(-1)

    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
        return np.asarray(y_prob, dtype=float).reshape(-1)

    raise ValueError("ML model has neither predict_proba nor decision_function")


def predict_proba_dl(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor).view(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return np.asarray(probs, dtype=float).reshape(-1)


def load_score_percentiles(model_name: str) -> dict:
    metrics_dir = PATHS.get("models_metrics")
    if metrics_dir is None:
        project_root = PATHS.get("project_root")
        if project_root is None:
            raise KeyError("PATHS에 'models_metrics' 또는 'project_root'가 없습니다.")
        metrics_dir = Path(project_root) / "models" / "metrics"

    path = Path(metrics_dir) / f"{model_name}_score_percentiles.json"
    if not path.exists():
        raise FileNotFoundError(f"score_percentiles.json 없음: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def interpret_percentile_with_gap(y_prob: float, percentiles: list[dict]):
    rows = sorted(percentiles, key=lambda x: int(x["pct"]))
    for row in rows:
        pct = int(row["pct"])
        cutoff = float(row["score"])
        if float(y_prob) >= cutoff:
            return f"상위 {pct}% 이내", float(y_prob) - cutoff, pct, cutoff

    # 50% 컷도 못 넘으면
    if rows:
        last = rows[-1]
        return "상위 50% 밖", None, int(last["pct"]), float(last["score"])

    return "상위 50% 밖", None, None, None


def interpret_percentile(y_prob: float, percentiles: list[dict]) -> str:
    label, _, _, _ = interpret_percentile_with_gap(y_prob, percentiles)
    return label



def interpret_risk_level(percentile_label: str) -> str:
    """
    percentile_label 예:
      - "상위 1% 이내"
      - "상위 5% 이내"
      - "상위 50% 밖"
    """
    m = re.search(r"상위\s*(\d+)\s*%", percentile_label)
    if not m:
        return "🟢 낮음 (Low Risk)"

    pct = int(m.group(1))

    if pct <= 5:
        return "🔴 매우 높음 (High Risk)"
    elif pct <= 20:
        return "🟠 높음 (Medium-High)"
    elif pct <= 30:
        return "🟡 주의 (Medium)"
    else:
        return "🟢 낮음 (Low Risk)"