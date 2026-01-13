# app/utils/save.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix

from app.utils.paths import PATHS

N_DECIMALS = 5


def trunc_n(x: float, n: int = N_DECIMALS) -> float:
    """반올림 없이 n자리까지 절삭(숫자 유지)."""
    p = 10 ** n
    return math.trunc(float(x) * p) / p


def _safe_filename(name: str) -> str:
    name = str(name).strip().replace(" ", "_")
    for ch in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name


def _metrics_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_pct: int,
    base_rate: float,
) -> tuple[float, float, float, int, float]:
    n = len(y_prob)
    n_sel = int(np.floor(n * k_pct / 100))
    n_sel = max(n_sel, 1)

    order = np.argsort(-y_prob)
    sel = order[:n_sel]

    tp = int(y_true[sel].sum())
    precision_at_k = tp / n_sel
    recall_at_k = tp / max(int(y_true.sum()), 1)
    lift_at_k = (precision_at_k / base_rate) if base_rate > 0 else 0.0

    # cutoff (raw)
    t_k = float(y_prob[sel[-1]])
    return precision_at_k, recall_at_k, lift_at_k, n_sel, t_k


def save_model_and_artifacts(
    *,
    model: Any,
    model_name: str,
    model_type: str,   # "ml" | "dl"
    model_id: str,     # 예: "dl__mlp_enhance"  -> eval 폴더는 "dlmlp_enhance"
    split: str,        # 예: "test"
    metrics: dict,
    y_true,
    y_prob,
    version: str = "baseline",
    scaler=None,
    figures: dict | None = None,
    config: dict | None = None,
) -> dict[str, str]:
    assert model_type in {"ml", "dl"}, "model_type must be 'ml' or 'dl'"

    # ----------------------------
    # 경로(팀 규칙: 폴더로만 구분)
    # ----------------------------
    model_root = PATHS["models_dl"] if model_type == "dl" else PATHS["models_ml"]

    MODEL_DIR   = Path(model_root) / model_name / version
    PREP_DIR    = Path(PATHS["models_preprocessing"]) / model_name / version
    METRICS_DIR = Path(PATHS["models_metrics"]) / model_name / version
    FIG_DIR     = Path(PATHS["assets_training"]) / model_name / version
    CFG_DIR     = Path(PATHS["models_configs"]) / model_name / version

    # ✅ eval 폴더 규칙: models/eval/dlmlp_enhance
    eval_folder = model_id.replace("__", "")
    EVAL_DIR = Path(PATHS["models_eval"]) / eval_folder

    for d in [MODEL_DIR, PREP_DIR, METRICS_DIR, FIG_DIR, CFG_DIR, EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 입력 정리
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    saved: dict[str, str] = {}

    # ----------------------------
    # 1) 모델 저장
    # ----------------------------
    if model_type == "ml":
        model_path = MODEL_DIR / "model.pkl"
        joblib.dump(model, model_path)
    else:
        model_path = MODEL_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
    saved["model"] = str(model_path)

    # ----------------------------
    # 2) scaler 저장
    # ----------------------------
    if scaler is not None:
        scaler_path = PREP_DIR / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        saved["scaler"] = str(scaler_path)

    # ----------------------------
    # 3) metrics 저장(원본 그대로)
    # ----------------------------
    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    saved["metrics"] = str(metrics_path)

    # ----------------------------
    # 4) figures 저장(넘긴 것만)
    # ----------------------------
    if figures:
        for name, fig in figures.items():
            if fig is None or not hasattr(fig, "savefig"):
                continue
            safe = _safe_filename(name)
            img_path = FIG_DIR / f"{safe}.png"
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
            saved[f"figure_{safe}"] = str(img_path)

    # ----------------------------
    # 5) config 저장 (✅ feature_cols 저장 안 함)
    # ----------------------------
    config_payload = config or {
        "model_name": model_name,
        "model_type": model_type,
        "version": version,
        "feature_source": "features_ml_clean.parquet",
    }
    config_path = CFG_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)
    saved["config"] = str(config_path)

    # ============================
    # ✅ EVAL 산출물 저장 (팀 규칙)
    # ============================

    # 1) model_card.json
    model_card = {
        "model_id": model_id,
        "display_name": f"{model_name} ({model_type.upper()})",
        "category": model_type.upper(),
        "split": split,
        "version": version,
    }
    with open(EVAL_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    # 2) pr_metrics.json
    pr_auc = metrics.get("PR-AUC (Average Precision)")
    if pr_auc is None:
        pr_auc = float(average_precision_score(y_true, y_prob))
    pr_auc = trunc_n(pr_auc)

    pr_metrics = {"model_id": model_id, "split": split, "pr_auc": pr_auc}
    with open(EVAL_DIR / "pr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pr_metrics, f, indent=2, ensure_ascii=False)

    # 3) topk_metrics.json + topk_cutoffs.json
    base_rate = float(y_true.mean())
    base_rate_s = trunc_n(base_rate)

    k_list = [5, 10, 15, 30]

    topk_metrics = {
        "model_id": model_id,
        "split": split,
        "base_rate": base_rate_s,
        "metrics_by_k": [],
    }
    topk_cutoffs = {
        "model_id": model_id,
        "split": split,
        "n_total": int(len(y_prob)),
        "n_selected_rule": "floor",
        "tie_policy": "sort_and_take_top_n",
        "cutoffs_by_k": [],
    }

    cutoffs_raw: list[float] = []
    for k in k_list:
        p_at_k, r_at_k, lift, n_sel, t_k_raw = _metrics_at_k(y_true, y_prob, k, base_rate)

        topk_metrics["metrics_by_k"].append({
            "k_pct": int(k),
            "precision_at_k": trunc_n(p_at_k),
            "recall_at_k": trunc_n(r_at_k),
            "lift_at_k": trunc_n(lift),
        })
        topk_cutoffs["cutoffs_by_k"].append({
            "k_pct": int(k),
            "n_selected": int(n_sel),
            "t_k": trunc_n(t_k_raw),  # 저장은 절삭
        })
        cutoffs_raw.append(t_k_raw)

    with open(EVAL_DIR / "topk_metrics.json", "w", encoding="utf-8") as f:
        json.dump(topk_metrics, f, indent=2, ensure_ascii=False)

    with open(EVAL_DIR / "topk_cutoffs.json", "w", encoding="utf-8") as f:
        json.dump(topk_cutoffs, f, indent=2, ensure_ascii=False)

    # 4) confusion_matrix.json (Top 5% 기준)
    thr_raw = float(cutoffs_raw[0])  # 예측은 raw 기준
    y_pred_5 = (y_prob >= thr_raw).astype(int)
    cm = confusion_matrix(y_true, y_pred_5)

    confusion_payload = {
        "model_id": model_id,
        "split": split,
        "threshold": trunc_n(thr_raw),
        "labels": ["non_m2", "m2"],
        "matrix": cm.tolist(),
    }
    with open(EVAL_DIR / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(confusion_payload, f, indent=2, ensure_ascii=False)

    saved["eval_dir"] = str(EVAL_DIR)
    return saved