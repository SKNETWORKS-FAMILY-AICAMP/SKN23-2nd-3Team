# app/utils/save.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

from app.utils.paths import DEFAULT_PATHS as P

N_DECIMALS = 5


def trunc_n(x: float, n: int = N_DECIMALS) -> float:
    """
    반올림 없이 n자리까지 절삭한다.
    저장되는 cutoffs, 지표를 사람이 읽기 좋게 고정 자릿수로 맞출 때 사용한다.
    """
    p = 10**n
    return math.trunc(float(x) * p) / p


def _safe_filename(name: str) -> str:
    """
    파일명으로 쓰기 위험한 문자를 치환한다.
    figures 저장 시 파일명 안전성을 확보하는 용도다.
    """
    s = str(name).strip().replace(" ", "_")
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s


def _metrics_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_pct: int,
    base_rate: float,
) -> tuple[float, float, float, int, float]:
    """
    Top-K 퍼센트 구간의 precision, recall, lift, 선택개수, cutoff score를 계산한다.
    cutoff score는 Top-K의 마지막 샘플 점수(정렬 후 경계값)다.
    """
    n = int(len(y_prob))
    n_sel = int(np.floor(n * (k_pct / 100.0)))
    n_sel = max(n_sel, 1)

    order = np.argsort(-y_prob)
    sel = order[:n_sel]

    tp = int(y_true[sel].sum())
    precision_at_k = float(tp / n_sel)
    recall_at_k = float(tp / max(int(y_true.sum()), 1))
    lift_at_k = float((precision_at_k / base_rate) if base_rate > 0 else 0.0)

    t_k = float(y_prob[sel[-1]])
    return precision_at_k, recall_at_k, lift_at_k, n_sel, t_k


def _normalize_version_for_save(model_root_for_name: Path, version: str | None) -> str:
    """
    저장할 버전 폴더명을 정한다.
    version이 비어있거나 baseline이면 기존 구조가 v1_baseline을 쓰는 경우 그쪽을 우선한다.
    """
    v = (version or "").strip() or "baseline"

    if v == "baseline":
        if (model_root_for_name / "v1_baseline").exists():
            return "v1_baseline"
        return "baseline"

    return v


def save_model_bundle(
    *,
    model_type: str,
    model_name: str,
    version: str | None,
    model: Any,
    scaler: Any | None,
    config: dict[str, Any] | None,
    eval_folder: str | None,
) -> dict[str, str]:
    """
    모델, 스케일러, config, eval_dir 경로를 프로젝트 규칙에 맞게 저장한다.

    저장 위치
    - ML 모델: models/ml/{model_name}/{version_dir}/model.pkl
    - DL 모델: models/dl/{model_name}/{version_dir}/model.pt
    - Scaler : models/preprocessing/{model_name}/{version_dir}/scaler.pkl
    - Config : models/configs/{model_name}/{version_dir}/config.json
    - Eval   : models/eval/{eval_folder}/
    """
    if model_type not in {"ml", "dl"}:
        raise ValueError(f"unknown model_type: {model_type}")

    model_root = P.models_dl_dir if model_type == "dl" else P.models_ml_dir
    base_dir = model_root / model_name
    version_dir = _normalize_version_for_save(base_dir, version)

    out_model_dir = model_root / model_name / version_dir
    out_prep_dir = P.models_preprocessing_dir / model_name / version_dir
    out_cfg_dir = P.models_configs_dir / model_name / version_dir

    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_prep_dir.mkdir(parents=True, exist_ok=True)
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    if model_type == "ml":
        model_path = out_model_dir / "model.pkl"
        joblib.dump(model, model_path)
        saved["model"] = str(model_path)
    else:
        import torch

        model_path = out_model_dir / "model.pt"
        state = model.state_dict() if hasattr(model, "state_dict") else model
        torch.save(state, model_path)
        saved["model"] = str(model_path)

    if scaler is not None:
        scaler_path = out_prep_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        saved["scaler"] = str(scaler_path)

    if config is not None:
        cfg_path = out_cfg_dir / "config.json"
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        saved["config"] = str(cfg_path)

    if eval_folder:
        out_eval_dir = P.models_eval_dir / eval_folder
        out_eval_dir.mkdir(parents=True, exist_ok=True)
        saved["eval_dir"] = str(out_eval_dir)

    saved["version_dir"] = version_dir
    return saved


def _save_eval_artifacts(
    *,
    out_eval_dir: Path,
    model_id: str,
    split: str,
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """
    팀 공통 eval 산출물을 models/eval/{eval_folder}/ 아래에 저장한다.
    """
    out_eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_eval_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    pr_auc = metrics.get("PR-AUC (Average Precision)")
    if pr_auc is None:
        pr_auc = float(average_precision_score(y_true, y_prob))
    pr_auc = trunc_n(float(pr_auc))

    pr_metrics = {"model_id": model_id, "split": split, "pr_auc": pr_auc}
    (out_eval_dir / "pr_metrics.json").write_text(
        json.dumps(pr_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_rate = float(y_true.mean())
    k_list = [5, 10, 15, 30]

    topk_metrics = {
        "model_id": model_id,
        "split": split,
        "base_rate": trunc_n(base_rate),
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
        p_at_k, r_at_k, lift, n_sel, t_k_raw = _metrics_at_k(y_true, y_prob, int(k), base_rate)

        topk_metrics["metrics_by_k"].append(
            {
                "k_pct": int(k),
                "precision_at_k": trunc_n(p_at_k),
                "recall_at_k": trunc_n(r_at_k),
                "lift_at_k": trunc_n(lift),
            }
        )
        topk_cutoffs["cutoffs_by_k"].append(
            {
                "k_pct": int(k),
                "n_selected": int(n_sel),
                "t_k": trunc_n(t_k_raw),
            }
        )
        cutoffs_raw.append(float(t_k_raw))

    (out_eval_dir / "topk_metrics.json").write_text(
        json.dumps(topk_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_eval_dir / "topk_cutoffs.json").write_text(
        json.dumps(topk_cutoffs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    thr_raw = float(cutoffs_raw[0]) if cutoffs_raw else 1.0
    y_pred = (y_prob >= thr_raw).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    confusion_payload = {
        "model_id": model_id,
        "split": split,
        "threshold": trunc_n(thr_raw),
        "labels": ["non_m2", "m2"],
        "matrix": cm.tolist(),
    }
    (out_eval_dir / "confusion_matrix.json").write_text(
        json.dumps(confusion_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_figures(
    *,
    model_name: str,
    version_dir: str,
    figures: dict[str, Any] | None,
) -> dict[str, str]:
    """
    matplotlib figure 객체들을 assets/training/{model_name}/{version_dir}/ 아래에 저장한다.
    """
    if not figures:
        return {}

    out_dir = P.assets_training_dir / model_name / version_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    for name, fig in figures.items():
        if fig is None or not hasattr(fig, "savefig"):
            continue
        safe = _safe_filename(name)
        path = out_dir / f"{safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved[f"figure_{safe}"] = str(path)

    return saved


def save_model_and_artifacts(
    *,
    model: Any,
    model_name: str,
    model_type: str,
    model_id: str,
    split: str,
    metrics: dict[str, Any],
    y_true,
    y_prob,
    version: str = "baseline",
    scaler: Any | None = None,
    figures: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    학습 결과물을 프로젝트 규칙에 맞게 저장한다.
    호출부는 y_true, y_prob를 반드시 넘겨서 eval 산출물이 같이 생성되도록 한다.
    """
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)

    eval_folder = model_id.replace("__", "")
    saved = save_model_bundle(
        model_type=model_type,
        model_name=model_name,
        version=version,
        model=model,
        scaler=scaler,
        config=config,
        eval_folder=eval_folder,
    )

    out_eval_dir = Path(saved["eval_dir"])
    _save_eval_artifacts(
        out_eval_dir=out_eval_dir,
        model_id=model_id,
        split=split,
        metrics=metrics,
        y_true=y_true_arr,
        y_prob=y_prob_arr,
    )

    fig_saved = _save_figures(model_name=model_name, version_dir=saved["version_dir"], figures=figures)
    saved.update(fig_saved)

    return saved


if __name__ == "__main__":
    root = P.root
    print(f"root: {root}  exists={root.exists()}")
    print(f"models_ml_dir: {P.models_ml_dir}  exists={P.models_ml_dir.exists()}")
    print(f"models_dl_dir: {P.models_dl_dir}  exists={P.models_dl_dir.exists()}")
    print(f"models_eval_dir: {P.models_eval_dir}  exists={P.models_eval_dir.exists()}")
    print(f"assets_training_dir: {P.assets_training_dir}  exists={P.assets_training_dir.exists()}")