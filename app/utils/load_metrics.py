from __future__ import annotations

import json
from pathlib import Path

from app.utils.paths import DEFAULT_PATHS as P


def _read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _model_eval_dir_name(model_type: str, model_name: str) -> str:
    t = (model_type or "").strip().lower()
    if t not in {"ml", "dl"}:
        raise ValueError(f"model_type must be 'ml' or 'dl'. got: {model_type}")
    return f"{t}{model_name}"


def resolve_topk_cutoffs_path(model_type: str, model_name: str, version: str | None) -> Path:
    """
    topk_cutoffs.json 위치를 해석한다.

    기본 규칙
    - models/eval/{mlhgb}/topk_cutoffs.json 같은 구조를 우선 사용한다

    확장 규칙
    - 버전별로 저장했을 가능성도 대비해 {eval_dir}/{version}/topk_cutoffs.json 후보도 같이 본다
    """
    eval_dir = P.models_eval_dir / _model_eval_dir_name(model_type, model_name)

    v = (version or "").strip()
    candidates: list[Path] = []

    if v:
        candidates.append(eval_dir / v / "topk_cutoffs.json")

    candidates.append(eval_dir / "topk_cutoffs.json")

    for p in candidates:
        if p.exists():
            return p

    return candidates[-1]


def load_topk_cutoffs(model_type: str, model_name: str, version: str | None) -> dict[int, float]:
    """
    topk_cutoffs.json을 읽어 {k_pct: threshold} 형태로 반환한다.

    허용 포맷
    - {"cutoffs_by_k": [{"k_pct": 5, "t_k": 0.94}, ...]}
    - {"5": 0.94, "10": 0.93, ...} 같은 단순 dict도 허용한다
    """
    path = resolve_topk_cutoffs_path(model_type, model_name, version)
    payload = _read_json(path)
    if payload is None:
        return {}

    if isinstance(payload, dict) and "cutoffs_by_k" in payload:
        rows = payload.get("cutoffs_by_k", [])
        out: dict[int, float] = {}
        for row in rows:
            out[int(row["k_pct"])] = float(row["t_k"])
        return out

    if isinstance(payload, dict):
        out2: dict[int, float] = {}
        for k, v in payload.items():
            try:
                out2[int(k)] = float(v)
            except Exception:
                continue
        return out2

    return {}


def resolve_score_percentiles_path(model_name: str, version: str | None) -> Path:
    """
    score_percentiles.json 위치를 해석한다.

    기본 규칙
    - models/metrics/{model_name}_score_percentiles.json

    확장 규칙
    - models/metrics/{model_name}_{version}_score_percentiles.json 후보도 같이 본다
    """
    v = (version or "").strip()
    candidates: list[Path] = []

    if v:
        candidates.append(P.models_metrics_dir / f"{model_name}_{v}_score_percentiles.json")

    candidates.append(P.models_metrics_dir / f"{model_name}_score_percentiles.json")

    for p in candidates:
        if p.exists():
            return p

    return candidates[-1]


def load_score_percentiles(model_name: str, version: str | None) -> list[dict] | None:
    """
    score_percentiles.json을 읽어 percentiles 리스트를 반환한다.

    허용 포맷
    - {"percentiles": [{"pct": 5, "score": 0.94}, ...]}
    - [{"pct": 5, "score": 0.94}, ...] 같은 리스트도 허용한다
    """
    path = resolve_score_percentiles_path(model_name, version)
    payload = _read_json(path)
    if payload is None:
        return None

    if isinstance(payload, dict) and "percentiles" in payload:
        val = payload.get("percentiles")
        if isinstance(val, list):
            return val
        return None

    if isinstance(payload, list):
        return payload

    return None