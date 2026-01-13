from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.utils.paths import DEFAULT_PATHS as P


def _list_version_dirs(base_dir: Path) -> list[str]:
    """
    주어진 base_dir 아래의 버전 폴더 목록을 반환한다.
    load_ml_model, load_dl_model, load_scaler, load_config에서 공통으로 사용한다.
    """
    if not base_dir.exists():
        return []
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])


def _resolve_version_dir(base_dir: Path, version: str | None) -> str:
    """
    version 입력을 실제 존재하는 버전 폴더명으로 해석한다.
    version이 None 또는 'baseline'이면 v1_baseline 또는 baseline 유사 폴더를 자동 선택한다.
    선택 실패 시 사용 가능한 버전 목록을 포함해 예외를 발생시킨다.
    """
    candidates = _list_version_dirs(base_dir)
    if not candidates:
        raise FileNotFoundError(f"no version dirs under: {base_dir}")

    v = (version or "").strip()

    if v and (base_dir / v).exists():
        return v

    if (not v) or (v == "baseline"):
        if (base_dir / "v1_baseline").exists():
            return "v1_baseline"

        baseline_like = [x for x in candidates if "baseline" in x]
        if len(baseline_like) == 1:
            return baseline_like[0]

        if len(candidates) == 1:
            return candidates[0]

        raise FileNotFoundError(
            "cannot resolve version\n"
            f"requested: {version}\n"
            f"available: {candidates}"
        )

    raise FileNotFoundError(
        "version dir not found\n"
        f"requested: {version}\n"
        f"available: {candidates}"
    )


def load_ml_model(model_name: str, version: str | None):
    """
    scikit-learn 계열 ML 모델을 로드한다.
    Streamlit 실시간 추론, 오프라인 평가 스크립트에서 공통으로 사용한다.
    경로 규칙은 models/ml/{model_name}/{version}/model.pkl 이다.
    """
    base_dir = P.models_ml_dir / model_name
    vdir = _resolve_version_dir(base_dir, version)
    model_path = base_dir / vdir / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"ML model file not found: {model_path}")

    return joblib.load(model_path)


def load_scaler(model_name: str, version: str | None):
    """
    전처리 스케일러(StandardScaler 등)를 로드한다.
    Streamlit 실시간 추론에서 입력 피처를 학습 시점과 동일한 스케일로 맞추는 용도다.
    경로 규칙은 models/preprocessing/{model_name}/{version}/scaler.pkl 이다.
    """
    base_dir = P.models_preprocessing_dir / model_name
    vdir = _resolve_version_dir(base_dir, version)
    scaler_path = base_dir / vdir / "scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler file not found: {scaler_path}")

    return joblib.load(scaler_path)


def load_config(model_name: str, version: str | None):
    """
    학습 설정과 메타데이터를 담은 config.json을 로드한다.
    UI 표시, 모델 카드 생성, 재현성 문서화에서 활용한다.
    경로 규칙은 models/configs/{model_name}/{version}/config.json 이다.
    """
    base_dir = P.models_configs_dir / model_name
    vdir = _resolve_version_dir(base_dir, version)
    config_path = base_dir / vdir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    return json.loads(config_path.read_text(encoding="utf-8"))


def _unwrap_state_dict(obj: object) -> dict:
    """
    torch.load 결과에서 state_dict 형태를 꺼낸다.
    체크포인트가 {'state_dict': ...} 형태거나 바로 state_dict인 경우를 모두 지원한다.
    DataParallel 저장 가중치의 'module.' prefix도 제거한다.
    """
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("invalid checkpoint format")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


def _infer_arch_from_state_dict(sd: dict) -> str:
    """
    state_dict 키 패턴을 보고 MLP 아키텍처 타입을 추정한다.
    사용자가 model_name을 잘못 지정했거나 weight 파일이 섞였을 때 자동 복구에 사용한다.
    """
    keys = list(sd.keys())

    if any(k.startswith("input_layer.") for k in keys) or any(k.startswith("blocks.") for k in keys):
        return "mlp_advanced"

    if any(("running_mean" in k) or ("running_var" in k) for k in keys):
        return "mlp_enhance"

    return "mlp_base"


def load_dl_model(
    model_name: str,
    version: str | None,
    input_dim: int,
    device: str = "cpu",
    auto_fix_arch: bool = True,
):
    """
    PyTorch DL 모델을 로드한다.
    Streamlit 실시간 추론에서 model.pt를 읽어 모델을 구성하고 eval 모드로 전환한다.
    auto_fix_arch가 True면 state_dict 패턴으로 아키텍처를 추정해 잘못된 model_name 입력을 보정한다.
    """
    import torch
    from models.model_definitions import MLP_base, MLP_enhance, MLP_advanced

    base_dir = P.models_dl_dir / model_name
    vdir = _resolve_version_dir(base_dir, version)
    weight_dir = base_dir / vdir

    candidates = [
        weight_dir / "model.pt",
        weight_dir / "weights.pt",
        weight_dir / f"{model_name}.pt",
    ]
    weight_path = next((p for p in candidates if p.exists()), None)
    if weight_path is None:
        raise FileNotFoundError(f"DL model file not found. searched: {candidates}")

    ckpt = torch.load(weight_path, map_location=device)
    sd = _unwrap_state_dict(ckpt)
    arch = _infer_arch_from_state_dict(sd)

    actual_name = model_name
    if auto_fix_arch and arch != model_name:
        actual_name = arch

    if actual_name == "mlp_base":
        model = MLP_base(input_dim)
    elif actual_name == "mlp_enhance":
        model = MLP_enhance(input_dim)
    elif actual_name == "mlp_advanced":
        model = MLP_advanced(input_dim)
    else:
        raise ValueError(f"unknown DL model: {actual_name}")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    return model, actual_name, str(weight_path)