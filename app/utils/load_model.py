import json
import joblib
import torch
from pathlib import Path

from app.utils.paths import PATHS
from models.model_definitions import MLP_base, MLP_enhance, MLP_advanced


#
# ML MODEL
#
def load_ml_model(model_name: str, version: str | None):
    """
    ML 모델 로드 (.pkl)

    팀 규칙(폴더로만 구분):
      models/ml/{model_name}/{version}/model.pkl
    """
    v = (version or "baseline").strip() or "baseline"

    # PATHS["models_ml"] == {PROJECT_ROOT}/models/ml 라고 가정
    model_path = Path(PATHS["models_ml"]) / model_name / v / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"ML 모델 파일이 없습니다: {model_path}")

    return joblib.load(model_path)


def _unwrap_state_dict(obj: object) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("Invalid checkpoint format (expected dict or {'state_dict': ...}).")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


def _infer_arch_from_state_dict(sd: dict) -> str:
    keys = list(sd.keys())

    if any(k.startswith("input_layer.") for k in keys) or any(k.startswith("blocks.") for k in keys):
        return "mlp_advanced"

    if any("running_mean" in k or "running_var" in k for k in keys):
        return "mlp_enhance"

    return "mlp_base"


#
# DL MODEL
#
def load_dl_model(
    model_name: str,
    version: str | None,
    input_dim: int,
    device: str = "cpu",
    auto_fix_arch: bool = True,
):
    if version:
        weight_dir = PATHS["models_dl"] / model_name / version
    else:
        weight_dir = PATHS["models_dl"] / model_name / "baseline"

    candidates = [
        weight_dir / "model.pt",
        weight_dir / "weights.pt",
        weight_dir / f"{model_name}.pt",
    ]
    weight_path = next((p for p in candidates if p.exists()), None)
    if weight_path is None:
        raise FileNotFoundError(f"DL 모델 파일이 없습니다. 탐색 후보: {candidates}")

    ckpt = torch.load(weight_path, map_location=device)
    sd = _unwrap_state_dict(ckpt)
    arch = _infer_arch_from_state_dict(sd)

    # 저장된 가중치 구조가 다른 모델이면 자동으로 맞춰 로드
    actual_name = model_name
    if auto_fix_arch and arch != model_name:
        actual_name = arch
        print(f"[WARN] weight file seems '{arch}' but requested '{model_name}'. path={weight_path}")

    if actual_name == "mlp_base":
        model = MLP_base(input_dim)
    elif actual_name == "mlp_enhance":
        model = MLP_enhance(input_dim)
    elif actual_name == "mlp_advanced":
        model = MLP_advanced(input_dim)
    else:
        raise ValueError(f"Unknown DL model: {actual_name}")

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    # ✅ 핵심: model + 실제 모델명 + weight 경로를 같이 리턴
    return model, actual_name, str(weight_path)


#
# SCALER
#

def load_scaler(model_name: str, version: str | None):
    """
    Scaler 로드 (.pkl)

    팀 규칙(폴더로만 구분):
      models/preprocessing/{model_name}/{version}/scaler.pkl
    """
    v = (version or "baseline").strip() or "baseline"

    base = Path(PATHS["models_preprocessing"])  # == {PROJECT_ROOT}/models/preprocessing
    scaler_path = base / model_name / v / "scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler 파일이 없습니다: {scaler_path}")

    return joblib.load(scaler_path)

#
# CONFIG
#
def load_config(model_name: str, version: str | None):
    if version is None:
        version = "baseline"

    config_path = PATHS["models_configs"] / model_name / version / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config 파일이 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)