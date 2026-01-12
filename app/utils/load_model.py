import os
import torch
import joblib
from app.utils.paths import PATHS


def load_model(model_name, model_type="dl", model_class=None):
    """
    저장된 모델 파일을 불러옵니다.

    Args:
        model_name (str): 모델 파일 이름 (확장자 제외, 예: 'mlp_enhance').
        model_type (str): 모델 타입 ('dl': PyTorch, 'ml': Sklearn). 기본값은 'dl'입니다.
        model_class (class, optional): PyTorch 모델의 경우, 초기화할 모델 클래스.
                                       None이면 전체 객체를 불러오려 시도합니다 (PyTorch에서는 권장되지 않음).

    Returns:
        불러온 모델 객체 (PyTorch 모델 또는 Sklearn 모델).

    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않을 경우 발생합니다.
        ValueError: 잘못된 model_type을 지정했을 경우 발생합니다.

    사용 예시:
        >>> model = load_model("mlp_enhance", model_type="dl")
    """
    if model_type == "dl":
        model_path = os.path.join(PATHS.MODELS_DL, f"{model_name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        if model_class:
            model = model_class()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        else:
            return torch.load(model_path)

    elif model_type == "ml":
        model_path = os.path.join(PATHS.MODELS_ML, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        return joblib.load(model_path)

    else:
        raise ValueError("model_type은 반드시 'dl' 또는 'ml'이어야 합니다.")


def load_scaler(model_name):
    """
    모델과 연관된 스케일러(Scaler) 객체를 불러옵니다.

    Args:
        model_name (str): 모델 이름 (스케일러 파일명은 {model_name}_scaler.pkl 형식이어야 함).

    Returns:
        불러온 스케일러 객체.

    Raises:
        FileNotFoundError: 스케일러 파일이 존재하지 않을 경우 발생합니다.

    사용 예시:
        >>> scaler = load_scaler("mlp_enhance")
    """
    # 스케일러는 PREPROCESSING 경로에 저장됨
    scaler_path = os.path.join(PATHS.PREPROCESSING, f"{model_name}_scaler.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}")

    return joblib.load(scaler_path)
