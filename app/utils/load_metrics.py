import json
import os
from app.utils.paths import PATHS


def load_metrics(model_name):
    """
    특정 모델의 성능 지표 JSON 파일을 불러옵니다.

    Args:
        model_name (str): 모델 이름 (예: 'mlp_enhance').

    Returns:
        dict: 성능 지표가 담긴 딕셔너리.

    Raises:
        FileNotFoundError: 지표 파일이 존재하지 않을 경우 발생합니다.

    사용 예시:
        >>> metrics = load_metrics("mlp_enhance")
    """
    metrics_path = os.path.join(PATHS.METRICS, f"{model_name}_metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"지표 파일을 찾을 수 없습니다: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics
