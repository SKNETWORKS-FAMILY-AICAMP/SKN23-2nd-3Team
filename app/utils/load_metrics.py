import json
from app.utils.paths import PATHS


def load_runtime_metrics(model_name: str, version: str):
    """
    실험/실시간 단계에서 저장한 metrics.json 로드
    models/metrics/{model_name}_{version}_metrics.json
    """
    metrics_path = PATHS["models_metrics"] / f"{model_name}_{version}_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics 파일이 없습니다: {metrics_path}")

    with open(metrics_path, "r") as f:
        return json.load(f)


def load_eval_metrics(model_id: str):
    """
    eval 단계 팀 공통 규칙 metrics 로드
    models/eval/<model_id>/
    """
    eval_dir = PATHS["models_eval"] / model_id

    if not eval_dir.exists():
        raise FileNotFoundError(f"eval 폴더가 없습니다: {eval_dir}")

    with open(eval_dir / "model_card.json") as f:
        model_card = json.load(f)

    with open(eval_dir / "pr_metrics.json") as f:
        pr_metrics = json.load(f)

    with open(eval_dir / "topk_metrics.json") as f:
        topk_metrics = json.load(f)

    with open(eval_dir / "topk_cutoffs.json") as f:
        topk_cutoffs = json.load(f)

    return {
        "model_card": model_card,
        "pr_metrics": pr_metrics,
        "topk_metrics": topk_metrics,
        "topk_cutoffs": topk_cutoffs,
    }