# app/utils/artifacts.py

from __future__ import annotations

from typing import Any

from app.utils.save import save_model_and_artifacts as _save_model_and_artifacts


def save_model_and_artifacts(
    model: Any,
    model_name: str,
    model_type: str,          # "ml" | "dl"
    metrics: dict,
    scaler: Any | None = None,
    figures: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    report: str | None = None,       # (현재 save.py는 md report 저장 안 함)
    *,
    version: str = "baseline",
    split: str = "test",
    model_id: str | None = None,     # None이면 자동 생성
) -> dict[str, str]:
    """
    [레거시 호환용 래퍼]
    예전 코드가 app.utils.artifacts.save_model_and_artifacts(...)를 호출하던 것을
    프로젝트 표준 저장 함수(app.utils.save.save_model_and_artifacts)로 연결한다.

    - 팀 규칙(폴더로만 구분) + eval 산출물 규칙은 app/utils/save.py가 책임진다.
    - report(md) 저장이 필요하면, 나중에 save.py에 기능을 추가하거나 별도 유틸로 분리한다.
    """
    if model_id is None:
        # 예: "ml__hgb", "dl__mlp_enhance"
        model_id = f"{model_type}__{model_name}"

    # NOTE: report 저장은 현재 표준 save.py에 없음(요구되면 다음 단계에서 추가)
    return _save_model_and_artifacts(
        model=model,
        model_name=model_name,
        model_type=model_type,
        model_id=model_id,
        split=split,
        metrics=metrics,
        y_true=metrics.get("_y_true_for_save", []),  # 호출부에서 넘기도록 바꾸는 게 정석
        y_prob=metrics.get("_y_prob_for_save", []),  # 호출부에서 넘기도록 바꾸는 게 정석
        version=version,
        scaler=scaler,
        figures=figures,
        config=config,
    )