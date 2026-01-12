import pandas as pd
from pathlib import Path

def load_feature_schema(
    feature_parquet_path: str,
    exclude_cols=("user_id", "anchor_time"),
):
    """
    학습에 사용한 feature parquet에서
    Streamlit 입력용 feature schema 자동 생성
    """
    df = pd.read_parquet(feature_parquet_path)

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    schema = {}
    for col in feature_cols:
        series = df[col].dropna()

        # 값 범위 자동 추정
        min_val = float(series.quantile(0.01)) 
        max_val = float(series.quantile(0.99))

        # 이상치 보호
        if min_val == max_val:
            max_val = min_val + 1.0

        schema[col] = {
            "label": col,            # 필요하면 한글 매핑 가능
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "step": round((max_val - min_val) / 100, 3),
            "default": round(series.median(), 3),
        }

    return schema