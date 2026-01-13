from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """
    프로젝트 루트를 계산한다.
    app/utils/paths.py 기준으로 2단계 위를 프로젝트 루트로 본다.
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """
    프로젝트 내 주요 경로를 한 곳에서 제공한다.
    다른 모듈은 하드코딩 대신 이 클래스를 통해 경로를 얻는다.
    """
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def data_raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def models_ml_dir(self) -> Path:
        return self.models_dir / "ml"

    @property
    def models_dl_dir(self) -> Path:
        return self.models_dir / "dl"

    @property
    def models_preprocessing_dir(self) -> Path:
        return self.models_dir / "preprocessing"

    @property
    def models_configs_dir(self) -> Path:
        return self.models_dir / "configs"

    @property
    def models_metrics_dir(self) -> Path:
        return self.models_dir / "metrics"

    @property
    def models_eval_dir(self) -> Path:
        return self.models_dir / "eval"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def assets_training_dir(self) -> Path:
        return self.assets_dir / "training"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def reports_training_dir(self) -> Path:
        return self.reports_dir / "training"

    def parquet_path(self, name: str) -> Path:
        """
        프로젝트 표준 parquet 4개의 경로를 이름으로 반환한다.
        name은 다음 중 하나여야 한다.
        base, anchors, labels, features_ml_clean
        """
        mapping = {
            "base": self.data_raw_dir / "base.parquet",
            "anchors": self.data_processed_dir / "anchors.parquet",
            "labels": self.data_processed_dir / "labels.parquet",
            "features_ml_clean": self.data_processed_dir / "features_ml_clean.parquet",
        }
        try:
            return mapping[name]
        except KeyError as e:
            raise KeyError(f"Unknown parquet name: {name}") from e

    def must_parquet_path(self, name: str) -> Path:
        """
        parquet_path로 경로를 얻고, 파일이 없으면 예외를 낸다.
        추론/학습 진입 전에 필수 파일 존재를 강제할 때 사용한다.
        """
        p = self.parquet_path(name)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")
        return p

    def debug_snapshot(self) -> dict[str, Any]:
        """
        경로 계산 결과와 존재 여부를 한 번에 확인하기 위한 스냅샷을 만든다.
        콘솔 출력이나 Streamlit 디버그 패널에서 그대로 사용한다.
        """
        items: list[tuple[str, Path]] = [
            ("root", self.root),
            ("data_raw_dir", self.data_raw_dir),
            ("data_processed_dir", self.data_processed_dir),
            ("models_ml_dir", self.models_ml_dir),
            ("models_dl_dir", self.models_dl_dir),
            ("models_preprocessing_dir", self.models_preprocessing_dir),
            ("models_configs_dir", self.models_configs_dir),
            ("models_metrics_dir", self.models_metrics_dir),
            ("models_eval_dir", self.models_eval_dir),
            ("assets_training_dir", self.assets_training_dir),
            ("reports_training_dir", self.reports_training_dir),
            ("parquet_base", self.parquet_path("base")),
            ("parquet_anchors", self.parquet_path("anchors")),
            ("parquet_labels", self.parquet_path("labels")),
            ("parquet_features_ml_clean", self.parquet_path("features_ml_clean")),
        ]

        out: dict[str, Any] = {}
        for k, p in items:
            out[k] = {"path": str(p), "exists": bool(p.exists())}
        return out

    def validate_required_parquets(self) -> None:
        """
        표준 parquet 4개가 모두 존재하는지 확인한다.
        실패하면 어떤 파일이 없는지 예외 메시지로 알려준다.
        """
        self.must_parquet_path("base")
        self.must_parquet_path("anchors")
        self.must_parquet_path("labels")
        self.must_parquet_path("features_ml_clean")


DEFAULT_PATHS = ProjectPaths(get_project_root())


def ensure_runtime_dirs() -> None:
    """
    런타임에서 생성되어야 하는 폴더만 만든다.
    폴더 생성은 최소 범위로 제한한다.
    """
    dirs = [
        DEFAULT_PATHS.assets_training_dir,
        DEFAULT_PATHS.reports_training_dir,
        DEFAULT_PATHS.models_metrics_dir,
        DEFAULT_PATHS.models_eval_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_debug_snapshot() -> None:
    """
    debug_snapshot을 사람이 보기 쉽게 콘솔로 출력한다.
    """
    snap = DEFAULT_PATHS.debug_snapshot()
    for k in sorted(snap.keys()):
        row = snap[k]
        print(f"{k}: {row['path']}  exists={row['exists']}")


if __name__ == "__main__":
    ensure_runtime_dirs()
    print_debug_snapshot()
    DEFAULT_PATHS.validate_required_parquets()
    print("required parquets ok")