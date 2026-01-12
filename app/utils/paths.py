import os
import pathlib

# Define project root (relative to this file)
# This file is in app/utils/paths.py, so project root is ../../
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()


class Paths:
    def __init__(self, root):
        self.ROOT = root
        self.MODELS_ML = root / "models" / "ml"
        self.MODELS_DL = root / "models" / "dl"
        self.PREPROCESSING = root / "models" / "preprocessing"
        self.METRICS = root / "models" / "metrics"
        self.ASSETS_TRAINING = root / "assets" / "training"
        self.REPORTS_TRAINING = root / "reports" / "training"
        self.OUTPUTS_SAMPLES = root / "outputs" / "samples"

        # Ensure directories exist
        for path in [
            self.MODELS_ML,
            self.MODELS_DL,
            self.PREPROCESSING,
            self.METRICS,
            self.ASSETS_TRAINING,
            self.REPORTS_TRAINING,
        ]:
            os.makedirs(path, exist_ok=True)


PATHS = Paths(PROJECT_ROOT)
