# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO-NAS models interface.

Example:
    ```python
    from ultralytics import NAS

    models = NAS("yolo_nas_s")
    results = models.predict("ultralytics/assets/bus.jpg")
    ```
"""

from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.torch_utils import model_info

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    """
    YOLO NAS models for object detection.

    This class provides an interface for the YOLO-NAS models and extends the `Model` class from Ultralytics engine.
    It is designed to facilitate the task of object detection using pre-trained or custom-trained YOLO-NAS models.

    Example:
        ```python
        from ultralytics import NAS

        models = NAS("yolo_nas_s")
        results = models.predict("ultralytics/assets/bus.jpg")
        ```

    Attributes:
        model (str): Path to the pre-trained models or models name. Defaults to 'yolo_nas_s.pt'.

    Note:
        YOLO-NAS models only support pre-trained models. Do not provide YAML configuration files.
    """

    def __init__(self, model="yolo_nas_s.pt") -> None:
        """Initializes the NAS models with the provided or default 'yolo_nas_s.pt' models."""
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        """Loads an existing NAS models weights or creates a new NAS models with pretrained weights if not provided."""
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model = torch.load(attempt_download_asset(weights))

        elif suffix == "":
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        # Override the forward method to ignore additional arguments
        def new_forward(x, *args, **kwargs):
            """Ignore additional __call__ arguments."""
            return self.model._original_forward(x)

        self.model._original_forward = self.model.forward
        self.model.forward = new_forward

        # Standardize models
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # for info()
        self.model.yaml = {}  # for info()
        self.model.pt_path = weights  # for export()
        self.model.task = "detect"  # for export()

    def info(self, detailed=False, verbose=True):
        """
        Logs models info.

        Args:
            detailed (bool): Show detailed information about models.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self):
        """Returns a dictionary mapping tasks to respective predictor and validator classes."""
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
