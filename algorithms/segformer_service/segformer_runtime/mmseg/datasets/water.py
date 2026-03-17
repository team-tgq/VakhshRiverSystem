from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class WaterDataset(CustomDataset):
    CLASSES = ('background', 'water')
    PALETTE = [[0, 0, 0], [255, 255, 255]]