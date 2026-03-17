from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class SnowDataset(CustomDataset):
    CLASSES = ('background', 'snow')
    PALETTE = [[0, 0, 0], [255, 255, 255]]