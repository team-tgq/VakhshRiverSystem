from app.base_plugin import BasePlugin
from .segformer_widget import SegFormerWidget


class Plugin(BasePlugin):
    def name(self):
        return "SegFormer专题识别"

    def widget(self):
        return SegFormerWidget()