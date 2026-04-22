from app.base_plugin import BasePlugin

from .flood_widget import FloodWidget


class Plugin(BasePlugin):
    def name(self):
        return "洪涝风险评估"

    def widget(self):
        return FloodWidget()
