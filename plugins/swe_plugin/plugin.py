from app.base_plugin import BasePlugin
from .swe_widget import SWEWidget


class Plugin(BasePlugin):
    def name(self):
        return "雪水当量估算"

    def widget(self):
        return SWEWidget()