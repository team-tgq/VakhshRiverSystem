from app.base_plugin import BasePlugin
from .snow_state_widget import SnowStateWidget


class Plugin(BasePlugin):
    def name(self):
        return "积雪状态识别"

    def order(self):
        return 20

    def widget(self):
        return SnowStateWidget()
