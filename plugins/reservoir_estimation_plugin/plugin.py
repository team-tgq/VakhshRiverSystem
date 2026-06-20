from app.base_plugin import BasePlugin
from .reservoir_estimation_widget import ReservoirEstimationWidget


class Plugin(BasePlugin):
    def name(self):
        return "\u5e93\u533a\u6c34\u91cf\u4f30\u7b97"

    def order(self):
        return 31

    def widget(self):
        return ReservoirEstimationWidget()
