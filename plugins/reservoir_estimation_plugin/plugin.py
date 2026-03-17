from app.base_plugin import BasePlugin
from .reservoir_estimation_widget import ReservoirEstimationWidget


class Plugin(BasePlugin):
    def name(self):
        return "库区水量估算"

    def widget(self):
        return ReservoirEstimationWidget()