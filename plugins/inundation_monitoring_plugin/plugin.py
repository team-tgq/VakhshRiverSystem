from app.base_plugin import BasePlugin
from .inundation_monitoring_widget import InundationMonitoringWidget


class Plugin(BasePlugin):
    def name(self):
        return "\u6df9\u6ca1\u533a\u76d1\u6d4b"

    def order(self):
        return 30

    def widget(self):
        return InundationMonitoringWidget()
