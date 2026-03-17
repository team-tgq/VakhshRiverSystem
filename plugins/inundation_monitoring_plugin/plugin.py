from app.base_plugin import BasePlugin
from .inundation_monitoring_widget import InundationMonitoringWidget


class Plugin(BasePlugin):
    def name(self):
        return "淹没区监测"

    def widget(self):
        return InundationMonitoringWidget()