# plugins/monitoring_plugin/plugin.py
from app.base_plugin import BasePlugin
from .monitoring_widget import MonitoringWidget


class Plugin(BasePlugin):
    def name(self):
        return "水文监测系统"

    def widget(self):
        return MonitoringWidget()