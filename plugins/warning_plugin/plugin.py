from algorithms.warning.core import build_warning_system
from app.base_plugin import BasePlugin
from .warning_widget import WarningWidget


class Plugin(BasePlugin):
    def name(self):
        return "洪水智能预警监控"

    def widget(self):
        warning_system = build_warning_system()
        return WarningWidget(warning_system=warning_system)