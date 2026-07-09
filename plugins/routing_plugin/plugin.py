# plugins/routing_plugin/plugin.py
from app.base_plugin import BasePlugin
from .routing_widget import RoutingWidget


class Plugin(BasePlugin):
    def name(self):
        return "洪水演进与汇流模拟"

    def order(self):
        return 40

    def widget(self):
        return RoutingWidget()
