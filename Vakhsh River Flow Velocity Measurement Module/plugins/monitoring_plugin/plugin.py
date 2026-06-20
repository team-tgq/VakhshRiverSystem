from app.base_plugin import BasePlugin

from .flow_velocity_widget import FlowVelocityWidget


class Plugin(BasePlugin):
    def name(self):
        return "水文监测"

    def order(self):
        return 10

    def widget(self):
        return FlowVelocityWidget()

