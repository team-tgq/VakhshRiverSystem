from app.base_plugin import BasePlugin
from .water_allocation_widget import WaterAllocationWidget


class Plugin(BasePlugin):
    def name(self):
        return "水资源分配"

    def order(self):
        return 60

    def widget(self):
        return WaterAllocationWidget()
