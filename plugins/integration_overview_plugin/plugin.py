from app.base_plugin import BasePlugin
from .integration_overview_widget import IntegrationOverviewWidget


class Plugin(BasePlugin):
    def name(self):
        return "数据整理与流程总览"

    def order(self):
        return 0

    def widget(self):
        return IntegrationOverviewWidget()
