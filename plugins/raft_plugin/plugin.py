from app.base_plugin import BasePlugin
from .raft_widget import RaftWidget


class Plugin(BasePlugin):
    def name(self):
        return "RAFT光流测速"

    def widget(self):
        return RaftWidget()
