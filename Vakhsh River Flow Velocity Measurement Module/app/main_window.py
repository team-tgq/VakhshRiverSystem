from PyQt5.QtWidgets import QMainWindow, QTabWidget

from app.plugin_manager import PluginManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vakhsh River System")
        self.resize(1200, 780)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.plugin_manager = PluginManager()
        self.load_plugins()

    def load_plugins(self):
        for plugin in self.plugin_manager.load_plugins():
            self.tabs.addTab(plugin.widget(), plugin.name())

