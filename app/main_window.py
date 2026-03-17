from PyQt5.QtWidgets import QMainWindow, QTabWidget
from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT
from .plugin_manager import PluginManager


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)

        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.tabs = QTabWidget()

        self.setCentralWidget(self.tabs)

        self.load_plugins()

    def load_plugins(self):

        manager = PluginManager()

        plugins = manager.load_plugins()

        for plugin in plugins:

            widget = plugin.widget()

            self.tabs.addTab(widget, plugin.name())