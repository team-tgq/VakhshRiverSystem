from PyQt5.QtWidgets import QWidget


class BasePlugin:

    def name(self):
        """
        插件名称
        """
        return "Plugin"

    def widget(self) -> QWidget:
        """
        返回插件UI
        """
        raise NotImplementedError