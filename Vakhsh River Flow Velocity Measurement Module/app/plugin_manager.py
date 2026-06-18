import importlib
from pathlib import Path

from app.base_plugin import BasePlugin


class PluginManager:
    def __init__(self, plugins_dir="plugins"):
        self.plugins_dir = Path(plugins_dir)

    def load_plugins(self):
        plugins = []
        if not self.plugins_dir.exists():
            return plugins

        for plugin_dir in sorted(self.plugins_dir.iterdir()):
            if not plugin_dir.is_dir() or plugin_dir.name.startswith("__"):
                continue

            module_name = f"plugins.{plugin_dir.name}.plugin"
            try:
                module = importlib.import_module(module_name)
                plugin_cls = getattr(module, "Plugin")
                plugin = plugin_cls()
                if not isinstance(plugin, BasePlugin):
                    raise TypeError(f"{module_name}.Plugin must inherit BasePlugin")
                plugins.append(plugin)
            except Exception as exc:
                print(f"[PluginManager] Failed to load {module_name}: {exc}")

        return sorted(plugins, key=lambda item: item.order())

