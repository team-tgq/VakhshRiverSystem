import os
import importlib
from config import PLUGIN_DIR


class PluginManager:

    def __init__(self):
        self.plugins = []

    def load_plugins(self):
        self.plugins = []

        for folder in sorted(os.listdir(PLUGIN_DIR)):

            if folder.startswith("__"):
                continue

            path = os.path.join(PLUGIN_DIR, folder)

            if not os.path.isdir(path):
                continue

            plugin_file = os.path.join(path, "plugin.py")

            if not os.path.exists(plugin_file):
                continue

            try:
                module = importlib.import_module(f"{PLUGIN_DIR}.{folder}.plugin")
                plugin = module.Plugin()
                self.plugins.append(plugin)

                order = plugin.order() if hasattr(plugin, "order") else 999
                print(f"加载插件: {plugin.name()} (order={order})")

            except Exception as e:
                print("插件加载失败:", folder, e)

        self.plugins.sort(
            key=lambda p: p.order() if hasattr(p, "order") else 999
        )

        return self.plugins