import os
import importlib
from config import PLUGIN_DIR


class PluginManager:

    def __init__(self):
        self.plugins = []

    def load_plugins(self):

        for folder in os.listdir(PLUGIN_DIR):

            if folder.startswith("__"):
                continue

            path = os.path.join(PLUGIN_DIR, folder)

            if not os.path.isdir(path):
                continue

            plugin_file = os.path.join(path, "plugin.py")

            if not os.path.exists(plugin_file):
                continue

            try:

                module = importlib.import_module(
                    f"{PLUGIN_DIR}.{folder}.plugin"
                )

                plugin = module.Plugin()

                self.plugins.append(plugin)

                print("加载插件:", plugin.name())

            except Exception as e:

                print("插件加载失败:", folder, e)

        return self.plugins