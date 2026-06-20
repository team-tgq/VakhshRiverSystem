import os
import importlib
import json
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

            metadata_file = os.path.join(path, "plugin.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r", encoding="utf-8-sig") as file:
                        metadata = json.load(file)
                    if metadata.get("enabled") is False:
                        print(f"跳过插件: {folder} (enabled=false)")
                        continue
                except Exception as e:
                    print("插件元数据读取失败:", folder, e)
                    continue

            plugin_file = os.path.join(path, "plugin.py")

            if not os.path.exists(plugin_file):
                continue

            try:
                module = importlib.import_module(f"{PLUGIN_DIR}.{folder}.plugin")
                if not hasattr(module, "Plugin"):
                    print(f"跳过插件: {folder} (未定义 Plugin 类)")
                    continue
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
