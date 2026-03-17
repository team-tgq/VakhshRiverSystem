import os
import sys

# 先设置环境变量，再导入 QApplication
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu --disable-gpu-compositing"
os.environ["QT_OPENGL"] = "software"

from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtWidgets import QApplication

from app.main_window import MainWindow


def main():
    # 必须在 QApplication 之前
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()