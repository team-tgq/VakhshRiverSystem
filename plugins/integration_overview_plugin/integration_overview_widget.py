from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.digital_twin_standard import (
    DATE_FIELD,
    DATE_FORMAT,
    DEFAULT_TWIN_DATA_ROOT,
    MODULE_SPECS,
    STANDARD_FIELDS,
    STUDY_YEAR_END,
    STUDY_YEAR_START,
    TARGET_CRS,
    TARGET_CRS_NAME,
    TIME_STEP,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARD_DOC = PROJECT_ROOT / "docs" / "digital_twin_integration_standard.md"


class IntegrationOverviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.status_label = QLabel()
        self._init_ui()
        self.refresh_status()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("瓦赫什流域数字孪生集成总览")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "统一管理流域基准数据、原始观测数据、模型成果目录和模块调用顺序。"
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新状态")
        self.open_sample_btn = QPushButton("打开样例数据目录")
        self.open_doc_btn = QPushButton("打开整改说明")
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.open_sample_btn)
        btn_row.addWidget(self.open_doc_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(self.status_label)
        layout.addWidget(self.summary)

        self.refresh_btn.clicked.connect(self.refresh_status)
        self.open_sample_btn.clicked.connect(lambda: self._open_path(DEFAULT_TWIN_DATA_ROOT))
        self.open_doc_btn.clicked.connect(lambda: self._open_path(STANDARD_DOC))

    def refresh_status(self):
        root = DEFAULT_TWIN_DATA_ROOT
        required_dirs = [root / "baseline", root / "raw", root / "processed"]
        missing = [str(path) for path in required_dirs if not path.exists()]
        if missing:
            self.status_label.setText("状态: 样例数据目录不完整")
        else:
            self.status_label.setText("状态: 样例数据目录已就绪")

        lines = [
            "一、统一空间与时间规范",
            f"- 坐标系: {TARGET_CRS_NAME} ({TARGET_CRS})",
            f"- 研究时段: {STUDY_YEAR_START}-{STUDY_YEAR_END}",
            f"- 时间步长: {TIME_STEP}",
            f"- 时间字段: {DATE_FIELD} ({DATE_FORMAT})",
            "",
            "二、标准数据目录",
            f"- baseline: {root / 'baseline'}",
            f"- raw: {root / 'raw'}",
            f"- processed: {root / 'processed'}",
            "",
            "三、模块调用链路",
        ]
        for spec in MODULE_SPECS:
            downstream = "、".join(spec.downstream) if spec.downstream else "无"
            lines.append(f"- {spec.code} {spec.name} [{spec.phase}/{spec.role}] -> 下游: {downstream}")

        lines.extend(["", "四、统一字段与单位"])
        for field, info in STANDARD_FIELDS.items():
            lines.append(f"- {field}: {info['name']}，单位 {info['unit']}")

        if missing:
            lines.extend(["", "缺失目录:"])
            lines.extend(f"- {item}" for item in missing)

        self.summary.setPlainText("\n".join(lines))

    def _open_path(self, path: Path):
        try:
            if not path.exists():
                raise FileNotFoundError(str(path))
            os.startfile(str(path))
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
