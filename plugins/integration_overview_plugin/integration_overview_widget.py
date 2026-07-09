from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    configured_twin_data_root,
)
from tools.validate_twin_data import validate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARD_DOC = PROJECT_ROOT / "docs" / "digital_twin_integration_standard.md"


class IntegrationOverviewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.status_label = QLabel()
        self.root_edit = QLineEdit(str(configured_twin_data_root()))
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

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("数据根目录:"))
        self.root_edit.setMinimumWidth(520)
        self.choose_root_btn = QPushButton("选择目录")
        root_row.addWidget(self.root_edit)
        root_row.addWidget(self.choose_root_btn)
        layout.addLayout(root_row)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新状态")
        self.validate_btn = QPushButton("校验数据目录")
        self.open_sample_btn = QPushButton("打开当前数据目录")
        self.open_doc_btn = QPushButton("打开整改说明")
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.validate_btn)
        btn_row.addWidget(self.open_sample_btn)
        btn_row.addWidget(self.open_doc_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(self.status_label)
        layout.addWidget(self.summary)

        self.choose_root_btn.clicked.connect(self.choose_root)
        self.refresh_btn.clicked.connect(self.refresh_status)
        self.validate_btn.clicked.connect(self.validate_current_root)
        self.open_sample_btn.clicked.connect(lambda: self._open_path(self.current_root()))
        self.open_doc_btn.clicked.connect(lambda: self._open_path(STANDARD_DOC))

    def current_root(self) -> Path:
        text = self.root_edit.text().strip()
        return Path(text).expanduser().resolve() if text else DEFAULT_TWIN_DATA_ROOT

    def choose_root(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择瓦赫什流域孪生数据根目录",
            str(self.current_root()),
        )
        if selected:
            self.root_edit.setText(selected)
            self.refresh_status()

    def refresh_status(self):
        root = self.current_root()
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
            f"- 根目录: {root}",
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

    def validate_current_root(self):
        root = self.current_root()
        report = validate(root)
        lines = [
            f"校验目录: {root}",
            f"检查通过项: {len(report.checked)}",
            f"警告: {len(report.warnings)}",
            f"错误: {len(report.errors)}",
            "",
        ]
        if report.errors:
            lines.append("错误列表:")
            lines.extend(f"- {item}" for item in report.errors[:80])
            if len(report.errors) > 80:
                lines.append(f"- 其余 {len(report.errors) - 80} 项略")
        elif report.warnings:
            lines.append("警告列表:")
            lines.extend(f"- {item}" for item in report.warnings[:80])
            if len(report.warnings) > 80:
                lines.append(f"- 其余 {len(report.warnings) - 80} 项略")
        else:
            lines.append("数据目录通过统一规范校验。")

        self.status_label.setText(
            "状态: 校验通过" if report.success else "状态: 校验存在错误"
        )
        self.summary.setPlainText("\n".join(lines))

    def _open_path(self, path: Path):
        try:
            if not path.exists():
                raise FileNotFoundError(str(path))
            os.startfile(str(path))
        except Exception as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
