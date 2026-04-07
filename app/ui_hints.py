from __future__ import annotations

from typing import Union

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget


def create_hint_badge(hint_text: str) -> QLabel:
    badge = QLabel("i")
    badge.setToolTip(hint_text)
    badge.setWhatsThis(hint_text)
    badge.setAlignment(Qt.AlignCenter)
    badge.setFixedSize(16, 16)
    badge.setCursor(Qt.PointingHandCursor)
    badge.setStyleSheet(
        "QLabel {"
        "background-color: #eef2f7;"
        "color: #4b5563;"
        "border: 1px solid #cbd5e1;"
        "border-radius: 8px;"
        "font-size: 11px;"
        "font-weight: 600;"
        "}"
    )
    return badge


def label_with_hint(label: Union[str, QLabel], hint_text: str, stretch: bool = True) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)

    label_widget = QLabel(label) if isinstance(label, str) else label
    row.addWidget(label_widget)
    row.addWidget(create_hint_badge(hint_text))
    if stretch:
        row.addStretch(1)

    return container


def attach_hint(widget: QWidget, hint_text: str) -> None:
    widget.setToolTip(hint_text)
    widget.setWhatsThis(hint_text)
