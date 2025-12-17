from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QFileDialog,
    QMessageBox,
)




# ======= Dialog Xuất lịch sử =======
class ExportHistoryDialog(QDialog):
    """
    Hộp thoại chọn cách xuất lịch sử:

        - Xuất dòng đã chọn   -> done(1)
        - Xuất toàn bộ bảng   -> done(2)
        - Hủy                 -> done(0)
    """




    # === Khởi tạo dialog =====
    def __init__(self, parent: QWidget | None = None, has_selection: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Xuất lịch sử")
        self.setModal(True)
        self.setMinimumWidth(430)

        self.setStyleSheet(
            """
            QDialog {
                background-color:#ffffff;
            }
            QLabel {
                color:#111827;
                font-weight:600;
            }
            QPushButton {
                height:34px;
                border-radius:8px;
                font-weight:600;
                padding:6px 12px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        label = QLabel("Bạn muốn xuất dữ liệu lịch sử như thế nào?")
        label.setWordWrap(True)
        layout.addWidget(label)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.btn_sel = QPushButton("Xuất dòng đã chọn")
        self.btn_all = QPushButton("Xuất toàn bộ trong bảng")
        self.btn_can = QPushButton("Hủy")

        self.btn_sel.setStyleSheet(
            "QPushButton{background:#dbeafe; border:1px solid #bfdbfe; color:#1d4ed8;}"
            "QPushButton:hover{background:#bfdbfe;}"
        )
        self.btn_all.setStyleSheet(
            "QPushButton{background:#fee2e2; border:1px solid #fecaca; color:#b91c1c;}"
            "QPushButton:hover{background:#fecaca;}"
        )
        self.btn_can.setStyleSheet(
            "QPushButton{background:#e5e7eb; border:1px solid #d1d5db; color:#111827;}"
            "QPushButton:hover{background:#d4d4d8;}"
        )

        # nếu không có dòng chọn thì disable nút xuất dòng đã chọn
        self.btn_sel.setEnabled(has_selection)

        row.addWidget(self.btn_sel)
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_can)
        layout.addLayout(row)

        self.btn_sel.clicked.connect(lambda: self.done(1))
        self.btn_all.clicked.connect(lambda: self.done(2))
        self.btn_can.clicked.connect(lambda: self.done(0))




# === Xuất DataFrame ra file Excel ===
def export_df_to_excel(parent: QWidget, df: pd.DataFrame) -> None:
    """
    Xuất DataFrame ra file Excel.
    df: DataFrame đã được lọc sẵn (nếu chỉ export một phần).
    """
    if df is None or df.empty:
        QMessageBox.information(parent, "Xuất", "Không có dữ liệu để xuất.")
        return

    # bỏ cột STT nếu còn
    if "STT" in df.columns:
        df = df.drop(columns=["STT"])

    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Lưu file Excel",
        "history.xlsx",
        "Excel Files (*.xlsx)",
    )
    if not path:
        return

    try:
        df.to_excel(path, index=False)
        QMessageBox.information(parent, "Xuất", f"Đã xuất dữ liệu ra:\n{path}")
    except Exception as e:
        QMessageBox.warning(parent, "Xuất", f"Lỗi khi xuất Excel:\n{e}")
