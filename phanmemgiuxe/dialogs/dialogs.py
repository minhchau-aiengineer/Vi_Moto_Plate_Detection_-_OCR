from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton



# ======= Dialog Xóa lịch sử ======
class DeleteDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xóa lịch sử")
        self.setModal(True)

        self.setStyleSheet("""
            QDialog {
                background: #ffffff;
                border-radius: 10px;
            }
            QLabel {
                font-weight: 600;
                font-size: 14px;
                color: #111827;
            }
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
        lay.addWidget(lab)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.btn_sel = QPushButton("Xóa dòng đã chọn")
        self.btn_all = QPushButton("Xóa tất cả")
        self.btn_can = QPushButton("Hủy")

        row.addWidget(self.btn_sel, 1)
        row.addWidget(self.btn_all, 1)
        row.addWidget(self.btn_can, 1)

        lay.addLayout(row)

        base = "height:36px; font-weight:600; border-radius:10px; padding:6px 12px;"

        # ❗ Nút xanh đậm hơn
        self.btn_sel.setStyleSheet(
            f"""
            QPushButton {{
                {base}
                background:#c7dcff;
                border:1px solid #90b6ff;
                color:#0b3357;
            }}
            QPushButton:hover {{
                background:#b3d0ff;
            }}
            """
        )

        # ❗ Nút đỏ đậm hơn
        self.btn_all.setStyleSheet(
            f"""
            QPushButton {{
                {base}
                background:#ffcccc;
                border:1px solid #ff9999;
                color:#7a1f1f;
            }}
            QPushButton:hover {{
                background:#ffb3b3;
            }}
            """
        )

        # ❗ Nút hủy xám đậm hơn
        self.btn_can.setStyleSheet(
            f"""
            QPushButton {{
                {base}
                background:#e5e7eb;
                border:1px solid #cbd0d6;
                color:#374151;
            }}
            QPushButton:hover {{
                background:#d8dbe0;
            }}
            """
        )

        self.btn_sel.clicked.connect(lambda: self.done(1))
        self.btn_all.clicked.connect(lambda: self.done(2))
        self.btn_can.clicked.connect(lambda: self.done(0))
