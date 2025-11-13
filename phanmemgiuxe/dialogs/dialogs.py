from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton




class DeleteDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xóa lịch sử")
        self.setModal(True)
        self.setStyleSheet(""" QDialog { background: #ffffff; border-radius: 10px; } QLabel { font-weight: 600; } """)
       
        lay = QVBoxLayout(self); 
        lay.setContentsMargins(16,16,16,16); 
        lay.setSpacing(12)
        lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
        lay.addWidget(lab)

        row = QHBoxLayout(); row.setSpacing(12)
        self.btn_sel = QPushButton("Xóa dòng đã chọn")
        self.btn_all = QPushButton("Xóa tất cả")
        self.btn_can = QPushButton("Hủy")

        row.addWidget(self.btn_sel, 1); 
        row.addWidget(self.btn_all, 1); 
        row.addWidget(self.btn_can, 1)
        lay.addLayout(row)

        base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
        self.btn_sel.setStyleSheet(f"QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}")
        self.btn_all.setStyleSheet(f"QPushButton {{ {base} background:#ffe0e0; border:1px solid #ffb3b3; }}")
        self.btn_can.setStyleSheet(f"QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}")
        
        self.btn_sel.clicked.connect(lambda: self.done(1))
        self.btn_all.clicked.connect(lambda: self.done(2))
        self.btn_can.clicked.connect(lambda: self.done(0))
