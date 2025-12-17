from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDialog, QFormLayout, QLineEdit, QCheckBox, QAbstractItemView
from PySide6.QtCore import Qt
from phanmemgiuxe.database.card_reader_config_db import CardReaderConfigDB





# ====== Card Reader Config Page ======
class CardReaderDialog(QDialog):
    
    
    
    # === Init dialog with form fields ===
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle("Đầu đọc thẻ")
        self.resize(350, 400)
        self._data = data or {}
        self.device_name = QLineEdit()
        self.port_name = QLineEdit()
        self.ip_address = QLineEdit()
        self.port_number = QLineEdit()
        self.serial_number = QLineEdit()
        self.reader_id = QLineEdit()
        self.status = QCheckBox("Đang sử dụng")
        self.device_type = QLineEdit()
        if data:
            self.device_name.setText(data.get("device_name", ""))
            self.port_name.setText(data.get("port_name", ""))
            self.ip_address.setText(data.get("ip_address", ""))
            self.port_number.setText(str(data.get("port_number", "")))
            self.serial_number.setText(data.get("serial_number", ""))
            self.reader_id.setText(str(data.get("reader_id", "")))
            self.status.setChecked(bool(int(data.get("status", 1))))
            self.device_type.setText(data.get("device_type", ""))
        else:
            self.status.setChecked(True)
        form = QFormLayout()
        form.addRow("Tên đầu đọc:", self.device_name)
        form.addRow("Tên cổng:", self.port_name)
        form.addRow("IP:", self.ip_address)
        form.addRow("Port:", self.port_number)
        form.addRow("SerialNumber:", self.serial_number)
        form.addRow("Reader ID:", self.reader_id)
        form.addRow("Trạng thái:", self.status)
        form.addRow("Loại đầu đọc:", self.device_type)
        btn_ok = QPushButton("Lưu")
        btn_cancel = QPushButton("Hủy")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        h = QHBoxLayout()
        h.addStretch()
        h.addWidget(btn_ok)
        h.addWidget(btn_cancel)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(h)
    def get_data(self):
        return {
            "device_name": self.device_name.text().strip(),
            "port_name": self.port_name.text().strip(),
            "ip_address": self.ip_address.text().strip(),
            "port_number": int(self.port_number.text()),
            "serial_number": self.serial_number.text().strip(),
            "reader_id": int(self.reader_id.text()),
            "status": 1 if self.status.isChecked() else 0,
            "device_type": self.device_type.text().strip()
        }





# ===== Card Reader Config Page ======
class CardReaderConfigPage(QWidget):
    
    
    
    
    # === Init main page with table and buttons ===
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = CardReaderConfigDB()
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Tên đầu đọc", "Tên cổng", "IP", "Port", "SerialNumber", "Reader ID", "Trạng thái"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.btn_refresh = QPushButton("Tải lại")
        self.btn_add = QPushButton("Thêm")
        self.btn_edit = QPushButton("Sửa")
        self.btn_delete = QPushButton("Xóa")
        btn_bar = QHBoxLayout()
        btn_bar.addWidget(self.btn_refresh)
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_add)
        btn_bar.addWidget(self.btn_edit)
        btn_bar.addWidget(self.btn_delete)
        layout = QVBoxLayout(self)
        layout.addLayout(btn_bar)
        layout.addWidget(self.table)
        self.btn_refresh.clicked.connect(self.load_data)
        self.btn_add.clicked.connect(self.on_add)
        self.btn_edit.clicked.connect(self.on_edit)
        self.btn_delete.clicked.connect(self.on_delete)
        self.load_data()
        
        
        
        
        
    # === Load current selected row ID ===
    def _current_row_id(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if not item:
            return None
        try:
            return int(item.text())
        except ValueError:
            return None
    
    
    
    
    
    # === Load data into table ===
    def load_data(self):
        self.table.setRowCount(0)
        rows = self.db.get_all()
        for r_idx, row in enumerate(rows):
            self.table.insertRow(r_idx)
            for c_idx in range(8):
                item = QTableWidgetItem(str(row[c_idx]))
                if c_idx == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r_idx, c_idx, item)
    
    
    
    
    
    # === Button Handlers ===
    def on_add(self):
        dlg = CardReaderDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["device_name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên đầu đọc không được để trống.")
                return
            self.db.add(data)
            self.load_data()
    
    
    
    
    
    # === Edit selected row ===
    def on_edit(self):
        row_id = self._current_row_id()
        if row_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để sửa.")
            return
        row = self.table.currentRow()
        status_text = self.table.item(row, 7).text()
        if status_text.strip() in ["Đang dùng", "Đang sử dụng"]:
            status_val = 1
        elif status_text.strip() in ["Không dùng", "Không sử dụng"]:
            status_val = 0
        else:
            try:
                status_val = int(status_text)
            except Exception:
                status_val = 1
        data = {
            "device_name": self.table.item(row, 1).text(),
            "port_name": self.table.item(row, 2).text(),
            "ip_address": self.table.item(row, 3).text(),
            "port_number": int(self.table.item(row, 4).text()),
            "serial_number": self.table.item(row, 5).text(),
            "reader_id": int(self.table.item(row, 6).text()),
            "status": status_val,
            "device_type": ""
        }
        dlg = CardReaderDialog(self, data)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_data = dlg.get_data()
            if not new_data["device_name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên đầu đọc không được để trống.")
                return
            self.db.update(row_id, new_data)
            self.load_data()
    
    
    
    
    
    
    # === Delete selected row ===
    def on_delete(self):
        row_id = self._current_row_id()
        if row_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để xóa.")
            return
        if QMessageBox.question(self, "Xác nhận", f"Bạn chắc chắn muốn xóa đầu đọc ID = {row_id}?",) != QMessageBox.StandardButton.Yes:
            return
        self.db.delete(row_id)
        self.load_data()
