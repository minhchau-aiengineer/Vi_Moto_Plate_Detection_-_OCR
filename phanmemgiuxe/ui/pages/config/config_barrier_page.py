from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDialog, QFormLayout, QLineEdit, QCheckBox, QAbstractItemView
from phanmemgiuxe.database.barrier_config_db import BarrierConfigDB


# ====== Barrier Config Page ======
class BarrierDialog(QDialog):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle("Barrier")
        self.resize(350, 600)
        self._data = data or {}
        self.name = QLineEdit()
        self.lane = QLineEdit()
        self.ip_address = QLineEdit()
        self.port_number = QLineEdit()
        self.serial_number = QLineEdit()
        self.account = QLineEdit()
        self.password = QLineEdit()
        self.port_name = QLineEdit()
        self.relay = QLineEdit()
        self.open_delay_ms = QLineEdit()
        self.close_delay_ms = QLineEdit()
        self.auto_open_on_match = QCheckBox()
        self.is_active = QCheckBox()
        self.note = QLineEdit()
        if data:
            self.name.setText(data.get("name", ""))
            self.lane.setText(data.get("lane", ""))
            self.ip_address.setText(data.get("ip_address", ""))
            self.port_number.setText(str(data.get("port_number", "")))
            self.serial_number.setText(data.get("serial_number", ""))
            self.account.setText(data.get("account", ""))
            self.password.setText(data.get("password", ""))
            self.port_name.setText(data.get("port_name", ""))
            self.relay.setText(data.get("relay", ""))
            self.open_delay_ms.setText(str(data.get("open_delay_ms", "")))
            self.close_delay_ms.setText(str(data.get("close_delay_ms", "")))
            self.auto_open_on_match.setChecked(bool(data.get("auto_open_on_match", False)))
            self.is_active.setChecked(bool(data.get("is_active", True)))
            self.note.setText(data.get("note", ""))
        form = QFormLayout()
        form.addRow("Tên:", self.name)
        form.addRow("Làn:", self.lane)
        form.addRow("IP:", self.ip_address)
        form.addRow("Port:", self.port_number)
        form.addRow("Serial:", self.serial_number)
        form.addRow("Account:", self.account)
        form.addRow("Password:", self.password)
        form.addRow("Cổng:", self.port_name)
        form.addRow("Relay:", self.relay)
        form.addRow("OpenDelay:", self.open_delay_ms)
        form.addRow("CloseDelay:", self.close_delay_ms)
        form.addRow("AutoOpen:", self.auto_open_on_match)
        form.addRow("Active:", self.is_active)
        form.addRow("Ghi chú:", self.note)
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
        
        
        
        
    # === Get data from dialog ===
    def get_data(self):
        data = {
            "name": self.name.text(),
            "lane": self.lane.text(),
            "ip_address": self.ip_address.text(),
            "port_number": int(self.port_number.text()) if self.port_number.text().isdigit() else 0,
            "serial_number": self.serial_number.text(),
            "account": self.account.text(),
            "password": self.password.text(),
            "port_name": self.port_name.text(),
            "relay": self.relay.text(),
            "open_delay_ms": int(self.open_delay_ms.text()) if self.open_delay_ms.text().isdigit() else 0,
            "close_delay_ms": int(self.close_delay_ms.text()) if self.close_delay_ms.text().isdigit() else 0,
            "auto_open_on_match": 1 if self.auto_open_on_match.isChecked() else 0,
            "is_active": 1 if self.is_active.isChecked() else 0,
            "note": self.note.text()
        }
        return data
    
    
    
    
    
# ======= Barrier Config Page =======
class BarrierConfigPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = BarrierConfigDB()
        self.table = QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
            "ID", "Tên", "Làn", "IP", "Port", "Serial", "Account", "Password", "Cổng", "Relay", "OpenDelay", "CloseDelay", "AutoOpen", "Active"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Button bar above table
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
            for c_idx in range(14):
                item = QTableWidgetItem(str(row[c_idx]))
                if c_idx == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r_idx, c_idx, item)
    
    
    
    
    # === Button Handlers ===
    def on_add(self):
        dlg = BarrierDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên barrier không được để trống.")
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
        auto_open_val = self.table.item(row, 12).text().strip().lower()
        if auto_open_val in ["true", "1"]:
            auto_open_on_match = 1
        else:
            auto_open_on_match = 0
        is_active_val = self.table.item(row, 13).text().strip().lower()
        if is_active_val in ["true", "1"]:
            is_active = 1
        else:
            is_active = 0
        data = {
            "name": self.table.item(row, 1).text(),
            "lane": self.table.item(row, 2).text(),
            "ip_address": self.table.item(row, 3).text(),
            "port_number": int(self.table.item(row, 4).text()),
            "serial_number": self.table.item(row, 5).text(),
            "account": self.table.item(row, 6).text(),
            "password": self.table.item(row, 7).text(),
            "port_name": self.table.item(row, 8).text(),
            "relay": self.table.item(row, 9).text(),
            "open_delay_ms": int(self.table.item(row, 10).text()),
            "close_delay_ms": int(self.table.item(row, 11).text()),
            "auto_open_on_match": auto_open_on_match,
            "is_active": is_active,
            "note": ""
        }
        dlg = BarrierDialog(self, data)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_data = dlg.get_data()
            if not new_data["name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên barrier không được để trống.")
                return
            self.db.update(row_id, new_data)
            self.load_data()
    
    
    
    
    # === Delete selected row ===
    def on_delete(self):
        row_id = self._current_row_id()
        if row_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để xóa.")
            return
        if QMessageBox.question(self, "Xác nhận", f"Bạn chắc chắn muốn xóa barrier ID = {row_id}?",) != QMessageBox.StandardButton.Yes:
            return
        self.db.delete(row_id)
        self.load_data()
