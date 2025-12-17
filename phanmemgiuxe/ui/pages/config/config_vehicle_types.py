from typing import List, Dict, Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QDialog,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QCheckBox,
    QAbstractItemView,
)
from PySide6.QtCore import Qt

from ....database.database import DB
from ....config.config import CONN_STR




# ====== Vehicle Types Config Page ======
class VehicleTypeDialog(QDialog):
    
    
    # === Init dialog with form fields ===
    def __init__(self, parent=None, data: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Loại xe")

        self._data = data or {}
        self.name_edit = QLineEdit()
        self.desc_edit = QTextEdit()
        self.active_chk = QCheckBox("Đang sử dụng")

        if data:
            self.name_edit.setText(data.get("name", ""))
            self.desc_edit.setPlainText(data.get("description", ""))
            self.active_chk.setChecked(bool(data.get("is_active", True)))
        else:
            self.active_chk.setChecked(True)

        form = QFormLayout()
        form.addRow("Tên loại xe:", self.name_edit)
        form.addRow("Mô tả:", self.desc_edit)
        form.addRow("", self.active_chk)

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

    
    
    
    
    # === Get data from form fields ===
    def get_data(self) -> Dict:
        return {
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip(),
            "is_active": self.active_chk.isChecked(),
        }







# ====== Vehicle Types Config Page ======
class VehicleTypesConfigPage(QWidget):
    
    
    # === Init page with table and buttons ===
    def __init__(self, parent=None):
        super().__init__(parent)

        self.db = DB(CONN_STR)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Tên loại xe", "Mô tả", "Đang sử dụng"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.btn_refresh = QPushButton("Tải lại")
        self.btn_add = QPushButton("Thêm loại xe")
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

    
    
    
    
    
    # === Get vehicle_type_id of current selected row ===
    def _current_row_id(self) -> Optional[int]:
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

    
    
    
    
    
    #  === Load data into table ===
    def load_data(self):
        self.table.setRowCount(0)
        if not getattr(self.db, "ok", False):
            return

        try:
            data: List[Dict] = self.db.get_vehicle_types()  # type: ignore[attr-defined]
        except AttributeError:
            data = []

        for r_idx, vt in enumerate(data):
            vt_id = vt.get("vehicle_type_id") or vt.get("id")
            if vt_id is None:
                continue

            self.table.insertRow(r_idx)
            id_item = QTableWidgetItem(str(vt_id))
            name_item = QTableWidgetItem(vt.get("name", ""))
            desc_item = QTableWidgetItem(vt.get("description", ""))
            active_item = QTableWidgetItem("Có" if vt.get("is_active", True) else "Không")

            # ID không cho sửa
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(r_idx, 0, id_item)
            self.table.setItem(r_idx, 1, name_item)
            self.table.setItem(r_idx, 2, desc_item)
            self.table.setItem(r_idx, 3, active_item)





    # === Button Handlers ===
    def on_add(self):
        dlg = VehicleTypeDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên loại xe không được để trống.")
                return
            try:
                vt_id = self.db.insert_vehicle_type(  
                    name=data["name"],
                    description=data["description"],
                    is_active=data["is_active"],
                )
                if vt_id is None:
                    QMessageBox.critical(
                        self,
                        "Lỗi",
                        "Không lưu được loại xe. Vui lòng kiểm tra lại cơ sở dữ liệu.",
                    )
            except AttributeError:
                QMessageBox.critical(self, "Lỗi", "Hàm insert_vehicle_type chưa được khai báo trong DB.")
            self.load_data()





    # === Edit selected row ===
    def on_edit(self):
        vt_id = self._current_row_id()
        if vt_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để sửa.")
            return

        row = self.table.currentRow()
        name_item = self.table.item(row, 1)
        desc_item = self.table.item(row, 2)
        active_item = self.table.item(row, 3)
        if not (name_item and desc_item and active_item):
            return

        name = name_item.text()
        desc = desc_item.text()
        active = active_item.text() == "Có"

        dlg = VehicleTypeDialog(
            self,
            data={
                "id": vt_id,
                "name": name,
                "description": desc,
                "is_active": active,
            },
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên loại xe không được để trống.")
                return
            try:
                self.db.update_vehicle_type(  
                    vt_id,
                    name=data["name"],
                    description=data["description"],
                    is_active=data["is_active"],
                )
            except AttributeError:
                QMessageBox.critical(self, "Lỗi", "Hàm update_vehicle_type chưa được khai báo trong DB.")
            self.load_data()






    # === Delete selected row ===
    def on_delete(self):
        vt_id = self._current_row_id()
        if vt_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để xóa.")
            return

        if (
            QMessageBox.question(
                self,
                "Xác nhận",
                f"Bạn chắc chắn muốn xóa loại xe ID = {vt_id}?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        try:
            self.db.delete_vehicle_type(vt_id)  
        except AttributeError:
            QMessageBox.critical(self, "Lỗi", "Hàm delete_vehicle_type chưa được khai báo trong DB.")
        self.load_data()
