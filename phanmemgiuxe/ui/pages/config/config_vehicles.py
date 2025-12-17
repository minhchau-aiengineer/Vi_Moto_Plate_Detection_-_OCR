from typing import List, Dict, Optional
from PySide6.QtCore import Qt
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
    QComboBox,
    QCheckBox,
    QAbstractItemView,
)

from ....database.database import DB
from ....config.config import CONN_STR




# ====== Vehicles Config Page ======
class VehicleDialog(QDialog):
    
    
    # === Init dialog with form fields ===
    def __init__(self, parent=None, data: Optional[Dict] = None, vehicle_types: Optional[List[Dict]] = None):
        super().__init__(parent)
        self.setWindowTitle("Xe nội bộ")

        self._data = data or {}
        self._vehicle_types = vehicle_types or []
        self.plate_edit = QLineEdit()
        self.owner_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.note_edit = QTextEdit()
        self.active_chk = QCheckBox("Đang sử dụng")

        self.type_combo.addItem("--- Không chọn ---", None)
        for vt in self._vehicle_types:
            self.type_combo.addItem(vt["name"], vt["id"])

        if data:
            self.plate_edit.setText(data.get("plate", ""))
            self.owner_edit.setText(data.get("owner_name", ""))
            self.note_edit.setPlainText(data.get("note", ""))
            self.active_chk.setChecked(bool(data.get("is_active", True)))

            vt_id = data.get("vehicle_type_id")
            if vt_id is not None:
                for i in range(self.type_combo.count()):
                    if self.type_combo.itemData(i) == vt_id:
                        self.type_combo.setCurrentIndex(i)
                        break
        else:
            self.active_chk.setChecked(True)

        form = QFormLayout()
        form.addRow("Biển số:", self.plate_edit)
        form.addRow("Chủ xe:", self.owner_edit)
        form.addRow("Loại xe:", self.type_combo)
        form.addRow("Ghi chú:", self.note_edit)
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
            "plate": self.plate_edit.text().strip(),
            "owner_name": self.owner_edit.text().strip(),
            "vehicle_type_id": self.type_combo.currentData(),
            "note": self.note_edit.toPlainText().strip(),
            "is_active": self.active_chk.isChecked(),
        }






# ====== Vehicles Config Page ======
class VehiclesConfigPage(QWidget):
    
    
    # === Init page with table and buttons ===
    def __init__(self, parent=None):
        super().__init__(parent)

        self.db = DB(CONN_STR)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Biển số", "Chủ xe", "Loại xe", "Ghi chú", "Đang sử dụng"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.btn_refresh = QPushButton("Tải lại")
        self.btn_add = QPushButton("Thêm xe nội bộ")
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






    # === Get vehicle_id of current selected row ===
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






    # === Load vehicle types for dialog ===
    def _load_vehicle_types(self) -> List[Dict]:
        if not getattr(self.db, "ok", False):
            return []
        try:
            return self.db.get_vehicle_types()  
        except AttributeError:
            return []





    # === Load data into table ===
    def load_data(self):
        self.table.setRowCount(0)
        if not getattr(self.db, "ok", False):
            return

        try:
            data: List[Dict] = self.db.get_vehicles_with_type()  
        except AttributeError:
            data = []

        for r_idx, v in enumerate(data):
            self.table.insertRow(r_idx)

            id_item = QTableWidgetItem(str(v["id"]))
            plate_item = QTableWidgetItem(v.get("plate", ""))
            owner_item = QTableWidgetItem(v.get("owner_name", ""))
            type_item = QTableWidgetItem(v.get("vehicle_type_name", ""))
            note_item = QTableWidgetItem(v.get("note", ""))
            active_item = QTableWidgetItem("Có" if v.get("is_active", True) else "Không")

            flags = id_item.flags()
            flags &= ~Qt.ItemFlag.ItemIsEditable
            id_item.setFlags(flags)


            self.table.setItem(r_idx, 0, id_item)
            self.table.setItem(r_idx, 1, plate_item)
            self.table.setItem(r_idx, 2, owner_item)
            self.table.setItem(r_idx, 3, type_item)
            self.table.setItem(r_idx, 4, note_item)
            self.table.setItem(r_idx, 5, active_item)





    # === Button Handlers ===
    def on_add(self):
        vtypes = self._load_vehicle_types()
        dlg = VehicleDialog(self, vehicle_types=vtypes)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["plate"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Biển số không được để trống.")
                return
            try:
                self.db.insert_vehicle( 
                    plate=data["plate"],
                    owner_name=data["owner_name"],
                    vehicle_type_id=data["vehicle_type_id"],
                    note=data["note"],
                    is_active=data["is_active"],
                )
            except AttributeError:
                QMessageBox.critical(self, "Lỗi", "Hàm insert_vehicle chưa được khai báo trong DB.")
            self.load_data()





    # === Edit selected row ===
    def on_edit(self):
        v_id = self._current_row_id()
        if v_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để sửa.")
            return

        row = self.table.currentRow()
        plate = self.table.item(row, 1).text()
        owner = self.table.item(row, 2).text()
        vtype_name = self.table.item(row, 3).text()
        note = self.table.item(row, 4).text()
        active = self.table.item(row, 5).text() == "Có"

        vtypes = self._load_vehicle_types()
        vehicle_type_id = None
        for vt in vtypes:
            if vt["name"] == vtype_name:
                vehicle_type_id = vt["id"]
                break

        dlg = VehicleDialog(
            self,
            data={
                "id": v_id,
                "plate": plate,
                "owner_name": owner,
                "vehicle_type_id": vehicle_type_id,
                "note": note,
                "is_active": active,
            },
            vehicle_types=vtypes,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["plate"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Biển số không được để trống.")
                return
            try:
                self.db.update_vehicle(  # type: ignore[attr-defined]
                    v_id,
                    plate=data["plate"],
                    owner_name=data["owner_name"],
                    vehicle_type_id=data["vehicle_type_id"],
                    note=data["note"],
                    is_active=data["is_active"],
                )
            except AttributeError:
                QMessageBox.critical(self, "Lỗi", "Hàm update_vehicle chưa được khai báo trong DB.")
            self.load_data()






    # === Delete selected row ===
    def on_delete(self):
        v_id = self._current_row_id()
        if v_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để xóa.")
            return

        if (
            QMessageBox.question(
                self,
                "Xác nhận",
                f"Bạn chắc chắn muốn xóa xe nội bộ ID = {v_id}?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

        try:
            self.db.delete_vehicle(v_id)  # type: ignore[attr-defined]
        except AttributeError:
            QMessageBox.critical(self, "Lỗi", "Hàm delete_vehicle chưa được khai báo trong DB.")
        self.load_data()
