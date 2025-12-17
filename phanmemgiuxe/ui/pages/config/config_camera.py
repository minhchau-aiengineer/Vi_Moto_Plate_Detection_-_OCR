
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
    QComboBox,
    QLabel,
    QComboBox,
    QLabel,
)
from PySide6.QtCore import Qt




# ====== Camera Config Dialog ======
class CameraDialog(QDialog):
    def __init__(self, parent=None, data: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera")
        self._data = data or {}

        from PySide6.QtWidgets import QComboBox
        self.name_edit = QLineEdit()
        self.type_edit = QLineEdit()
        self.source_index_edit = QLineEdit()
        self.ip_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.url_path_edit = QLineEdit()
        self.full_url_edit = QLineEdit()
        self.username_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.direction_edit = QLineEdit()
        self.view_role_edit = QLineEdit()
        self.is_active_edit = QComboBox()
        self.is_active_edit.addItems(["1", "0"])
        self.note_edit = QTextEdit()

        if data:
            self.name_edit.setText(data.get("camera_name", ""))
            self.type_edit.setText(data.get("camera_type", ""))
            self.source_index_edit.setText(str(data.get("source_index", "")))
            self.ip_edit.setText(data.get("ip_address", ""))
            self.port_edit.setText(str(data.get("port", "")))
            self.url_path_edit.setText(data.get("url_path", ""))
            self.full_url_edit.setText(data.get("full_url", ""))
            self.username_edit.setText(data.get("username", ""))
            self.password_edit.setText(data.get("password", ""))
            self.direction_edit.setText(data.get("direction", ""))
            self.view_role_edit.setText(str(data.get("view_role", "")))
            idx_active = self.is_active_edit.findText(str(data.get("is_active", 1)))
            if idx_active >= 0:
                self.is_active_edit.setCurrentIndex(idx_active)
            self.note_edit.setPlainText(data.get("note", ""))
        else:
            self.is_active_edit.setCurrentIndex(0)

        form = QFormLayout()
        form.addRow("Tên camera:", self.name_edit)
        form.addRow("Loại:", self.type_edit)
        form.addRow("Source Index:", self.source_index_edit)
        form.addRow("IP:", self.ip_edit)
        form.addRow("Port:", self.port_edit)
        form.addRow("URL Path:", self.url_path_edit)
        form.addRow("Full URL:", self.full_url_edit)
        form.addRow("Username:", self.username_edit)
        form.addRow("Password:", self.password_edit)
        form.addRow("Hướng:", self.direction_edit)
        form.addRow("Vai trò 4view:", self.view_role_edit)
        form.addRow("is_active:", self.is_active_edit)
        form.addRow("Ghi chú:", self.note_edit)

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
    def get_data(self) -> Dict:
        def parse_int(val):
            val = val.strip()
            if val == "" or val.lower() == "none":
                return None
            try:
                return int(val)
            except Exception:
                return None
        return {
            "camera_name": self.name_edit.text().strip(),
            "camera_type": self.type_edit.text().strip(),
            "source_index": parse_int(self.source_index_edit.text()),
            "ip_address": self.ip_edit.text().strip(),
            "port": parse_int(self.port_edit.text()),
            "url_path": self.url_path_edit.text().strip(),
            "full_url": self.full_url_edit.text().strip(),
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text().strip(),
            "direction": self.direction_edit.text().strip(),
            "view_role": self.view_role_edit.text().strip() or None,
            "is_active": int(self.is_active_edit.currentText()),
            "note": self.note_edit.toPlainText().strip(),
        }


from ....database.camera_config_db import CameraConfigDB
from ....config.config import CONN_STR




# ======= Camera Config Page =======
class CameraConfigPage(QWidget):
    
    
    # ==== Init UI =======
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_db = CameraConfigDB(CONN_STR)

        self.table = QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
            "ID", "Tên camera", "Loại", "Source Index", "IP", "Port", "URL Path", "Full URL", "Username", "Password", "Hướng", "Vai trò 4view", "is_active", "Ghi chú"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.btn_refresh = QPushButton("Tải lại")
        self.btn_add = QPushButton("Thêm camera")
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

        # --- Khung mapping chức năng camera ---
        self.mapping_functions = [
            ("vao_truoc", "Cam vào trước"),
            ("vao_sau", "Cam vào sau"),
            ("ra_truoc", "Cam ra trước"),
            ("ra_sau", "Cam ra sau"),
        ]
        self.combo_mapping = {}
        self._update_camera_comboboxes()
        h_mapping = QHBoxLayout()
        h_mapping.setSpacing(24)
        for func, label in self.mapping_functions:
            box = QHBoxLayout()
            lbl = QLabel(label)
            combo = QComboBox()
            combo.setFixedWidth(200)
            box.addWidget(lbl)
            box.addWidget(combo)
            h_mapping.addLayout(box)
            self.combo_mapping[func] = combo
        btn_save_mapping = QPushButton("Lưu mapping camera")
        btn_save_mapping.setFixedWidth(180)
        btn_save_mapping.clicked.connect(self.save_camera_mapping)
        h_mapping.addWidget(btn_save_mapping)
        layout.addLayout(h_mapping)

        self.btn_refresh.clicked.connect(self.load_and_update_table)
        self.btn_add.clicked.connect(self.on_add)
        self.btn_edit.clicked.connect(self.on_edit)
        self.btn_delete.clicked.connect(self.on_delete)

        self.load_and_update_table()





    # === Update camera comboboxes for 4view mapping ===
    def _update_camera_comboboxes(self):
        cameras = self.camera_db.get_all_active_cameras()
        names = [cam.get("camera_name", "") for cam in cameras]
        mapping = self.camera_db.get_camera_mapping_configs() if hasattr(self.camera_db, 'get_camera_mapping_configs') else {}
        for func, combo in self.combo_mapping.items():
            combo.clear()
            combo.addItems(names)
            mapped_cam = mapping.get(func)
            mapped_name = mapped_cam.get("camera_name") if mapped_cam else None
            if mapped_name and mapped_name in names:
                idx = combo.findText(mapped_name)
                if idx >= 0:
                    combo.setCurrentIndex(idx)





    # === Save camera mapping for 4view functions ===
    def save_camera_mapping(self):
        from ....database.camera_config_db import CameraConfigDB
        db = CameraConfigDB(CONN_STR)
        cameras = db.get_all_active_cameras()
        name_to_id = {cam.get("camera_name", ""): cam.get("camera_id") for cam in cameras}
        success = True
        for func, combo in self.combo_mapping.items():
            cam_name = combo.currentText()
            cam_id = name_to_id.get(cam_name)
            if not cam_id:
                continue
            db._execute("DELETE FROM dbo.CameraMapping WHERE function_type = ?", (func,))
            ok = db._execute(
                "INSERT INTO dbo.CameraMapping (function_type, camera_id) VALUES (?, ?)",
                (func, cam_id)
            )
            if not ok:
                success = False
        if success:
            QMessageBox.information(self, "Lưu mapping camera", "Đã lưu mapping thành công!")
        else:
            QMessageBox.critical(self, "Lỗi", "Lưu mapping thất bại!")





    # === Load and update table data ===
    def load_and_update_table(self):
        self.table.setRowCount(0)
        cameras = self.camera_db.get_all_active_cameras()
        for r_idx, cam in enumerate(cameras):
            self.table.insertRow(r_idx)
            id_item = QTableWidgetItem(str(cam.get("camera_id", "")))
            name_item = QTableWidgetItem(cam.get("camera_name", ""))
            type_item = QTableWidgetItem(cam.get("camera_type", ""))
            source_index_item = QTableWidgetItem(str(cam.get("source_index", "")))
            ip_item = QTableWidgetItem(cam.get("ip_address", ""))
            port_item = QTableWidgetItem(str(cam.get("port", "")))
            url_path_item = QTableWidgetItem(cam.get("url_path", ""))
            full_url_item = QTableWidgetItem(cam.get("full_url", ""))
            user_item = QTableWidgetItem(cam.get("username", ""))
            pass_item = QTableWidgetItem(cam.get("password", ""))
            dir_item = QTableWidgetItem(cam.get("direction", ""))
            role_item = QTableWidgetItem(str(cam.get("view_role", "")))
            is_active_item = QTableWidgetItem(str(cam.get("is_active", "")))
            note_item = QTableWidgetItem(cam.get("note", ""))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r_idx, 0, id_item)
            self.table.setItem(r_idx, 1, name_item)
            self.table.setItem(r_idx, 2, type_item)
            self.table.setItem(r_idx, 3, source_index_item)
            self.table.setItem(r_idx, 4, ip_item)
            self.table.setItem(r_idx, 5, port_item)
            self.table.setItem(r_idx, 6, url_path_item)
            self.table.setItem(r_idx, 7, full_url_item)
            self.table.setItem(r_idx, 8, user_item)
            self.table.setItem(r_idx, 9, pass_item)
            self.table.setItem(r_idx, 10, dir_item)
            self.table.setItem(r_idx, 11, role_item)
            self.table.setItem(r_idx, 12, is_active_item)
            self.table.setItem(r_idx, 13, note_item)
        # Cập nhật lại combobox chọn camera cho 4view
        self._update_camera_comboboxes()





    # === Load current selected row ID ===
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





    # === Load data into table ===
    def load_data(self):
        self.table.setRowCount(0)
        cameras = self.camera_db.get_all_active_cameras()
        for r_idx, cam in enumerate(cameras):
            self.table.insertRow(r_idx)
            id_item = QTableWidgetItem(str(cam.get("camera_id", "")))
            name_item = QTableWidgetItem(cam.get("camera_name", ""))
            type_item = QTableWidgetItem(cam.get("camera_type", ""))
            ip_item = QTableWidgetItem(cam.get("ip_address", ""))
            port_item = QTableWidgetItem(str(cam.get("port", "")))
            url_item = QTableWidgetItem(cam.get("full_url", ""))
            user_item = QTableWidgetItem(cam.get("username", ""))
            pass_item = QTableWidgetItem(cam.get("password", ""))
            dir_item = QTableWidgetItem(cam.get("direction", ""))
            active_item = QTableWidgetItem("Có" if cam.get("is_active", True) else "Không")
            note_item = QTableWidgetItem(cam.get("note", ""))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r_idx, 0, id_item)
            self.table.setItem(r_idx, 1, name_item)
            self.table.setItem(r_idx, 2, type_item)
            self.table.setItem(r_idx, 3, ip_item)
            self.table.setItem(r_idx, 4, port_item)
            self.table.setItem(r_idx, 5, url_item)
            self.table.setItem(r_idx, 6, user_item)
            self.table.setItem(r_idx, 7, pass_item)
            self.table.setItem(r_idx, 8, dir_item)
            self.table.setItem(r_idx, 9, active_item)
            self.table.setItem(r_idx, 10, note_item)
            

    
    
    
    # === Button Handlers ===
    def on_add(self):
        dlg = CameraDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["camera_name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên camera không được để trống.")
                return
            ok = self.camera_db.add_camera(data)
            if not ok:
                QMessageBox.critical(self, "Lỗi", "Không thêm được camera. Kiểm tra lại kết nối DB.")
            self.load_data()

    
    
    
    
    # === Edit selected row ===
    def on_edit(self):
        cam_id = self._current_row_id()
        if cam_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để sửa.")
            return
        row = self.table.currentRow()
        def get_item_text(row, col):
            item = self.table.item(row, col)
            return item.text() if item else ""
        cam_data = {
            "camera_name": get_item_text(row, 1),
            "camera_type": get_item_text(row, 2),
            "source_index": get_item_text(row, 3),
            "ip_address": get_item_text(row, 4),
            "port": get_item_text(row, 5),
            "url_path": get_item_text(row, 6),
            "full_url": get_item_text(row, 7),
            "username": get_item_text(row, 8),
            "password": get_item_text(row, 9),
            "direction": get_item_text(row, 10),
            "view_role": get_item_text(row, 11),
            "is_active": int(get_item_text(row, 12)) if get_item_text(row, 12).isdigit() else 1,
            "note": get_item_text(row, 13),
        }
        dlg = CameraDialog(self, cam_data)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if not data["camera_name"]:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Tên camera không được để trống.")
                return
            ok = self.camera_db.update_camera(cam_id, data)
            if not ok:
                QMessageBox.critical(self, "Lỗi", "Không cập nhật được camera. Kiểm tra lại kết nối DB.")
            self.load_and_update_table()





    # === Delete selected row ===
    def on_delete(self):
        cam_id = self._current_row_id()
        if cam_id is None:
            QMessageBox.information(self, "Thông báo", "Hãy chọn 1 dòng để xóa.")
            return
        if (
            QMessageBox.question(
                self,
                "Xác nhận",
                f"Bạn chắc chắn muốn xóa camera ID = {cam_id}?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        ok = self.camera_db.delete_camera(cam_id)
        if not ok:
            QMessageBox.critical(self, "Lỗi", "Không xóa được camera. Kiểm tra lại kết nối DB.")
        self.load_data()
