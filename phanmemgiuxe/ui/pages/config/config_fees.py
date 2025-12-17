from __future__ import annotations

from typing import Optional, Dict, List, Any

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
    QComboBox,
    QLineEdit,
    QTextEdit,
    QCheckBox,
    QSizePolicy,
)

from ...theme import normalize_button, apply_button_style
from ....config.config import CONN_STR
from ....database.database import DB, SESSION_CAT_INTERNAL, SESSION_CAT_TRANSIENT





# === Choices for combo boxes ===
RULE_TYPE_CHOICES: List[tuple[str, str]] = [
    ("DAYTIME", "Gửi ban ngày (DAYTIME)"),
    ("OVERNIGHT_24H", "Qua đêm ≤24h (OVERNIGHT_24H)"),
    ("PER_DAY", "Theo ngày (PER_DAY)"),
    ("FREE", "Miễn phí (FREE)"),
]





# === Choices for unit combo box ===
UNIT_CHOICES: List[tuple[str, str]] = [
    ("", "— Không đặt đơn vị —"),
    ("Lượt", "Lượt"),
    ("Giờ", "Giờ"),
    ("Ngày", "Ngày"),
]





# === Dialog Thêm / Sửa rule phí gửi xe ===
class FeeRuleDialog(QDialog):
    """
    Dialog thêm / sửa 1 dòng cấu hình phí gửi xe.

    Mapping với DB FeeRules:

        dbo.FeeRules(
            fee_rule_id INT IDENTITY PRIMARY KEY,
            vehicle_type_id INT NOT NULL,
            category NVARCHAR(32) NULL,      -- VISITOR / INTERNAL / ALL
            rule_type NVARCHAR(64) NULL,     -- loại rule (VD: DAYTIME, PER_DAY,...)
            price INT NOT NULL,
            effective_from DATETIME NULL,
            effective_to DATETIME NULL,
            is_active BIT NOT NULL DEFAULT 1,
            description NVARCHAR(MAX) NULL,
            created_at DATETIME DEFAULT GETDATE(),
            unit NVARCHAR(32) NULL           -- đơn vị (Lượt, Giờ, Ngày,...)
        )
    """





    # === Init dialog ===
    def __init__(
        self,
        parent: QWidget | None,
        vehicle_type_map: Dict[int, str],
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cấu hình phí gửi xe")
        self.setModal(True)
        self.resize(420, 320)

        self._vehicle_type_map = vehicle_type_map
        self._data = data or {}

        main = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # ================== LOẠI XE ==================
        self.cbo_vehicle_type = QComboBox(self)
        self.cbo_vehicle_type.setEditable(False)
        # KHÔNG còn '-- Chọn loại xe --' nữa
        for vt_id, vt_name in sorted(self._vehicle_type_map.items()):
            self.cbo_vehicle_type.addItem(vt_name, vt_id)

        # ================== LOẠI KHÁCH ==================
        self.cbo_session_category = QComboBox(self)
        self.cbo_session_category.addItem("Vãng lai (VISITOR)", SESSION_CAT_TRANSIENT)
        self.cbo_session_category.addItem("Nội bộ (INTERNAL)", SESSION_CAT_INTERNAL)

        # ================== LOẠI RULE ==================
        self.cbo_rule_type = QComboBox(self)
        self.cbo_rule_type.setEditable(False)
        # KHÔNG còn '-- Chọn loại rule --'
        for code, label in RULE_TYPE_CHOICES:
            self.cbo_rule_type.addItem(label, code)

        # ================== GIÁ ==================
        self.edt_fee_amount = QLineEdit(self)
        self.edt_fee_amount.setPlaceholderText("Nhập số tiền, ví dụ: 5000")
        self.edt_fee_amount.setMaximumWidth(160)

        # ================== ĐƠN VỊ ==================
        self.cbo_unit = QComboBox(self)
        self.cbo_unit.setEditable(False)
        for code, label in UNIT_CHOICES:
            self.cbo_unit.addItem(label, code)

        # ================== GHI CHÚ ==================
        self.txt_description = QTextEdit(self)
        self.txt_description.setPlaceholderText("Ghi chú thêm cho rule phí (tuỳ chọn).")
        self.txt_description.setFixedHeight(80)

        # ================== TRẠNG THÁI ==================
        self.chk_active = QCheckBox("Đang sử dụng", self)
        self.chk_active.setChecked(True)

        form.addRow("Loại xe:", self.cbo_vehicle_type)
        form.addRow("Loại khách:", self.cbo_session_category)
        form.addRow("Loại rule:", self.cbo_rule_type)
        form.addRow("Giá:", self.edt_fee_amount)
        form.addRow("Đơn vị:", self.cbo_unit)
        form.addRow("Ghi chú:", self.txt_description)
        form.addRow("", self.chk_active)
        main.addLayout(form)

        # ================== NÚT LƯU / HỦY ==================
        row_btn = QHBoxLayout()
        row_btn.addStretch(1)
        self.btn_ok = QPushButton("Lưu", self)
        self.btn_cancel = QPushButton("Hủy", self)

        normalize_button(self.btn_ok, self.btn_cancel)
        primary_css = """
        QPushButton {
            background-color:#2563eb;
            color:#ffffff;
            border-radius:6px;
            padding:6px 16px;
            font-weight:600;
        }
        QPushButton:hover {
            background-color:#1e40af;
        }
        QPushButton:pressed {
            background-color:#1d4ed8;
        }
        """
        apply_button_style(self.btn_ok, primary_css)

        row_btn.addWidget(self.btn_ok)
        row_btn.addWidget(self.btn_cancel)
        main.addLayout(row_btn)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        # ====== Nếu là sửa: fill data; nếu là thêm mới: chọn gợi ý giống "Loại khách" ======
        if self._data:
            self._load_from_data(self._data)
        else:
            if self.cbo_vehicle_type.count() > 0:
                self.cbo_vehicle_type.setCurrentIndex(0)      # loại xe đầu tiên
            self.cbo_session_category.setCurrentIndex(0)       # Vãng lai (VISITOR)
            if self.cbo_rule_type.count() > 0:
                self.cbo_rule_type.setCurrentIndex(0)          # Gửi ban ngày (DAYTIME)
            if self.cbo_unit.count() > 0:
                self.cbo_unit.setCurrentIndex(0)               # Không đặt đơn vị

    
    
    
    
    # === Load data vào dialog khi sửa ===
    def _load_from_data(self, d: Dict[str, Any]) -> None:
        # Loại xe
        vt_id = d.get("vehicle_type_id")
        if vt_id is None:
            if self.cbo_vehicle_type.count() > 0:
                self.cbo_vehicle_type.setCurrentIndex(0)
        else:
            for i in range(self.cbo_vehicle_type.count()):
                if self.cbo_vehicle_type.itemData(i) == vt_id:
                    self.cbo_vehicle_type.setCurrentIndex(i)
                    break

        # Loại khách
        cat = (d.get("session_category") or "").upper()
        for i in range(self.cbo_session_category.count()):
            if str(self.cbo_session_category.itemData(i)).upper() == cat:
                self.cbo_session_category.setCurrentIndex(i)
                break

        # Loại rule
        rt = (d.get("rule_type") or "").upper()
        for i in range(self.cbo_rule_type.count()):
            if str(self.cbo_rule_type.itemData(i)).upper() == rt:
                self.cbo_rule_type.setCurrentIndex(i)
                break

        # Giá
        self.edt_fee_amount.setText(str(d.get("fee_amount") or ""))

        # Đơn vị
        unit = str(d.get("unit") or "")
        for i in range(self.cbo_unit.count()):
            if str(self.cbo_unit.itemData(i)) == unit:
                self.cbo_unit.setCurrentIndex(i)
                break

        # Ghi chú + trạng thái
        self.txt_description.setPlainText(str(d.get("description") or ""))
        self.chk_active.setChecked(bool(d.get("is_active", True)))

    
    
    
    
    
    # === Lấy dữ liệu người dùng nhập ===
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Lấy dữ liệu người dùng nhập. Trả về None nếu validate fail."""

        # Loại xe
        vt_id = self.cbo_vehicle_type.currentData()
        if vt_id is None:
            QMessageBox.warning(self, "Lỗi", "Chưa cấu hình loại xe để chọn.")
            return None

        # Loại rule
        rule_type_code = self.cbo_rule_type.currentData()
        if rule_type_code is None:
            QMessageBox.warning(self, "Lỗi", "Chưa cấu hình loại rule.")
            return None

        # Giá
        fee_text = self.edt_fee_amount.text().strip()
        if not fee_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập giá.")
            return None
        try:
            fee_amount = int(float(fee_text))
        except Exception:
            QMessageBox.warning(self, "Lỗi", "Giá phải là số.")
            return None

        session_category = self.cbo_session_category.currentData()
        unit_code = self.cbo_unit.currentData() or ""
        description = self.txt_description.toPlainText().strip()
        is_active = self.chk_active.isChecked()

        result: Dict[str, Any] = {
            "vehicle_type_id": vt_id,
            "session_category": session_category,
            "rule_type": rule_type_code,   # lưu code, hiển thị label
            "fee_amount": fee_amount,
            "unit": unit_code,
            "description": description,
            "is_active": is_active,
        }

        if "fee_id" in self._data:
            result["fee_id"] = self._data["fee_id"]
        elif "fee_rule_id" in self._data:
            result["fee_id"] = self._data["fee_rule_id"]

        return result



# === Tab 'Phí gửi xe' trong CẤU HÌNH ===
class FeesConfigPage(QWidget):
    """
    Tab 'Phí gửi xe' trong CẤU HÌNH.

    - Bên trên: dãy nút Reload / Thêm phí / Sửa / Xóa
    - Bên dưới: bảng QTableWidget hiển thị các rule.
    """





    # === Init page ===
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.db: Optional[DB] = None
        self._vehicle_type_map: Dict[int, str] = {}
        self._rules_cache: Dict[int, Dict[str, Any]] = {}

        self._load_db()
        self._load_vehicle_types()
        self._build_ui()
        self.reload_table()

    
    
    
    
    # === Load DB và loại xe từ DB ===
    def _load_db(self) -> None:
        self.db = DB(CONN_STR)
        if not (self.db and self.db.ok):
            print("[FeesConfigPage] Không kết nối được DB (phí gửi xe).")





    # === Load vehicle types from DB ===
    def _load_vehicle_types(self) -> None:
        """Lấy danh sách loại xe để map id -> name."""
        if not (self.db and self.db.ok):
            self._vehicle_type_map = {}
            return
        try:
            vts = self.db.get_vehicle_types(include_inactive=False)
            self._vehicle_type_map = {int(v["vehicle_type_id"]): v["name"] for v in vts}
        except Exception as e:
            print("[FeesConfigPage] _load_vehicle_types error:", e)
            self._vehicle_type_map = {}





    # === Build UI ===
    def _build_ui(self) -> None:
        self.setObjectName("FeesConfigPageRoot")
        self.setStyleSheet(
            """
            QWidget#FeesConfigPageRoot {
                background-color:#f5f5f7;
            }
            QTableWidget {
                background-color:#ffffff;
                alternate-background-color:#f9fafb;
                gridline-color:#9ca3af;
                color:#111827;
            }
            QTableWidget::item:selected {
                background-color:#dbeafe;
                color:#111827;
            }
            QHeaderView::section {
                background-color:#e5e7eb;
                border:1px solid #9ca3af;
                padding:4px;
                color:#111827;
                font-weight:bold;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ===== Thanh nút trên =====
        top = QHBoxLayout()
        self.btn_reload = QPushButton("Tải lại", self)
        self.btn_add = QPushButton("Thêm phí", self)
        self.btn_edit = QPushButton("Sửa", self)
        self.btn_delete = QPushButton("Xóa", self)

        normalize_button(self.btn_reload, self.btn_add, self.btn_edit, self.btn_delete)

        for btn in (self.btn_reload, self.btn_add, self.btn_edit, self.btn_delete):
            btn.setMinimumWidth(90)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # primary style cho các nút action bên phải
        primary_css = """
        QPushButton {
            background-color:#4b5563;
            color:#ffffff;
            border-radius:4px;
            padding:4px 10px;
            font-weight:500;
        }
        QPushButton:hover {
            background-color:#374151;
        }
        QPushButton:pressed {
            background-color:#111827;
        }
        """
        apply_button_style(self.btn_add, primary_css)
        apply_button_style(self.btn_edit, primary_css)
        apply_button_style(self.btn_delete, primary_css)

        top.addWidget(self.btn_reload)
        top.addStretch(1)
        top.addWidget(self.btn_add)
        top.addWidget(self.btn_edit)
        top.addWidget(self.btn_delete)

        root.addLayout(top)

        # ===== Bảng phí =====
        self.tbl = QTableWidget(0, 8, self)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.tbl.verticalHeader().setVisible(False)

        self.tbl.setHorizontalHeaderLabels(
            [
                "ID",
                "Loại xe",
                "Loại khách",
                "Loại rule",
                "Giá",
                "Đơn vị",
                "Ghi chú",
                "Đang sử dụng",
            ]
        )

        header: QHeaderView = self.tbl.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        for col in (1, 2, 3, 5, 6):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        root.addWidget(self.tbl, 1)

        # ===== Signals =====
        self.btn_reload.clicked.connect(self.reload_table)
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_edit.clicked.connect(self._on_edit_clicked)
        self.btn_delete.clicked.connect(self._on_delete_clicked)





    # === Label cho loại khách (session category) ===
    def _session_category_label(self, cat: str) -> str:
        cat = (cat or "").upper()
        if cat == SESSION_CAT_INTERNAL:
            return "Nội bộ"
        if cat == SESSION_CAT_TRANSIENT:
            return "Vãng lai"
        return cat or ""

    
    
    
    
    
    # === Reload table from DB ===
    def reload_table(self) -> None:
        """Load danh sách rule phí từ DB lên bảng."""
        self.tbl.setRowCount(0)
        self._rules_cache.clear()

        if not (self.db and self.db.ok):
            return

        rules = self.db.get_fee_rules()
        if not rules:
            return

        self.tbl.setRowCount(len(rules))

        for row_idx, r in enumerate(rules):
            fee_id_raw = r.get("fee_id")
            if fee_id_raw is None:
                continue  # Skip this record if fee_id is None
            fee_id = int(fee_id_raw)
            self._rules_cache[fee_id] = r

            # ID
            item_id = QTableWidgetItem(str(fee_id))
            item_id.setFlags(item_id.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 0, item_id)

            # Loại xe
            vt_id = r.get("vehicle_type_id")
            if vt_id is None:
                vt_name = "Tất cả loại xe"
            else:
                vt_name = self._vehicle_type_map.get(int(vt_id), f"ID={vt_id}")
            item_vt = QTableWidgetItem(vt_name)
            item_vt.setFlags(item_vt.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 1, item_vt)

            # Loại khách
            cat_label = self._session_category_label(r.get("session_category") or "")
            item_cat = QTableWidgetItem(cat_label)
            item_cat.setFlags(item_cat.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 2, item_cat)

            # Loại rule
            item_rule = QTableWidgetItem(str(r.get("rule_type") or ""))
            item_rule.setFlags(item_rule.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 3, item_rule)

            # Giá
            try:
                fee_val = int(r.get("fee_amount", 0))
                fee_text = f"{fee_val:,}"
            except Exception:
                fee_text = str(r.get("fee_amount") or "")
            item_fee = QTableWidgetItem(fee_text)
            item_fee.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item_fee.setFlags(item_fee.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 4, item_fee)

            # Đơn vị
            item_unit = QTableWidgetItem(str(r.get("unit") or ""))
            item_unit.setFlags(item_unit.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 5, item_unit)

            # Ghi chú
            item_desc = QTableWidgetItem(str(r.get("description") or ""))
            item_desc.setFlags(item_desc.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 6, item_desc)

            # Đang sử dụng
            active = bool(r.get("is_active", True))
            status_text = "Có" if active else "Không"
            item_active = QTableWidgetItem(status_text)
            item_active.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_active.setFlags(item_active.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.tbl.setItem(row_idx, 7, item_active)

        self.tbl.resizeRowsToContents()

    
    
    
    
    
    # === Lấy fee_id của dòng đang chọn ===
    def _get_selected_fee_id(self) -> Optional[int]:
        selected = self.tbl.selectedIndexes()
        if not selected:
            return None
        row = selected[0].row()
        item_id = self.tbl.item(row, 0)
        if not item_id:
            return None
        try:
            return int(item_id.text())
        except Exception:
            return None

    
    
    
    
    
    # === Thêm phí mới ===
    def _on_add_clicked(self) -> None:
        if not (self.db and self.db.ok):
            QMessageBox.warning(self, "Lỗi", "Chưa kết nối DB.")
            return

        dlg = FeeRuleDialog(self, self._vehicle_type_map, data=None)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        data = dlg.get_data()
        if data is None:
            return

        fee_id = self.db.insert_fee_rule(
            session_category=data["session_category"],
            rule_type=data["rule_type"],
            fee_amount=data["fee_amount"],
            unit=data["unit"],
            description=data["description"],
            is_active=data["is_active"],
            vehicle_type_id=data["vehicle_type_id"],
        )

        if fee_id is None:
            QMessageBox.warning(
                self, "Lỗi", "Không thêm được rule phí. Xem log console để biết chi tiết."
            )
        self.reload_table()

    
    
    
    
    
    # === Sửa phí đang chọn ===
    def _on_edit_clicked(self) -> None:
        """
        Sửa 1 rule phí đang chọn trong bảng.
        """
        if not (self.db and self.db.ok):
            QMessageBox.warning(self, "Lỗi", "Chưa kết nối DB.")
            return

        fee_id = self._get_selected_fee_id()
        if fee_id is None:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn 1 dòng để sửa.")
            return

        current_rule = self._rules_cache.get(fee_id, {})
        dlg = FeeRuleDialog(self, self._vehicle_type_map, data=current_rule)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        data = dlg.get_data()
        if data is None:
            return

        # Nếu sau này dialog có thêm ngày hiệu lực, ưu tiên data; nếu không thì giữ giá trị cũ
        eff_from = data.get("effective_from", current_rule.get("effective_from"))
        eff_to = data.get("effective_to", current_rule.get("effective_to"))
        description = data.get("description", current_rule.get("description"))

        try:
            self.db.update_fee_rule(
                fee_rule_id=fee_id,
                vehicle_type_id=data["vehicle_type_id"],
                session_category=data["session_category"],
                rule_type=data["rule_type"],
                price=data["fee_amount"],
                unit=data["unit"],
                effective_from=eff_from,
                effective_to=eff_to,
                is_active=data["is_active"],
                description=description,
            )
        except Exception as e:
            print("[FeesConfigPage] _on_edit_clicked update_fee_rule error:", e)
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không cập nhật được rule phí. Xem log console để biết chi tiết.",
            )
            return

        # Load lại bảng
        self.reload_table()





    # === Xóa phí đang chọn ===
    def _on_delete_clicked(self) -> None:
        if not (self.db and self.db.ok):
            QMessageBox.warning(self, "Lỗi", "Chưa kết nối DB.")
            return

        fee_id = self._get_selected_fee_id()
        if fee_id is None:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn 1 dòng để xóa.")
            return

        ans = QMessageBox.question(
            self,
            "Xác nhận",
            f"Bạn chắc chắn muốn xóa rule phí ID = {fee_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return

        self.db.delete_fee_rule(fee_id)
        self.reload_table()
