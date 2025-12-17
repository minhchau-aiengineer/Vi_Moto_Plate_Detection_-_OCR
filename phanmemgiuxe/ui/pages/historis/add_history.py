# ui/pages/add_history.py
"""
AddHistoryDialog

Dialog THÊM XE (2 cột: vào / ra + trạng thái + checkbox Vào/Ra).

- Người dùng chọn:
    + Áp dụng cho: Vào / Ra (1 hoặc 2 ô)
    + Loại xe: Tự động / Vãng lai / Nội bộ
    + Đường dẫn ảnh, biển số, ngày, giờ tương ứng
    + Trạng thái (PENDING / KHOP-BIEN-SO / KHONG-KHOP-BIEN-SO)
- Khi LƯU:
    + Ít nhất phải tick Vào hoặc Ra
    + Nếu tick Vào -> kiểm tra biển số vào, tạo dữ liệu vào
    + Nếu tick Ra   -> kiểm tra biển số ra, tạo dữ liệu ra
    + Tính toán session_category + vehicle_id + vehicle_type_id
    + Gọi db.insert_history_record(record) để lưu xuống DB
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QDate, QTime, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ....database.database import (  # type: ignore[import-not-found]
    DB,
    SESSION_CAT_TRANSIENT,
    SESSION_CAT_INTERNAL,
)





# === Hàm chuẩn hoá session_category trước khi ghi DB ===
def _normalize_session_category_for_db(raw_value) -> Optional[str]:
    """
    Chuẩn hoá session_category trước khi ghi DB.

    Bảng ParkingSessions có CHECK:
        session_category = 'VISITOR'
        OR session_category = 'INTERNAL'
        OR session_category IS NULL

    Hàm này nhận vào:
        - hằng SESSION_CAT_TRANSIENT / SESSION_CAT_INTERNAL (có thể là 0/1 hoặc string)
        - hoặc chính chuỗi 'VISITOR' / 'INTERNAL'
    và luôn trả về chuỗi hợp lệ cho DB: 'VISITOR' / 'INTERNAL' / None.
    """
    if raw_value is None or raw_value == "":
        return None

    # trường hợp constants là chuỗi
    if isinstance(raw_value, str):
        v = raw_value.strip().upper()
        if v in ("VISITOR", "INTERNAL"):
            return v

    # trường hợp constants là số (0/1) nhưng DB yêu cầu chuỗi
    try:
        v_int = int(raw_value)
    except Exception:
        v_int = None

    if v_int is not None:
        # tuỳ cách bạn khai báo constants trong database.py
        if v_int == SESSION_CAT_INTERNAL:
            return "INTERNAL"
        if v_int == SESSION_CAT_TRANSIENT:
            return "VISITOR"

    # fallback an toàn: để NULL (pass CHECK constraint)
    return None






# === Dialog Thêm lịch sử xe (vào / ra) ===
class AddHistoryDialog(QDialog):
    
    
    # === Khởi tạo ===
    def __init__(self, parent: QWidget | None = None, db: Optional[DB] = None) -> None:
        super().__init__(parent)
        self.db = db

        self.setWindowTitle("Thêm xe")
        self.setModal(True)
        self.setMinimumWidth(720)
        self.new_id: Optional[int] = None

        self.setStyleSheet(
            """
            QDialog {
                background-color:#ffffff;
                color:#111827;
            }

            QLabel {
                color:#111827;
                font-size:13px;
            }

            /* Ô nhập / date / time / trạng thái – nền trắng, chữ đen */
            QLineEdit, QDateEdit, QTimeEdit, QComboBox {
                background-color:#ffffff;
                color:#111827;
                border:1px solid #d1d5db;
                border-radius:4px;
                padding:4px 6px;
            }

            QDateEdit::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width:18px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width:18px;
            }

            /* Menu xổ xuống của combobox Trạng thái / Loại xe */
            QComboBox QAbstractItemView {
                background-color:#ffffff;
                color:#111827;
                border:1px solid #d1d5db;
                selection-background-color:#e5e7eb;
                selection-color:#111827;
            }

            /* Checkbox VÀO / RA – giữ nguyên text, dùng indicator mặc định */
            QCheckBox {
                spacing: 6px;
                font-weight:600;
                color:#111827;
            }

            /* Ẩn nút up/down của time edit như cũ */
            QTimeEdit::up-button,
            QTimeEdit::down-button {
                width:0px;
                height:0px;
                border:none;
            }
            """
        )

        self._build_ui()

    
    
    
    
    
    # === Xây dựng giao diện ===
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)

        # ====== TIÊU ĐỀ ======
        title = QLabel("THÊM XE")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet(
            "font-size:18px; font-weight:700; letter-spacing:1px; color:#111827;"
        )
        root.addWidget(title)

        # ====== CHỌN VÀO / RA + LOẠI XE ======
        top_row = QHBoxLayout()
        top_row.setSpacing(18)
        top_row.addStretch(1)

        # --- Áp dụng cho (Vào / Ra) ---
        chk_row = QHBoxLayout()
        chk_row.setSpacing(8)

        lbl_apply = QLabel("Áp dụng cho:")
        self.chk_in = QCheckBox("Vào")
        self.chk_out = QCheckBox("Ra")

        self.chk_in.setChecked(True)   # mặc định thêm xe VÀO
        self.chk_out.setChecked(False)

        chk_row.addWidget(lbl_apply)
        chk_row.addWidget(self.chk_in)
        chk_row.addWidget(self.chk_out)

        # --- Loại xe ---
        type_row = QHBoxLayout()
        type_row.setSpacing(8)
        lbl_type = QLabel("Loại xe:")
        self.cmb_vehicle_mode = QComboBox()
        self.cmb_vehicle_mode.addItem("Tự động (theo DB)", "auto")
        self.cmb_vehicle_mode.addItem("Vãng lai", "transient")
        self.cmb_vehicle_mode.addItem("Nội bộ", "internal")
        self.cmb_vehicle_mode.setCurrentIndex(0)
        self.cmb_vehicle_mode.setFixedWidth(200)

        type_row.addWidget(lbl_type)
        type_row.addWidget(self.cmb_vehicle_mode)

        top_row.addLayout(chk_row)
        top_row.addSpacing(24)
        top_row.addLayout(type_row)
        top_row.addStretch(1)

        root.addLayout(top_row)

        # ====== HAI CỘT ======
        row = QHBoxLayout()
        row.setSpacing(24)
        root.addLayout(row)

        # ---- CỘT VÀO ----
        self.left_box = QWidget()
        left_layout = QFormLayout(self.left_box)
        left_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        left_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self.edt_img_in = QLineEdit()
        self.edt_img_in.setPlaceholderText("Chọn ảnh vào...")
        btn_browse_in = QPushButton("...")
        btn_browse_in.setFixedWidth(32)
        hl_img_in = QHBoxLayout()
        hl_img_in.addWidget(self.edt_img_in, 1)
        hl_img_in.addWidget(btn_browse_in)
        w_img_in = QWidget()
        w_img_in.setLayout(hl_img_in)
        left_layout.addRow("Đường dẫn ảnh (vào)", w_img_in)

        self.edt_plate_in = QLineEdit()
        self.edt_plate_in.setPlaceholderText("Biển số xe (vào)...")
        left_layout.addRow("Biển số xe (vào)", self.edt_plate_in)

        self.edt_date_in = QDateEdit()
        self.edt_date_in.setCalendarPopup(True)
        self.edt_date_in.setDate(QDate.currentDate())
        left_layout.addRow("Ngày (vào)", self.edt_date_in)

        self.edt_time_in = QTimeEdit()
        self.edt_time_in.setTime(QTime.currentTime())
        left_layout.addRow("Giờ (vào)", self.edt_time_in)

        row.addWidget(self.left_box, 1)

        # ---- CỘT RA ----
        self.right_box = QWidget()
        right_layout = QFormLayout(self.right_box)
        right_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        right_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        self.edt_img_out = QLineEdit()
        self.edt_img_out.setPlaceholderText("Chọn ảnh ra (có thể bỏ trống)...")
        btn_browse_out = QPushButton("...")
        btn_browse_out.setFixedWidth(32)
        hl_img_out = QHBoxLayout()
        hl_img_out.addWidget(self.edt_img_out, 1)
        hl_img_out.addWidget(btn_browse_out)
        w_img_out = QWidget()
        w_img_out.setLayout(hl_img_out)
        right_layout.addRow("Đường dẫn ảnh (ra)", w_img_out)

        self.edt_plate_out = QLineEdit()
        self.edt_plate_out.setPlaceholderText("Biển số xe (ra) nếu có...")
        right_layout.addRow("Biển số xe (ra)", self.edt_plate_out)

        self.edt_date_out = QDateEdit()
        self.edt_date_out.setCalendarPopup(True)
        self.edt_date_out.setDate(QDate.currentDate())
        right_layout.addRow("Ngày (ra)", self.edt_date_out)

        self.edt_time_out = QTimeEdit()
        self.edt_time_out.setTime(QTime.currentTime())
        right_layout.addRow("Giờ (ra)", self.edt_time_out)

        row.addWidget(self.right_box, 1)

        # ====== TRẠNG THÁI ======
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addStretch(1)

        lbl_status = QLabel("Trạng thái:")
        self.cmb_status = QComboBox()
        self.cmb_status.addItems(["PENDING", "KHOP-BIEN-SO", "KHONG-KHOP-BIEN-SO"])
        self.cmb_status.setCurrentIndex(0)
        self.cmb_status.setFixedWidth(220)

        status_row.addWidget(lbl_status)
        status_row.addWidget(self.cmb_status)
        status_row.addStretch(1)
        root.addLayout(status_row)

        # ====== NÚT ======
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)
        btn_row.addStretch(1)

        self.btn_cancel = QPushButton("Hủy")
        self.btn_save = QPushButton("Lưu")
        self.btn_save.setDefault(True)

        self.btn_cancel.setStyleSheet(
            "QPushButton {background-color:#e5e7eb; color:#111827; "
            "border-radius:6px; padding:6px 18px; font-weight:500;}"
            "QPushButton:hover {background-color:#d1d5db;}"
        )
        self.btn_save.setStyleSheet(
            "QPushButton {background-color:#3b82f6; color:white; "
            "border-radius:6px; padding:6px 18px; font-weight:600;}"
            "QPushButton:hover {background-color:#2563eb;}"
        )

        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # connect
        btn_browse_in.clicked.connect(self._on_browse_in)
        btn_browse_out.clicked.connect(self._on_browse_out)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.chk_in.toggled.connect(self._update_enabled)
        self.chk_out.toggled.connect(self._update_enabled)

        self._update_enabled()

    
    
    
    
    
    # === Cập nhật trạng thái enabled/disabled của 2 cột VÀO/RA ===
    def _update_enabled(self) -> None:
        self.left_box.setEnabled(self.chk_in.isChecked())
        self.right_box.setEnabled(self.chk_out.isChecked())





    # === Chuẩn hoá đường dẫn ảnh ===
    def _normalize_path(self, path: str) -> str:
        path = (path or "").strip()
        if not path:
            return ""
        return os.path.abspath(path)

    
    
    
    
    
    # === Nút Browse ảnh vào / ra ===
    @Slot()
    def _on_browse_in(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh vào", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.edt_img_in.setText(path)

    
    
    
    
    
    # === Nút Browse ảnh vào / ra ===
    @Slot()
    def _on_browse_out(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh ra", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.edt_img_out.setText(path)

    
    
    
    
    
    # === Nút Lưu ===
    @Slot()
    def _on_save_clicked(self) -> None:
        # 1) kiểm tra DB
        if not (self.db and getattr(self.db, "ok", False)):
            QMessageBox.warning(self, "Thêm", "Chưa kết nối DB, không thể lưu.")
            return

        # 2) ít nhất phải chọn Vào hoặc Ra
        if not self.chk_in.isChecked() and not self.chk_out.isChecked():
            QMessageBox.warning(self, "Thêm", "Hãy chọn VÀO và/hoặc RA để thêm.")
            return
        record: dict = {}

        # ----- PHẦN VÀO -----
        if self.chk_in.isChecked():
            plate_in = self.edt_plate_in.text().strip()
            if not plate_in:
                QMessageBox.warning(self, "Thêm", "Vui lòng nhập BIỂN SỐ XE (vào).")
                return

            record["Ảnh vào"] = self._normalize_path(self.edt_img_in.text())
            record["Biển số vào"] = plate_in
            record["Ngày vào"] = self.edt_date_in.date().toString("dd/MM/yyyy")
            record["Giờ vào"] = self.edt_time_in.time().toString("HH:mm:ss")

        # ----- PHẦN RA -----
        if self.chk_out.isChecked():
            plate_out = self.edt_plate_out.text().strip()
            if not plate_out:
                QMessageBox.warning(self, "Thêm", "Vui lòng nhập BIỂN SỐ XE (ra).")
                return

            record["Ảnh ra"] = self._normalize_path(self.edt_img_out.text())
            record["Biển số ra"] = plate_out
            record["Ngày ra"] = self.edt_date_out.date().toString("dd/MM/yyyy")
            record["Giờ ra"] = self.edt_time_out.time().toString("HH:mm:ss")

        # Trạng thái
        status = self.cmb_status.currentText().strip() or "PENDING"
        record["Trạng thái"] = status

        plate_for_lookup = (
            record.get("Biển số vào") or record.get("Biển số ra") or ""
        ).strip()
        
        mode = self.cmb_vehicle_mode.currentData()  
        session_category_raw = SESSION_CAT_TRANSIENT  
        vehicle_id = None
        vehicle_type_id = None

        # Vãng lai
        if mode == "transient":
            session_category_raw = SESSION_CAT_TRANSIENT

        # Nội bộ
        elif mode == "internal":
            session_category_raw = SESSION_CAT_INTERNAL
            if plate_for_lookup and hasattr(self.db, "get_vehicle_by_plate"):
                vinfo = self.db.get_vehicle_by_plate(plate_for_lookup)
                if vinfo:
                    vehicle_id = vinfo["id"]
                    vehicle_type_id = vinfo["vehicle_type_id"]

        # Tự động: lookup Vehicles, có -> nội bộ, không -> vãng lai
        elif mode == "auto":
            if plate_for_lookup and hasattr(self.db, "get_vehicle_by_plate"):
                vinfo = self.db.get_vehicle_by_plate(plate_for_lookup)
                if vinfo:
                    session_category_raw = SESSION_CAT_INTERNAL
                    vehicle_id = vinfo["id"]
                    vehicle_type_id = vinfo["vehicle_type_id"]
                else:
                    session_category_raw = SESSION_CAT_TRANSIENT

        # Chuẩn hoá sang giá trị hợp lệ cho DB: 'VISITOR' / 'INTERNAL' / None
        session_category_db = _normalize_session_category_for_db(session_category_raw)

        record["session_category"] = session_category_db
        record["vehicle_id"] = vehicle_id
        record["vehicle_type_id"] = vehicle_type_id

        # 3) gọi DB.insert_history_record
        if not hasattr(self.db, "insert_history_record"):
            QMessageBox.warning(
                self,
                "Thêm",
                "DB chưa có hàm insert_history_record(record).\n"
                "Hãy thêm method này trong lớp DB để lưu bản ghi.",
            )
            return

        try:
            self.new_id = self.db.insert_history_record(record)  # type: ignore[attr-defined]
        except Exception as e:
            QMessageBox.warning(self, "Thêm", f"Lỗi khi ghi DB:\n{e}")
            return

        QMessageBox.information(self, "Thêm", "Đã thêm 1 lượt gửi mới vào lịch sử.")
        self.accept()
