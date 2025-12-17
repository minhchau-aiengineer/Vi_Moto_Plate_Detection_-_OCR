# ui/pages/search.py
"""
SearchPageMixin + SearchPage

SearchPageMixin:
    - Dùng nội bộ trong HistoryPage để hiển thị BỘ LỌC TÌM KIẾM lịch sử.

SearchPage:
    - Trang "stub" đơn giản để MainWindow import được SearchPage cũ.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, cast
from datetime import datetime

from PySide6.QtCore import Qt, QDate, QTime, QDateTime, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QDateEdit,
    QTimeEdit,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QStackedWidget,
)

if TYPE_CHECKING:
    from ....database.database import DB  # noqa: F401

from ...theme import normalize_button, apply_button_style



# ====== Search Page Mixin ======
class SearchPageMixin:
    """
    Mixin cung cấp UI + logic cho trang bộ lọc tìm kiếm lịch sử.

    Lớp dùng mixin này cần có:
        - self.stacked               : QStackedWidget
        - self.refresh_history_data  : hàm load lại bảng lịch sử
        - self.show_history_view_only()
        - self.current_filter_start / end / status / plate
        - self.current_filter_vehicle_group  (list[str] hoặc None)
    """

   
    
    
    
    
    # === Tạo trang bộ lọc tìm kiếm lịch sử ===
    def build_search_page(self, common_btn_style: str) -> QWidget:
        """Tạo QWidget search_filter_view (bộ lọc tìm kiếm)."""

        self.search_filter_view = QWidget()
        self.search_filter_view.setObjectName("SearchPageRoot")

        sfv_layout = QVBoxLayout(self.search_filter_view)
        sfv_layout.setContentsMargins(20, 20, 20, 20)
        sfv_layout.setSpacing(15)

        # Tiêu đề
        title = QLabel("Bộ lọc tìm kiếm lịch sử")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:22px; font-weight:700; color:#111827;")
        sfv_layout.addWidget(title)

        # Khung form
        form = QFrame()
        form.setObjectName("SearchFilterFrame")
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(25, 25, 25, 25)
        form_layout.setSpacing(18)

        # ===== HÀNG 1: TỪ NGÀY / GIỜ =====
        row_start = QHBoxLayout()
        row_start.setSpacing(10)

        lbl_from_date = QLabel("TỪ NGÀY:")
        lbl_from_time = QLabel("GIỜ:")
        lbl_from_date.setMinimumWidth(80)
        lbl_from_time.setMinimumWidth(40)

        self.sfv_date_start = QDateEdit(QDate.currentDate().addDays(-1))
        self.sfv_date_start.setCalendarPopup(True)
        self.sfv_date_start.setDisplayFormat("dd/MM/yyyy")
        self.sfv_date_start.setFixedHeight(34)
        self.sfv_date_start.setFixedWidth(140)

        self.sfv_time_start = QTimeEdit(QTime(0, 0, 0))
        self.sfv_time_start.setDisplayFormat("HH:mm:ss")
        self.sfv_time_start.setFixedHeight(34)
        self.sfv_time_start.setFixedWidth(100)

        row_start.addWidget(lbl_from_date)
        row_start.addWidget(self.sfv_date_start)
        row_start.addSpacing(20)
        row_start.addWidget(lbl_from_time)
        row_start.addWidget(self.sfv_time_start)
        row_start.addStretch(1)
        form_layout.addLayout(row_start)

        # ===== HÀNG 2: ĐẾN NGÀY / GIỜ =====
        row_end = QHBoxLayout()
        row_end.setSpacing(10)

        lbl_to_date = QLabel("ĐẾN NGÀY:")
        lbl_to_time = QLabel("GIỜ:")
        lbl_to_date.setMinimumWidth(80)
        lbl_to_time.setMinimumWidth(40)

        self.sfv_date_end = QDateEdit(QDate.currentDate())
        self.sfv_date_end.setCalendarPopup(True)
        self.sfv_date_end.setDisplayFormat("dd/MM/yyyy")
        self.sfv_date_end.setFixedHeight(34)
        self.sfv_date_end.setFixedWidth(140)

        self.sfv_time_end = QTimeEdit(QTime.currentTime())
        self.sfv_time_end.setDisplayFormat("HH:mm:ss")
        self.sfv_time_end.setFixedHeight(34)
        self.sfv_time_end.setFixedWidth(100)

        row_end.addWidget(lbl_to_date)
        row_end.addWidget(self.sfv_date_end)
        row_end.addSpacing(20)
        row_end.addWidget(lbl_to_time)
        row_end.addWidget(self.sfv_time_end)
        row_end.addStretch(1)
        form_layout.addLayout(row_end)

        # ---- Trạng thái ----
        row_status = QHBoxLayout()
        row_status.setSpacing(12)

        lbl_status = QLabel("TRẠNG THÁI:")
        lbl_status.setMinimumWidth(80)
        row_status.addWidget(lbl_status)

        self.sfv_chk_pending = QCheckBox("CHỜ XỬ LÍ (PENDING)")
        self.sfv_chk_match = QCheckBox("KHOP-BIEN-SO")
        self.sfv_chk_mismatch = QCheckBox("KHONG-KHOP-BIEN-SO")

        self.sfv_chk_pending.setChecked(True)
        self.sfv_chk_match.setChecked(True)
        self.sfv_chk_mismatch.setChecked(True)

        row_status.addWidget(self.sfv_chk_pending)
        row_status.addWidget(self.sfv_chk_match)
        row_status.addWidget(self.sfv_chk_mismatch)
        row_status.addStretch(1)
        form_layout.addLayout(row_status)

        # ---- Nhóm xe (Vãng lai / Nội bộ) ----
        row_group = QHBoxLayout()
        row_group.setSpacing(12)

        lbl_group = QLabel("NHÓM XE:")
        lbl_group.setMinimumWidth(80)
        row_group.addWidget(lbl_group)

        self.sfv_chk_group_transient = QCheckBox("Vãng lai")
        self.sfv_chk_group_internal = QCheckBox("Nội bộ")

        # mặc định tick cả 2 -> không giới hạn nhóm xe
        self.sfv_chk_group_transient.setChecked(True)
        self.sfv_chk_group_internal.setChecked(True)

        row_group.addWidget(self.sfv_chk_group_transient)
        row_group.addWidget(self.sfv_chk_group_internal)
        row_group.addStretch(1)
        form_layout.addLayout(row_group)

        # ---- Biển số ----
        row_plate = QHBoxLayout()
        row_plate.setSpacing(10)

        lbl_plate = QLabel("BIỂN SỐ:")
        lbl_plate.setMinimumWidth(80)
        row_plate.addWidget(lbl_plate)

        self.sfv_txt_plate = QLineEdit()
        self.sfv_txt_plate.setPlaceholderText("Nhập biển số bạn muốn tìm...")
        self.sfv_txt_plate.setFixedHeight(34)
        row_plate.addWidget(self.sfv_txt_plate)

        form_layout.addLayout(row_plate)

        sfv_layout.addWidget(form)

        # ---- Nút ----
        row_btn = QHBoxLayout()
        self.sfv_btn_back = QPushButton("⬅ Quay lại bảng lịch sử")
        self.sfv_btn_search = QPushButton("Tìm kiếm")

        normalize_button(self.sfv_btn_back, self.sfv_btn_search)

        back_css = (
            "QPushButton{ "
            f"{common_btn_style} background-color:#e5e7eb; "
            "border:1px solid #d1d5db; }"
            "QPushButton:hover{ background-color:#d1d5db; }"
        )
        search_css = (
            "QPushButton{ "
            f"{common_btn_style} background-color:#dbeafe; "
            "border:1px solid #bfdbfe; }"
            "QPushButton:hover{ background-color:#bfdbfe; }"
        )

        apply_button_style(self.sfv_btn_back, back_css)
        apply_button_style(self.sfv_btn_search, search_css)

        row_btn.addStretch(1)
        row_btn.addWidget(self.sfv_btn_back)
        row_btn.addSpacing(80)
        row_btn.addWidget(self.sfv_btn_search)
        row_btn.addStretch(3)

        sfv_layout.addLayout(row_btn)
        sfv_layout.addStretch(1)

        # ---- STYLE NỀN & CONTROL ----
        self.search_filter_view.setStyleSheet(
            """
            QWidget#SearchPageRoot {
                background-color: #f5f5f7;
            }
            QFrame#SearchFilterFrame {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }
            QLabel {
                color: #111827;
            }
            QLineEdit,
            QDateEdit,
            QTimeEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 2px 6px;
            }
            /* Ẩn 2 mũi tên lên/xuống trong khung giờ */
            QTimeEdit::up-button,
            QTimeEdit::down-button {
                width: 0px;
                height: 0px;
                border: none;
            }
            QLineEdit:disabled,
            QDateEdit:disabled,
            QTimeEdit:disabled {
                background-color: #e5e7eb;
                color: #6b7280;
            }
            QCheckBox {
                color: #111827;
                spacing: 6px;
            }
            """
        )

        return self.search_filter_view

    
    
    
    
    
    
    # === Xử lý chuyển tab lớn ===
    @Slot()
    def on_search_history_clicked(self) -> None:
        stacked = getattr(self, "stacked", None)
        search_view = getattr(self, "search_filter_view", None)

        if isinstance(stacked, QStackedWidget) and isinstance(search_view, QWidget):
            stacked.setCurrentWidget(search_view)

    
    
    
    
    
    # === Xử lý nút Tìm kiếm ===
    @Slot()
    def on_run_search_from_page(self) -> None:
        qdate_start = self.sfv_date_start.date()
        qtime_start = self.sfv_time_start.time()
        qdate_end = self.sfv_date_end.date()
        qtime_end = self.sfv_time_end.time()

        start_dt_obj = QDateTime(qdate_start, qtime_start).toPython()
        end_dt_obj = QDateTime(qdate_end, qtime_end).toPython()

        start_dt = cast(datetime, start_dt_obj)
        end_dt = cast(datetime, end_dt_obj)

        if start_dt > end_dt:
            parent = cast(QWidget, self)
            QMessageBox.warning(
                parent,
                "Lỗi nhập liệu",
                "'Từ ngày/giờ' không được lớn hơn 'Đến ngày/giờ'.",
            )
            return

        statuses: list[str] = []
        if self.sfv_chk_pending.isChecked():
            statuses.append("PENDING")
        if self.sfv_chk_match.isChecked():
            statuses.append("KHOP-BIEN-SO")
        if self.sfv_chk_mismatch.isChecked():
            statuses.append("KHONG-KHOP-BIEN-SO")

        plate_text = self.sfv_txt_plate.text().strip()

        # nhóm xe
        vehicle_groups: list[str] = []
        if self.sfv_chk_group_transient.isChecked():
            vehicle_groups.append("Vãng lai")
        if self.sfv_chk_group_internal.isChecked():
            vehicle_groups.append("Nội bộ")

        # lưu filter hiện tại trên object
        self.current_filter_start = start_dt
        self.current_filter_end = end_dt
        self.current_filter_status = statuses if statuses else None
        self.current_filter_plate = plate_text if plate_text else None
        self.current_filter_vehicle_group = vehicle_groups if vehicle_groups else None

        # gọi hàm load dữ liệu (HistoryPage cung cấp)
        refresh = getattr(self, "refresh_history_data", None)
        if callable(refresh):
            refresh(
                start_time=self.current_filter_start,
                end_time=self.current_filter_end,
                status_filter=self.current_filter_status,
                plate_filter=self.current_filter_plate,
                vehicle_group_filter=self.current_filter_vehicle_group,
                clear_filters=False,
            )

        # quay về bảng lịch sử (giữ kết quả lọc)
        show_history = getattr(self, "show_history_view_only", None)
        if callable(show_history):
            show_history()






# ======= Search Page (Stub) ======
class SearchPage(QWidget):
    """
    Trang stub rất đơn giản, chỉ để tránh ImportError.
    Tính năng tìm kiếm thật đã nằm trong HistoryPage
    (nút 'Tìm kiếm' ở sidebar trái).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SearchPageStub")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel(
            "Chức năng TÌM KIẾM NÂNG CAO đã được chuyển sang trang LỊCH SỬ.\n"
            "Vui lòng mở tab 'Lịch sử' và dùng nút 'Tìm kiếm' ở bên trái."
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)

        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)

        self.setStyleSheet(
            """
            QWidget#SearchPageStub {
                background-color: #f5f5f7;
            }
            QLabel {
                font-size: 15px;
                color: #111827;
            }
            """
        )
