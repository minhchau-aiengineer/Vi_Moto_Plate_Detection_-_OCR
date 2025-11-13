# ui/pages/search.py
"""
SearchPageMixin

Chịu trách nhiệm:
- Trang BỘ LỌC TÌM KIẾM lịch sử (search_filter_view).
- Logic xử lý khi người dùng bấm "Tìm kiếm" từ sidebar
  và từ ngay trên trang filter.

YÊU CẦU MainWindow (class kế thừa mixin này) có:
- self.stacked                        : QStackedWidget
- self.btn_show_history, btn_hide_history
- self.refresh_history_data(...)      : hàm từ HistoryPageMixin
- self.show_history_view_only()       : hàm từ HistoryPageMixin
- self.current_filter_start/end/status/plate : biến filter
"""

from __future__ import annotations

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
)

from ..theme import normalize_button, apply_button_style


class SearchPageMixin:
    """
    Mixin cung cấp UI + logic cho trang bộ lọc tìm kiếm lịch sử.
    """

    # ======================================================================
    #  BUILD SEARCH FILTER PAGE
    # ======================================================================

    def build_search_page(self, common_btn_style: str) -> QWidget:
        """
        Tạo trang bộ lọc tìm kiếm lịch sử (search_filter_view).

        Trả về:
            search_filter_view (QWidget)
        """

        self.search_filter_view = QWidget()
        sfv_layout = QVBoxLayout(self.search_filter_view)
        sfv_layout.setContentsMargins(20, 20, 20, 20)
        sfv_layout.setSpacing(15)

        # 1. Tiêu đề
        sfv_title = QLabel("Bộ lọc tìm kiếm lịch sử")
        sfv_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #333;")
        sfv_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sfv_layout.addWidget(sfv_title)

        # 2. Form chứa các bộ lọc
        sfv_form = QFrame()
        sfv_form.setStyleSheet(
            "QFrame { background: #f9f9f9; border: 1px solid #eee; border-radius: 10px; }"
            "QLabel { font-weight: 600; }"
        )
        sfv_form_layout = QVBoxLayout(sfv_form)
        sfv_form_layout.setContentsMargins(25, 25, 25, 25)
        sfv_form_layout.setSpacing(18)

        # ---- Hàng "Từ ngày/giờ" ----
        row_start = QHBoxLayout()
        row_start.setSpacing(10)

        row_start.addWidget(QLabel("TỪ NGÀY:"))
        self.sfv_date_start = QDateEdit(QDate.currentDate().addDays(-1))
        self.sfv_date_start.setCalendarPopup(True)
        self.sfv_date_start.setDisplayFormat("dd/MM/yyyy")
        self.sfv_date_start.setFixedHeight(34)
        row_start.addWidget(self.sfv_date_start)

        row_start.addWidget(QLabel("GIỜ:"))
        self.sfv_time_start = QTimeEdit(QTime(0, 0, 0))
        self.sfv_time_start.setDisplayFormat("HH:mm:ss")
        self.sfv_time_start.setFixedHeight(34)
        row_start.addWidget(self.sfv_time_start)

        row_start.addStretch(1)
        sfv_form_layout.addLayout(row_start)

        # ---- Hàng "Đến ngày/giờ" ----
        row_end = QHBoxLayout()
        row_end.setSpacing(10)

        row_end.addWidget(QLabel("ĐẾN NGÀY:"))
        self.sfv_date_end = QDateEdit(QDate.currentDate())
        self.sfv_date_end.setCalendarPopup(True)
        self.sfv_date_end.setDisplayFormat("dd/MM/yyyy")
        self.sfv_date_end.setFixedHeight(34)
        row_end.addWidget(self.sfv_date_end)

        row_end.addWidget(QLabel("GIỜ:"))
        self.sfv_time_end = QTimeEdit(QTime.currentTime())
        self.sfv_time_end.setDisplayFormat("HH:mm:ss")
        self.sfv_time_end.setFixedHeight(34)
        row_end.addWidget(self.sfv_time_end)

        row_end.addStretch(1)
        sfv_form_layout.addLayout(row_end)

        # ---- Hàng "Trạng thái" ----
        row_status = QHBoxLayout()
        row_status.setSpacing(15)

        row_status.addWidget(QLabel("TRẠNG THÁI:"))

        self.sfv_chk_pending = QCheckBox("CHỜ XỬ LÍ (PENDING)")
        self.sfv_chk_match = QCheckBox("KHOP-BIEN-SO")
        self.sfv_chk_mismatch = QCheckBox("KHONG-KHOP-BIEN-SO")

        # Mặc định chọn tất cả
        self.sfv_chk_pending.setChecked(True)
        self.sfv_chk_match.setChecked(True)
        self.sfv_chk_mismatch.setChecked(True)

        row_status.addWidget(self.sfv_chk_pending)
        row_status.addWidget(self.sfv_chk_match)
        row_status.addWidget(self.sfv_chk_mismatch)
        row_status.addStretch(1)
        sfv_form_layout.addLayout(row_status)

        # ---- Hàng "Biển số" ----
        row_plate = QHBoxLayout()
        row_plate.setSpacing(10)

        row_plate.addWidget(QLabel("BIỂN SỐ:"))
        self.sfv_txt_plate = QLineEdit()
        self.sfv_txt_plate.setPlaceholderText("Nhập biển số bạn muốn tìm vào đây...")
        self.sfv_txt_plate.setFixedHeight(34)
        row_plate.addWidget(self.sfv_txt_plate)

        sfv_form_layout.addLayout(row_plate)

        sfv_layout.addWidget(sfv_form)

        # ---- Hàng nút (Quay lại, Tìm kiếm) ----
        sfv_row_btn = QHBoxLayout()

        self.sfv_btn_back = QPushButton("Quay lại")
        self.sfv_btn_search = QPushButton("Tìm kiếm")

        normalize_button(self.sfv_btn_back, self.sfv_btn_search)

        back_css = (
            f"QPushButton{{ {common_btn_style} background:#f3f4f6; border:1px solid #e5e7eb; }}"
            "QPushButton:hover{ background:#eef0f3; }"
        )
        search_css = (
            f"QPushButton{{ {common_btn_style} background:#e0ecff; border:1px solid #c7dcff; }}"
            "QPushButton:hover{ background:#d4e5ff; }"
        )

        apply_button_style(self.sfv_btn_back, back_css)
        apply_button_style(self.sfv_btn_search, search_css)

        sfv_row_btn.addWidget(self.sfv_btn_back)
        sfv_row_btn.addStretch(1)
        sfv_row_btn.addWidget(self.sfv_btn_search)

        sfv_layout.addLayout(sfv_row_btn)
        sfv_layout.addStretch(1)

        return self.search_filter_view

    # ======================================================================
    #  SLOT: NÚT TÌM KIẾM Ở SIDEBAR
    # ======================================================================

    @Slot()
    def on_search_history_clicked(self) -> None:
        """
        Khi nhấn nút "Tìm kiếm" bên sidebar (trong section BẢNG LỊCH SỬ):
        - Chuyển sang trang search_filter_view (index 3).
        - Hiển thị nút "Tắt bảng lịch sử".
        """
        if hasattr(self, "stacked"):
            self.stacked.setCurrentWidget(self.search_filter_view)

        if hasattr(self, "btn_show_history") and hasattr(self, "btn_hide_history"):
            self.btn_show_history.hide()
            self.btn_hide_history.show()

    # ======================================================================
    #  SLOT: TÌM KIẾM TỪ TRANG FILTER
    # ======================================================================

    @Slot()
    def on_run_search_from_page(self) -> None:
        """
        Khi nhấn nút "Tìm kiếm" trên trang filter:

        - Đọc dữ liệu từ:
            + sfv_date_start, sfv_time_start
            + sfv_date_end, sfv_time_end
            + sfv_chk_pending, sfv_chk_match, sfv_chk_mismatch
            + sfv_txt_plate
        - Validate khoảng thời gian (start <= end).
        - Gán vào self.current_filter_*.
        - Gọi refresh_history_data(...) với filter này.
        - Quay về bảng lịch sử (show_history_view_only).
        """
        print(">>> Entering on_run_search_from_page")

        # Lấy dữ liệu thời gian
        qdate_start = self.sfv_date_start.date()
        qtime_start = self.sfv_time_start.time()
        qdate_end = self.sfv_date_end.date()
        qtime_end = self.sfv_time_end.time()

        start_dt = QDateTime(qdate_start, qtime_start).toPython()
        end_dt = QDateTime(qdate_end, qtime_end).toPython()

        if start_dt > end_dt:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                self,
                "Lỗi nhập liệu",
                "'Từ ngày/giờ' không được lớn hơn 'Đến ngày/giờ'.\nVui lòng kiểm tra lại.",
            )
            print("<<< Exiting on_run_search_from_page (Date Error)")
            return

        # Lấy trạng thái được chọn
        selected_statuses: list[str] = []
        plate_text = self.sfv_txt_plate.text().strip()

        if self.sfv_chk_pending.isChecked():
            selected_statuses.append("PENDING")
        if self.sfv_chk_match.isChecked():
            selected_statuses.append("KHOP-BIEN-SO")
        if self.sfv_chk_mismatch.isChecked():
            selected_statuses.append("KHONG-KHOP-BIEN-SO")

        # LƯU LẠI BỘ LỌC HIỆN TẠI
        self.current_filter_start = start_dt
        self.current_filter_end = end_dt
        self.current_filter_status = selected_statuses if selected_statuses else None
        self.current_filter_plate = plate_text if plate_text else None

        print(">>> Filters JUST SET in on_run_search:")
        print(
            f"    Start: {self.current_filter_start}, "
            f"End: {self.current_filter_end}, "
            f"Status: {self.current_filter_status}, "
            f"Plate: {self.current_filter_plate}"
        )

        # Gọi hàm tải dữ liệu với bộ lọc mới
        print(">>> Calling refresh_history_data...")
        self.refresh_history_data(
            start_time=self.current_filter_start,
            end_time=self.current_filter_end,
            status_filter=self.current_filter_status,
            plate_filter=self.current_filter_plate,
            clear_filters=False,
        )
        print(">>> Returned from refresh_history_data.")
        print("<<< Exiting on_run_search_from_page (Success)")

        # Thông báo nhỏ ở statusBar (nếu có)
        try:
            if hasattr(self, "stats_range_combo"):
                current_range = (
                    f"{self.sfv_date_start.text()} {self.sfv_time_start.text()} -> "
                    f"{self.sfv_date_end.text()} {self.sfv_time_end.text()}"
                )
            else:
                current_range = (
                    f"{self.sfv_date_start.text()} -> {self.sfv_date_end.text()}"
                )

            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"✅ Đã áp dụng bộ lọc: {current_range}", 2000)
        except Exception:
            # Không cần die nếu có lỗi minor ở status bar
            pass

        # Quay về bảng lịch sử
        self.show_history_view_only()
