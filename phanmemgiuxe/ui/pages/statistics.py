# ui/pages/statistics.py
"""
StatisticsPageMixin

Chịu trách nhiệm:
- Trang THỐNG KÊ: KPI, tổng quan, xe trong bãi, top lâu nhất, thống kê theo ngày.
- Làm việc với self.stats_service (instance ParkingStatistics).
- Các nút:
    + btn_show_statistics  (ở sidebar, được tạo trong CameraPageMixin)
    + btn_stats_back       (quay lại trang chính)
    + btn_stats_refresh    (làm mới)
    + btn_stats_export     (export báo cáo)
    + stats_range_combo    (chọn Hôm nay / 7 ngày / Tháng này)

YÊU CẦU MainWindow (class kế thừa mixin này) có:
- self.stats_service          : ParkingStatistics | None
- self.stacked                : QStackedWidget
- self.btn_show_history,
  self.btn_hide_history       : nút ở sidebar (từ CameraPageMixin)
- self.statusBar()            : optional, để hiển thị thông báo nhỏ
- self._stats_last_reload     : float (timestamp) – đã khai báo trong MainWindow
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Slot, QDateTime
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QScrollArea,
    QFrame,
    QMessageBox,
    QFileDialog,
)

from ..theme import normalize_button, apply_button_style
from ..widgets import StatsCard, KPIChip


class StatisticsPageMixin:
    """
    Mixin cung cấp UI + logic cho trang THỐNG KÊ.
    """

    # ======================================================================
    #  BUILD STATISTICS PAGE
    # ======================================================================

    def build_statistics_page(self, common_btn_style: str) -> QWidget:
        """
        Xây dựng trang thống kê (statistics_view).

        Luôn tạo widget, nhưng nếu self.stats_service = None thì các nút
        sẽ không hoạt động (đã disable ở sidebar trong CameraPageMixin).
        """

        # Reset trạng thái
        self.statistics_view: QWidget | None = None
        self._stats_last_reload = 0.0

        self.statistics_view = QWidget()
        self.statistics_view.setSizePolicy(
            self.statistics_view.sizePolicy().horizontalPolicy(),
            self.statistics_view.sizePolicy().verticalPolicy(),
        )

        root_layout = QVBoxLayout(self.statistics_view)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Scroll toàn trang
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll)

        page = QWidget()
        page.setSizePolicy(
            page.sizePolicy().horizontalPolicy(),
            page.sizePolicy().verticalPolicy(),
        )
        scroll.setWidget(page)

        self.page_layout = QVBoxLayout(page)
        self.page_layout.setContentsMargins(5, 5, 5, 5)
        self.page_layout.setSpacing(10)

        # ---------------- Header title ----------------
        title = QLabel("THỐNG KÊ")
        title.setObjectName("StatsPageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.page_layout.addWidget(title)

        # ---------------- Filter row ----------------
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        self.page_layout.addLayout(filter_row)

        filter_row.addWidget(QLabel("Khoảng thời gian:"))
        self.stats_range_combo = QComboBox()
        self.stats_range_combo.addItems(["Hôm nay", "7 ngày", "Tháng này"])
        filter_row.addWidget(self.stats_range_combo)

        # Buttons
        self.btn_stats_back = QPushButton("⬅ Quay lại")
        self.btn_stats_refresh = QPushButton("Làm mới")
        self.btn_stats_export = QPushButton("Export")

        self.btn_stats_back.setToolTip("Quay lại trang chính")
        self.btn_stats_refresh.setToolTip("Làm mới dữ liệu thống kê theo khoảng thời gian đã chọn")
        self.btn_stats_export.setToolTip("Xuất báo cáo thống kê")

        normalize_button(self.btn_stats_back, self.btn_stats_refresh, self.btn_stats_export)
        for btn in (self.btn_stats_back, self.btn_stats_refresh, self.btn_stats_export):
            btn.setMinimumHeight(38)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

        filter_row.addStretch(1)
        filter_row.addWidget(self.btn_stats_refresh)
        filter_row.addWidget(self.btn_stats_export)
        filter_row.addWidget(self.btn_stats_back)

        # ---------------- KPI row ----------------
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(32)
        self.page_layout.addLayout(kpi_row)
        kpi_row.addStretch(1)

        # Dữ liệu ban đầu
        initial_current = "0"
        initial_in = "0"
        initial_out = "0"

        if getattr(self, "stats_service", None):
            try:
                overview = self.stats_service.get_overview_statistics()
                if not overview.get("error"):
                    initial_current = str(overview.get("current_cars_inside", 0) or 0)
                    initial_in = str(overview.get("today", {}).get("entries_today", 0) or 0)
                    initial_out = str(overview.get("today", {}).get("exits_today", 0) or 0)
            except Exception as e:
                print(f"Không thể load thống kê ban đầu: {e}")

        self.kpi_inpark = KPIChip("Xe trong bãi", initial_current, "#E6F7EC")
        self.kpi_in = KPIChip("Xe vào hôm nay", initial_in, "#E5F0FF")
        self.kpi_out = KPIChip("Xe ra hôm nay", initial_out, "#FFEFE6")

        for chip in (self.kpi_inpark, self.kpi_in, self.kpi_out):
            chip.setMinimumHeight(140)
            chip.setMinimumWidth(200)
            kpi_row.addWidget(chip)

        kpi_row.addStretch(1)

        # ---------------- Tổng quan card ----------------
        self.card_overview = StatsCard("Tổng quan")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.ov_labels: list[QLabel] = []

        initial_total = "0"
        initial_pending = "0"
        initial_unmatched = "0"
        initial_matched = "0"
        initial_avg = "0"

        if getattr(self, "stats_service", None):
            try:
                overview = self.stats_service.get_overview_statistics()
                if not overview.get("error"):
                    initial_total = str(overview.get("total_sessions", 0) or 0)
                    initial_pending = str(overview.get("today", {}).get("pending_cars", 0) or 0)
                    initial_unmatched = str(
                        overview.get("today", {}).get("unmatched_exits", 0) or 0
                    )
                    initial_matched = str(
                        overview.get("today", {}).get("matched_exits", 0) or 0
                    )
                    initial_avg = str(overview.get("avg_parking_minutes", 0) or 0)
            except Exception as e:
                print(f"Không thể load overview ban đầu: {e}")

        pairs = [
            ("Tổng lượt gửi:", initial_total),
            ("Xe đang đợi PENDING:", initial_pending),
            ("Xe đang trong bãi:", initial_current),
            ("Không khớp:", initial_unmatched),
            ("Xe vào:", initial_in),
            ("Khớp (OCR):", initial_matched),
            ("Xe ra:", initial_out),
            ("Thời gian trung bình/xe (phút):", initial_avg),
        ]

        for i, (k, v) in enumerate(pairs):
            r, c = divmod(i, 2)
            key_label = QLabel(k)
            key_label.setObjectName("OvKey")
            grid.addWidget(key_label, r, c * 2)

            val_label = QLabel(v)
            val_label.setObjectName("OvVal")
            grid.addWidget(val_label, r, c * 2 + 1, alignment=Qt.AlignmentFlag.AlignLeft)
            self.ov_labels.append(val_label)

        self.card_overview.layout().addLayout(grid)
        self.page_layout.addWidget(self.card_overview)

        # ---------------- Card: Xe đang trong bãi ----------------
        self.card_inpark = StatsCard("Xe đang trong bãi")
        self.tbl_stats_cars_inside = self._make_stats_table(
            ["ID", "Biển số", "Ngày vào", "Giờ vào", "Thời gian (phút)"]
        )
        self.tbl_stats_cars_inside.setMinimumHeight(350)
        self.card_inpark.layout().addWidget(self.tbl_stats_cars_inside)
        self.page_layout.addWidget(self.card_inpark)

        # ---------------- Card: TOP 5 gửi lâu nhất ----------------
        self.card_longest = StatsCard("TOP 5 lượt gửi lâu nhất (đang chờ)")
        self.tbl_stats_longest = self._make_stats_table(
            ["Biển số", "Ngày vào", "Giờ vào", "Thời gian (phút)", "Thời gian (giờ)"]
        )
        self.tbl_stats_longest.setMinimumHeight(300)
        self.card_longest.layout().addWidget(self.tbl_stats_longest)
        self.page_layout.addWidget(self.card_longest)

        # ---------------- Card: 7 ngày gần nhất ----------------
        self.card_weekly = StatsCard("Thống kê theo thời gian")
        self.tbl_stats_frequent = self._make_stats_table(
            ["Ngày", "Thứ", "Xe vào", "Xe ra", "Khớp", "Không khớp", "Tỉ lệ khớp (%)"]
        )
        self.tbl_stats_frequent.setMinimumHeight(320)

        self.footer_weekly = QLabel("Tổng vào: — · Tổng ra: — · Khớp: — · Tỉ lệ: —")
        self.footer_weekly.setObjectName("StatsCardFooter")

        self.card_weekly.layout().addWidget(self.tbl_stats_frequent)
        self.card_weekly.layout().addWidget(self.footer_weekly)
        self.page_layout.addWidget(self.card_weekly)

        # ---------------- Label "Cập nhật lần cuối" ----------------
        self.lbl_stats_last_update = QLabel("Cập nhật: --")
        self.lbl_stats_last_update.setObjectName("StatsLastUpdate")
        self.page_layout.addWidget(self.lbl_stats_last_update)

        # ---------------- Styles ----------------
        self._apply_stats_styles()

        # ---------------- Kết nối nút (cross-page signals sẽ gắn ở MainWindow) ----------------
        # Ở đây chỉ trả về statistics_view, các connect được MainWindow xử lý.

        return self.statistics_view

    # ======================================================================
    #  TABLE HELPERS
    # ======================================================================

    def _make_stats_table(self, headers: list[str]) -> QTableWidget:
        """
        Tạo QTableWidget cho trang thống kê.
        """
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        return table

    def _apply_stats_styles(self) -> None:
        """
        Áp dụng stylesheet modern cho statistics_view.
        """
        if not self.statistics_view:
            return

        self.statistics_view.setStyleSheet(
            """
        QWidget { 
            background: #F4F6FA; 
            font-family: 'Segoe UI','Inter',sans-serif; 
            color: #1F2937; 
        }
        
        #StatsPageTitle { 
            font-size: 42px; 
            font-weight: 900; 
            letter-spacing: 2px; 
            margin-top: 10px; 
            margin-bottom: 20px; 
        }

        QFrame#StatsCard { 
            background: #FFFFFF; 
            border: 1px solid #E5E7EB; 
            border-radius: 14px; 
        }
        
        QLabel#StatsCardTitle { 
            font-size: 16px; 
            font-weight: 800; 
            color: #111827; 
        }

        QFrame#KPIChip { 
            border: 1px solid #E5E7EB; 
            border-radius: 12px; 
        }
        
        QLabel#KpiTitle { 
            font-size: 13px; 
            font-weight: 700; 
            color: #374151; 
        }
        
        QLabel#KpiValue { 
            font-size: 48px; 
            font-weight: 900; 
            color: #DC2626; 
        }

        QLabel#OvKey { 
            font-size: 16px; 
            color: #1F2937; 
            font-weight: 700; 
        }
        
        QLabel#OvVal { 
            font-size: 16px; 
            color: #111827; 
            font-weight: 700; 
            padding-left: 8px; 
        }

        QHeaderView::section {
            background: #F3F4F6; 
            padding: 10px; 
            border: none; 
            border-right: 1px solid #E5E7EB;
            font-weight: 700; 
            color: #374151;
        }
        
        QTableWidget {
            background: #FFFFFF; 
            border: 1px solid #E5E7EB; 
            border-radius: 10px;
            alternate-background-color: #FAFAFA;
        }
        
        QTableWidget::item { 
            padding: 8px; 
        }
        
        QTableWidget::item:selected { 
            background: #EEF2FF; 
            color: #111827; 
        }

        QLabel#StatsCardFooter { 
            background: #F3F4F6; 
            padding: 8px 12px; 
            border-radius: 8px; 
            color: #374151; 
            font-size: 12px; 
        }
        
        QLabel#StatsLastUpdate {
            color: #6B7280; 
            font: 600 12px 'Segoe UI';
            text-align: center;
        }
        
        QPushButton { 
            background: #2563EB; 
            color: white; 
            border: none; 
            border-radius: 10px; 
            padding: 8px 18px; 
        }
        
        QPushButton:hover { 
            background: #1E4FD6; 
        }
        """
        )

    def _fill_stats_table(self, table: QTableWidget, rows: list[list]) -> None:
        """
        Fill data vào table.
        """
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                table.setItem(r, c, item)

    # ======================================================================
    #  REFRESH STATISTICS (CORE)
    # ======================================================================

    def _refresh_statistics(self, force: bool = False) -> None:
        """
        Làm mới thống kê với dữ liệu thật từ self.stats_service.

        - Tự động được gọi:
            + Khi mở trang thống kê (on_show_statistics_clicked).
            + Khi bấm "Làm mới".
            + Mỗi 5s khi đang xem trang thống kê (từ HistoryPageMixin.on_history_signal_refresh).
        """
        if not getattr(self, "stats_service", None):
            return

        now = time.time()
        if not force and (now - self._stats_last_reload) < 1.0:
            return

        try:
            # range_type từ combo box
            range_type = "today"
            if getattr(self, "stats_range_combo", None):
                selected = self.stats_range_combo.currentText()
                if selected == "Hôm nay":
                    range_type = "today"
                elif selected == "7 ngày":
                    range_type = "7days"
                elif selected == "Tháng này":
                    range_type = "month"

            # Lấy thống kê theo khoảng thời gian
            range_stats = self.stats_service.get_statistics_by_range(range_type)
            if range_stats.get("error"):
                print(f"Lỗi lấy thống kê theo range: {range_stats['error']}")
                return

            # Overview
            overview = self.stats_service.get_overview_statistics()
            if overview.get("error"):
                print(f"Lỗi lấy overview: {overview['error']}")
                return

            # KPI chips
            current_cars = overview.get("current_cars_inside", 0) or 0
            entries_period = range_stats.get("entries_today", 0) or 0
            exits_period = range_stats.get("exits_today", 0) or 0

            # KPI: Xe trong bãi
            if hasattr(self, "kpi_inpark"):
                self.kpi_inpark.update_value(str(current_cars))

            # KPI: Xe vào
            if hasattr(self, "kpi_in"):
                if range_type == "today":
                    self.kpi_in.update_title("Xe vào hôm nay")
                elif range_type == "7days":
                    self.kpi_in.update_title("Xe vào 7 ngày")
                elif range_type == "month":
                    self.kpi_in.update_title("Xe vào tháng này")
                self.kpi_in.update_value(str(entries_period))

            # KPI: Xe ra
            if hasattr(self, "kpi_out"):
                if range_type == "today":
                    self.kpi_out.update_title("Xe ra hôm nay")
                elif range_type == "7days":
                    self.kpi_out.update_title("Xe ra 7 ngày")
                elif range_type == "month":
                    self.kpi_out.update_title("Xe ra tháng này")
                self.kpi_out.update_value(str(exits_period))

            # Overview labels
            if getattr(self, "ov_labels", None):
                total_sessions = overview.get("total_sessions", 0) or 0
                range_matched = range_stats.get("matched_exits", 0) or 0
                range_unmatched = range_stats.get("unmatched_exits", 0) or 0
                pending_cars = range_stats.get("pending_cars", 0) or 0
                avg_minutes = overview.get("avg_parking_minutes", 0) or 0

                values = [
                    total_sessions,
                    pending_cars,
                    current_cars,
                    range_unmatched,
                    entries_period,
                    range_matched,
                    exits_period,
                    avg_minutes,
                ]

                for lbl, v in zip(self.ov_labels, values):
                    lbl.setText(str(v))

            # Xe đang trong bãi (bảng 1)
            cars_inside = self.stats_service.get_cars_currently_inside()
            if not cars_inside.get("error") and hasattr(self, "tbl_stats_cars_inside"):
                cars_data = cars_inside.get("list", [])
                rows = []
                for i, car in enumerate(cars_data, 1):
                    duration_minutes = car.get("duration_minutes", 0) or 0
                    rows.append(
                        [
                            i,
                            car.get("plate", ""),
                            car.get("date_in", ""),
                            car.get("time_in", ""),
                            duration_minutes,
                        ]
                    )
                self._fill_stats_table(self.tbl_stats_cars_inside, rows)

            # TOP 5 lâu nhất (bảng 2)
            if hasattr(self, "tbl_stats_longest"):
                longest_data = self.stats_service.get_cars_currently_inside()
                if not longest_data.get("error"):
                    cars = longest_data.get("list", [])
                    cars.sort(key=lambda x: x.get("duration_minutes", 0), reverse=True)
                    rows = []
                    for car in cars[:5]:
                        minutes = car.get("duration_minutes", 0)
                        hours = round(minutes / 60, 1) if minutes else 0
                        rows.append(
                            [
                                car.get("plate", ""),
                                car.get("date_in", ""),
                                car.get("time_in", ""),
                                minutes,
                                hours,
                            ]
                        )
                    self._fill_stats_table(self.tbl_stats_longest, rows)

            # Thống kê theo thời gian (weekly) (bảng 3)
            if hasattr(self, "tbl_stats_frequent"):
                weekly_data = self.stats_service.get_weekly_real_data()
                if not weekly_data.get("error"):
                    days = weekly_data.get("weekly_data", [])
                    summary = weekly_data.get("summary", {})

                    rows = []
                    for day in days:
                        rows.append(
                            [
                                day.get("date", ""),
                                day.get("day_name", ""),
                                day.get("entries", 0),
                                day.get("exits", 0),
                                day.get("matched", 0),
                                day.get("unmatched", 0),
                                f"{day.get('success_rate', 0):.2f}",
                            ]
                        )
                    self._fill_stats_table(self.tbl_stats_frequent, rows)

                    if hasattr(self, "footer_weekly"):
                        total_in = summary.get("total_in", 0)
                        total_out = summary.get("total_out", 0)
                        total_matched = summary.get("total_matched", 0)
                        overall_success = summary.get("overall_success", 0)
                        self.footer_weekly.setText(
                            f"Tổng vào: {total_in} · Tổng ra: {total_out} · "
                            f"Khớp: {total_matched} · Tỉ lệ: {overall_success:.2f}%"
                        )

            # Cập nhật thời gian
            if hasattr(self, "lbl_stats_last_update"):
                self.lbl_stats_last_update.setText(
                    "Cập nhật: "
                    + QDateTime.currentDateTime().toString("dd/MM/yyyy HH:mm:ss")
                )

            self._stats_last_reload = time.time()

        except Exception as e:
            print(f"Lỗi refresh statistics: {e}")
            import traceback

            traceback.print_exc()

    # ======================================================================
    #  SLOTS CHO NÚT / COMBO
    # ======================================================================

    @Slot()
    def on_show_statistics_clicked(self) -> None:
        """
        Khi nhấn nút "Xem thống kê" ở sidebar.
        """
        if not getattr(self, "stats_service", None):
            QMessageBox.information(
                self,
                "Thống kê",
                "Chức năng thống kê yêu cầu bật kết nối cơ sở dữ liệu.",
            )
            return

        # Đảm bảo nút lịch sử ở trạng thái "bình thường"
        if hasattr(self, "btn_hide_history"):
            self.btn_hide_history.hide()
        if hasattr(self, "btn_show_history"):
            self.btn_show_history.show()

        self._refresh_statistics(force=True)

        index = self.stacked.indexOf(self.statistics_view)
        if index != -1:
            self.stacked.setCurrentIndex(index)

    @Slot()
    def on_refresh_statistics_clicked(self) -> None:
        """
        Khi nhấn nút "Làm mới".
        """
        if not hasattr(self, "btn_stats_refresh"):
            return

        self.btn_stats_refresh.setEnabled(False)
        original_text = self.btn_stats_refresh.text()
        self.btn_stats_refresh.setText("🔄 Đang tải...")

        try:
            self._refresh_statistics(force=True)
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("✅ Đã cập nhật thống kê", 2000)
        except Exception as e:
            print(f"Lỗi refresh thống kê: {e}")
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("❌ Lỗi cập nhật thống kê", 3000)
        finally:
            self.btn_stats_refresh.setText(original_text)
            self.btn_stats_refresh.setEnabled(True)

    @Slot()
    def on_export_statistics_report(self) -> None:
        """
        Khi nhấn nút Export trong trang thống kê.
        """
        if not getattr(self, "stats_service", None):
            QMessageBox.information(
                self,
                "Thống kê",
                "Chức năng thống kê yêu cầu bật kết nối cơ sở dữ liệu.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu báo cáo thống kê",
            "parking_report.txt",
            "Text Files (*.txt)",
        )
        if not path:
            return

        if self.stats_service.export_comprehensive_report(path):
            QMessageBox.information(
                self,
                "Thống kê",
                f"Đã lưu báo cáo tại:\n{path}",
            )
        else:
            QMessageBox.warning(
                self,
                "Thống kê",
                "Không thể tạo báo cáo, vui lòng thử lại.",
            )

    @Slot()
    def on_stats_range_changed(self) -> None:
        """
        Khi đổi giá trị combo (Hôm nay / 7 ngày / Tháng này).
        """
        if not getattr(self, "stats_range_combo", None):
            return

        # Disable combo và nút refresh tạm thời
        self.stats_range_combo.setEnabled(False)

        original_refresh_text = None
        if getattr(self, "btn_stats_refresh", None):
            self.btn_stats_refresh.setEnabled(False)
            original_refresh_text = self.btn_stats_refresh.text()
            self.btn_stats_refresh.setText("🔄 Đang tải...")

        try:
            self._refresh_statistics(force=True)
            current_range = self.stats_range_combo.currentText()
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(f"✅ Đã cập nhật thống kê: {current_range}", 2000)
        except Exception as e:
            print(f"Lỗi khi thay đổi khoảng thời gian: {e}")
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("❌ Lỗi cập nhật thống kê", 3000)
        finally:
            self.stats_range_combo.setEnabled(True)
            if getattr(self, "btn_stats_refresh", None):
                self.btn_stats_refresh.setText(
                    original_refresh_text if original_refresh_text is not None else "Làm mới"
                )
                self.btn_stats_refresh.setEnabled(True)
