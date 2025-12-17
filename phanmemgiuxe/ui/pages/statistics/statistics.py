# phanmemgiuxe/ui/pages/statistics/statistics.py
"""
StatisticsPageMixin

Trang THỐNG KÊ: hiển thị KPI, tổng quan, xe trong bãi, thống kê theo ngày.

Thiết kế dựa trên các bảng:
- ParkingSessions
- FeeRules
- Vehicles
- VehicleTypes

stats_service (do MainWindow gán vào) cần cung cấp tối thiểu các hàm:

1) get_overview_statistics() -> dict
2) get_statistics_by_range(range_type: str) -> dict
3) get_cars_currently_inside() -> dict
4) export_comprehensive_report(path: str) -> bool
"""

from __future__ import annotations

import time
from typing import List, Any, cast

from PySide6.QtCore import Qt, Slot, QDateTime, QObject
from PySide6.QtGui import QPainter
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
    QTabWidget,
    QMessageBox,
    QFileDialog,
)


try:  
    from PySide6.QtCharts import (
        QChart,
        QChartView,
        QLineSeries,
        QBarCategoryAxis,
        QValueAxis,
    )

    HAVE_QTCHARTS = True
except Exception:  
    QChart = QChartView = QLineSeries = QBarCategoryAxis = QValueAxis = None  
    HAVE_QTCHARTS = False

# ---- QtCharts (biểu đồ) – optional ----
try:  # type: ignore
    from PySide6 import QtCharts

    HAVE_QTCHARTS = True
except Exception:
    QtCharts = None  # type: ignore
    HAVE_QTCHARTS = False


from ...theme import normalize_button
from ...widgets import StatsCard, KPIChip





# ====== Statistics Page Mixin ======
class StatisticsPages:
    """
    Mixin cung cấp UI + logic cho trang THỐNG KÊ, với các TAB con:

    - Tab 1: Tổng quan
    - Tab 2: Xe trong bãi
    - Tab 3: Thống kê theo ngày
    """

    # Khai báo trước cho Pylance đỡ báo lỗi "unknown attribute"
    stats_service: Any
    _stats_last_reload: float
    statistics_view: QWidget
    stats_tabs: QTabWidget

    # toolbar controls (giữ lại tham chiếu để dùng khi cần)
    stats_range_combo: QComboBox
    btn_stats_refresh: QPushButton
    btn_stats_export: QPushButton
    _range_combos: List[QComboBox]

    # KPI
    kpi_inpark: KPIChip
    kpi_in: KPIChip
    kpi_out: KPIChip

    # cards
    card_overview: StatsCard
    card_inpark: StatsCard
    card_weekly: StatsCard

    # labels / tables / footer
    ov_labels: List[QLabel]
    tbl_stats_cars_inside: QTableWidget
    tbl_stats_frequent: QTableWidget
    footer_weekly: QLabel
    footer_inpark: QLabel
    lbl_stats_last_update: QLabel

    # chart
    chart_weekly: "QChartView"  # type: ignore

    
    
    
    # === Stub cho type checker để dùng self.statusBar() ===
    def statusBar(self) -> Any:  # pragma: no cover
        ...

    
    
    # === Stub cho type checker để dùng self.sender() ===
    def sender(self) -> QObject | None:  # pragma: no cover
        ...

   





    # === Xây dựng trang thống kê ===
    def build_statistics_page(self, common_btn_style: str) -> QWidget:  
        """
        Xây dựng trang thống kê (statistics_view).
        """
        self.statistics_view = QWidget()
        self.statistics_view.setObjectName("StatisticsRoot")
        self._stats_last_reload = 0.0
        self._range_combos = []

        root_layout = QVBoxLayout(self.statistics_view)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(4)

        # QTabWidget giống như Cấu hình
        self.stats_tabs = QTabWidget()
        self.stats_tabs.setDocumentMode(True)
        self.stats_tabs.setTabPosition(QTabWidget.TabPosition.North)
        root_layout.addWidget(self.stats_tabs)

        # Tổng quan
        tab_overview = QWidget()
        tab_overview_layout = QVBoxLayout(tab_overview)
        tab_overview_layout.setContentsMargins(4, 4, 4, 4)
        tab_overview_layout.setSpacing(8)

        # Thanh công cụ: khoảng thời gian + nút
        (
            toolbar_overview,
            combo_overview,
            btn_refresh_overview,
            btn_export_overview,
        ) = self._create_stats_toolbar_row()
        tab_overview_layout.addLayout(toolbar_overview)

        # lưu lại bộ của tab đầu tiên để tương thích với các thuộc tính cũ
        self.stats_range_combo = combo_overview
        self.btn_stats_refresh = btn_refresh_overview
        self.btn_stats_export = btn_export_overview

        # KPI row 
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(24)
        tab_overview_layout.addLayout(kpi_row)
        kpi_row.addStretch(1)

        # Giá trị mặc định trước khi gọi DB
        initial_inpark = "0"
        initial_entries = "0"
        initial_revenue = "0"

        # Thử lấy overview ban đầu nếu stats_service đã được gán
        if getattr(self, "stats_service", None):
            try:
                overview = self.stats_service.get_overview_statistics()
                if not overview.get("error"):
                    totals = overview.get("totals", {}) or {}
                    revenue = overview.get("revenue", {}) or {}
                    initial_inpark = str(totals.get("current_inpark", 0) or 0)
                    initial_entries = str(totals.get("total_sessions", 0) or 0)
                    initial_revenue = str(int(revenue.get("total", 0) or 0))
            except Exception as e:
                print(f"Không thể load thống kê ban đầu: {e}")

        # 3 KPI chính
        self.kpi_inpark = KPIChip("Xe đang trong bãi", initial_inpark, "#E6F7EC")
        self.kpi_in = KPIChip("Tổng lượt gửi (tất cả)", initial_entries, "#E5F0FF")
        self.kpi_out = KPIChip("Doanh thu hôm nay (đ)", initial_revenue, "#FFF7E6")

        for chip in (self.kpi_inpark, self.kpi_in, self.kpi_out):
            chip.setMinimumHeight(120)
            chip.setMinimumWidth(220)
            kpi_row.addWidget(chip)

        kpi_row.addStretch(1)

        # Card tổng quan 
        self.card_overview = StatsCard("Tổng quan hệ thống")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        self.ov_labels = []

        # Giá trị khởi tạo
        total_sessions = "0"
        total_internal = "0"
        total_visitor = "0"
        current_inpark = initial_inpark
        rev_total = initial_revenue
        rev_internal = "0"
        rev_visitor = "0"
        unpaid_amount = "0"
        unpaid_count = "0"

        if getattr(self, "stats_service", None):
            try:
                overview = self.stats_service.get_overview_statistics()
                if not overview.get("error"):
                    totals = overview.get("totals", {}) or {}
                    revenue = overview.get("revenue", {}) or {}

                    total_sessions = str(totals.get("total_sessions", 0) or 0)
                    total_internal = str(totals.get("internal_sessions", 0) or 0)
                    total_visitor = str(totals.get("visitor_sessions", 0) or 0)
                    current_inpark = str(totals.get("current_inpark", 0) or 0)

                    rev_total = str(int(revenue.get("total", 0) or 0))
                    rev_internal = str(int(revenue.get("internal", 0) or 0))
                    rev_visitor = str(int(revenue.get("visitor", 0) or 0))
                    unpaid_amount = str(int(revenue.get("unpaid_amount", 0) or 0))
                    unpaid_count = str(revenue.get("unpaid_count", 0) or 0)
            except Exception as e:
                print(f"Không thể load overview ban đầu: {e}")

        # 8 dòng tổng quan
        pairs = [
            ("Tổng lượt gửi (tất cả):", total_sessions),
            ("Lượt nội bộ (tất cả):", total_internal),
            ("Lượt vãng lai (tất cả):", total_visitor),
            ("Xe đang trong bãi:", current_inpark),
            ("Doanh thu (lọc):", rev_total),
            ("Doanh thu nội bộ (lọc):", rev_internal),
            ("Doanh thu vãng lai (lọc):", rev_visitor),
            ("Chưa thanh toán (số tiền / số phiên):", f"{unpaid_amount} / {unpaid_count}"),
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

        
        
        
        
        # === Thêm grid vào card tổng quan ===
        from typing import cast  
        layout = self.card_overview.layout()
        if layout is not None:
            cast(QVBoxLayout, layout).addLayout(grid)
        tab_overview_layout.addWidget(self.card_overview)

        # Label cập nhật 
        self.lbl_stats_last_update = QLabel("Cập nhật: --")
        self.lbl_stats_last_update.setObjectName("StatsLastUpdate")
        tab_overview_layout.addWidget(self.lbl_stats_last_update)

        tab_overview_layout.addStretch(1)
        self.stats_tabs.addTab(tab_overview, "Tổng quan")

        # ===== Tab 2: XE TRONG BÃI =====
        tab_inpark = QWidget()
        tab_inpark_layout = QVBoxLayout(tab_inpark)
        tab_inpark_layout.setContentsMargins(4, 4, 4, 4)
        tab_inpark_layout.setSpacing(8)

        toolbar_inpark, _, _, _ = self._create_stats_toolbar_row()
        tab_inpark_layout.addLayout(toolbar_inpark)

        self.card_inpark = StatsCard("Xe đang trong bãi")
        self.tbl_stats_cars_inside = self._make_stats_table(
            [
                "STT",
                "Biển số",
                "Loại phiên",
                "Loại xe",
                "Ngày vào",
                "Giờ vào",
                "Thời gian (phút)",
            ]
        )
        self.tbl_stats_cars_inside.setMinimumHeight(350)
        self.card_inpark.layout().addWidget(self.tbl_stats_cars_inside)

        # footer tóm tắt
        self.footer_inpark = QLabel("Tổng xe đang trong bãi: 0")
        self.footer_inpark.setObjectName("StatsCardFooter")
        self.card_inpark.layout().addWidget(self.footer_inpark)

        tab_inpark_layout.addWidget(self.card_inpark)
        tab_inpark_layout.addStretch(1)
        self.stats_tabs.addTab(tab_inpark, "Xe trong bãi")

        # ===== Tab 3: THEO THỜI GIAN ======
        tab_time = QWidget()
        tab_time_layout = QVBoxLayout(tab_time)
        tab_time_layout.setContentsMargins(4, 4, 4, 4)
        tab_time_layout.setSpacing(8)

        toolbar_time, _, _, _ = self._create_stats_toolbar_row()
        tab_time_layout.addLayout(toolbar_time)

        self.card_weekly = StatsCard("Thống kê theo ngày")
        self.tbl_stats_frequent = self._make_stats_table(
            [
                "Ngày",
                "Thứ",
                "Tổng lượt",
                "Nội bộ",
                "Vãng lai",
                "Doanh thu",
                "DT nội bộ",
                "DT vãng lai",
            ]
        )
        self.tbl_stats_frequent.setMinimumHeight(260)
        self.card_weekly.layout().addWidget(self.tbl_stats_frequent)

        # Biểu đồ theo ngày (nếu có QtCharts)
        if HAVE_QTCHARTS and QtCharts is not None:
            self.chart_weekly = QtCharts.QChartView()
            self.chart_weekly.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.chart_weekly.setMinimumHeight(220)
            self.chart_weekly.setObjectName("StatsWeeklyChart")
            self.card_weekly.layout().addWidget(self.chart_weekly)

        else:
            lbl_no_chart = QLabel(
                "Biểu đồ không khả dụng (thiếu PySide6.QtCharts). "
                "Có thể cài thêm QtCharts nếu muốn xem biểu đồ."
            )
            lbl_no_chart.setObjectName("StatsCardFooter")
            self.card_weekly.layout().addWidget(lbl_no_chart)

        self.footer_weekly = QLabel(
            "Tổng lượt: 0 · Nội bộ: 0 · Vãng lai: 0 · Doanh thu: 0 đ"
        )
        self.footer_weekly.setObjectName("StatsCardFooter")

        self.card_weekly.layout().addWidget(self.footer_weekly)

        tab_time_layout.addWidget(self.card_weekly)
        tab_time_layout.addStretch(1)

        self.stats_tabs.addTab(tab_time, "Theo thời gian")

        # STYLE
        self._apply_stats_styles()

        # load lần đầu
        self._refresh_statistics(force=True, range_type=self._get_current_range_type())

        return self.statistics_view







    # === Tạo 1 hàng thanh công cụ cho thống kê ===
    def _create_stats_toolbar_row(
        self,
    ) -> tuple[QHBoxLayout, QComboBox, QPushButton, QPushButton]:
        """
        Tạo 1 thanh công cụ giống Config:
        [Khoảng thời gian: (combo)] .................. [Làm mới] [Export báo cáo]
        """
        layout = QHBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(8, 0, 8, 4)

        lbl_range = QLabel("Khoảng thời gian")
        lbl_range.setObjectName("StatsRangeLabel")
        layout.addWidget(lbl_range)

        combo = QComboBox()
        combo.setObjectName("StatsRangeCombo")
        combo.addItems(["Hôm nay", "7 ngày", "Tháng này"])
        combo.currentIndexChanged.connect(self.on_stats_range_changed)
        combo.setFixedHeight(28)
        combo.setFixedWidth(140)
        layout.addWidget(combo)

        layout.addStretch(1)

        btn_refresh = QPushButton("Làm mới")
        btn_refresh.setObjectName("StatsToolbarButton")

        btn_export = QPushButton("Export báo cáo")
        btn_export.setObjectName("StatsToolbarButton")

        # vẫn dùng normalize_button để đồng nhất font, padding cơ bản
        normalize_button(btn_refresh, btn_export)

        btn_refresh.clicked.connect(self.on_refresh_statistics_clicked)
        btn_export.clicked.connect(self.on_export_statistics_report)

        layout.addWidget(btn_refresh)
        layout.addWidget(btn_export)

        # lưu lại để tìm combo đang visible
        if not hasattr(self, "_range_combos"):
            self._range_combos = []
        self._range_combos.append(combo)

        return layout, combo, btn_refresh, btn_export


    
    
    
    
    
    # === Tạo QTableWidget cho thống kê ===
    def _make_stats_table(self, headers: list[str]) -> QTableWidget:
        """Tạo QTableWidget cho trang thống kê."""
        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setDefaultSectionSize(26)

        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        return table






    # === Áp dụng stylesheet cho thống kê ===
    def _apply_stats_styles(self) -> None:
        """Áp dụng stylesheet cho card & KPI, không thay nền / nút chung toàn app."""
        if not getattr(self, "statistics_view", None):
            return

        self.statistics_view.setStyleSheet(
            """
        QWidget#StatisticsRoot {
            background: #F5F5F7;
        }

        /* Toolbar: label + combo + button */
        QLabel#StatsRangeLabel {
            font-size: 13px;
            font-weight: 600;
            color: #111827;
            padding-right: 6px;
        }

        QComboBox#StatsRangeCombo {
            background: #E5E7EB;        /* xám giống nút */
            color: #111827;             /* chữ đen */
            border-radius: 6px;
            border: 1px solid #D1D5DB;
            padding: 2px 10px;
        }
        QComboBox#StatsRangeCombo::drop-down {
            border: none;
            width: 20px;
        }

        /* Popup list của combobox */
        QComboBox#StatsRangeCombo QAbstractItemView {
            background: #FFFFFF;         /* nền trắng */
            color: #111827;              /* chữ đen */
            selection-background-color: #E5E7EB;
            selection-color: #111827;
            border: 1px solid #D1D5DB;
            padding: 2px;
        }

        /* Bảng: xen kẽ trắng / xám nhạt, không còn xanh đậm khó nhìn */
        QWidget#StatisticsRoot QTableView,
        QWidget#StatisticsRoot QTableWidget {
            background-color: #FFFFFF;
            alternate-background-color: #F3F4F6;
            color: #111827;
            gridline-color: #D1D5DB;
        }

        QPushButton#StatsToolbarButton {
            background-color: #E5E7EB;
            color: #111827;
            border-radius: 6px;
            padding: 4px 14px;
            border: 1px solid #D1D5DB;
            font-weight: 600;
        }
        QPushButton#StatsToolbarButton:hover {
            background-color: #D1D5DB;
        }
        QPushButton#StatsToolbarButton:pressed {
            background-color: #9CA3AF;
        }
        QPushButton#StatsToolbarButton:disabled {
            background-color: #9CA3AF;
            border-color: #9CA3AF;
            color: #F3F4F6;
        }

        /* Card & KPI */
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
            font-size: 32px;
            font-weight: 900;
            color: #DC2626;
        }

        QLabel#OvKey {
            font-size: 14px;
            color: #1F2937;
            font-weight: 600;
        }

        QLabel#OvVal {
            font-size: 14px;
            color: #111827;
            font-weight: 700;
            padding-left: 6px;
        }

        QLabel#StatsCardFooter {
            background: #F3F4F6;
            padding: 6px 10px;
            border-radius: 8px;
            color: #374151;
            font-size: 12px;
        }

        QLabel#StatsLastUpdate {
            color: #6B7280;
            font: 600 12px 'Segoe UI';
        }
        """
        )






    # === Điền dữ liệu vào bảng thống kê ===
    def _fill_stats_table(self, table: QTableWidget, rows: list[list]) -> None:
        """Fill data vào table."""
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                )
                table.setItem(r, c, item)






    # === Lấy range_type hiện tại từ combo đang hiển thị ===
    def _get_current_range_type(self) -> str:
        """
        Lấy range_type ('today' / '7days' / 'month') dựa vào combo đang hiển thị.
        """
        text = "Hôm nay"
        for combo in getattr(self, "_range_combos", []):
            if combo.isVisible():
                text = combo.currentText()
                break

        if text in ("Hôm nay", "today"):
            return "today"
        if text in ("7 ngày", "7days"):
            return "7days"
        if text in ("Tháng này", "month"):
            return "month"
        return "today"






    # === Làm mới thống kê từ service ===
    def _refresh_statistics(self, force: bool = False, range_type: str | None = None) -> None:
        """Làm mới thống kê với dữ liệu thật từ self.stats_service."""
        if not getattr(self, "stats_service", None):
            return

        now = time.time()
        if not force and (now - self._stats_last_reload) < 1.0:
            return

        if range_type is None:
            range_type = self._get_current_range_type()

        try:
            # Lấy thống kê theo khoảng thời gian
            range_stats = self.stats_service.get_statistics_by_range(range_type)
            if range_stats.get("error"):
                print(f"Lỗi lấy thống kê theo range: {range_stats['error']}")
                return

            # Overview (tổng toàn hệ thống)
            overview = self.stats_service.get_overview_statistics()
            if overview.get("error"):
                print(f"Lỗi lấy overview: {overview['error']}")
                return

            totals = overview.get("totals", {}) or {}
            revenue_overall = overview.get("revenue", {}) or {}

            # KPI chips
            current_cars = totals.get("current_inpark", 0) or 0
            total_sessions = totals.get("total_sessions", 0) or 0
            revenue_total_range = int(range_stats.get("revenue_total", 0) or 0)

            if hasattr(self, "kpi_inpark"):
                self.kpi_inpark.update_value(str(current_cars))

            if hasattr(self, "kpi_in"):
                self.kpi_in.update_title("Tổng lượt gửi (tất cả)")
                self.kpi_in.update_value(str(total_sessions))

            if hasattr(self, "kpi_out"):
                if range_type == "today":
                    title = "Doanh thu hôm nay (đ)"
                elif range_type == "7days":
                    title = "Doanh thu 7 ngày (đ)"
                else:
                    title = "Doanh thu tháng này (đ)"
                self.kpi_out.update_title(title)
                self.kpi_out.update_value(str(revenue_total_range))

            # Overview labels
            if getattr(self, "ov_labels", None):
                total_internal = totals.get("internal_sessions", 0) or 0
                total_visitor = totals.get("visitor_sessions", 0) or 0

                unpaid_amount = int(revenue_overall.get("unpaid_amount", 0) or 0)
                unpaid_count = revenue_overall.get("unpaid_count", 0) or 0

                # phần doanh thu theo khoảng (range)
                rev_total_range = revenue_total_range
                rev_internal_range = int(range_stats.get("revenue_internal", 0) or 0)
                rev_visitor_range = int(range_stats.get("revenue_visitor", 0) or 0)

                values = [
                    total_sessions,
                    total_internal,
                    total_visitor,
                    current_cars,
                    rev_total_range,
                    rev_internal_range,
                    rev_visitor_range,
                    f"{unpaid_amount} / {unpaid_count}",
                ]

                for lbl, v in zip(self.ov_labels, values):
                    lbl.setText(str(v))

            # Xe đang trong bãi (tab 2)
            cars_inside = self.stats_service.get_cars_currently_inside()
            if not cars_inside.get("error") and hasattr(self, "tbl_stats_cars_inside"):
                cars_data = cars_inside.get("list", []) or []
                rows = []
                for i, car in enumerate(cars_data, 1):
                    duration_minutes = car.get("duration_minutes", 0) or 0
                    session_cat = car.get("session_category", "") or ""
                    if session_cat == "INTERNAL":
                        session_cat_label = "Nội bộ"
                    elif session_cat == "VISITOR":
                        session_cat_label = "Vãng lai"
                    else:
                        session_cat_label = session_cat

                    rows.append(
                        [
                            i,
                            car.get("plate", ""),
                            session_cat_label,
                            car.get("vehicle_type_name", ""),
                            car.get("date_in", ""),
                            car.get("time_in", ""),
                            duration_minutes,
                        ]
                    )
                self._fill_stats_table(self.tbl_stats_cars_inside, rows)

                # cập nhật footer
                if hasattr(self, "footer_inpark"):
                    self.footer_inpark.setText(
                        f"Tổng xe đang trong bãi: {len(rows)}"
                    )

            # Thống kê theo ngày (tab 3)
            if hasattr(self, "tbl_stats_frequent"):
                daily = range_stats.get("daily", []) or []
                rows = []

                total_all = 0
                total_internal_sum = 0
                total_visitor_sum = 0
                total_revenue_sum = 0

                for item in daily:
                    total = item.get("total_sessions", 0) or 0
                    internal = item.get("internal_sessions", 0) or 0
                    visitor = item.get("visitor_sessions", 0) or 0
                    rev_total_d = int(item.get("revenue_total", 0) or 0)
                    rev_internal_d = int(item.get("revenue_internal", 0) or 0)
                    rev_visitor_d = int(item.get("revenue_visitor", 0) or 0)

                    total_all += total
                    total_internal_sum += internal
                    total_visitor_sum += visitor
                    total_revenue_sum += rev_total_d

                    rows.append(
                        [
                            item.get("date", ""),
                            item.get("day_name", ""),
                            total,
                            internal,
                            visitor,
                            rev_total_d,
                            rev_internal_d,
                            rev_visitor_d,
                        ]
                    )

                self._fill_stats_table(self.tbl_stats_frequent, rows)

                if hasattr(self, "footer_weekly"):
                    self.footer_weekly.setText(
                        f"Tổng lượt: {total_all} · Nội bộ: {total_internal_sum} · "
                        f"Vãng lai: {total_visitor_sum} · Doanh thu: {total_revenue_sum} đ"
                    )

                # ---- Cập nhật biểu đồ line: tổng lượt / ngày ----
                if HAVE_QTCHARTS and QtCharts is not None and hasattr(self, "chart_weekly"):
                    chart = QtCharts.QChart()
                    chart.setTitle("Biểu đồ tổng lượt gửi theo ngày")

                    series = QtCharts.QLineSeries()
                    series.setName("Tổng lượt")

                    categories: list[str] = []
                    for idx, item in enumerate(daily):
                        total = int(item.get("total_sessions", 0) or 0)
                        series.append(float(idx), float(total))
                        categories.append(str(item.get("date", "")))

                    chart.addSeries(series)

                    axis_x = QtCharts.QBarCategoryAxis()
                    axis_x.append(categories)
                    axis_y = QtCharts.QValueAxis()
                    axis_y.setTitleText("Lượt gửi")

                    chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
                    chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
                    series.attachAxis(axis_x)
                    series.attachAxis(axis_y)

                    chart.legend().setVisible(True)
                    chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

                    self.chart_weekly.setChart(chart)


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

   
   
   
   
   
    # === Slots / Handlers ===
    @Slot()
    def on_refresh_statistics_clicked(self) -> None:
        """Khi nhấn nút 'Làm mới' ở bất kỳ tab nào."""
        sender_obj = self.sender()
        btn = sender_obj if isinstance(sender_obj, QPushButton) else None

        original_text = "Làm mới"
        if btn is not None:
            original_text = btn.text()
            btn.setEnabled(False)
            btn.setText("🔄 Đang tải...")

        try:
            range_type = self._get_current_range_type()
            self._refresh_statistics(force=True, range_type=range_type)
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("✅ Đã cập nhật thống kê", 2000)
        except Exception as e:
            print(f"Lỗi refresh thống kê: {e}")
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("❌ Lỗi cập nhật thống kê", 3000)
        finally:
            if btn is not None:
                btn.setText(original_text)
                btn.setEnabled(True)






    # === Slots / Handlers ===
    @Slot()
    def on_export_statistics_report(self) -> None:
        """Khi nhấn nút Export trong trang thống kê (bất kỳ tab)."""
        if not getattr(self, "stats_service", None):
            parent = cast(QWidget, self)
            QMessageBox.information(
                parent,
                "Thống kê",
                "Chức năng thống kê yêu cầu bật kết nối cơ sở dữ liệu.",
            )
            return

        parent = cast(QWidget, self)
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Lưu báo cáo thống kê",
            "parking_report.txt",
            "Text Files (*.txt)",
        )
        if not path:
            return

        ok = False
        try:
            ok = self.stats_service.export_comprehensive_report(path)
        except TypeError:
            try:
                range_type = self._get_current_range_type()
                ok = self.stats_service.export_comprehensive_report_with_filters(
                    path, range_type
                )
            except Exception:
                ok = False

        if ok:
            QMessageBox.information(
                parent,
                "Thống kê",
                f"Đã lưu báo cáo tại:\n{path}",
            )
        else:
            QMessageBox.warning(
                parent,
                "Thống kê",
                "Không thể tạo báo cáo, vui lòng thử lại.",
            )






    # === Slots / Handlers ===
    @Slot()
    def on_stats_range_changed(self) -> None:
        """Khi đổi giá trị combo (Hôm nay / 7 ngày / Tháng này) ở bất kỳ tab."""
        try:
            range_type = self._get_current_range_type()
            self._refresh_statistics(force=True, range_type=range_type)

            if hasattr(self, "statusBar") and self.statusBar():
                text = "Hôm nay"
                for combo in getattr(self, "_range_combos", []):
                    if combo.isVisible():
                        text = combo.currentText()
                        break
                self.statusBar().showMessage(
                    f"✅ Đã cập nhật thống kê: {text}", 2000
                )
        except Exception as e:
            print(f"Lỗi khi thay đổi khoảng thời gian: {e}")
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage("❌ Lỗi cập nhật thống kê", 3000)
