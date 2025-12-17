# ui/pages/historis/history.py
"""
HistoryPage

Trang LỊCH SỬ cho guard/manager:

- Trên cùng: 1 thanh chứa
    + Bên trái: QTabWidget 2 tab lớn  ->  "Xem bảng" / "Tìm kiếm"
    + Bên phải: các nút thao tác -> Thêm, Sửa, Xuất, Xóa
  (giống giao diện tab Cấu hình: "Loại xe / Xe nội bộ" + dãy nút bên phải)

- Bên dưới: QStackedWidget:
    * Trang 1: Bảng lịch sử (HistoryPageMixin từ history_table.py)
    * Trang 2: Trang chi tiết 1 lượt gửi
    * Trang 3: Trang bộ lọc tìm kiếm (SearchPageMixin từ search.py)

Chỉ cập nhật giao diện, giữ nguyên logic xử lý dữ liệu.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QStackedWidget,
    QDialog,
    QTabWidget,
)

from ...theme import normalize_button, apply_button_style
from ....config.config import USE_SQL, CONN_STR
from ....database.database import DB

from .history_table import HistoryPageMixin
from .search import SearchPageMixin
from .add_history import AddHistoryDialog
from .edit_history import EditHistoryDialog




# ====== HISTORY PAGE ======
class HistoryPage(QWidget, HistoryPageMixin, SearchPageMixin):
    """
    Trang lịch sử tự chứa DB, stack nội bộ.
    MainWindow chỉ việc addWidget(HistoryPage).
    """

    # === Init page ===
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # backend
        self.db: Optional[DB] = None
        self.history_df: pd.DataFrame = pd.DataFrame()
        self._hist_last_reload: float = 0.0

        # filter hiện tại
        self.current_filter_start = None
        self.current_filter_end = None
        self.current_filter_status = None
        self.current_filter_plate = None
        self.current_filter_vehicle_group = None  # filter nhóm xe

        self._load_db()
        self._build_ui()

        # lần đầu mở trang: load dữ liệu 1 lần
        if self.db and getattr(self.db, "ok", False):
            self.refresh_history_data(clear_filters=True)

    
    
    
    
    
    # === Load DB connection ===
    def _load_db(self) -> None:
        if USE_SQL:
            self.db = DB(CONN_STR)
            if not self.db.ok:
                print("[HistoryPage] Không kết nối được DB.")
        else:
            self.db = None
            print("[HistoryPage] USE_SQL = False, không dùng DB.")

    
    
    
    
    
    # === Build UI ===
    def _build_ui(self) -> None:
        self.setObjectName("HistoryPageRoot")
        self.setStyleSheet(
            """
            QWidget#HistoryPageRoot {
                background-color:#f5f5f7;
            }

            QGroupBox {
                background-color:#ffffff;
                border:1px solid #d1d5db;
                border-radius:8px;
                margin-top:18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left:12px;
                padding:0 6px;
                color:#111827;
                font-weight:600;
                font-size:14px;
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

            /* nút Back trong trang chi tiết (objectName do mixin đặt) */
            QPushButton#BackToHistoryBtn {
                background-color:#2563eb;
                color:#ffffff;
                border-radius:6px;
                padding:6px 14px;
                font-weight:600;
            }
            QPushButton#BackToHistoryBtn:hover {
                background-color:#1e40af;
            }
            QPushButton#BackToHistoryBtn:pressed {
                background-color:#1d4ed8;
            }

            /* khung thanh trên của lịch sử – mô phỏng khung xám của Cấu hình */
            QFrame#HistoryTopBar {
                background-color:#4b4b4b;
                border:1px solid #3f3f3f;
                border-radius:0px;
            }
            """
        )

        # -------- ROOT LAYOUT: THẲNG ĐỨNG (thanh trên + stack dưới) --------
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(4)

        # ===================== THANH TRÊN: TAB LỚN + NÚT ACTION =====================
        top_bar = QFrame(self)
        top_bar.setObjectName("HistoryTopBar")
        top_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 4, 6, 4)
        top_layout.setSpacing(6)

        self.subtabs = QTabWidget(top_bar)
        self.subtabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self._tab_view = QWidget()
        self._tab_search = QWidget()
        self.subtabs.addTab(self._tab_view, "Xem bảng")
        self.subtabs.addTab(self._tab_search, "Tìm kiếm")

        self.subtabs.setStyleSheet(
            """
            QTabWidget::pane {
                border:0px;
                background-color:transparent;
            }
            QTabBar::tab {
                background-color:#e5e7eb;
                color:#111827;
                padding:4px 12px;
                margin-right:2px;
            }
            QTabBar::tab:selected {
                background-color:#ffffff;
                color:#111827;
            }
            QTabBar::tab:hover {
                background-color:#f3f4f6;
            }
            """
        )

        top_layout.addWidget(self.subtabs, 1)
        self.btn_add = QPushButton("Thêm")
        self.btn_edit = QPushButton("Sửa")
        self.btn_export = QPushButton("Xuất")
        self.btn_delete = QPushButton("Xóa")

        normalize_button(self.btn_add, self.btn_edit, self.btn_export, self.btn_delete)

        for btn in (self.btn_add, self.btn_edit, self.btn_export, self.btn_delete):
            btn.setMinimumWidth(90)
            btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        top_layout.addStretch(1)
        top_layout.addWidget(self.btn_add)
        top_layout.addWidget(self.btn_edit)
        top_layout.addWidget(self.btn_export)
        top_layout.addWidget(self.btn_delete)

        root.addWidget(top_bar)

        # ===================== STACK NỘI DUNG BÊN DƯỚI =====================
        self.stacked = QStackedWidget(self)
        self.stacked.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.stacked, 1)

        common_btn_style = "padding:6px 10px; font-size:13px;"

        # Tạo các trang từ mixin (GIỮ NGUYÊN LOGIC)
        history_view, detail_view = self.build_history_pages(common_btn_style)
        search_filter_view = self.build_search_page(common_btn_style)

        self.stacked.addWidget(history_view)        # index 0
        self.stacked.addWidget(detail_view)         # index 1
        self.stacked.addWidget(search_filter_view)  # index 2

        self.stacked.setCurrentWidget(history_view)
        self.subtabs.setCurrentIndex(0)
        self._style_search_page_buttons()

        # ===================== NỐI SIGNAL (KHÔNG ĐỔI LOGIC) =====================
        # từ trang chi tiết + filter
        self.btn_back_to_history.clicked.connect(self.show_history_view_only)
        self.sfv_btn_back.clicked.connect(self.show_history_view_only)
        self.sfv_btn_search.clicked.connect(self.on_run_search_from_page)

        # Tab lớn: Xem bảng / Tìm kiếm
        self.subtabs.currentChanged.connect(self._on_subtab_changed)

        # Action buttons
        self.btn_add.clicked.connect(self._on_click_add)
        self.btn_edit.clicked.connect(self._on_click_edit)
        self.btn_export.clicked.connect(self.on_export_excel)   # hàm trong mixin
        self.btn_delete.clicked.connect(self.on_delete_history)


    
    
    
    
    # === Style nút trong trang tìm kiếm ===
    def _style_search_page_buttons(self) -> None:
        """
        Làm 2 nút 'Quay lại bảng lịch sử' và 'Tìm kiếm' đậm hơn.
        Giả định SearchPageMixin tạo:
            - self.sfv_btn_back
            - self.sfv_btn_search
        """
        primary_css = """
        QPushButton {
            background-color:#2563eb;
            color:#ffffff;
            border-radius:6px;
            padding:8px 20px;
            font-weight:600;
        }
        QPushButton:hover {
            background-color:#1e40af;
        }
        QPushButton:pressed {
            background-color:#1d4ed8;
        }
        """

        if hasattr(self, "sfv_btn_back") and isinstance(self.sfv_btn_back, QPushButton):
            apply_button_style(self.sfv_btn_back, primary_css)
        if hasattr(self, "sfv_btn_search") and isinstance(self.sfv_btn_search, QPushButton):
            apply_button_style(self.sfv_btn_search, primary_css)

    
    
    
    
    
    # === Xử lý chuyển tab lớn ===
    def _on_subtab_changed(self, index: int) -> None:
        """
        Giữ nguyên logic cũ:
        - Tab 0 -> Xem bảng (list / detail dùng self.stacked như trước)
        - Tab 1 -> Tìm kiếm (trang filter trong self.stacked)
        """
        if index == 0:
            self.show_history_view_only()
        else:
            self.on_search_history_clicked()

    
    
    
    
    
    # === Xử lý nút Thêm / Sửa ===
    def _on_click_add(self) -> None:
        if not (self.db and getattr(self.db, "ok", False)):
            QMessageBox.warning(self, "Thêm", "Chưa kết nối DB, không thể thêm mới.")
            return

        dlg = AddHistoryDialog(self, db=self.db)
        res = dlg.exec()
        if res == QDialog.DialogCode.Accepted:
            # reload theo filter hiện tại (kể cả nhóm xe)
            self.refresh_history_data(
                start_time=self.current_filter_start,
                end_time=self.current_filter_end,
                status_filter=self.current_filter_status,
                plate_filter=self.current_filter_plate,
                vehicle_group_filter=self.current_filter_vehicle_group,
            )
            # quay về tab "Xem bảng"
            self.subtabs.setCurrentIndex(0)
            self.show_history_view_only()






    # === Xử lý nút Sửa ===
    def _on_click_edit(self) -> None:
        if not (self.db and getattr(self.db, "ok", False)):
            QMessageBox.warning(self, "Sửa", "Chưa kết nối DB, không thể sửa.")
            return

        selected = self.tbl_hist.selectedIndexes()
        if not selected:
            QMessageBox.information(self, "Sửa", "Vui lòng chọn 1 dòng trong bảng để sửa.")
            return

        row_view = selected[0].row()
        id_item = self.tbl_hist.item(row_view, 0)  # cột 0 = ID
        if not id_item:
            QMessageBox.warning(self, "Sửa", "Không lấy được ID bản ghi.")
            return

        try:
            record_id = int(id_item.text())
        except Exception:
            QMessageBox.warning(self, "Sửa", "ID bản ghi không hợp lệ.")
            return

        if self.history_df.empty or "ID" not in self.history_df.columns:
            QMessageBox.warning(self, "Sửa", "Không tìm thấy dữ liệu lịch sử.")
            return

        row_series = self.history_df[self.history_df["ID"] == record_id]
        if row_series.empty:
            QMessageBox.warning(self, "Sửa", "Không tìm thấy bản ghi trong DataFrame.")
            return

        row_data = row_series.iloc[0]
        record = row_data.to_dict()

        dlg = EditHistoryDialog(self, db=self.db, record=record)
        res = dlg.exec()
        if res == QDialog.DialogCode.Accepted:
            self.refresh_history_data(
                start_time=self.current_filter_start,
                end_time=self.current_filter_end,
                status_filter=self.current_filter_status,
                plate_filter=self.current_filter_plate,
                vehicle_group_filter=self.current_filter_vehicle_group,
            )
            # quay lại tab "Xem bảng"
            self.subtabs.setCurrentIndex(0)
            self.show_history_view_only()
