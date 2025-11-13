# ui/pages/history.py
"""
HistoryPageMixin

Chịu trách nhiệm:
- Trang BẢNG LỊCH SỬ (history_view) với QTableWidget.
- Trang CHI TIẾT (detail_view) hiển thị thông tin 1 lượt gửi và ảnh IN/OUT.
- Tải dữ liệu lịch sử từ DB bằng HistoryLoaderWorker (workers.HistoryLoaderWorker).
- Export lịch sử ra Excel.
- Xoá lịch sử (dòng chọn hoặc toàn bộ).
- Tự động refresh định kỳ (on_history_signal_refresh).

YÊU CẦU MainWindow (class kế thừa mixin này) có:
- self.db                  : DB instance
- self.history_df          : pandas.DataFrame
- self.history_worker      : HistoryLoaderWorker | None
- self._hist_last_reload   : float (timestamp)
- self.current_filter_*    : các biến filter (start, end, status, plate)
- self.stacked             : QStackedWidget
- self.clear_detail_view() : hàm này nằm trong mixin này
- self.show_main_view()    : hàm trong MainWindow (hoặc mixin khác)
- Các helper ảnh từ CameraPageMixin:
    + self._set_centered_pixmap(...)
    + self._get_valid_image_path_internal(...)
    + self.qpix_logo()
"""

from __future__ import annotations

import time
import pandas as pd

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QLineEdit,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtWidgets import QLabel
from ..widgets import make_card
from ...workers.workers import HistoryLoaderWorker
from ...dialogs.dialogs import DeleteDialog


class HistoryPageMixin:
    """
    Mixin cung cấp UI + logic cho phần BẢNG LỊCH SỬ + TRANG CHI TIẾT.
    """

    # ======================================================================
    #  BUILD HISTORY & DETAIL PAGES
    # ======================================================================

    def build_history_pages(self, common_btn_style: str) -> tuple[QWidget, QWidget]:
        """
        Tạo:
        - history_view: trang bảng lịch sử (ParkingSessions)
        - detail_view : trang chi tiết 1 lượt gửi

        Trả về:
            (history_view, detail_view)
        """

        # ======================= HISTORY VIEW =======================
        self.history_view = QWidget()
        hist_layout = QVBoxLayout(self.history_view)

        hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)")
        hist_v = QVBoxLayout(hist_group)

        # Bảng lịch sử
        self.tbl_hist = QTableWidget(0, 10)
        self.tbl_hist.setHorizontalHeaderLabels(
            [
                "ID",
                "Ảnh vào",
                "Biển số vào",
                "Ngày vào",
                "Giờ vào",
                "Ảnh ra",
                "Biển số ra",
                "Ngày ra",
                "Giờ ra",
                "Trạng thái",
            ]
        )

        header = self.tbl_hist.horizontalHeader()

        hfont = QFont(header.font())
        hfont.setBold(True)
        header.setFont(hfont)

        # Cột resize theo nội dung
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Ngày vào
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Giờ vào
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Ngày ra
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Giờ ra
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Trạng thái

        # Các cột còn lại stretch
        for j in range(1, 10):
            if header.sectionResizeMode(j) != QHeaderView.ResizeToContents:
                header.setSectionResizeMode(j, QHeaderView.Stretch)

        self.tbl_hist.setSizePolicy(
            self.tbl_hist.sizePolicy().horizontalPolicy(),
            self.tbl_hist.sizePolicy().verticalPolicy(),
        )
        self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.tbl_hist.setAlternatingRowColors(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected)

        hist_v.addWidget(self.tbl_hist)
        hist_layout.addWidget(hist_group)

        # ======================= DETAIL VIEW =======================
        self.detail_view = QWidget()
        detail_layout = QVBoxLayout(self.detail_view)

        # Hàng nút quay lại
        row_btn_back = QHBoxLayout()
        self.btn_back_to_history = getattr(self, "btn_back_to_history", None)
        if self.btn_back_to_history is None:
            # sẽ được kết nối trong MainWindow._connect_cross_page_signals
            from PySide6.QtWidgets import QPushButton
            self.btn_back_to_history = QPushButton("⬅ Quay lại bảng lịch sử", self.detail_view)

        row_btn_back.addWidget(self.btn_back_to_history)
        row_btn_back.addStretch(1)
        detail_layout.addLayout(row_btn_back)

        # Hàng ảnh: scene IN / scene OUT
        row_images = QHBoxLayout()

        self.lbl_detail_scene = QLabel()
        self.lbl_detail_roi = QLabel()

        for lbl in (self.lbl_detail_scene, self.lbl_detail_roi):
            lbl.setScaledContents(False)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
            lbl.setMinimumHeight(320)
            lbl.setSizePolicy(
                lbl.sizePolicy().horizontalPolicy(),
                lbl.sizePolicy().verticalPolicy(),
            )

        detail_scene_card, _ = make_card("Ảnh Chụp Vào (Image_IN)", self.lbl_detail_scene)
        detail_roi_card, _ = make_card("Ảnh Chụp Ra (Image_OUT)", self.lbl_detail_roi)

        row_images.addWidget(detail_scene_card, 1)
        row_images.addWidget(detail_roi_card, 1)
        detail_layout.addLayout(row_images, 1)

        # Group thông tin lượt gửi
        gb_detail_info = QGroupBox("Thông tin Lượt Gửi")
        gl_detail = QGridLayout(gb_detail_info)

        self.lbl_detail_plate_in = QLineEdit()
        self.lbl_detail_plate_in.setReadOnly(True)
        self.lbl_detail_date_in = QLineEdit()
        self.lbl_detail_date_in.setReadOnly(True)
        self.lbl_detail_time_in = QLineEdit()
        self.lbl_detail_time_in.setReadOnly(True)

        self.lbl_detail_plate_out = QLineEdit()
        self.lbl_detail_plate_out.setReadOnly(True)
        self.lbl_detail_date_out = QLineEdit()
        self.lbl_detail_date_out.setReadOnly(True)
        self.lbl_detail_time_out = QLineEdit()
        self.lbl_detail_time_out.setReadOnly(True)

        self.lbl_detail_match = QLineEdit()
        self.lbl_detail_match.setReadOnly(True)

        self.lbl_detail_plate_in.setStyleSheet(
            "color: #ff0000; font-size: 14px; font-weight: 700;"
        )
        self.lbl_detail_plate_out.setStyleSheet(
            "color: #ff0000; font-size: 14px; font-weight: 700;"
        )
        self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")

        gl_detail.addWidget(QLabel("Biển số vào:"), 0, 0)
        gl_detail.addWidget(self.lbl_detail_plate_in, 0, 1)
        gl_detail.addWidget(QLabel("Ngày vào:"), 1, 0)
        gl_detail.addWidget(self.lbl_detail_date_in, 1, 1)
        gl_detail.addWidget(QLabel("Giờ vào:"), 2, 0)
        gl_detail.addWidget(self.lbl_detail_time_in, 2, 1)

        gl_detail.addWidget(QLabel("Biển số ra:"), 0, 2)
        gl_detail.addWidget(self.lbl_detail_plate_out, 0, 3)
        gl_detail.addWidget(QLabel("Ngày ra:"), 1, 2)
        gl_detail.addWidget(self.lbl_detail_date_out, 1, 3)
        gl_detail.addWidget(QLabel("Giờ ra:"), 2, 2)
        gl_detail.addWidget(self.lbl_detail_time_out, 2, 3)

        gl_detail.addWidget(QLabel("Trạng thái:"), 3, 0)
        gl_detail.addWidget(self.lbl_detail_match, 3, 1, 1, 3)

        detail_layout.addWidget(gb_detail_info)

        return self.history_view, self.detail_view

    # ======================================================================
    #  VIEW SWITCHERS
    # ======================================================================

    @Slot()
    def on_show_all_history_clicked(self) -> None:
        """
        Slot kết nối với btn_show_history.
        - Chuyển sang tab bảng lịch sử (index 1).
        - Tải lại dữ liệu history (xoá filter).
        """
        print("\n--- DEBUG: on_show_all_history_clicked just called refresh_history_data ---\n")
        if self.stacked.currentIndex() != 1:
            self.show_history_view_only()
        self.refresh_history_data(clear_filters=True)

    @Slot()
    def show_history_view_only(self) -> None:
        """
        Chỉ chuyển tab sang trang bảng lịch sử, KHÔNG tải lại dữ liệu.
        """
        self.stacked.setCurrentIndex(1)
        if hasattr(self, "btn_show_history") and hasattr(self, "btn_hide_history"):
            self.btn_show_history.hide()
            self.btn_hide_history.show()

    # ======================================================================
    #  EXPORT & DELETE
    # ======================================================================

    @Slot()
    def on_export_excel(self) -> None:
        """
        Export self.history_df ra file Excel.
        """
        df_to_export = self.history_df.copy()
        if not df_to_export.empty and "STT" in df_to_export.columns:
            df_to_export = df_to_export.drop(columns=["STT"])

        if df_to_export.empty:
            QMessageBox.information(self, "Export", "Không có dữ liệu để export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu Excel",
            "history.xlsx",
            "Excel Files (*.xlsx)",
        )
        if not path:
            return

        try:
            df_to_export.to_excel(path, index=False)
            QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

    @Slot()
    def on_delete_history(self) -> None:
        """
        Xử lý xoá lịch sử:
        - Xoá các dòng chọn theo ID.
        - Hoặc xoá toàn bộ bảng (delete_all).
        """
        if not (self.db and getattr(self.db, "ok", False)):
            QMessageBox.warning(self, "Xóa", "Chưa kết nối DB.")
            return

        dlg = DeleteDialog(self)
        dlg.setModal(True)
        dlg.adjustSize()

        # Căn giữa dialog theo cửa sổ chính
        parent_center = self.geometry().center()
        dlg_rect = dlg.frameGeometry()
        dlg_rect.moveCenter(self.mapToGlobal(parent_center))
        dlg.move(dlg_rect.topLeft())

        res = dlg.exec()
        ids_to_delete: list[str] = []

        # res == 1: xoá dòng chọn
        if res == 1:
            rows_view = sorted(set(idx.row() for idx in self.tbl_hist.selectedIndexes()))
            if not rows_view:
                QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào.")
                return

            for r_view in rows_view:
                id_item = self.tbl_hist.item(r_view, 0)  # cột 0 là ID
                if id_item:
                    ids_to_delete.append(id_item.text())

            if not ids_to_delete:
                QMessageBox.warning(self, "Xóa", "Không lấy được ID các dòng chọn.")
                return

            self.db.delete_by_ids(ids_to_delete)

        # res == 2: xoá toàn bộ
        elif res == 2:
            confirm = QMessageBox.question(
                self,
                "Xác nhận",
                "Bạn chắc chắn muốn xóa TOÀN BỘ lịch sử?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.db.delete_all()
            else:
                return

        else:
            # Người dùng bấm Cancel
            return

        # Sau khi xoá:
        # - Dọn trang chi tiết.
        # - Quay về bảng lịch sử.
        # - Tải lại dữ liệu với bộ lọc hiện tại.
        self.clear_detail_view()
        self.show_history_view_only()

        self.refresh_history_data(
            start_time=self.current_filter_start,
            end_time=self.current_filter_end,
            status_filter=self.current_filter_status,
            plate_filter=self.current_filter_plate,
        )

    # ======================================================================
    #  CLEAR DETAIL VIEW
    # ======================================================================

    def clear_detail_view(self) -> None:
        """
        Xoá nội dung trang chi tiết, đưa ảnh về logo mặc định, xoá selection trong bảng.
        """
        for w in (
            self.lbl_detail_plate_in,
            self.lbl_detail_date_in,
            self.lbl_detail_time_in,
            self.lbl_detail_plate_out,
            self.lbl_detail_date_out,
            self.lbl_detail_time_out,
            self.lbl_detail_match,
        ):
            w.setText("")

        # Đổi ảnh về logo mặc định
        self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())
        self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

        # Bỏ chọn dòng trong bảng lịch sử
        self.tbl_hist.clearSelection()

    # ======================================================================
    #  REFRESH HISTORY DATA (LOAD BẰNG WORKER)
    # ======================================================================

    def refresh_history_data(
        self,
        start_time=None,
        end_time=None,
        status_filter=None,
        plate_filter=None,
        clear_filters: bool = False,
    ) -> None:
        """
        Tải lại lịch sử dùng HistoryLoaderWorker.

        Nếu clear_filters=True:
        - Xoá bộ lọc đang dùng (tất cả self.current_filter_* = None).
        - Worker được tạo với các tham số None (tức là không filter).
        """

        # XÓA BỘ LỌC NẾU CÓ YÊU CẦU
        if clear_filters:
            print("--- Clearing filters because clear_filters=True ---")
            self.current_filter_start = None
            self.current_filter_end = None
            self.current_filter_status = None
            self.current_filter_plate = None

            start_time = None
            end_time = None
            status_filter = None
            plate_filter = None

        # Nếu worker đang chạy thì thôi, tránh đè nhau
        if self.history_worker and self.history_worker.isRunning():
            print("History worker is already running.")
            return

        print(
            f"+++ Starting HistoryLoaderWorker with filters: "
            f"Start={start_time}, End={end_time}, "
            f"Status={status_filter}, Plate={plate_filter} +++"
        )

        self.history_worker = HistoryLoaderWorker(
            self.db,
            start_time,
            end_time,
            status_filter,
            plate_filter,
            self,
        )
        self.history_worker.resultReady.connect(self.update_history_table)
        self.history_worker.finished.connect(self.history_worker.deleteLater)
        self.history_worker.start()

    # ======================================================================
    #  AUTO REFRESH (TIMER / WORKER SIGNAL)
    # ======================================================================

    @Slot()
    def on_history_signal_refresh(self) -> None:
        """
        Được gọi bởi:
        - QTimer trong MainWindow
        - histSignal từ VideoWorker

        Hành vi:
        - Nếu đang ở trang thống kê -> refresh thống kê real-time.
        - Nếu đang ở trang lịch sử -> mỗi 5s reload history theo filter hiện tại.
        """
        # Cập nhật statistics real-time (nếu có trang thống kê)
        if hasattr(self, "statistics_view") and self.statistics_view is not None:
            current_widget = self.stacked.currentWidget()
            if current_widget == self.statistics_view and hasattr(self, "_refresh_statistics"):
                # Đang xem trang thống kê -> cập nhật ngay
                self._refresh_statistics(force=True)

        # History refresh: chỉ khi đang ở index 1
        if self.stacked.currentIndex() != 1:
            return

        now = time.time()
        if now - self._hist_last_reload < 5.0:
            return

        self._hist_last_reload = now

        self.refresh_history_data(
            start_time=self.current_filter_start,
            end_time=self.current_filter_end,
            status_filter=self.current_filter_status,
            plate_filter=self.current_filter_plate,
        )

    # ======================================================================
    #  NHẬN DATAFRAME TỪ WORKER
    # ======================================================================

    @Slot(pd.DataFrame)
    def update_history_table(self, df: pd.DataFrame) -> None:
        """
        Nhận DataFrame từ HistoryLoaderWorker và fill vào tbl_hist.
        """
        print(f"+++ update_history_table received {len(df)} rows +++")
        self.history_df = df.copy()

        df_display = df.drop(columns=["STT"], errors="ignore")

        self.tbl_hist.setUpdatesEnabled(False)
        self.tbl_hist.setSortingEnabled(False)

        cols = list(df_display.columns)
        self.tbl_hist.clearContents()
        self.tbl_hist.setColumnCount(len(cols))
        self.tbl_hist.setHorizontalHeaderLabels(cols)
        self.tbl_hist.setRowCount(len(df_display))

        for i in range(len(df_display)):
            for j, col in enumerate(cols):
                if j >= self.tbl_hist.columnCount():
                    continue
                val = df_display.iloc[i, j]
                item = QTableWidgetItem()
                if j == 0:
                    # Cột ID -> hiển thị dạng số nếu có thể
                    try:
                        item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                    except Exception:
                        item.setText(str(val))
                else:
                    item.setText(str(val))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.tbl_hist.setItem(i, j, item)

        # Sắp xếp theo ID giảm dần
        self.tbl_hist.setSortingEnabled(True)
        self.tbl_hist.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.tbl_hist.setSortingEnabled(True)
        self.tbl_hist.setUpdatesEnabled(True)

        self.history_worker = None
        print("--- History worker reference released ---")

    # ======================================================================
    #  CHỌN HÀNG -> MỞ TRANG CHI TIẾT
    # ======================================================================

    @Slot()
    def on_history_row_selected(self) -> None:
        """
        Khi người dùng chọn 1 hàng trong bảng lịch sử:
        - Tìm row tương ứng trong self.history_df theo ID.
        - Đổ dữ liệu vào các field của trang detail.
        - Đọc ảnh vào/ra và hiển thị.
        - Chuyển sang trang detail (stacked index 2).
        """
        selected_items = self.tbl_hist.selectedItems()
        if not selected_items or self.history_df.empty or "ID" not in self.history_df.columns:
            return

        try:
            row_index_view = selected_items[0].row()
            id_item = self.tbl_hist.item(row_index_view, 0)
            if not id_item:
                return

            row_id = int(id_item.text())
            row_data_series = self.history_df[self.history_df["ID"] == row_id]

            if row_data_series.empty:
                return

            row_data = row_data_series.iloc[0]

            # Cập nhật thông tin text
            self.lbl_detail_plate_in.setText(str(row_data.get("Biển số vào", "")))
            self.lbl_detail_date_in.setText(str(row_data.get("Ngày vào", "")))
            self.lbl_detail_time_in.setText(str(row_data.get("Giờ vào", "")))

            self.lbl_detail_plate_out.setText(str(row_data.get("Biển số ra", "")))
            self.lbl_detail_date_out.setText(str(row_data.get("Ngày ra", "")))
            self.lbl_detail_time_out.setText(str(row_data.get("Giờ ra", "")))

            match_status = str(row_data.get("Trạng thái", "")).replace("-", " ").title()
            self.lbl_detail_match.setText(match_status)

            if "KHOP-BIEN-SO" in match_status:
                self.lbl_detail_match.setStyleSheet("color: #007700; font-weight: 700;")
            elif "KHONG-KHOP-BIEN-SO" in match_status:
                self.lbl_detail_match.setStyleSheet("color: #ff0000; font-weight: 700;")
            else:
                self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")

            # Cập nhật ảnh (dùng helper từ CameraPageMixin)
            valid_in_path = self._get_valid_image_path_internal(
                str(row_data.get("Ảnh vào", ""))
            )
            valid_out_path = self._get_valid_image_path_internal(
                str(row_data.get("Ảnh ra", ""))
            )

            import cv2

            if valid_in_path:
                self._set_centered_pixmap(
                    self.lbl_detail_scene,
                    cv2.imread(valid_in_path),
                )
            else:
                self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())

            if valid_out_path:
                self._set_centered_pixmap(
                    self.lbl_detail_roi,
                    cv2.imread(valid_out_path),
                )
            else:
                self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

            # Chuyển sang trang chi tiết (giả định index 2)
            # Chỉ cần setCurrentWidget(detail_view) là an toàn hơn
            self.stacked.setCurrentWidget(self.detail_view)

        except Exception as e:
            print(f"Lỗi khi chọn hàng: {e}")
            import traceback

            traceback.print_exc()
