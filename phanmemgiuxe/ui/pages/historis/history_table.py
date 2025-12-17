# ui/pages/historis/history_table.py
"""
HistoryTablePageMixin (HistoryPageMixin)

- Xây UI:
    + Trang bảng lịch sử (QTableWidget)
    + Trang chi tiết 1 lượt gửi (ảnh + thông tin)
- Logic:
    + load/update bảng trực tiếp từ DB (không dùng HistoryLoaderWorker nữa)
    + chọn dòng để xem chi tiết
    + export excel
    + xóa bản ghi

Được dùng bên trong HistoryPage (history.py).

YÊU CẦU lớp cha (HistoryPage) phải có các thuộc tính:
    - self.stacked: QStackedWidget
    - self.db: DB hoặc None
    - self.history_df: pd.DataFrame
    - self.current_filter_start, self.current_filter_end,
      self.current_filter_status, self.current_filter_plate,
      self.current_filter_vehicle_group
    - self._hist_last_reload: float
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import pandas as pd

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
)

from ...widgets import make_card
from ....dialogs.dialogs import DeleteDialog
from ....dialogs.export_history import ExportHistoryDialog, export_df_to_excel
from ....utils.utils import bgr_to_qimage
from ....database.database import DB  # Import DB class





# ====== History Table + Detail Page Mixin ======
class HistoryPageMixin:
    """
    Mixin xây UI bảng lịch sử + trang chi tiết.
    Dùng trong HistoryPage.
    """

    # Khai báo cho Pylance biết là sẽ có 2 thuộc tính này
    stacked: QStackedWidget
    db: Optional["DB"]

    
    
    
    
    
    # === Chuẩn hoá đường dẫn ảnh, trả về path tuyệt đối nếu tồn tại ===
    def _get_valid_image_path_internal(self, path: str) -> Optional[str]:
        """Chuẩn hoá đường dẫn ảnh, trả về path tuyệt đối nếu tồn tại."""
        if not path:
            return None

        path = str(path).strip()
        if not path:
            return None

        if os.path.isabs(path) and os.path.exists(path):
            return os.path.abspath(path)

        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

        return None






    # === Đưa ảnh vào QLabel, căn giữa & scale giữ tỉ lệ ===
    def _set_centered_pixmap(self, label: QLabel, img_or_pix) -> None:
        """
        Đưa ảnh vào QLabel, căn giữa & scale giữ tỉ lệ.
        - img_or_pix: numpy BGR hoặc QPixmap.
        """
        if img_or_pix is None:
            label.clear()
            return

        if isinstance(img_or_pix, QPixmap):
            pix = img_or_pix
        else:
            try:
                qimg = bgr_to_qimage(img_or_pix)
                pix = QPixmap.fromImage(qimg)
            except Exception:
                pix = self.qpix_logo()

        if pix.isNull():
            label.clear()
            return

        scaled = pix.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    
    
    
    
    
    # === Tạo pixmap placeholder (nền xám) ===
    def qpix_logo(self) -> QPixmap:
        """Tạo pixmap placeholder (nền xám)."""
        pix = QPixmap(320, 240)
        pix.fill(Qt.GlobalColor.lightGray)
        return pix

    
    
    
    
    
    # === Xây UI trang lịch sử + chi tiết ===
    def build_history_pages(self, _common_btn_style: str):
        # ====== TRANG BẢNG ======
        self.history_view = QWidget()
        self.history_view.setObjectName("HistoryRoot")
        hist_layout = QVBoxLayout(self.history_view)
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.setSpacing(0)

        # 12 cột (thêm "Tiền phí")
        self.tbl_hist = QTableWidget(0, 12)
        self.tbl_hist.setObjectName("HistoryTable")
        self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Scrollbar dọc màu trắng cho bảng lịch sử
        self.tbl_hist.setStyleSheet(
            """
            QScrollBar:vertical {
                background: white;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #d0d0d0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #bcbcbc;
            }
            QScrollBar::add-line,
            QScrollBar::sub-line {
                background: white;
                height: 0px;
            }
            QScrollBar::add-page,
            QScrollBar::sub-page {
                background: white;
            }
        """
        )

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
                "Nhóm xe",
                "Tiền phí",
                "Trạng thái",
            ]
        )

        header = self.tbl_hist.horizontalHeader()
        hfont = QFont(header.font())
        hfont.setBold(True)
        header.setFont(hfont)

        # ID, Ngày vào/ra, Giờ vào/ra, Tiền phí, Trạng thái auto size
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(10, QHeaderView.ResizeMode.ResizeToContents)  # Tiền phí
        header.setSectionResizeMode(11, QHeaderView.ResizeMode.ResizeToContents)  # Trạng thái

        # Các cột còn lại stretch
        for j in range(1, 12):
            if header.sectionResizeMode(j) != QHeaderView.ResizeMode.ResizeToContents:
                header.setSectionResizeMode(j, QHeaderView.ResizeMode.Stretch)

        header.setStretchLastSection(True)

        self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.tbl_hist.setAlternatingRowColors(True)
        self.tbl_hist.verticalHeader().setVisible(False)

        self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected)
        hist_layout.addWidget(self.tbl_hist, 1)

        # ====== TRANG CHI TIẾT ======
        self.detail_view = QWidget()
        self.detail_view.setObjectName("HistoryDetailRoot")
        detail_layout = QVBoxLayout(self.detail_view)
        detail_layout.setContentsMargins(8, 4, 8, 4)
        detail_layout.setSpacing(8)

        # nút quay lại
        row_back = QHBoxLayout()
        self.btn_back_to_history = QPushButton("⬅ Quay lại bảng lịch sử", self.detail_view)
        self.btn_back_to_history.setObjectName("BackToHistoryBtn")
        row_back.addWidget(self.btn_back_to_history)
        row_back.addStretch(1)
        detail_layout.addLayout(row_back)

        # ảnh
        row_imgs = QHBoxLayout()
        self.lbl_detail_scene = QLabel()
        self.lbl_detail_roi = QLabel()
        for lbl in (self.lbl_detail_scene, self.lbl_detail_roi):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
            lbl.setMinimumHeight(320)

        scene_card, _ = make_card("Ảnh Chụp Vào (Image_IN)", self.lbl_detail_scene)
        roi_card, _ = make_card("Ảnh Chụp Ra (Image_OUT)", self.lbl_detail_roi)
        row_imgs.addWidget(scene_card, 1)
        row_imgs.addWidget(roi_card, 1)
        detail_layout.addLayout(row_imgs, 1)

        # info group
        gb_detail = QGroupBox("Thông tin Lượt Gửi")
        gb_detail.setObjectName("HistoryDetailGroup")
        gb_detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        gl = QGridLayout(gb_detail)
        gl.setHorizontalSpacing(16)
        gl.setVerticalSpacing(8)

        self.lbl_detail_plate_in = QLineEdit()
        self.lbl_detail_date_in = QLineEdit()
        self.lbl_detail_time_in = QLineEdit()
        self.lbl_detail_plate_out = QLineEdit()
        self.lbl_detail_date_out = QLineEdit()
        self.lbl_detail_time_out = QLineEdit()
        self.lbl_detail_match = QLineEdit()
        self.lbl_detail_vehicle_group = QLineEdit()
        # mới: hiển thị tiền phí
        self.lbl_detail_fee_amount = QLineEdit()

        base_lineedit_style = (
            "background-color:#ffffff; color:#111827; "
            "border:1px solid #d1d5db; border-radius:3px; padding:2px 6px;"
        )

        for w in (
            self.lbl_detail_plate_in,
            self.lbl_detail_date_in,
            self.lbl_detail_time_in,
            self.lbl_detail_plate_out,
            self.lbl_detail_date_out,
            self.lbl_detail_time_out,
            self.lbl_detail_match,
            self.lbl_detail_vehicle_group,
            self.lbl_detail_fee_amount,
        ):
            w.setReadOnly(True)
            w.setStyleSheet(base_lineedit_style)

        self.lbl_detail_plate_in.setStyleSheet(
            base_lineedit_style + "color:#ff0000; font-size:14px; font-weight:700;"
        )
        self.lbl_detail_plate_out.setStyleSheet(
            base_lineedit_style + "color:#ff0000; font-size:14px; font-weight:700;"
        )
        self.lbl_detail_match.setStyleSheet(
            base_lineedit_style + "color:#0000ff; font-weight:700;"
        )
        self.lbl_detail_fee_amount.setStyleSheet(
            base_lineedit_style + "color:#047857; font-weight:700;"
        )

        r = 0
        gl.addWidget(QLabel("Biển số vào:"), r, 0)
        gl.addWidget(self.lbl_detail_plate_in, r, 1)
        gl.addWidget(QLabel("Biển số ra:"), r, 2)
        gl.addWidget(self.lbl_detail_plate_out, r, 3)
        r += 1

        gl.addWidget(QLabel("Ngày vào:"), r, 0)
        gl.addWidget(self.lbl_detail_date_in, r, 1)
        gl.addWidget(QLabel("Ngày ra:"), r, 2)
        gl.addWidget(self.lbl_detail_date_out, r, 3)
        r += 1

        gl.addWidget(QLabel("Giờ vào:"), r, 0)
        gl.addWidget(self.lbl_detail_time_in, r, 1)
        gl.addWidget(QLabel("Giờ ra:"), r, 2)
        gl.addWidget(self.lbl_detail_time_out, r, 3)
        r += 1

        # Trạng thái + Nhóm xe trên cùng 1 hàng
        gl.addWidget(QLabel("Trạng thái:"), r, 0)
        gl.addWidget(self.lbl_detail_match, r, 1)
        gl.addWidget(QLabel("Nhóm xe:"), r, 2)
        gl.addWidget(self.lbl_detail_vehicle_group, r, 3)
        r += 1

        # Hàng tiền phí
        gl.addWidget(QLabel("Tiền phí:"), r, 0)
        gl.addWidget(self.lbl_detail_fee_amount, r, 1)
        r += 1

        detail_layout.addWidget(gb_detail)

        # màu chữ đen cho mọi label trong trang chi tiết
        self.detail_view.setStyleSheet(
            """
            QWidget#HistoryDetailRoot QLabel {
                color:#111827;
            }
            """
        )

        return self.history_view, self.detail_view






    # === Chuyển sang trang bảng lịch sử (không reload) ===
    @Slot()
    def show_history_view_only(self) -> None:
        """Chỉ chuyển sang trang bảng, KHÔNG tự reload/filter."""
        self.stacked.setCurrentWidget(self.history_view)






    # === Nút 'Xem tất cả' lịch sử ===
    @Slot()
    def on_show_all_history_clicked(self) -> None:
        """Nếu cần nút 'Xem tất cả' riêng thì gọi hàm này."""
        if self.stacked.currentWidget() is not self.history_view:
            self.show_history_view_only()
        self.refresh_history_data(clear_filters=True)

    
    
    
    
    
    # === Xuất Excel ===
    @Slot()
    def on_export_excel(self) -> None:
        """
        Xuất lịch sử ra Excel:
            - Hộp thoại hỏi:
                + Xuất dòng đã chọn
                + Xuất toàn bộ bảng hiện tại
        """
        if self.history_df is None or self.history_df.empty:
            QMessageBox.information(self.history_view, "Xuất", "Không có dữ liệu để xuất.")
            return

        # DataFrame dùng để export (bỏ cột STT nếu có)
        df_base = self.history_df.drop(columns=["STT"], errors="ignore")
        has_selection = bool(self.tbl_hist.selectedIndexes())
        dlg = ExportHistoryDialog(self.history_view, has_selection=has_selection)
        result = dlg.exec()

        if result == 0:
            return

        if result == 1:
            # Xuất dòng đã chọn
            selected_rows = sorted({idx.row() for idx in self.tbl_hist.selectedIndexes()})
            if not selected_rows:
                QMessageBox.information(self.history_view, "Xuất", "Bạn chưa chọn dòng nào.")
                return

            # Lấy ID từ các dòng chọn, dùng ID map ngược về DataFrame
            selected_ids: list[int] = []
            for r in selected_rows:
                item = self.tbl_hist.item(r, 0)  # cột 0 = ID
                if item is None:
                    continue
                try:
                    selected_ids.append(int(item.text()))
                except Exception:
                    continue

            if not selected_ids:
                QMessageBox.warning(
                    self.history_view,
                    "Xuất",
                    "Không lấy được ID từ các dòng đang chọn.",
                )
                return

            if "ID" not in df_base.columns:
                QMessageBox.warning(
                    self.history_view,
                    "Xuất",
                    "DataFrame không có cột 'ID', không thể map dòng.",
                )
                return

            df_to_export = df_base[df_base["ID"].isin(selected_ids)].copy()
        else:
            # Xuất toàn bộ bảng hiện tại
            df_to_export = df_base.copy()

        export_df_to_excel(self.history_view, df_to_export)

    
    
    
    
    
    # === Xóa lịch sử ===
    @Slot()
    def on_delete_history(self) -> None:
        if not (self.db and getattr(self.db, "ok", False)):
            QMessageBox.warning(self.history_view, "Xóa", "Chưa kết nối DB.")
            return

        dlg = DeleteDialog(self.history_view)
        res = dlg.exec()
        ids_to_delete: list[str] = []

        if res == 1:
            # xóa theo các dòng đang chọn
            rows_view = sorted(set(idx.row() for idx in self.tbl_hist.selectedIndexes()))
            if not rows_view:
                QMessageBox.information(self.history_view, "Xóa", "Bạn chưa chọn dòng nào.")
                return

            for r_view in rows_view:
                id_item = self.tbl_hist.item(r_view, 0)
                if id_item:
                    ids_to_delete.append(id_item.text())

            if not ids_to_delete:
                QMessageBox.warning(self.history_view, "Xóa", "Không lấy được ID các dòng chọn.")
                return

            self.db.delete_by_ids(ids_to_delete)

        elif res == 2:
            # xóa toàn bộ
            confirm = QMessageBox.question(
                self.history_view,
                "Xác nhận",
                "Bạn chắc chắn muốn xóa TOÀN BỘ lịch sử?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.db.delete_all()
            else:
                return
        else:
            return

        self.clear_detail_view()
        self.show_history_view_only()

        # load lại theo filter hiện tại (kể cả nhóm xe)
        self.refresh_history_data(
            start_time=self.current_filter_start,
            end_time=self.current_filter_end,
            status_filter=self.current_filter_status,
            plate_filter=self.current_filter_plate,
            vehicle_group_filter=self.current_filter_vehicle_group,
        )

    
    
    
    
    
    # === Clear detail view ===
    def clear_detail_view(self) -> None:
        for w in (
            self.lbl_detail_plate_in,
            self.lbl_detail_date_in,
            self.lbl_detail_time_in,
            self.lbl_detail_plate_out,
            self.lbl_detail_date_out,
            self.lbl_detail_time_out,
            self.lbl_detail_match,
            self.lbl_detail_vehicle_group,
            self.lbl_detail_fee_amount,
        ):
            w.setText("")
        self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())
        self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())
        self.tbl_hist.clearSelection()

    
    
    
    
    
    
    # === Load lịch sử từ DB với filter ===
    def refresh_history_data(
        self,
        start_time=None,
        end_time=None,
        status_filter=None,
        plate_filter=None,
        vehicle_group_filter=None,
        clear_filters: bool = False,
    ) -> None:
        """
        Load dữ liệu lịch sử trực tiếp từ DB.
        Hỗ trợ filter theo:
            - khoảng thời gian
            - trạng thái (list[str] hoặc None)
            - biển số
            - nhóm xe (list[str] hoặc None) – ví dụ: ["Vãng lai", "Nội bộ"]
        """
        if not (self.db and getattr(self.db, "ok", False)):
            self.update_history_table(pd.DataFrame(columns=["STT"]))
            self.clear_detail_view()
            return

        if clear_filters:
            self.current_filter_start = None
            self.current_filter_end = None
            self.current_filter_status = None
            self.current_filter_plate = None
            self.current_filter_vehicle_group = None
        else:
            if start_time is not None:
                self.current_filter_start = start_time
            if end_time is not None:
                self.current_filter_end = end_time
            if status_filter is not None:
                self.current_filter_status = status_filter
            if plate_filter is not None:
                self.current_filter_plate = plate_filter
            if vehicle_group_filter is not None:
                self.current_filter_vehicle_group = vehicle_group_filter

        try:
            df = self.db.fetch_history_df(
                limit=10000,
                start_time=self.current_filter_start,
                end_time=self.current_filter_end,
                status_filter=self.current_filter_status,
                plate_filter=self.current_filter_plate,
                vehicle_group_filter=self.current_filter_vehicle_group,
            )
        except Exception as e:
            print("[HistoryPageMixin] refresh_history_data error:", e)
            QMessageBox.warning(
                self.history_view,
                "Lịch sử",
                f"Lỗi khi tải lịch sử từ DB:\n{e}",
            )
            return

        self.update_history_table(df)
        self._hist_last_reload = time.time()






    # === Slot refresh từ timer ngoài ===
    @Slot()
    def on_history_signal_refresh(self) -> None:
        """
        Có thể gọi từ timer ngoài để refresh định kỳ.
        """
        if self.stacked.currentWidget() is not self.history_view:
            return

        now = time.time()
        if now - getattr(self, "_hist_last_reload", 0.0) < 5.0:
            return

        self.refresh_history_data(
            start_time=None,
            end_time=None,
            status_filter=None,
            plate_filter=None,
            vehicle_group_filter=self.current_filter_vehicle_group,
            clear_filters=False,
        )

    
    
    
    
    
    # === Cập nhật bảng lịch sử từ DataFrame ===
    @Slot(pd.DataFrame)
    def update_history_table(self, df: pd.DataFrame) -> None:
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
                    # cột ID -> sort số
                    try:
                        if isinstance(val, (int, float, str)) and not isinstance(val, complex):
                            try:
                                item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                            except Exception:
                                item.setText(str(val))
                        else:
                            item.setText(str(val))
                    except Exception:
                        item.setText(str(val))
                else:
                    item.setText(str(val))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.tbl_hist.setItem(i, j, item)

        self.tbl_hist.setSortingEnabled(True)
        if cols:
            self.tbl_hist.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.tbl_hist.setUpdatesEnabled(True)

    
    
    
    
    
    # === Chọn dòng trong bảng lịch sử ===
    @Slot()
    def on_history_row_selected(self) -> None:
        """Click vào 1 dòng -> chuyển sang trang chi tiết."""
        selected_items = self.tbl_hist.selectedItems()
        if (
            not selected_items
            or self.history_df.empty
            or "ID" not in self.history_df.columns
        ):
            return

        try:
            row_index_view = selected_items[0].row()
            id_item = self.tbl_hist.item(row_index_view, 0)
            if not id_item:
                return

            row_id = int(id_item.text())
            row_series = self.history_df[self.history_df["ID"] == row_id]
            if row_series.empty:
                return
            row_data = row_series.iloc[0]

            self.lbl_detail_plate_in.setText(str(row_data.get("Biển số vào", "")))
            self.lbl_detail_date_in.setText(str(row_data.get("Ngày vào", "")))
            self.lbl_detail_time_in.setText(str(row_data.get("Giờ vào", "")))

            self.lbl_detail_plate_out.setText(str(row_data.get("Biển số ra", "")))
            self.lbl_detail_date_out.setText(str(row_data.get("Ngày ra", "")))
            self.lbl_detail_time_out.setText(str(row_data.get("Giờ ra", "")))

            match_status = str(row_data.get("Trạng thái", "")).replace("-", " ").upper()
            self.lbl_detail_match.setText(match_status)

            base = (
                "background-color:#ffffff; border:1px solid #d1d5db; "
                "border-radius:3px; padding:2px 6px; font-weight:700;"
            )
            if "KHOP" in match_status and "KHONG" not in match_status:
                self.lbl_detail_match.setStyleSheet(base + "color:#007700;")
            elif "KHONG" in match_status:
                self.lbl_detail_match.setStyleSheet(base + "color:#ff0000;")
            else:
                self.lbl_detail_match.setStyleSheet(base + "color:#0000ff;")

            # Nhóm xe
            self.lbl_detail_vehicle_group.setText(str(row_data.get("Nhóm xe", "")))

            # Tiền phí: lấy từ cột "Tiền phí" (đã được format sẵn trong DataFrame)
            if "Tiền phí" in row_data.index:
                fee_val = str(row_data.get("Tiền phí", "") or "")
                if fee_val:
                    self.lbl_detail_fee_amount.setText(f"{fee_val} VND")
                else:
                    self.lbl_detail_fee_amount.setText("")
            else:
                self.lbl_detail_fee_amount.setText("")

            in_path = self._get_valid_image_path_internal(str(row_data.get("Ảnh vào", "")))
            out_path = self._get_valid_image_path_internal(str(row_data.get("Ảnh ra", "")))

            if in_path:
                self._set_centered_pixmap(self.lbl_detail_scene, cv2.imread(in_path))
            else:
                self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())

            if out_path:
                self._set_centered_pixmap(self.lbl_detail_roi, cv2.imread(out_path))
            else:
                self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

            self.stacked.setCurrentWidget(self.detail_view)

        except Exception as e:
            print(f"Lỗi khi chọn hàng lịch sử: {e}")
            QMessageBox.warning(
                self.history_view,
                "Lỗi",
                f"Lỗi khi hiển thị chi tiết lịch sử:\n{e}",
            )
