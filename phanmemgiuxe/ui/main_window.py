# ui/main_window.py

import os
import time
import traceback

import numpy as np
import cv2
import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
    QSizePolicy,
    QMessageBox,
)

from ..config.config import (
    API_MAP,
    LOGO_PATH,
    DETECT_MODEL_PATH,
    OCR_MODEL_PATH,
    SOUND_IN_PATH,
    SOUND_OUT_PATH,
    CONN_STR,
    USE_SQL,
)
from ..database.database import DB
from ..models.models import Models, GEMINI_READY
from ..workers.workers import VideoWorker, HistoryLoaderWorker
from ..dialogs.dialogs import DeleteDialog
from ..utils.utils import bgr_to_qimage, letterbox
from ..statistics.statistics import ParkingStatistics

# Theme & widgets
from .theme import apply_global_theme

# Các "page" / mixin cho từng phần UI
from .pages.camera import CameraPageMixin
from .pages.history import HistoryPageMixin
from .pages.search import SearchPageMixin
from .pages.statistics import StatisticsPageMixin


class MainWindow(
    QMainWindow,
    CameraPageMixin,
    HistoryPageMixin,
    SearchPageMixin,
    StatisticsPageMixin,
):
    """
    Cửa sổ chính của ứng dụng.

    Vai trò của MainWindow:
    - Khởi tạo core (models, DB, statistics service).
    - Khởi tạo và ghép các "page" (camera, history, search, statistics) vào QStackedWidget.
    - Gắn các signal giữa sidebar, các page, timer.
    - Xử lý vòng đời ứng dụng (closeEvent).

    Toàn bộ code UI chi tiết được chia sang:
    - ui/theme.py                    : global stylesheet
    - ui/widgets.py                  : component UI nhỏ
    - ui/pages/camera.py             : camera + sidebar
    - ui/pages/history.py            : bảng lịch sử + detail view
    - ui/pages/search.py             : trang bộ lọc tìm kiếm
    - ui/pages/statistics.py         : trang thống kê
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---------- Cấu hình cửa sổ ----------
        self.setWindowTitle("Desktop App (Giữ xe)")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        # ---------- Áp dụng theme chung ----------
        apply_global_theme(self)

        # ---------- Khởi tạo core: model, DB, statistics ----------
        self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
        if not self.models.ok:
            QMessageBox.warning(
                self,
                "Model error",
                f"Không load được model:\n{self.models.err}",
            )

        # DB & Statistics service
        self.db = DB(CONN_STR) if USE_SQL else DB("")
        self.stats_service = ParkingStatistics() if USE_SQL else None
        if self.stats_service:
            self.stats_service.db = self.db

        # ---------- Trạng thái chung cho Statistics ----------
        self._stats_last_reload = 0.0
        self.statistics_view = None  # sẽ được build trong build_statistics_page

        # ---------- Trạng thái camera ----------
        self.cam1_worker: VideoWorker | None = None
        self.cam2_worker: VideoWorker | None = None

        # Hướng làn xe
        self.lane1_dir = "VÀO"
        self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True
        self.two_way_toggle = True

        # OCR mode
        self.current_ocr_mode = "yolo"

        # ---------- Lịch sử / History ----------
        self.history_df = pd.DataFrame()
        self.current_filter_start = None
        self.current_filter_end = None
        self.current_filter_status = None
        self.current_filter_plate = None
        self.history_worker: HistoryLoaderWorker | None = None
        self._hist_last_reload = 0.0

        # ---------- Logo mặc định ----------
        # (qpix_logo được định nghĩa trong CameraPageMixin)
        self._logo_pm: QPixmap | None = None

        # ---------- Âm thanh ----------
        # (được sử dụng trong CameraPageMixin)
        self.sound_in = None
        self.sound_out = None
        self._init_sounds()

        # ---------- Xây UI ----------
        self._build_ui()

        # Sau khi UI camera đã build xong, khởi tạo logo
        self._logo_pm = self.qpix_logo()
        self.show_logo(1)
        self.show_logo(2)

        # ---------- Timer tự động refresh history & statistics ----------
        self.hist_timer = QTimer(self)
        self.hist_timer.timeout.connect(self.on_history_signal_refresh)
        self.hist_timer.start(5000)

    # ======================================================================
    #  AUDIO INIT
    # ======================================================================

    def _init_sounds(self) -> None:
        """
        Khởi tạo QSoundEffect cho xe vào / xe ra.

        Được gọi trong __init__, trước khi các page dùng.
        Thực tế xử lý play sound nằm trong CameraPageMixin (on_play_sound).
        """
        from PySide6.QtMultimedia import QSoundEffect
        from PySide6.QtCore import QUrl

        self.sound_in = QSoundEffect(self)
        sound_in_abs = os.path.abspath(SOUND_IN_PATH)
        if os.path.exists(sound_in_abs):
            self.sound_in.setSource(QUrl.fromLocalFile(sound_in_abs))
        else:
            print(f"Lỗi: Không tìm thấy file âm thanh: {sound_in_abs}")

        self.sound_out = QSoundEffect(self)
        sound_out_abs = os.path.abspath(SOUND_OUT_PATH)
        if os.path.exists(sound_out_abs):
            self.sound_out.setSource(QUrl.fromLocalFile(sound_out_abs))
        else:
            print(f"Lỗi: Không tìm thấy file âm thanh: {sound_out_abs}")

    # ======================================================================
    #  BUILD UI
    # ======================================================================

    def _build_ui(self) -> None:
        """
        Ghép tất cả phần UI lại:

        - Tạo central widget + layout root.
        - Nhờ CameraPageMixin build sidebar + main_view (camera).
        - Tạo QStackedWidget chứa:
            index 0: main_view (camera)
            index 1: history_view
            index 2: detail_view
            index 3: search_filter_view
            index 4: statistics_view (nếu có)
        - Gắn cross-page signals (history, search, statistics).
        """
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Style chung cho các nút (button) – truyền cho các page nếu cần
        common_btn_style = (
            "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;"
        )

        # ---------------------------------------------------------
        # 1) Sidebar + Main Camera View (camera page)
        # ---------------------------------------------------------
        # CameraPageMixin phải cung cấp:
        #   build_camera_page(common_btn_style) -> (sidebar_scroll, main_view)
        sidebar_scroll, self.main_view = self.build_camera_page(common_btn_style)

        # Sidebar nằm bên trái, không giãn (stretch 0)
        root.addWidget(sidebar_scroll, 0)

        # ---------------------------------------------------------
        # 2) Stacked Widget cho phần bên phải
        # ---------------------------------------------------------
        self.stacked = QStackedWidget()
        self.stacked.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # index 0: Main camera view
        self.stacked.addWidget(self.main_view)  # index 0

        # ---------------------------------------------------------
        # 3) History + Detail pages
        # ---------------------------------------------------------
        # HistoryPageMixin phải cung cấp:
        #   build_history_pages(common_btn_style) -> (history_view, detail_view)
        self.history_view, self.detail_view = self.build_history_pages(common_btn_style)
        self.stacked.addWidget(self.history_view)  # index 1
        self.stacked.addWidget(self.detail_view)   # index 2

        # ---------------------------------------------------------
        # 4) Search filter page
        # ---------------------------------------------------------
        # SearchPageMixin phải cung cấp:
        #   build_search_page(common_btn_style) -> search_filter_view
        self.search_filter_view = self.build_search_page(common_btn_style)
        self.stacked.addWidget(self.search_filter_view)  # index 3

        # ---------------------------------------------------------
        # 5) Statistics page
        # ---------------------------------------------------------
        # StatisticsPageMixin phải cung cấp:
        #   build_statistics_page(common_btn_style) -> statistics_view | None
        self.statistics_view = self.build_statistics_page(common_btn_style)
        if self.statistics_view is not None:
            self.stacked.addWidget(self.statistics_view)  # index 4

        self.stacked.setCurrentIndex(0)

        # ---------------------------------------------------------
        # 6) Đặt stacked vào right_container
        # ---------------------------------------------------------
        right_container = QVBoxLayout()
        right_container.setContentsMargins(0, 0, 0, 0)
        right_container.setSpacing(0)
        right_container.addWidget(self.stacked, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_container)
        right_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        # Right widget chiếm toàn bộ phần còn lại (stretch 1)
        root.addWidget(right_widget, 1)

        # ---------------------------------------------------------
        # 7) Cập nhật title camera (theo làn) sau khi UI xong
        # ---------------------------------------------------------
        # Hàm này được định nghĩa trong CameraPageMixin
        self.update_titles_and_modes()

        # ---------------------------------------------------------
        # 8) Gắn các signal liên quan tới nhiều page
        # ---------------------------------------------------------
        self._connect_cross_page_signals()

    # ======================================================================
    #  CROSS PAGE SIGNALS
    # ======================================================================

    def _connect_cross_page_signals(self) -> None:
        """
        Gắn các signal giữa sidebar và các trang:

        - Các nút lịch sử (show/hide, export, delete, search).
        - Nút quay lại từ trang detail và trang search filter.
        - Nút thống kê và các nút trong trang thống kê.
        """

        # ---------------- BẢNG LỊCH SỬ ----------------
        # Các thuộc tính này được tạo trong build_camera_page / build_history_pages / build_search_page
        if hasattr(self, "btn_show_history"):
            self.btn_show_history.clicked.connect(self.on_show_all_history_clicked)

        if hasattr(self, "btn_hide_history"):
            self.btn_hide_history.clicked.connect(self.show_main_view)

        if hasattr(self, "btn_export_hist"):
            self.btn_export_hist.clicked.connect(self.on_export_excel)

        if hasattr(self, "btn_delete_hist"):
            self.btn_delete_hist.clicked.connect(self.on_delete_history)

        if hasattr(self, "btn_search_hist"):
            self.btn_search_hist.clicked.connect(self.on_search_history_clicked)

        if hasattr(self, "btn_back_to_history"):
            self.btn_back_to_history.clicked.connect(self.show_history_view_only)

        # ---------------- TRANG TÌM KIẾM ----------------
        if hasattr(self, "sfv_btn_back"):
            self.sfv_btn_back.clicked.connect(self.show_history_view_only)

        if hasattr(self, "sfv_btn_search"):
            self.sfv_btn_search.clicked.connect(self.on_run_search_from_page)

        # ---------------- TRANG THỐNG KÊ ----------------
        if hasattr(self, "btn_show_statistics"):
            self.btn_show_statistics.clicked.connect(self.on_show_statistics_clicked)

        if getattr(self, "btn_stats_back", None):
            self.btn_stats_back.clicked.connect(self.show_main_view)

        if getattr(self, "btn_stats_refresh", None):
            self.btn_stats_refresh.clicked.connect(self.on_refresh_statistics_clicked)

        if getattr(self, "btn_stats_export", None):
            self.btn_stats_export.clicked.connect(self.on_export_statistics_report)

        if getattr(self, "stats_range_combo", None):
            self.stats_range_combo.currentIndexChanged.connect(
                self.on_stats_range_changed
            )

    # ======================================================================
    #  VIEW SWITCH HELPERS
    #  (có thể cũng được override / dùng lại trong HistoryPageMixin)
    # ======================================================================

    def show_main_view(self) -> None:
        """
        Chuyển về trang chính (camera).

        Hàm này được gọi từ:
        - Nút "Tắt bảng lịch sử"
        - Nút "Quay lại" trong trang thống kê
        """
        if hasattr(self, "stacked"):
            self.stacked.setCurrentIndex(0)

        # Cập nhật nút history nếu tồn tại
        if hasattr(self, "btn_hide_history"):
            self.btn_hide_history.hide()
        if hasattr(self, "btn_show_history"):
            self.btn_show_history.show()

    # ======================================================================
    #  CLOSE EVENT
    # ======================================================================

    def closeEvent(self, event) -> None:
        """
        Khi đóng cửa sổ chính:
        - Dừng camera worker (nếu đang chạy).
        - Gọi closeEvent gốc của QMainWindow.
        """
        try:
            if hasattr(self, "stop_cam_generic"):
                # Được định nghĩa trong CameraPageMixin
                self.stop_cam_generic(1)
                self.stop_cam_generic(2)
        except Exception:
            traceback.print_exc()

        super().closeEvent(event)
