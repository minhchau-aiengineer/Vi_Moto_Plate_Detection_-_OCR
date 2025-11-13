# ui/pages/camera.py
"""
CameraPageMixin

Chịu trách nhiệm:
- Xây sidebar bên trái (camera control, lane control, OCR, info vào/ra,
  nút lịch sử, nút thống kê).
- Xây main_view (2 camera, 2 ảnh nhỏ, group info chi tiết).
- Điều khiển camera (start/stop), cập nhật hướng làn.
- Nhận frame/scene/roi/info từ VideoWorker và hiển thị.
- Quản lý OCR mode, âm thanh in/out.

YÊU CẦU MainWindow (class kế thừa mixin này) có:
- self.models          : Models(...)
- self.db              : DB instance
- self.sound_in        : QSoundEffect (đã init trong MainWindow)
- self.sound_out       : QSoundEffect
- self.current_ocr_mode: str ("yolo" hoặc "gemini")
- self.lane1_dir, self.lane2_dir, self.one_way_toggle_vao, self.two_way_toggle
- self._logo_pm        : QPixmap (sau khi _build_ui xong, MainWindow set bằng self.qpix_logo())
- self.on_history_signal_refresh() : slot (định nghĩa ở history mixin)
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget,
    QScrollArea,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QPushButton,
    QRadioButton,
    QGridLayout,
    QLineEdit,
    QSizePolicy,
    QTableWidget,
    QHeaderView,
    QMessageBox,
)

from ...config.config import API_MAP, LOGO_PATH, USE_SQL
from ...models.models import GEMINI_READY
from ...workers.workers import VideoWorker
from ...utils.utils import bgr_to_qimage, letterbox
from ..theme import normalize_button, apply_button_style
from ..widgets import make_card


class CameraPageMixin:
    """
    Mixin cung cấp UI + logic cho phần CAMERA + sidebar.

    Được sử dụng bởi MainWindow.
    """

    # ======================================================================
    #  BUILD CAMERA PAGE
    # ======================================================================

    def build_camera_page(self, common_btn_style: str) -> tuple[QScrollArea, QWidget]:
        """
        Xây dựng:
        - Sidebar (QScrollArea -> QWidget#SideBar) chứa các group:
            + CAMERA CONTROL
            + ĐIỀU KHIỂN LÀN
            + OCR MODEL
            + THÔNG TIN XE VÀO
            + THÔNG TIN XE RA
            + BẢNG LỊCH SỬ (các nút)
            + THỐNG KÊ (nút xem thống kê)
        - Main view (widget bên phải): 2 camera trên, 2 ảnh nhỏ dưới, group thông tin chi tiết.

        Trả về:
            (sidebar_scroll, main_view)
        """

        # ========================= SIDEBAR =========================
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        sidebar_scroll.setMinimumWidth(420)

        side = QWidget(objectName="SideBar")
        side.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        side.setMinimumWidth(420)

        vside = QVBoxLayout(side)
        vside.setContentsMargins(18, 10, 20, 10)
        vside.setSpacing(12)

        # --------------- CAMERA CONTROL ---------------
        gb_camctl = QGroupBox("CAMERA CONTROL")
        vl_camctl = QVBoxLayout(gb_camctl)
        vl_camctl.setSpacing(10)

        self.spin_cam1 = QSpinBox()
        self.spin_cam1.setRange(0, 9)
        self.spin_cam1.setValue(0)

        self.spin_cam2 = QSpinBox()
        self.spin_cam2.setRange(0, 9)
        self.spin_cam2.setValue(0)

        row_indices = QHBoxLayout()
        row_indices.setSpacing(10)
        row_indices.addWidget(QLabel("Index Cam 1"))
        row_indices.addWidget(self.spin_cam1, 1)
        row_indices.addWidget(QLabel("Index Cam 2"))
        row_indices.addWidget(self.spin_cam2, 1)
        vl_camctl.addLayout(row_indices)

        self.btn_start1 = QPushButton("Bật Cam 1")
        self.btn_stop1 = QPushButton("Tắt Cam 1")
        self.btn_start2 = QPushButton("Bật Cam 2")
        self.btn_stop2 = QPushButton("Tắt Cam 2")

        normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)

        # Style nút cam in/out
        start_css = (
            f"QPushButton{{ {common_btn_style} background:#d1fadf; border:1px solid #a6f4c5; }}"
            "QPushButton:hover{ background:#c3f7d6; }"
            "QPushButton:disabled{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }"
        )
        stop_css = (
            f"QPushButton{{ {common_btn_style} background:#ffe0e0; border:1px solid #ffb3b3; }}"
            "QPushButton:hover{ background:#ffd1d1; }"
            "QPushButton:disabled{ background:#fff2f2; color:#777; border-color:#ffdede; }"
        )
        apply_button_style(self.btn_start1, start_css)
        apply_button_style(self.btn_stop1, stop_css)
        apply_button_style(self.btn_start2, start_css)
        apply_button_style(self.btn_stop2, stop_css)

        # Kết nối sự kiện camera
        self.btn_start1.clicked.connect(self.start_cam1)
        self.btn_stop1.clicked.connect(self.stop_cam1)
        self.btn_start2.clicked.connect(self.start_cam2)
        self.btn_stop2.clicked.connect(self.stop_cam2)

        row_btn1 = QHBoxLayout()
        row_btn1.setSpacing(12)
        row_btn1.addWidget(self.btn_start1)
        row_btn1.addWidget(self.btn_stop1)
        vl_camctl.addLayout(row_btn1)

        row_btn2 = QHBoxLayout()
        row_btn2.setSpacing(12)
        row_btn2.addWidget(self.btn_start2)
        row_btn2.addWidget(self.btn_stop2)
        vl_camctl.addLayout(row_btn2)

        vside.addWidget(gb_camctl)

        # --------------- ĐIỀU KHIỂN LÀN ---------------
        gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
        vl_lane = QVBoxLayout(gb_lane)
        vl_lane.setSpacing(10)

        row_lane = QHBoxLayout()
        row_lane.setSpacing(12)

        self.btn_oneway = QPushButton("1 chiều")
        self.btn_twoway = QPushButton("2 chiều")
        self.btn_reset_lane = QPushButton("Reset làn")

        normalize_button(self.btn_oneway, self.btn_twoway, self.btn_reset_lane)

        one_two_css = (
            f"QPushButton{{ {common_btn_style} background:#dbeafe; border:1px solid #bfdbfe; }}"
            "QPushButton:hover{ background:#cfe3fd; }"
        )
        reset_css = (
            f"QPushButton{{ {common_btn_style} background:#fff3bf; border:1px solid #ffe066; }}"
            "QPushButton:hover{ background:#ffeda3; }"
        )

        apply_button_style(self.btn_oneway, one_two_css)
        apply_button_style(self.btn_twoway, one_two_css)
        apply_button_style(self.btn_reset_lane, reset_css)

        row_lane.addWidget(self.btn_oneway)
        row_lane.addWidget(self.btn_twoway)
        vl_lane.addLayout(row_lane)
        vl_lane.addWidget(self.btn_reset_lane)

        self.btn_oneway.clicked.connect(self.on_one_way_clicked)
        self.btn_twoway.clicked.connect(self.on_two_way_clicked)
        self.btn_reset_lane.clicked.connect(self.on_reset_lanes)

        vside.addWidget(gb_lane)

        # --------------- OCR MODEL ---------------
        gb_ocr = QGroupBox("OCR MODEL")
        vb_ocr = QVBoxLayout(gb_ocr)

        self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)")
        self.rb_yolo.setChecked(True)
        self.rb_gem = QRadioButton("Dùng Gemini AI")

        vb_ocr.addWidget(self.rb_yolo)
        vb_ocr.addWidget(self.rb_gem)

        self.rb_yolo.toggled.connect(self.on_ocr_mode_changed)
        self.rb_gem.toggled.connect(self.on_ocr_mode_changed)

        if not GEMINI_READY:
            self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY")

        vside.addWidget(gb_ocr)

        # --------------- THÔNG TIN XE VÀO ---------------
        gb_in = QGroupBox("THÔNG TIN XE VÀO")
        gl_in = QGridLayout(gb_in)

        self.ed_date_in = QLineEdit()
        self.ed_time_in = QLineEdit()
        self.ed_plate_in = QLineEdit()
        self.ed_plate_in.setStyleSheet(
            "color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;"
        )

        gl_in.addWidget(QLabel("Ngày vào:"), 0, 0)
        gl_in.addWidget(self.ed_date_in, 0, 1)
        gl_in.addWidget(QLabel("Giờ vào:"), 1, 0)
        gl_in.addWidget(self.ed_time_in, 1, 1)
        gl_in.addWidget(QLabel("Biển số vào:"), 2, 0)
        gl_in.addWidget(self.ed_plate_in, 2, 1)

        vside.addWidget(gb_in)

        # --------------- THÔNG TIN XE RA ---------------
        gb_out = QGroupBox("THÔNG TIN XE RA")
        gl_out = QGridLayout(gb_out)

        self.ed_date_out = QLineEdit()
        self.ed_time_out = QLineEdit()
        self.ed_plate_out = QLineEdit()
        self.ed_plate_out.setStyleSheet(
            "color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;"
        )

        gl_out.addWidget(QLabel("Ngày ra:"), 0, 0)
        gl_out.addWidget(self.ed_date_out, 0, 1)
        gl_out.addWidget(QLabel("Giờ ra:"), 1, 0)
        gl_out.addWidget(self.ed_time_out, 1, 1)
        gl_out.addWidget(QLabel("Biển số ra:"), 2, 0)
        gl_out.addWidget(self.ed_plate_out, 2, 1)

        vside.addWidget(gb_out)

        # --------------- NHÓM NÚT LỊCH SỬ ---------------
        gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
        v_hist_btns = QVBoxLayout(gb_hist_btns)

        self.btn_show_history = QPushButton("Xem bảng lịch sử")
        self.btn_export_hist = QPushButton("Export Excel")
        self.btn_delete_hist = QPushButton("Xóa bảng")
        self.btn_search_hist = QPushButton("Tìm kiếm")
        self.btn_hide_history = QPushButton("Tắt bảng lịch sử")
        self.btn_hide_history.hide()

        normalize_button(
            self.btn_show_history,
            self.btn_export_hist,
            self.btn_delete_hist,
            self.btn_search_hist,
            self.btn_hide_history,
        )

        show_hist_css = (
            f"QPushButton{{ {common_btn_style} background:#E6F4EA; border:1px solid #cde9d6; }}"
            "QPushButton:hover{ background:#d9efe0; }"
        )
        hide_hist_css = (
            f"QPushButton{{ {common_btn_style} background:#fff3bf; border:1px solid #f5c6c2; }}"
            "QPushButton:hover{ background:#ffeda3; }"
        )
        export_css = (
            f"QPushButton{{ {common_btn_style} background:#e0ecff; border:1px solid #c7dcff; }}"
            "QPushButton:hover{ background:#d4e5ff; }"
        )
        delete_css = (
            f"QPushButton{{ {common_btn_style} background:#ffe0e0; border:1px solid #ffb3b3; }}"
            "QPushButton:hover{ background:#ffd1d1; }"
        )
        search_css = export_css  # giống nút export

        apply_button_style(self.btn_show_history, show_hist_css)
        apply_button_style(self.btn_hide_history, hide_hist_css)
        apply_button_style(self.btn_export_hist, export_css)
        apply_button_style(self.btn_delete_hist, delete_css)
        apply_button_style(self.btn_search_hist, search_css)

        row_cmd = QHBoxLayout()
        row_cmd.addWidget(self.btn_search_hist)
        row_cmd.addWidget(self.btn_export_hist)
        row_cmd.addWidget(self.btn_delete_hist)

        v_hist_btns.addWidget(self.btn_show_history)
        v_hist_btns.addLayout(row_cmd)
        v_hist_btns.addWidget(self.btn_hide_history)

        vside.addWidget(gb_hist_btns)

        # --------------- NHÓM NÚT THỐNG KÊ ---------------
        gb_stats = QGroupBox("THỐNG KÊ")
        v_stats = QVBoxLayout(gb_stats)

        self.btn_show_statistics = QPushButton("Xem thống kê")
        normalize_button(self.btn_show_statistics)

        stats_css = (
            f"QPushButton{{ {common_btn_style} background:#ede9fe; border:1px solid #c4b5fd; }}"
            "QPushButton:hover{ background:#ddd6fe; }"
        )
        apply_button_style(self.btn_show_statistics, stats_css)

        v_stats.addWidget(self.btn_show_statistics)

        if not USE_SQL:
            self.btn_show_statistics.setEnabled(False)
            self.btn_show_statistics.setToolTip("Chức năng thống kê cần kết nối cơ sở dữ liệu")

        v_stats.addStretch(1)
        vside.addWidget(gb_stats)
        vside.addStretch(1)

        sidebar_scroll.setWidget(side)

        # ========================= MAIN VIEW (CAMERA PAGE) =========================
        self.main_view = QWidget()
        main_layout = QVBoxLayout(self.main_view)

        # ---------- Hàng trên: 2 camera ----------
        top = QHBoxLayout()

        self.lbl_cam1 = QLabel()
        self.lbl_cam2 = QLabel()

        for lbl in (self.lbl_cam1, self.lbl_cam2):
            lbl.setScaledContents(False)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
            lbl.setMinimumHeight(220)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        cam1_card, self.cam1_title = make_card("Cam 1 (Vào)", self.lbl_cam1)
        cam2_card, self.cam2_title = make_card("Cam 2 (Vào)", self.lbl_cam2)

        top.addWidget(cam1_card, 1)
        top.addWidget(cam2_card, 1)
        main_layout.addLayout(top)

        # ---------- Hàng dưới: 2 ảnh nhỏ (scene + roi) ----------
        bottom = QHBoxLayout()
        self.lbl_scene = QLabel()
        self.lbl_roi = QLabel()

        for lbl in (self.lbl_scene, self.lbl_roi):
            lbl.setScaledContents(False)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
            lbl.setMinimumHeight(220)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scene_card, _ = make_card("Image_BOX", self.lbl_scene)
        roi_card, _ = make_card("ROI_Plate", self.lbl_roi)

        bottom.addWidget(scene_card, 1)
        bottom.addWidget(roi_card, 1)
        main_layout.addLayout(bottom)

        # ---------- Group thông tin chi tiết ----------
        self.info_group = QGroupBox("Thông tin chi tiết")
        info_layout = QGridLayout(self.info_group)

        self.txt_date_in = QLabel("--/--/----")
        self.txt_time_in = QLabel("--:--:--")
        self.txt_plate_in = QLabel("---")
        self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")

        self.txt_date_out = QLabel("--/--/----")
        self.txt_time_out = QLabel("--:--:--")
        self.txt_plate_out = QLabel("---")
        self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")

        self.txt_match = QLineEdit()
        self.txt_match.setReadOnly(True)
        self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")

        r = 0
        info_layout.addWidget(QLabel("Ngày vào:"), r, 0)
        info_layout.addWidget(self.txt_date_in, r, 1)
        info_layout.addWidget(QLabel("Giờ vào:"), r, 2)
        info_layout.addWidget(self.txt_time_in, r, 3)
        info_layout.addWidget(QLabel("Biển số vào:"), r, 4)
        info_layout.addWidget(self.txt_plate_in, r, 5)
        r += 1

        info_layout.addWidget(QLabel("Ngày ra:"), r, 0)
        info_layout.addWidget(self.txt_date_out, r, 1)
        info_layout.addWidget(QLabel("Giờ ra:"), r, 2)
        info_layout.addWidget(self.txt_time_out, r, 3)
        info_layout.addWidget(QLabel("Biển số ra:"), r, 4)
        info_layout.addWidget(self.txt_plate_out, r, 5)
        r += 1

        info_layout.addWidget(QLabel("So khớp biển số:"), r, 0)
        info_layout.addWidget(self.txt_match, r, 1, 1, 2)

        main_layout.addWidget(self.info_group)

        return sidebar_scroll, self.main_view

    # ======================================================================
    #  CAMERA & LANE LOGIC
    # ======================================================================

    def update_titles_and_modes(self) -> None:
        """
        Cập nhật tiêu đề card camera và mode của worker theo hướng làn.
        """
        self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir == 'VÀO' else 'Ra'})")
        self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir == 'VÀO' else 'Ra'})")

        if self.cam1_worker:
            self.cam1_worker.set_mode("in" if self.lane1_dir == "VÀO" else "out")
        if self.cam2_worker:
            self.cam2_worker.set_mode("in" if self.lane2_dir == "VÀO" else "out")

    @Slot()
    def on_reset_lanes(self) -> None:
        self.lane1_dir = "VÀO"
        self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True
        self.two_way_toggle = True
        self.update_titles_and_modes()
        self.show_logo(1)
        self.show_logo(2)

    @Slot()
    def on_one_way_clicked(self) -> None:
        """
        Chế độ 1 chiều: cả 2 làn đều VÀO hoặc RA.
        """
        if self.one_way_toggle_vao:
            self.lane1_dir = "VÀO"
            self.lane2_dir = "VÀO"
        else:
            self.lane1_dir = "RA"
            self.lane2_dir = "RA"
        self.one_way_toggle_vao = not self.one_way_toggle_vao
        self.update_titles_and_modes()

    @Slot()
    def on_two_way_clicked(self) -> None:
        """
        Chế độ 2 chiều: 1 làn vào, 1 làn ra (đảo chiều mỗi lần nhấn).
        """
        if self.two_way_toggle:
            self.lane1_dir = "VÀO"
            self.lane2_dir = "RA"
        else:
            self.lane1_dir = "RA"
            self.lane2_dir = "VÀO"
        self.two_way_toggle = not self.two_way_toggle
        self.update_titles_and_modes()

    # ======================================================================
    #  MATCH + SOUND + OCR MODE
    # ======================================================================

    @Slot(str)
    def update_match_status(self, status: str) -> None:
        """
        Cập nhật text và màu sắc cho field so khớp biển số (txt_match).
        """
        display_status = status.replace("-", " ").title()
        self.txt_match.setText(display_status)

        if "KHOP-BIEN-SO" in display_status:
            self.txt_match.setStyleSheet("color: #007700; font-weight: 700;")
        elif "KHONG-KHOP-BIEN-SO" in display_status:
            self.txt_match.setStyleSheet("color: #ff0000; font-weight: 700;")
        else:
            self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")

    @Slot(str)
    def on_play_sound(self, mode: str) -> None:
        """
        Phát âm thanh tương ứng với in/out.
        """
        if mode == "in" and self.sound_in:
            self.sound_in.play()
        elif mode == "out" and self.sound_out:
            self.sound_out.play()

    @Slot()
    def on_ocr_mode_changed(self) -> None:
        """
        Thay đổi mode OCR (yolo / gemini).
        """
        self.current_ocr_mode = (
            "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
        )
        if self.rb_gem.isChecked() and not GEMINI_READY:
            QMessageBox.information(
                self,
                "Gemini",
                "Chưa cấu hình GEMINI_API_KEY. Sẽ dùng YOLO OCR.",
            )
            self.rb_yolo.setChecked(True)
            self.current_ocr_mode = "yolo"

        if self.cam1_worker:
            self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
        if self.cam2_worker:
            self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

    # ======================================================================
    #  HIỂN THỊ ẢNH & LOGO
    # ======================================================================

    def _set_centered_pixmap(self, lbl: QLabel, src) -> None:
        """
        Hiển thị QPixmap/QImage/ndarray lên QLabel với tỉ lệ đúng, canh giữa.
        """
        pm = None
        if isinstance(src, np.ndarray):
            pm = QPixmap.fromImage(bgr_to_qimage(src))
        elif isinstance(src, QPixmap):
            pm = src
        else:
            # giả sử là QImage
            try:
                pm = QPixmap.fromImage(src)
            except Exception:
                pm = None

        if pm is None or pm.isNull():
            lbl.clear()
            return

        rect = lbl.contentsRect()
        avail = rect.size()
        dpr = getattr(lbl, "devicePixelRatioF", lambda: 1.0)()

        target_w = max(1, int(avail.width() * dpr))
        target_h = max(1, int(avail.height() * dpr))

        scaled = pm.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        if hasattr(scaled, "setDevicePixelRatio"):
            scaled.setDevicePixelRatio(dpr)

        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setPixmap(scaled)

    def qpix_logo(self) -> QPixmap:
        """
        Tạo QPixmap logo mặc định. Nếu không có LOGO_PATH thì tạo ảnh rỗng từ utils.letterbox.
        """
        if os.path.exists(LOGO_PATH):
            return QPixmap(LOGO_PATH)
        return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))

    def show_logo(self, which: int) -> None:
        """
        Hiển thị logo mặc định lên cam 1 hoặc cam 2.
        """
        pm = self._logo_pm if hasattr(self, "_logo_pm") and self._logo_pm else self.qpix_logo()
        if which == 1:
            self._set_centered_pixmap(self.lbl_cam1, pm)
        else:
            self._set_centered_pixmap(self.lbl_cam2, pm)

    # ======================================================================
    #  SLOT NHẬN FRAME / ẢNH TỪ WORKER
    # ======================================================================

    @Slot(np.ndarray, str)
    def on_frame(self, frame_bgr, title: str) -> None:
        sender = self.sender()
        if sender is self.cam1_worker:
            self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
        elif sender is self.cam2_worker:
            self._set_centered_pixmap(self.lbl_cam2, frame_bgr)

    def _get_valid_image_path_internal(self, path_from_db: str | None) -> str | None:
        """
        Kiểm tra đường dẫn ảnh có tồn tại hay không, thử cả path tuyệt đối.
        Dùng cho cả camera (scene/roi) và history detail.
        """
        if not path_from_db:
            return None
        if os.path.exists(path_from_db):
            return path_from_db

        maybe_path = os.path.abspath(path_from_db)
        if os.path.exists(maybe_path):
            return maybe_path

        print(
            f"Cảnh báo: Không tìm thấy ảnh tại '{path_from_db}' "
            f"hoặc '{maybe_path}'"
        )
        return None

    @Slot(str)
    def on_scene(self, path: str) -> None:
        valid_path = self._get_valid_image_path_internal(path)
        if valid_path:
            bgr = cv2.imread(valid_path)
            self._set_centered_pixmap(self.lbl_scene, bgr)
        else:
            self._set_centered_pixmap(self.lbl_scene, self.qpix_logo())

    @Slot(str, str)
    def on_roi(self, roi_path: str, mode: str) -> None:
        valid_path = self._get_valid_image_path_internal(roi_path)
        if valid_path:
            bgr = cv2.imread(valid_path)
            self._set_centered_pixmap(self.lbl_roi, bgr)
        else:
            self._set_centered_pixmap(self.lbl_roi, self.qpix_logo())

    @Slot(dict)
    def on_info(self, info: dict) -> None:
        if "date_in" in info:
            self.txt_date_in.setText(info["date_in"])
            self.ed_date_in.setText(info["date_in"])
        if "time_in" in info:
            self.txt_time_in.setText(info["time_in"])
            self.ed_time_in.setText(info["time_in"])
        if "plate_text_in" in info:
            self.txt_plate_in.setText(info["plate_text_in"])
            self.ed_plate_in.setText(info["plate_text_in"])
        if "date_out" in info:
            self.txt_date_out.setText(info["date_out"])
            self.ed_date_out.setText(info["date_out"])
        if "time_out" in info:
            self.txt_time_out.setText(info["time_out"])
            self.ed_time_out.setText(info["time_out"])
        if "plate_text_out" in info:
            self.txt_plate_out.setText(info["plate_text_out"])
            self.ed_plate_out.setText(info["plate_text_out"])

    @Slot(str)
    def on_match(self, txt: str) -> None:
        self.txt_match.setText(txt.upper())

    # ======================================================================
    #  KẾT NỐI WORKER
    # ======================================================================

    def _connect_worker(self, w: VideoWorker) -> None:
        w.frameSignal.connect(self.on_frame)
        w.sceneSignal.connect(self.on_scene)
        w.roiSignal.connect(self.on_roi)
        w.infoSignal.connect(self.on_info)
        w.matchSignal.connect(self.on_match)
        # histSignal dùng để gợi ý refresh history/statistics
        w.histSignal.connect(self.on_history_signal_refresh)
        w.playSoundSignal.connect(self.on_play_sound)

    # ======================================================================
    #  START / STOP CAMERA
    # ======================================================================

    def start_cam_generic(self, which: int) -> None:
        """
        Khởi động camera cho kênh 1 hoặc 2.
        """
        if not self.models.ok:
            QMessageBox.warning(
                self,
                "Model error",
                f"Không load được model:\n{self.models.err}",
            )
            return

        # Nếu worker đang chạy thì không làm gì
        if which == 1 and self.cam1_worker and self.cam1_worker.isRunning():
            return
        if which == 2 and self.cam2_worker and self.cam2_worker.isRunning():
            return

        ocr_mode = self.current_ocr_mode
        default_api = API_MAP["DSHOW(Windows)"]

        if which == 1:
            idx = int(self.spin_cam1.value())
            mode = "in" if self.lane1_dir == "VÀO" else "out"
            title = f"Cam 1 ({'Vào' if mode == 'in' else 'Ra'})"
            self.cam1_worker = VideoWorker(
                idx, default_api, mode, self.models, self.db, 1.2,
                ocr_mode=ocr_mode, title=title
            )
            self._connect_worker(self.cam1_worker)
            self.cam1_worker.start()
        else:
            idx = int(self.spin_cam2.value())
            mode = "in" if self.lane2_dir == "VÀO" else "out"
            title = f"Cam 2 ({'Vào' if mode == 'in' else 'Ra'})"
            self.cam2_worker = VideoWorker(
                idx, default_api, mode, self.models, self.db, 1.2,
                ocr_mode=ocr_mode, title=title
            )
            self._connect_worker(self.cam2_worker)
            self.cam2_worker.start()

    def stop_cam_generic(self, which: int) -> None:
        """
        Dừng camera worker 1 hoặc 2.
        """
        worker = self.cam1_worker if which == 1 else self.cam2_worker
        if worker and worker.isRunning():
            worker.stop()
            worker.wait(1000)

        if which == 1:
            self.cam1_worker = None
            self.show_logo(1)
        else:
            self.cam2_worker = None
            self.show_logo(2)

    def start_cam1(self) -> None:
        self.start_cam_generic(1)

    def stop_cam1(self) -> None:
        self.stop_cam_generic(1)

    def start_cam2(self) -> None:
        self.start_cam_generic(2)

    def stop_cam2(self) -> None:
        self.stop_cam_generic(2)
