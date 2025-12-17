from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot, QDateTime
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QMessageBox,
    QSizePolicy,
    QGroupBox,
    QScrollArea,
    QLineEdit,
    QGridLayout,
    QRadioButton,
    QSpinBox,
)

from ....config.config import (
    DETECT_MODEL_PATH,
    OCR_MODEL_PATH,
    USE_SQL,
    CONN_STR,
    API_MAP,
    LOGO_PATH,
)
from ....models.models import Models, GEMINI_READY
from ....database.database import DB
from ....utils.utils import letterbox, bgr_to_qimage
from ....workers.workers import VideoWorker





# ===== Manager Camera Page ======
class ManagerCameraPage(QWidget):
    """
    Giao diện Camera dành cho QUẢN LÝ:
      - Chỉnh index cam 1/2, OCR model.
      - Điều khiển làn, thông tin xe vào/ra.
      - Nhóm nút Lịch sử (xem bảng, tìm kiếm, export, xoá, ẩn).
      - Nút xem Thống kê.
      - 4 khung hình (2 cam, Image_BOX, ROI_Plate) – đều có logo mặc định lúc đầu.
    """





    # === Khởi tạo Manager Camera Page ===
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # BACKEND
        self.models: Optional[Models] = None
        self.db: Optional[DB] = None
        self._load_backend()

        # WORKER / LÀN
        self.cam1_worker: Optional[VideoWorker] = None
        self.cam2_worker: Optional[VideoWorker] = None
        # Cho phép Cam 2 "dùng chung" Cam 1 (mirror)
        self.cam2_mirror_active: bool = False

        self.lane1_dir = "VÀO"
        self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True
        self.two_way_toggle = True

        self.current_ocr_mode: str = "yolo"
        self._logo_pm: Optional[QPixmap] = None

        self._build_ui()

        # Logo cho 4 khung
        self._logo_pm = self.qpix_logo()
        self.show_logo(1)
        self.show_logo(2)
        self._set_centered_pixmap(self.lbl_scene, self._logo_pm)
        self._set_centered_pixmap(self.lbl_roi, self._logo_pm)

        if self.models is None or not self.models.ok:
            QMessageBox.warning(
                self,
                "Lỗi model",
                f"Không thể load model YOLO.\n\nChi tiết: {getattr(self.models, 'err', '')}",
            )
        if USE_SQL and (self.db is None or not self.db.ok):
            QMessageBox.warning(
                self,
                "Lỗi cơ sở dữ liệu",
                "Không kết nối được SQL Server.\n"
                "Lịch sử / thống kê vẫn chạy ở chế độ offline (không lưu DB).",
            )

    
    
    
    
    # === Load backend: Models, DB ===
    def _load_backend(self) -> None:
        self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
        if not self.models.ok:
            print("[ManagerCameraPage] Lỗi load models:", self.models.err)

        if USE_SQL:
            self.db = DB(CONN_STR)
            if not self.db.ok:
                print("[ManagerCameraPage] Lỗi kết nối DB.")
        else:
            self.db = None
            print("[ManagerCameraPage] USE_SQL = False, không dùng DB.")

    
    
    
    
    
    # === Xây dựng giao diện UI ===
    def _build_ui(self) -> None:
        self.setStyleSheet("background-color: #ffffff;")

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ========== SIDEBAR ==========
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setMinimumWidth(320)
        sidebar_scroll.setMaximumWidth(340)

        side = QWidget()
        side.setObjectName("SideBarManager")
        vside = QVBoxLayout(side)
        vside.setContentsMargins(12, 10, 12, 10)
        vside.setSpacing(10)

        side.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 14px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                margin-left: 4px;
                background: #f8fafc;
                color: #0f172a;
            }
            QLabel {
                color: #0f172a;
            }
            QLineEdit {
                height: 28px;
                border-radius: 4px;
                border: 1px solid #e2e8f0;
                padding: 2px 6px;
                background: #ffffff;
                color: #000000;
            }
            """
        )

        plate_font = QFont("Segoe UI", 14, QFont.Weight.Bold)

        # CAMERA CONTROL
        gb_camctl = QGroupBox("CAMERA CONTROL")
        vl_camctl = QVBoxLayout(gb_camctl)
        vl_camctl.setSpacing(8)

        self.spin_cam1 = QSpinBox()
        self.spin_cam1.setRange(0, 9)
        self.spin_cam1.setValue(0)

        self.spin_cam2 = QSpinBox()
        self.spin_cam2.setRange(0, 9)
        self.spin_cam2.setValue(0)

        row_idx = QHBoxLayout()
        row_idx.setSpacing(6)
        row_idx.addWidget(QLabel("Index Cam 1:"))
        row_idx.addWidget(self.spin_cam1)
        row_idx.addSpacing(6)
        row_idx.addWidget(QLabel("Cam 2:"))
        row_idx.addWidget(self.spin_cam2)
        vl_camctl.addLayout(row_idx)

        self.btn_start1 = QPushButton("Bật Cam 1")
        self.btn_stop1 = QPushButton("Tắt Cam 1")
        self.btn_start2 = QPushButton("Bật Cam 2")
        self.btn_stop2 = QPushButton("Tắt Cam 2")

        green_btn_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #22c55e;
                color: #ffffff;
                border: 1px solid #16a34a;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #16a34a;
            }
        """
        red_btn_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #ef4444;
                color: #ffffff;
                border: 1px solid #b91c1c;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #dc2626;
            }
        """
        self.btn_start1.setStyleSheet(green_btn_css)
        self.btn_start2.setStyleSheet(green_btn_css)
        self.btn_stop1.setStyleSheet(red_btn_css)
        self.btn_stop2.setStyleSheet(red_btn_css)

        row_btn1 = QHBoxLayout()
        row_btn1.setSpacing(8)
        row_btn1.addWidget(self.btn_start1)
        row_btn1.addWidget(self.btn_stop1)
        vl_camctl.addLayout(row_btn1)

        row_btn2 = QHBoxLayout()
        row_btn2.setSpacing(8)
        row_btn2.addWidget(self.btn_start2)
        row_btn2.addWidget(self.btn_stop2)
        vl_camctl.addLayout(row_btn2)

        self.btn_start1.clicked.connect(lambda: self.start_cam_generic(1))
        self.btn_stop1.clicked.connect(lambda: self.stop_cam_generic(1))
        self.btn_start2.clicked.connect(lambda: self.start_cam_generic(2))
        self.btn_stop2.clicked.connect(lambda: self.stop_cam_generic(2))

        vside.addWidget(gb_camctl)

        # ĐIỀU KHIỂN LÀN
        gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
        vl_lane = QVBoxLayout(gb_lane)
        vl_lane.setSpacing(6)

        self.btn_oneway = QPushButton("1 chiều")
        self.btn_twoway = QPushButton("2 chiều")
        self.btn_reset_lane = QPushButton("Reset làn")

        blue_btn_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #3b82f6;
                color: #ffffff;
                border: 1px solid #1d4ed8;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2563eb;
            }
        """
        yellow_btn_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #facc15;
                color: #78350f;
                border: 1px solid #eab308;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #eab308;
            }
        """
        self.btn_oneway.setStyleSheet(blue_btn_css)
        self.btn_twoway.setStyleSheet(blue_btn_css)
        self.btn_reset_lane.setStyleSheet(yellow_btn_css)

        row_lane = QHBoxLayout()
        row_lane.setSpacing(8)
        row_lane.addWidget(self.btn_oneway)
        row_lane.addWidget(self.btn_twoway)
        vl_lane.addLayout(row_lane)
        vl_lane.addWidget(self.btn_reset_lane)

        self.btn_oneway.clicked.connect(self.on_one_way_clicked)
        self.btn_twoway.clicked.connect(self.on_two_way_clicked)
        self.btn_reset_lane.clicked.connect(self.on_reset_lanes)

        vside.addWidget(gb_lane)

        # OCR MODEL
        gb_ocr = QGroupBox("OCR MODEL")
        vl_ocr = QVBoxLayout(gb_ocr)
        vl_ocr.setSpacing(4)

        self.rb_yolo = QRadioButton("Dùng YOLO OCR")
        self.rb_gem = QRadioButton("Dùng Gemini AI")

        self.rb_yolo.setChecked(True)
        if not GEMINI_READY:
            self.rb_gem.setToolTip("Chưa cấu hình GEMINI_API_KEY")

        self.rb_yolo.toggled.connect(self.on_ocr_mode_changed)
        self.rb_gem.toggled.connect(self.on_ocr_mode_changed)

        vl_ocr.addWidget(self.rb_yolo)
        vl_ocr.addWidget(self.rb_gem)
        vside.addWidget(gb_ocr)

        # THÔNG TIN XE VÀO
        gb_in = QGroupBox("THÔNG TIN XE VÀO")
        gl_in = QGridLayout(gb_in)
        gl_in.setHorizontalSpacing(8)
        gl_in.setVerticalSpacing(6)

        self.ed_date_in = QLineEdit()
        self.ed_time_in = QLineEdit()
        self.ed_plate_in = QLineEdit()
        self.ed_plate_in.setFont(plate_font)
        self.ed_plate_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_plate_in.setStyleSheet(
            "color:#ff0000; font-weight:700; height:32px; background:#fff7f7;"
        )

        # Ô hiển thị loại xe vào (Nội bộ / Vãng lai)
        self.ed_group_in = QLineEdit()
        self.ed_group_in.setReadOnly(True)
        self.ed_group_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_group_in.setStyleSheet(
            "color:#0f172a; font-weight:600; height:28px; "
            "border-radius:4px; border:1px solid #e2e8f0; background:#f9fafb;"
        )

        gl_in.addWidget(QLabel("Ngày vào:"), 0, 0)
        gl_in.addWidget(self.ed_date_in, 0, 1)
        gl_in.addWidget(QLabel("Giờ vào:"), 1, 0)
        gl_in.addWidget(self.ed_time_in, 1, 1)
        gl_in.addWidget(QLabel("Biển số vào:"), 2, 0)
        gl_in.addWidget(self.ed_plate_in, 2, 1)
        gl_in.addWidget(QLabel("Loại xe:"), 3, 0)
        gl_in.addWidget(self.ed_group_in, 3, 1)
        gl_in.setColumnStretch(1, 1)

        vside.addWidget(gb_in)

        # THÔNG TIN XE RA
        gb_out = QGroupBox("THÔNG TIN XE RA")
        gl_out = QGridLayout(gb_out)
        gl_out.setHorizontalSpacing(8)
        gl_out.setVerticalSpacing(6)

        self.ed_date_out = QLineEdit()
        self.ed_time_out = QLineEdit()
        self.ed_plate_out = QLineEdit()
        self.ed_plate_out.setFont(plate_font)
        self.ed_plate_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_plate_out.setStyleSheet(
            "color:#ff0000; font-weight:700; height:32px; background:#fff7f7;"
        )

        # Ô hiển thị loại xe ra (Nội bộ / Vãng lai)
        self.ed_group_out = QLineEdit()
        self.ed_group_out.setReadOnly(True)
        self.ed_group_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_group_out.setStyleSheet(
            "color:#0f172a; font-weight:600; height:28px; "
            "border-radius:4px; border:1px solid #e2e8f0; background:#f9fafb;"
        )

        gl_out.addWidget(QLabel("Ngày ra:"), 0, 0)
        gl_out.addWidget(self.ed_date_out, 0, 1)
        gl_out.addWidget(QLabel("Giờ ra:"), 1, 0)
        gl_out.addWidget(self.ed_time_out, 1, 1)
        gl_out.addWidget(QLabel("Biển số ra:"), 2, 0)
        gl_out.addWidget(self.ed_plate_out, 2, 1)
        gl_out.addWidget(QLabel("Loại xe:"), 3, 0)
        gl_out.addWidget(self.ed_group_out, 3, 1)
        gl_out.setColumnStretch(1, 1)

        vside.addWidget(gb_out)

        # BẢNG LỊCH SỬ
        gb_hist = QGroupBox("BẢNG LỊCH SỬ")
        vl_hist = QVBoxLayout(gb_hist)
        vl_hist.setSpacing(6)

        self.btn_show_history = QPushButton("Xem bảng lịch sử")
        self.btn_export_hist = QPushButton("Export Excel")
        self.btn_delete_hist = QPushButton("Xóa bảng")
        self.btn_search_hist = QPushButton("Tìm kiếm")
        self.btn_hide_history = QPushButton("Ẩn bảng lịch sử")

        for b in (
            self.btn_show_history,
            self.btn_export_hist,
            self.btn_delete_hist,
            self.btn_search_hist,
            self.btn_hide_history,
        ):
            b.setMinimumHeight(30)
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        primary_btn_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #e5e7eb;
                color: #111827;
                border: 1px solid #cbd5e1;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #d4d4d8;
            }
        """
        danger_btn_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #fee2e2;
                color: #b91c1c;
                border: 1px solid #fecaca;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #fecaca;
            }
        """
        self.btn_show_history.setStyleSheet(primary_btn_css)
        self.btn_export_hist.setStyleSheet(primary_btn_css)
        self.btn_search_hist.setStyleSheet(primary_btn_css)
        self.btn_hide_history.setStyleSheet(primary_btn_css)
        self.btn_delete_hist.setStyleSheet(danger_btn_css)

        vl_hist.addWidget(self.btn_show_history)
        row_hist = QHBoxLayout()
        row_hist.setSpacing(6)
        row_hist.addWidget(self.btn_search_hist)
        row_hist.addWidget(self.btn_export_hist)
        row_hist.addWidget(self.btn_delete_hist)
        vl_hist.addLayout(row_hist)
        vl_hist.addWidget(self.btn_hide_history)

        self.btn_show_history.clicked.connect(lambda: self._switch_parent_page("history"))
        self.btn_search_hist.clicked.connect(lambda: self._switch_parent_page("search"))
        self.btn_export_hist.clicked.connect(self._show_hist_message)
        self.btn_delete_hist.clicked.connect(self._show_hist_message)
        self.btn_hide_history.clicked.connect(lambda: self._switch_parent_page("camera"))

        vside.addWidget(gb_hist)

        # THỐNG KÊ
        gb_stats = QGroupBox("THỐNG KÊ")
        vl_stats = QVBoxLayout(gb_stats)

        self.btn_show_stats = QPushButton("Xem thống kê")
        self.btn_show_stats.setMinimumHeight(30)
        self.btn_show_stats.setCursor(Qt.CursorShape.PointingHandCursor)
        stats_css = """
            QPushButton {
                height: 30px;
                border-radius: 6px;
                background: #ede9fe;
                color: #3730a3;
                border: 1px solid #c4b5fd;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #ddd6fe;
            }
        """
        self.btn_show_stats.setStyleSheet(stats_css)

        if USE_SQL:
            self.btn_show_stats.clicked.connect(
                lambda: self._switch_parent_page("statistics")
            )
        else:
            self.btn_show_stats.setEnabled(False)
            self.btn_show_stats.setToolTip(
                "Cần cấu hình kết nối SQL Server để xem thống kê."
            )

        vl_stats.addWidget(self.btn_show_stats)
        vside.addWidget(gb_stats)
        vside.addStretch(1)

        sidebar_scroll.setWidget(side)
        root.addWidget(sidebar_scroll)

        # ========== MAIN VIEW ==========
        right = QWidget()
        right.setStyleSheet("background-color: #ffffff;")
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # trên: 2 cam
        top = QHBoxLayout()
        top.setSpacing(8)

        self.lbl_cam1 = QLabel()
        self.lbl_cam2 = QLabel()
        for lbl in (self.lbl_cam1, self.lbl_cam2):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "background:#ffffff; border-radius:16px; border:1px solid #e2e8f0;"
            )
            lbl.setMinimumHeight(260)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        cam1_box = QVBoxLayout()
        self.cam1_title = QLabel("Cam 1 (Vào)")
        self.cam1_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam1_title.setStyleSheet("font-weight:600;")
        cam1_box.addWidget(self.cam1_title)
        cam1_box.addWidget(self.lbl_cam1, 1)
        cam1_frame = QFrame()
        cam1_frame.setLayout(cam1_box)

        cam2_box = QVBoxLayout()
        self.cam2_title = QLabel("Cam 2 (Vào)")
        self.cam2_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam2_title.setStyleSheet("font-weight:600;")
        cam2_box.addWidget(self.cam2_title)
        cam2_box.addWidget(self.lbl_cam2, 1)
        cam2_frame = QFrame()
        cam2_frame.setLayout(cam2_box)

        top.addWidget(cam1_frame, 1)
        top.addWidget(cam2_frame, 1)
        right_layout.addLayout(top, 1)

        # dưới: scene + roi
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.lbl_scene = QLabel()
        self.lbl_roi = QLabel()
        for lbl in (self.lbl_scene, self.lbl_roi):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "background:#ffffff; border-radius:16px; border:1px solid #e2e8f0;"
            )
            lbl.setMinimumHeight(200)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        scene_box = QVBoxLayout()
        scene_title = QLabel("Image_BOX")
        scene_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scene_title.setStyleSheet("font-weight:600;")
        scene_box.addWidget(scene_title)
        scene_box.addWidget(self.lbl_scene, 1)
        scene_frame = QFrame()
        scene_frame.setLayout(scene_box)

        roi_box = QVBoxLayout()
        roi_title = QLabel("ROI_Plate")
        roi_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        roi_title.setStyleSheet("font-weight:600;")
        roi_box.addWidget(roi_title)
        roi_box.addWidget(self.lbl_roi, 1)
        roi_frame = QFrame()
        roi_frame.setLayout(roi_box)

        bottom.addWidget(scene_frame, 1)
        bottom.addWidget(roi_frame, 1)
        right_layout.addLayout(bottom, 1)

        # status match
        self.txt_match = QLineEdit()
        self.txt_match.setReadOnly(True)
        self.txt_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_match.setStyleSheet(
            "color:#1d4ed8; font-weight:700; height:28px; border-radius:6px; "
            "border:1px solid #bfdbfe; background:#e0f2fe;"
        )
        self.txt_match.setText("TRẠNG THÁI SO KHỚP BIỂN SỐ")
        right_layout.addWidget(self.txt_match)

        root.addWidget(right, 1)






    # === Thử nhiều backend để mở camera ===
    def _probe_camera_backend(self, idx: int, preferred_api: int) -> Optional[int]:
        """Thử mở camera với nhiều backend và trả về API hoạt động được.
        Thứ tự ưu tiên: preferred -> DSHOW -> MSMF -> ANY. Trả về None nếu không mở được.
        """
        tried = []
        candidates = [preferred_api, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        # Loại bỏ trùng lặp và giá trị âm
        uniq = []
        for a in candidates:
            if a is None or a in tried:
                continue
            tried.append(a)
            if a not in uniq and int(a) >= 0:
                uniq.append(int(a))

        for api in uniq:
            try:
                cap = cv2.VideoCapture(int(idx), api)
                ok = cap is not None and cap.isOpened()
            except Exception:
                ok = False
                cap = None
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
            if ok:
                return api
        return None






    # === Hiển thị logo mặc định ===
    def _switch_parent_page(self, key: str) -> None:
        parent = self.parent()
        while parent is not None and not hasattr(parent, "_switch_page"):
            parent = parent.parent()
        if parent is not None and hasattr(parent, "_switch_page"):
            parent._switch_page(key)  # type: ignore[attr-defined]






    # === Thông báo lịch sử / thống kê ===
    def _show_hist_message(self) -> None:
        QMessageBox.information(
            self,
            "Lịch sử",
            "Chức năng Export/Xóa lịch sử chi tiết được thực hiện trong tab 'Lịch sử' hoặc 'Thống kê'.",
        )

    
    
    
    
    
    # === CAM 2 MIRROR CAM 1 ===
    @Slot(np.ndarray, str)
    def _on_frame_mirror_cam2(self, frame: np.ndarray, title: str) -> None:
        if frame is None or frame.size == 0:
            return
        # Luôn đẩy frame của Cam 1 sang khung Cam 2
        self._set_centered_pixmap(self.lbl_cam2, frame)






    # === Bật / Tắt CAM 2 MIRROR CAM 1 ===
    def _enable_cam2_mirror(self) -> None:
        if not (self.cam1_worker and self.cam1_worker.isRunning()):
            return
        if self.cam2_mirror_active:
            return
        try:
            self.cam1_worker.frameSignal.connect(self._on_frame_mirror_cam2)
            self.cam2_mirror_active = True
            # Cập nhật tiêu đề cho dễ hiểu
            self.cam2_title.setText("Cam 2 (Mirror Cam 1)")
        except Exception:
            self.cam2_mirror_active = False






    # === Bật / Tắt CAM 2 MIRROR CAM 1 ===
    def _disable_cam2_mirror(self) -> None:
        if not self.cam2_mirror_active:
            return
        try:
            if self.cam1_worker:
                self.cam1_worker.frameSignal.disconnect(self._on_frame_mirror_cam2)
        except Exception:
            pass
        self.cam2_mirror_active = False
        self.show_logo(2)

    
    
    
    
    
    # === Bật camera chung ===
    def update_titles_and_modes(self) -> None:
        self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir == 'VÀO' else 'Ra'})")
        self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir == 'VÀO' else 'Ra'})")
        if self.cam1_worker:
            self.cam1_worker.set_mode("in" if self.lane1_dir == "VÀO" else "out")
        if self.cam2_worker:
            self.cam2_worker.set_mode("in" if self.lane2_dir == "VÀO" else "out")






    # === XỬ LÝ NÚT LÀN ===
    @Slot()
    def on_reset_lanes(self) -> None:
        self.lane1_dir = "VÀO"
        self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True
        self.two_way_toggle = True
        self.update_titles_and_modes()
        self.show_logo(1)
        self.show_logo(2)






    # === XỬ LÝ NÚT LÀN ===
    @Slot()
    def on_one_way_clicked(self) -> None:
        if self.one_way_toggle_vao:
            self.lane1_dir = "VÀO"
            self.lane2_dir = "VÀO"
        else:
            self.lane1_dir = "RA"
            self.lane2_dir = "RA"
        self.one_way_toggle_vao = not self.one_way_toggle_vao
        self.update_titles_and_modes()






    # === XỬ LÝ NÚT LÀN ===
    @Slot()
    def on_two_way_clicked(self) -> None:
        if self.two_way_toggle:
            self.lane1_dir = "VÀO"
            self.lane2_dir = "RA"
        else:
            self.lane1_dir = "RA"
            self.lane2_dir = "VÀO"
        self.two_way_toggle = not self.two_way_toggle
        self.update_titles_and_modes()






    # === XỬ LÝ THAY ĐỔI OCR MODE ===
    @Slot()
    def on_ocr_mode_changed(self) -> None:
        if self.rb_gem.isChecked() and GEMINI_READY:
            self.current_ocr_mode = "gemini"
        else:
            self.current_ocr_mode = "yolo"

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

    
    
    
    
    
    # === Hiển thị ảnh trong QLabel, căn giữa và scale phù hợp ===
    def _set_centered_pixmap(self, lbl: QLabel, src) -> None:
        pm = None
        if isinstance(src, np.ndarray):
            pm = QPixmap.fromImage(bgr_to_qimage(src))
        elif isinstance(src, QPixmap):
            pm = src
        else:
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






    # === Lấy QPixmap logo mặc định ===
    def qpix_logo(self) -> QPixmap:
        if os.path.exists(LOGO_PATH):
            return QPixmap(LOGO_PATH)
        return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))






    # === Hiển thị logo mặc định vào khung camera/scene/roi ===
    def show_logo(self, which: int) -> None:
        pm = self._logo_pm or self.qpix_logo()
        if which == 1:
            self._set_centered_pixmap(self.lbl_cam1, pm)
        else:
            self._set_centered_pixmap(self.lbl_cam2, pm)

    
    
    
    
    
    # === XỬ LÝ TÍN HIỆU TỪ WORKER ===
    @Slot(np.ndarray, str)
    def _on_frame(self, frame: np.ndarray, title: str) -> None:
        if frame is None or frame.size == 0:
            return
        if "1" in title:
            self._set_centered_pixmap(self.lbl_cam1, frame)
        else:
            self._set_centered_pixmap(self.lbl_cam2, frame)






    # === CẬP NHẬT THÔNG TIN XE VÀO/RA ===
    @Slot(dict)
    def _on_info(self, info: dict) -> None:
        now = QDateTime.currentDateTime()
        date_now = now.toString("dd/MM/yyyy")
        time_now = now.toString("HH:mm:ss")

        # XE VÀO
        if "plate_text_in" in info:
            plate_in = info["plate_text_in"]
            self.ed_plate_in.setText(plate_in)
            self.ed_date_in.setText(info.get("date_in", date_now))
            self.ed_time_in.setText(info.get("time_in", time_now))

            # Cập nhật loại xe vào (Nội bộ / Vãng lai) nếu có DB
            if self.db and self.db.ok:
                try:
                    group_label = self.db.get_vehicle_group_label_by_plate(plate_in)
                except Exception:
                    group_label = ""
            else:
                group_label = ""
            self.ed_group_in.setText(group_label)

        # XE RA
        if "plate_text_out" in info:
            plate_out = info["plate_text_out"]
            self.ed_plate_out.setText(plate_out)
            self.ed_date_out.setText(info.get("date_out", date_now))
            self.ed_time_out.setText(info.get("time_out", time_now))

            # Cập nhật loại xe ra (Nội bộ / Vãng lai) nếu có DB
            if self.db and self.db.ok:
                try:
                    group_label = self.db.get_vehicle_group_label_by_plate(plate_out)
                except Exception:
                    group_label = ""
            else:
                group_label = ""
            self.ed_group_out.setText(group_label)






    # === CẬP NHẬT TRẠNG THÁI SO KHỚP BIỂN SỐ ===
    @Slot(str)
    def _on_match(self, status: str) -> None:
        display_status = status.replace("-", " ").upper() if status else "-"
        self.txt_match.setText(display_status)
        if "KHOP" in display_status:
            self.txt_match.setStyleSheet(
                "color:#007700; font-weight:700; height:28px; "
                "border-radius:6px; border:1px solid #bbf7d0; background:#dcfce7;"
            )
        elif "KHONG" in display_status:
            self.txt_match.setStyleSheet(
                "color:#b91c1c; font-weight:700; height:28px; "
                "border-radius:6px; border:1px solid #fecaca; background:#fee2e2;"
            )
        else:
            self.txt_match.setStyleSheet(
                "color:#1d4ed8; font-weight:700; height:28px; "
                "border-radius:6px; border:1px solid #bfdbfe; background:#e0f2fe;"
            )






    # === CẬP NHẬT ẢNH SCENE ===
    @Slot(str)
    def _on_scene(self, path: str) -> None:
        if not path:
            self._set_centered_pixmap(self.lbl_scene, self.qpix_logo())
            return
        if os.path.exists(path):
            bgr = cv2.imread(path)
            self._set_centered_pixmap(self.lbl_scene, bgr)
        else:
            self._set_centered_pixmap(self.lbl_scene, self.qpix_logo())






    # === CẬP NHẬT ẢNH ROI ===
    @Slot(str, str)
    def _on_roi(self, roi_path: str, mode: str) -> None:
        if not roi_path:
            self._set_centered_pixmap(self.lbl_roi, self.qpix_logo())
            return
        if os.path.exists(roi_path):
            bgr = cv2.imread(roi_path)
            self._set_centered_pixmap(self.lbl_roi, bgr)
        else:
            self._set_centered_pixmap(self.lbl_roi, self.qpix_logo())

    
    
    
    
    
    
    # === KẾT NỐI TÍN HIỆU TỪ WORKER ===
    def _connect_worker(self, w: VideoWorker) -> None:
        w.frameSignal.connect(self._on_frame)
        w.infoSignal.connect(self._on_info)
        w.matchSignal.connect(self._on_match)
        if hasattr(w, "sceneSignal"):
            w.sceneSignal.connect(self._on_scene)
        if hasattr(w, "roiSignal"):
            w.roiSignal.connect(self._on_roi)
        if hasattr(w, "histSignal"):
            w.histSignal.connect(lambda: print("[ManagerCameraPage] history changed"))






    # === BẬT / TẮT CAMERA CHUNG ===
    def start_cam_generic(self, which: int) -> None:
        if not self.models or not self.models.ok:
            QMessageBox.warning(
                self,
                "Model error",
                f"Không load được model:\n{getattr(self.models, 'err', '')}",
            )
            return

        ocr_mode = self.current_ocr_mode
        api_pref = API_MAP.get("ANY", cv2.CAP_ANY)

        if which == 1:
            if self.cam1_worker and self.cam1_worker.isRunning():
                return
            idx = int(self.spin_cam1.value())
            # Nếu Cam 2 đang mirror, không có xung đột thực với thiết bị
            resolved_api = self._probe_camera_backend(idx, api_pref)
            if resolved_api is None:
                QMessageBox.critical(
                    self,
                    "Không mở được Cam 1",
                    f"Không thể mở camera index {idx}.\n"
                    f"Hãy kiểm tra: thiết bị có tồn tại không, quyền truy cập camera,\n"
                    f"hoặc chọn đúng index/nguồn (USB/RTSP).",
                )
                return
            mode = "in" if self.lane1_dir == "VÀO" else "out"
            title = f"Cam 1 ({'Vào' if mode == 'in' else 'Ra'})"
            self.cam1_worker = VideoWorker(
                cam_idx=idx,
                api=resolved_api,
                mode=mode,
                models=self.models,
                db=self.db,
                stable_seconds=1.2,
                ocr_mode=ocr_mode,
                title=title,
                parent=self,
            )
            self._connect_worker(self.cam1_worker)
            self.cam1_worker.start()
        else:
            if self.cam2_worker and self.cam2_worker.isRunning():
                return
            idx = int(self.spin_cam2.value())
            # Nếu Cam 1 đang chạy và index trùng, dùng chế độ MIRROR cho Cam 2
            if self.cam1_worker and self.cam1_worker.isRunning():
                try:
                    if int(getattr(self.cam1_worker, "cam_idx", -1)) == idx:
                        # Tắt mirror cũ nếu đang bật
                        self._disable_cam2_mirror()
                        self.cam2_worker = None  
                        self._enable_cam2_mirror()
                        return
                except Exception:
                    pass
            resolved_api = self._probe_camera_backend(idx, api_pref)
            if resolved_api is None:
                QMessageBox.critical(
                    self,
                    "Không mở được Cam 2",
                    f"Không thể mở camera index {idx}.\n"
                    f"Hãy kiểm tra: thiết bị có tồn tại không, quyền truy cập camera,\n"
                    f"hoặc chọn đúng index/nguồn (USB/RTSP).",
                )
                return
            mode = "in" if self.lane2_dir == "VÀO" else "out"
            title = f"Cam 2 ({'Vào' if mode == 'in' else 'Ra'})"
            self.cam2_worker = VideoWorker(
                cam_idx=idx,
                api=resolved_api,
                mode=mode,
                models=self.models,
                db=self.db,
                stable_seconds=1.2,
                ocr_mode=ocr_mode,
                title=title,
                parent=self,
            )
            self._connect_worker(self.cam2_worker)
            self.cam2_worker.start()






    # === DỪNG CAMERA CHUNG ===
    def stop_cam_generic(self, which: int) -> None:
        if which == 2 and self.cam2_mirror_active:
            # Tắt chế độ mirror trước
            self._disable_cam2_mirror()
        else:
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






    # === DỪNG TẤT CẢ CAMERA ===
    def stop_all(self) -> None:
        self.stop_cam_generic(1)
        self.stop_cam_generic(2)






    # === OVERRIDE CLOSE EVENT ĐỂ DỪNG CAMERA ===
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_all()
        super().closeEvent(event)
