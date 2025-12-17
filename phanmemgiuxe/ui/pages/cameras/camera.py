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
)

from ....config.config import (
    DETECT_MODEL_PATH,
    OCR_MODEL_PATH,
    USE_SQL,
    CONN_STR,
    API_MAP,
    LOGO_PATH,
)
from ....models.models import Models
from ....database.database import DB
from ....utils.utils import letterbox, bgr_to_qimage
from ....workers.workers import VideoWorker





# ===== Camera Page ======
class CameraPage(QWidget):
        
    """
    Đặc điểm:
      * Bật Cam xanh lá, Tắt Cam đỏ (hover).
      * 1 chiều / 2 chiều xanh dương, Reset làn vàng (hover).
      * 4 khung đều hiển thị logo mặc định khi chưa có hình.
      * Ngày/giờ chỉ hiển thị khi có biển số tương ứng
        (ưu tiên date/time từ worker, nếu không có thì dùng thời điểm hiện tại).
      * Biển số màu đỏ, chữ khác màu đen.
      * Hiển thị "Loại xe" (Nội bộ / Vãng lai) cho Xe vào / Xe ra.
      * THÊM: GroupBox "TIỀN PHÍ" hiển thị tiền tự động (nếu worker gửi fee_amount).
    """

    
    
    
    
    # === Lấy tên camera từ bảng CameraMapping theo function_type ===
    def _get_camera_name_by_function(self, function_type: str) -> str:
        """
        Lấy tên camera từ bảng CameraMapping theo function_type (VD: vao_truoc, ra_truoc, ...)
        Nếu không có thì trả về tên mặc định.
        """
        if not self.db or not self.db.ok:
            return ""
        try:
            row = self.db._execute_one(
                """
                SELECT c.camera_name
                FROM dbo.CameraMapping m
                JOIN dbo.Cameras c ON m.camera_id = c.camera_id
                WHERE m.function_type = ? AND c.is_active = 1
                """,
                (function_type,)
            )
            return row["camera_name"] if row and row.get("camera_name") else ""
        except Exception as e:
            print(f"[CameraPage] _get_camera_name_by_function error ({function_type}):", e)
            return ""






    # == Khởi tạo Camera Page ===
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # ===== BACKEND =====
        self.models: Optional[Models] = None
        self.db: Optional[DB] = None
        self._load_backend()

        # ===== WORKER / LÀN =====
        self.cam1_worker: Optional[VideoWorker] = None
        self.cam2_worker: Optional[VideoWorker] = None
        # Cho phép Cam 2 "dùng chung" Cam 1 (mirror)
        self.cam2_mirror_active: bool = False

        self.lane1_dir = "VÀO"
        self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True
        self.two_way_toggle = True

        # Cấu hình camera lấy từ bảng dbo.Cameras
        # cam*_cfg là dict: {camera_type, source_index, full_url, camera_name, ...}
        self.cam1_cfg: Optional[dict] = None
        self.cam2_cfg: Optional[dict] = None


        # Index camera cố định
        self.cam1_idx: int = 0
        self.cam2_idx: int = 1

        # OCR mode (nếu sau này dùng)
        self.current_ocr_mode: str = "yolo"

        # Logo mặc định
        self._logo_pm: Optional[QPixmap] = None

        # ===== UI =====
        self._build_ui()

        # Sau khi UI xong, set logo mặc định cho cả 4 khung
        self._logo_pm = self.qpix_logo()
        self.show_logo(1)
        self.show_logo(2)
        self._set_centered_pixmap(self.lbl_scene, self._logo_pm)
        self._set_centered_pixmap(self.lbl_roi, self._logo_pm)

        # Sau khi load DB, thử tải cấu hình camera từ bảng dbo.Cameras
        self._load_camera_configs()


        # Cảnh báo backend nếu lỗi
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

    
    
    
    
    
    # === Load backend (models, DB) ===
    def _load_backend(self) -> None:
        self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
        if not self.models.ok:
            print("[CameraPage] Lỗi load models:", self.models.err)

        if USE_SQL:
            self.db = DB(CONN_STR)
            if not self.db.ok:
                print("[CameraPage] Lỗi kết nối DB.")
        else:
            self.db = None
            print("[CameraPage] USE_SQL = False, không dùng DB.")


    
    
    
    
    
    # === Lấy cấu hình camera từ DB theo camera_name ===
    def _load_camera_cfg_from_db(self, camera_name: str) -> Optional[dict]:
        """
        Lấy 1 dòng camera theo camera_name trong bảng dbo.Cameras.
        Yêu cầu:
          - DB.ok
          - Bảng Cameras đã tạo như script trước đó.
        """
        if not self.db or not self.db.ok:
            return None
        try:
            row = self.db._execute_one(
                """
                SELECT TOP (1)
                    camera_id,
                    camera_name,
                    camera_type,
                    source_index,
                    ip_address,
                    port,
                    url_path,
                    full_url,
                    username,
                    password,
                    direction,
                    is_active,
                    note
                FROM dbo.Cameras
                WHERE camera_name = ? AND is_active = 1
                ORDER BY camera_id
                """,
                (camera_name,),
            )
            return row
        except Exception as e:
            print(f"[CameraPage] _load_camera_cfg_from_db error ({camera_name}):", e)
            return None






    # === Load cấu hình Cam 1 / Cam 2 từ bảng Cameras ===
    def _load_camera_configs(self) -> None:
        """
        Thử load cấu hình Cam 1 / Cam 2 từ bảng Cameras.
        Nếu có file cấu hình 4view thì lấy tên từ đó, không thì dùng mặc định.
        """

        if not self.db or not self.db.ok:
            print("[CameraPage] DB không OK, không thể tải cấu hình camera từ DB.")
            return
        from ....database.camera_config_db import CameraConfigDB
        mapping = CameraConfigDB(CONN_STR).get_camera_mapping_configs()

        self.cam1_cfg = mapping.get("vao_truoc")
        self.cam2_cfg = mapping.get("ra_truoc")
        if self.cam1_cfg:
            print(f"[CameraPage] Cam1 config (vao_truoc):", self.cam1_cfg)
        else:
            print(f"[CameraPage] Không tìm thấy camera 'vao_truoc' trong mapping.")
        if self.cam2_cfg:
            print(f"[CameraPage] Cam2 config (ra_truoc):", self.cam2_cfg)
        else:
            print(f"[CameraPage] Không tìm thấy camera 'ra_truoc' trong mapping.")


    
    
    
    
    # === Tạo giao diện UI ===
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
        sidebar_scroll.setMinimumWidth(260)
        sidebar_scroll.setMaximumWidth(280)

        side = QWidget()
        side.setObjectName("SideBar")
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

        plate_font_sidebar = QFont("Segoe UI", 14, QFont.Weight.Bold)

        # ----- CAMERA CONTROL -----
        gb_camctl = QGroupBox("CAMERA CONTROL")
        vl_camctl = QVBoxLayout(gb_camctl)
        vl_camctl.setSpacing(8)

        # index cố định, không cho bảo vệ chỉnh
        self.cam1_idx = 0
        self.cam2_idx = 1

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

        self.btn_start1.clicked.connect(self.start_cam1)
        self.btn_stop1.clicked.connect(self.stop_cam1)
        self.btn_start2.clicked.connect(self.start_cam2)
        self.btn_stop2.clicked.connect(self.stop_cam2)

        vside.addWidget(gb_camctl)

        # ----- ĐIỀU KHIỂN LÀN -----
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

        # ----- THÔNG TIN XE VÀO -----
        gb_in = QGroupBox("THÔNG TIN XE VÀO")
        gl_in = QGridLayout(gb_in)
        gl_in.setHorizontalSpacing(8)
        gl_in.setVerticalSpacing(6)

        self.ed_date_in = QLineEdit()
        self.ed_time_in = QLineEdit()
        self.ed_plate_in = QLineEdit()
        self.ed_plate_in.setFont(plate_font_sidebar)
        self.ed_plate_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_plate_in.setStyleSheet(
            "color: #ff0000; font-weight: 700; height: 32px; background:#fff7f7;"
        )

        # ô loại xe vào
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

        # ----- THÔNG TIN XE RA -----
        gb_out = QGroupBox("THÔNG TIN XE RA")
        gl_out = QGridLayout(gb_out)
        gl_out.setHorizontalSpacing(8)
        gl_out.setVerticalSpacing(6)

        self.ed_date_out = QLineEdit()
        self.ed_time_out = QLineEdit()
        self.ed_plate_out = QLineEdit()
        self.ed_plate_out.setFont(plate_font_sidebar)
        self.ed_plate_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_plate_out.setStyleSheet(
            "color: #ff0000; font-weight: 700; height: 32px; background:#fff7f7;"
        )

        # ô loại xe ra
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

        # ----- TIỀN PHÍ -----
        gb_fee = QGroupBox("TIỀN PHÍ")
        gl_fee = QGridLayout(gb_fee)
        gl_fee.setHorizontalSpacing(8)
        gl_fee.setVerticalSpacing(6)

        self.ed_fee = QLineEdit()
        self.ed_fee.setReadOnly(True)
        self.ed_fee.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_fee.setStyleSheet(
            "color:#047857; font-weight:700; height:32px; "
            "border-radius:4px; border:1px solid #bbf7d0; background:#dcfce7;"
        )

        gl_fee.addWidget(QLabel("Tiền phí:"), 0, 0)
        gl_fee.addWidget(self.ed_fee, 0, 1)
        gl_fee.setColumnStretch(1, 1)

        vside.addWidget(gb_fee)

        vside.addStretch(1)

        sidebar_scroll.setWidget(side)
        root.addWidget(sidebar_scroll)

        # ========== MAIN VIEW ==========
        right = QWidget()
        right.setStyleSheet("background-color: #ffffff;")
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # trên: 2 camera
        top = QHBoxLayout()
        top.setSpacing(8)

        self.lbl_cam1 = QLabel()
        self.lbl_cam2 = QLabel()
        for lbl in (self.lbl_cam1, self.lbl_cam2):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "background:#ffffff; border-radius:16px; border:1px solid #e2e8f0;"
            )

            # Để label không tự kéo layout cao lên
            lbl.setScaledContents(False)
            lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding,   # ngang giãn
                QSizePolicy.Policy.Expanding,   # dọc giãn trong KHUNG, KHÔNG đẩy frame to hơn
            )

        # Chiều cao cố định cho khung camera trên
        fixed_cam_h = 280

        cam1_box = QVBoxLayout()
        self.cam1_title = QLabel("Cam 1 (Vào)")
        self.cam1_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam1_title.setStyleSheet("font-weight:600;")
        cam1_box.addWidget(self.cam1_title)
        cam1_box.addWidget(self.lbl_cam1, 1)
        cam1_frame = QFrame()
        cam1_frame.setLayout(cam1_box)
        
        cam1_frame.setMinimumHeight(fixed_cam_h)
        cam1_frame.setMaximumHeight(fixed_cam_h)

        cam2_box = QVBoxLayout()
        self.cam2_title = QLabel("Cam 2 (Vào)")
        self.cam2_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam2_title.setStyleSheet("font-weight:600;")
        cam2_box.addWidget(self.cam2_title)
        cam2_box.addWidget(self.lbl_cam2, 1)
        cam2_frame = QFrame()
        cam2_frame.setLayout(cam2_box)

        cam2_frame.setMinimumHeight(fixed_cam_h)
        cam2_frame.setMaximumHeight(fixed_cam_h)

        top.addWidget(cam1_frame, 1)
        top.addWidget(cam2_frame, 1)
        right_layout.addLayout(top, 0)

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
            lbl.setScaledContents(False)
            lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )

        fixed_scene_h = 220   # thấp hơn khung trên một chút

        scene_box = QVBoxLayout()
        scene_title = QLabel("Image_BOX")
        scene_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scene_title.setStyleSheet("font-weight:600;")
        scene_box.addWidget(scene_title)
        scene_box.addWidget(self.lbl_scene, 1)
        scene_frame = QFrame()
        scene_frame.setLayout(scene_box)

        scene_frame.setMinimumHeight(fixed_scene_h)
        scene_frame.setMaximumHeight(fixed_scene_h)

        roi_box = QVBoxLayout()
        roi_title = QLabel("ROI_Plate")
        roi_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        roi_title.setStyleSheet("font-weight:600;")
        roi_box.addWidget(roi_title)
        roi_box.addWidget(self.lbl_roi, 1)
        roi_frame = QFrame()
        roi_frame.setLayout(roi_box)

        roi_frame.setMinimumHeight(fixed_scene_h)
        roi_frame.setMaximumHeight(fixed_scene_h)

        bottom.addWidget(scene_frame, 1)
        bottom.addWidget(roi_frame, 1)
        right_layout.addLayout(bottom, 0)

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

    
    
    
    
    
    # === Lấy nhãn loại xe (Nội bộ / Vãng lai) từ bảng Vehicles ===
    def _get_vehicle_group_label(self, plate: str) -> str:
        """
        Dựa vào bảng Vehicles:
          - Nếu tìm thấy plate -> 'Nội bộ'
          - Không thấy       -> 'Vãng lai'
        Nếu DB tắt hoặc lỗi thì trả về "" (để trống, không crash).
        """
        plate = (plate or "").strip()
        if not plate or not self.db or not self.db.ok:
            return ""
        try:
            v = self.db.get_vehicle_by_plate(plate)
        except Exception as e:
            print("[CameraPage] _get_vehicle_group_label error:", e)
            return ""
        return "Nội bộ" if v else "Vãng lai"

    
    
    
    
    
    
    # === Định dạng tiền phí hiển thị ===
    def _format_fee(self, fee: Optional[int]) -> str:
        if fee is None:
            return ""
        try:
            return f"{int(fee):,} VND"
        except Exception:
            return str(fee)

    
    
    
    
    
    # === CẬP NHẬT TIÊU ĐỀ VÀ CHẾ ĐỘ LÀN ===
    def update_titles_and_modes(self) -> None:
        self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir == 'VÀO' else 'Ra'})")
        if self.cam2_mirror_active:
            self.cam2_title.setText("Cam 2 (Mirror Cam 1)")
        else:
            self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir == 'VÀO' else 'Ra'})")
        if self.cam1_worker:
            self.cam1_worker.set_mode("in" if self.lane1_dir == "VÀO" else "out")
        if self.cam2_worker:
            self.cam2_worker.set_mode("in" if self.lane2_dir == "VÀO" else "out")





    # === XỬ LÝ NÚT BẤM LÀN ===
    @Slot()
    def on_reset_lanes(self) -> None:
        self.lane1_dir = "VÀO"
        self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True
        self.two_way_toggle = True
        self.update_titles_and_modes()
        self.show_logo(1)
        self.show_logo(2)
        # reset luôn thông tin + tiền phí
        self.ed_fee.clear()






    # === XỬ LÝ NÚT BẤM LÀN ===
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






    # === XỬ LÝ NÚT BẤM LÀN ===
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

    
    
    
    
    
    # === Hiển thị QPixmap / np.ndarray vào QLabel, căn giữa và giữ tỉ lệ ===
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






    # === Hiển thị logo mặc định vào khung camera ===
    def show_logo(self, which: int) -> None:
        pm = self._logo_pm or self.qpix_logo()
        if which == 1:
            self._set_centered_pixmap(self.lbl_cam1, pm)
        else:
            self._set_centered_pixmap(self.lbl_cam2, pm)

    
    
    
    
    
    
    # === XỬ LÝ MIRROR CAM 2 TỪ CAM 1 ===
    @Slot(np.ndarray, str)
    def _on_frame_mirror_cam2(self, frame: np.ndarray, title: str) -> None:
        if frame is None or frame.size == 0:
            return
        self._set_centered_pixmap(self.lbl_cam2, frame)






    # === Bật / Tắt chế độ mirror Cam 2 từ Cam 1 ===
    def _enable_cam2_mirror(self) -> None:
        if not (self.cam1_worker and self.cam1_worker.isRunning()):
            return
        if self.cam2_mirror_active:
            return
        try:
            self.cam1_worker.frameSignal.connect(self._on_frame_mirror_cam2)
            self.cam2_mirror_active = True
            self.cam2_title.setText("Cam 2 (Mirror Cam 1)")
        except Exception:
            self.cam2_mirror_active = False





    # === Tắt chế độ mirror Cam 2 ===
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






    # === Kiểm tra backend camera (DirectShow, MSMF, ANY) ===
    def _probe_camera_backend(self, idx: int, preferred_api: int) -> Optional[int]:
        tried = []
        candidates = [preferred_api, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
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

    
    
    
    
    
    
    # === XỬ LÝ NHẬN FRAME TỪ WORKER ===
    @Slot(np.ndarray, str)
    def _on_frame(self, frame: np.ndarray, title: str) -> None:
        if frame is None or frame.size == 0:
            return
        if "1" in title:
            self._set_centered_pixmap(self.lbl_cam1, frame)
        else:
            self._set_centered_pixmap(self.lbl_cam2, frame)







    # === XỬ LÝ NHẬN THÔNG TIN TỪ WORKER ===
    @Slot(dict)
    def _on_info(self, info: dict) -> None:
        """
        Nhận thông tin từ worker:
          - Cập nhật ngày/giờ/biển số vào/ra.
          - TỰ ĐỘNG cập nhật loại xe (Nội bộ / Vãng lai) cho vào/ra.
          - Nếu có 'fee_amount' -> hiển thị ở ô TIỀN PHÍ.
          - Nếu không có 'fee_amount' nhưng đã có biển số ra -> tự lấy fee từ DB.
        """
        now = QDateTime.currentDateTime()
        date_now = now.toString("dd/MM/yyyy")
        time_now = now.toString("HH:mm:ss")

        # ================= XE VÀO =================
        if "plate_text_in" in info:
            plate_in = info["plate_text_in"]
            self.ed_plate_in.setText(plate_in)
            self.ed_date_in.setText(info.get("date_in", date_now))
            self.ed_time_in.setText(info.get("time_in", time_now))

            group_label_in = self._get_vehicle_group_label(plate_in)
            self.ed_group_in.setText(group_label_in)

        # ================= XE RA ==================
        if "plate_text_out" in info:
            plate_out = info["plate_text_out"]
            self.ed_plate_out.setText(plate_out)
            self.ed_date_out.setText(info.get("date_out", date_now))
            self.ed_time_out.setText(info.get("time_out", time_now))

            group_label_out = self._get_vehicle_group_label(plate_out)
            self.ed_group_out.setText(group_label_out)

            # Nếu worker chưa gửi fee_amount thì tự lấy từ DB
            if ("fee_amount" not in info) and self.db and self.db.ok:
                try:
                    # Lấy phiên gửi xe mới nhất có plate_out này
                    row = self.db._execute_one(
                        """
                        SELECT TOP (1) id, fee_amount
                        FROM dbo.ParkingSessions
                        WHERE plate_out = ?
                        ORDER BY id DESC
                        """,
                        (plate_out,),
                    )
                    if row and row.get("fee_amount") is not None:
                        self.ed_fee.setText(self._format_fee(row["fee_amount"]))
                except Exception as e:
                    print("[CameraPage] load fee_amount from DB error:", e)

        # ================= TIỀN PHÍ =================
        # Ưu tiên fee_amount mà worker gửi trực tiếp (nếu có)
        if "fee_amount" in info:
            fee_val = info.get("fee_amount")
            self.ed_fee.setText(self._format_fee(fee_val))






    # === XỬ LÝ NHẬN TRẠNG THÁI SO KHỚP BIỂN SỐ TỪ WORKER ===
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






    # === XỬ LÝ NHẬN ẢNH SCENE TỪ WORKER ===
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






    # === XỬ LÝ NHẬN ẢNH ROI TỪ WORKER ===
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

    
    
    
    
    
    # === KẾT NỐI SIGNALS TỪ WORKER ===
    def _connect_worker(self, w: VideoWorker) -> None:
        w.frameSignal.connect(self._on_frame)
        w.infoSignal.connect(self._on_info)
        w.matchSignal.connect(self._on_match)
        if hasattr(w, "sceneSignal"):
            w.sceneSignal.connect(self._on_scene)
        if hasattr(w, "roiSignal"):
            w.roiSignal.connect(self._on_roi)






    # === BẮT ĐẦU CHẠY CAMERA (CHUNG) ===
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

        # ------------------- CHỌN CONFIG CAMERA -------------------
        cfg: Optional[dict] = None
        if which == 1:
            cfg = self.cam1_cfg
        else:
            cfg = self.cam2_cfg

        # Nếu không có config DB -> fallback dùng webcam index cũ
        use_db_cfg = bool(cfg and isinstance(cfg, dict))

        # Giá trị mặc định cho webcam fallback
        idx_default = int(self.cam1_idx if which == 1 else self.cam2_idx)

        # Xác định mode ('in' / 'out') cho worker
        if which == 1:
            mode = "in" if self.lane1_dir == "VÀO" else "out"
        else:
            mode = "in" if self.lane2_dir == "VÀO" else "out"

        title = f"Cam {which} ({'Vào' if mode == 'in' else 'Ra'})"

        # ------------------- NẾU ĐANG CHẠY RỒI THÌ THÔI -------------------
        worker = self.cam1_worker if which == 1 else self.cam2_worker
        if worker and worker.isRunning():
            return

        # ------------------- CASE: DÙNG CONFIG TỪ DB -------------------
        if use_db_cfg:
            cam_type = (cfg.get("camera_type") or "WEBCAM").upper()
            src_index = cfg.get("source_index", None)
            full_url = (cfg.get("full_url") or "").strip()

            # WEBCAM: dùng index + API -> probe backend như cũ
            if cam_type == "WEBCAM":
                try:
                    idx = int(src_index if src_index is not None else idx_default)
                except Exception:
                    idx = idx_default

                resolved_api = self._probe_camera_backend(idx, api_pref)
                if resolved_api is None:
                    QMessageBox.critical(
                        self,
                        f"Không mở được Cam {which}",
                        f"Không thể mở camera WEBCAM index {idx}.\n"
                        f"Hãy kiểm tra thiết bị/quyền truy cập hoặc chọn index khác.",
                    )
                    return

                w = VideoWorker(
                    cam_idx=idx,
                    api=resolved_api,
                    mode=mode,
                    models=self.models,
                    db=self.db,
                    stable_seconds=1.2,
                    ocr_mode=ocr_mode,
                    title=title,
                    parent=self,
                    camera_type="WEBCAM",
                    full_url="",
                )

            # RTSP / HTTP: dùng full_url, bỏ qua cam_idx & api
            else:
                if not full_url:
                    QMessageBox.critical(
                        self,
                        f"Không mở được Cam {which}",
                        "Camera kiểu RTSP/HTTP nhưng full_url rỗng.\n"
                        "Hãy kiểm tra lại cấu hình bảng Cameras.",
                    )
                    return

                w = VideoWorker(
                    cam_idx=-1,
                    api=cv2.CAP_ANY,
                    mode=mode,
                    models=self.models,
                    db=self.db,
                    stable_seconds=1.2,
                    ocr_mode=ocr_mode,
                    title=title,
                    parent=self,
                    camera_type=cam_type,
                    full_url=full_url,
                )

            # Gán worker cho cam1/cam2 rồi start
            if which == 1:
                self.cam1_worker = w
            else:
                self.cam2_worker = w
            self._connect_worker(w)
            w.start()
            return

        # ------------------- CASE: FALLBACK WEBCAM NHƯ CŨ -------------------
        # (Khi chưa có bản ghi Cameras tương ứng)
        if which == 1:
            idx = idx_default
        else:
            idx = idx_default

        # Mirror logic cho Cam 2 như code cũ
        if which == 2:
            # Nếu trùng index với Cam 1 đang chạy -> dùng mirror
            if self.cam1_worker and self.cam1_worker.isRunning():
                try:
                    if int(getattr(self.cam1_worker, "cam_idx", -1)) == idx:
                        self._disable_cam2_mirror()
                        self.cam2_worker = None
                        self._enable_cam2_mirror()
                        return
                except Exception:
                    pass

        resolved_api = self._probe_camera_backend(idx, api_pref)
        if resolved_api is None:
            if which == 2:
                # Nếu không mở được Cam2 nhưng Cam1 đang chạy -> mirror
                if self.cam1_worker and self.cam1_worker.isRunning():
                    self._disable_cam2_mirror()
                    self.cam2_worker = None
                    self._enable_cam2_mirror()
                    return

                # Thử mở Cam1 rồi mirror
                try:
                    cam1_idx = int(self.cam1_idx)
                except Exception:
                    cam1_idx = 0
                resolved_api_cam1 = self._probe_camera_backend(cam1_idx, api_pref)
                if resolved_api_cam1 is not None:
                    mode1 = "in" if self.lane1_dir == "VÀO" else "out"
                    title1 = f"Cam 1 ({'Vào' if mode1 == 'in' else 'Ra'})"
                    self.cam1_worker = VideoWorker(
                        cam_idx=cam1_idx,
                        api=resolved_api_cam1,
                        mode=mode1,
                        models=self.models,
                        db=self.db,
                        stable_seconds=1.2,
                        ocr_mode=ocr_mode,
                        title=title1,
                        parent=self,
                        camera_type="WEBCAM",
                        full_url="",
                    )
                    self._connect_worker(self.cam1_worker)
                    self.cam1_worker.start()
                    self._disable_cam2_mirror()
                    self.cam2_worker = None
                    self._enable_cam2_mirror()
                    return

            QMessageBox.critical(
                self,
                f"Không mở được Cam {which}",
                f"Không thể mở camera index {idx}.\n"
                f"Hãy kiểm tra thiết bị/quyền truy cập hoặc chọn index khác.",
            )
            return

        # Fallback worker WEBCAM
        w = VideoWorker(
            cam_idx=idx,
            api=resolved_api,
            mode=mode,
            models=self.models,
            db=self.db,
            stable_seconds=1.2,
            ocr_mode=ocr_mode,
            title=title,
            parent=self,
            camera_type="WEBCAM",
            full_url="",
        )
        if which == 1:
            self.cam1_worker = w
        else:
            self.cam2_worker = w
        self._connect_worker(w)
        w.start()






    # === DỪNG CHẠY CAMERA (CHUNG) ===
    def stop_cam_generic(self, which: int) -> None:
        if which == 2 and self.cam2_mirror_active:
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






    # === BẮT ĐẦU / DỪNG CAMERA 1 & 2 ===
    def start_cam1(self) -> None:
        self.start_cam_generic(1)

    def stop_cam1(self) -> None:
        self.stop_cam_generic(1)

    def start_cam2(self) -> None:
        self.start_cam_generic(2)

    def stop_cam2(self) -> None:
        self.stop_cam_generic(2)

    def stop_all(self) -> None:
        self.stop_cam_generic(1)
        self.stop_cam_generic(2)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_all()
        super().closeEvent(event)
