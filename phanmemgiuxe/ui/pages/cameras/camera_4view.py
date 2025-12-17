from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QLineEdit,
    QSizePolicy,
    QFrame,
)

from ....config.config import LOGO_PATH
from ....utils.utils import bgr_to_qimage, letterbox
from phanmemgiuxe.models.models import Models
from phanmemgiuxe.database.database import DB
from phanmemgiuxe.utils.utils import save_image
from datetime import datetime
from PySide6.QtCore import QDateTime
from phanmemgiuxe.utils.utils import plate_norm





# ===== Camera 4 View Page ======
class Camera4ViewPage(QWidget):
    """
    Trang CAMERA 2: 4 camera + 4 khung ảnh chụp + thanh thông tin bên dưới.

    Bố cục:
      - Trên: 2 hàng x 4 cột
            + Hàng 1: 4 camera (vào trước, vào sau, ra trước, ra sau)
            + Hàng 2: 4 ảnh chụp (mặt / biển số vào / ra)
      - Dưới: 3 groupbox nằm ngang:
            + THÔNG TIN XE VÀO  (2x2: Ngày/BS + Giờ/Loại)
            + THÔNG TIN XE RA   (2x2: Ngày/BS + Giờ/Loại)
            + TIỀN PHÍ          (1 ô duy nhất)
    """
    
    
    
    
    # === Init UI ===
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        try:
            from ....config.config import DETECT_MODEL_PATH, OCR_MODEL_PATH
        except ImportError:
            DETECT_MODEL_PATH = ""
            OCR_MODEL_PATH = ""
        self.models = Models(DETECT_MODEL_PATH or "", OCR_MODEL_PATH or "")
        self.db = DB()
        from phanmemgiuxe.utils.utils import save_image
        self.save_image = save_image
        self._build_ui()
        self._init_logo()
        self.start_cameras()





    # === Load logo image ===
    def trigger_in(self, role: str):
        """
        Sự kiện xe vào: quẹt thẻ -> chụp ảnh, detect biển số, ocr, lưu DB, cập nhật UI.
        role: 'vao_truoc' hoặc 'vao_sau'
        """
        cam_idx = 0 if role == 'vao_truoc' else 1
        cap = self.cam_streams[cam_idx] if cam_idx < len(self.cam_streams) else None
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                self._show_error(f"Không chụp được ảnh từ camera {role}")
                return
            self._set_centered_pixmap(self.lbl_img_in_face, frame)
            img_face_path = self.save_image(frame, f"in_face_{role}") or ""
            plates, boxed = self.models.detect_plates(frame)
            
            if not plates:
                self._show_error("Không phát hiện được biển số!")
                self._set_centered_pixmap(self.lbl_img_in_plate, boxed)
                return

            (x1, y1, x2, y2), roi = plates[0]
            self._set_centered_pixmap(self.lbl_img_in_plate, roi)
            img_plate_path = self.save_image(roi, f"in_plate_{role}") or ""
            
            # OCR biển số
            plate_text, _ = self.models.ocr_plate_yolo(roi)
            self.ed_plate_in.setText(plate_text)
            now = QDateTime.currentDateTime()
            self.ed_date_in.setText(now.toString("dd/MM/yyyy"))
            self.ed_time_in.setText(now.toString("HH:mm:ss"))
            
            # Loại xe
            group_label = self.db.get_vehicle_group_label_by_plate(plate_text)
            self.ed_group_in.setText(group_label)
            
            # Lưu DB
            self.db.insert_in(plate_text, now.toString("dd/MM/yyyy"), now.toString("HH:mm:ss"), img_plate_path)
        else:
            self._show_error(f"Camera {role} không mở hoặc không có cấu hình!")






    # === Xử lý xe ra ===
    def trigger_out(self, role: str):
        """
        Sự kiện xe ra: quẹt thẻ -> chụp ảnh, detect biển số, ocr, lưu DB, cập nhật UI, so khớp, tính phí.
        role: 'ra_truoc' hoặc 'ra_sau'
        """
        cam_idx = 2 if role == 'ra_truoc' else 3
        cap = self.cam_streams[cam_idx] if cam_idx < len(self.cam_streams) else None
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                self._show_error(f"Không chụp được ảnh từ camera {role}")
                return
            
            # Hiển thị ảnh mặt ra
            self._set_centered_pixmap(self.lbl_img_out_face, frame)
            img_face_path = self.save_image(frame, f"out_face_{role}") or ""
           
            # Detect biển số
            plates, boxed = self.models.detect_plates(frame)
            if not plates:
                self._show_error("Không phát hiện được biển số!")
                self._set_centered_pixmap(self.lbl_img_out_plate, boxed)
                return
            (x1, y1, x2, y2), roi = plates[0]
            self._set_centered_pixmap(self.lbl_img_out_plate, roi)
            img_plate_path = self.save_image(roi, f"out_plate_{role}") or ""
           
            # OCR biển số
            plate_text, _ = self.models.ocr_plate_yolo(roi)
            self.ed_plate_out.setText(plate_text)
            now = QDateTime.currentDateTime()
            self.ed_date_out.setText(now.toString("dd/MM/yyyy"))
            self.ed_time_out.setText(now.toString("HH:mm:ss"))
           
            # Loại xe
            group_label = self.db.get_vehicle_group_label_by_plate(plate_text)
            self.ed_group_out.setText(group_label)
            match_status = self.db.attach_out(plate_text, now.toString("dd/MM/yyyy"), now.toString("HH:mm:ss"), img_plate_path)
            match_status = self.db.attach_out(plate_text, now.toString("dd/MM/yyyy"), now.toString("HH:mm:ss"), img_plate_path)
           
            # Hiển thị trạng thái khớp biển số
            if match_status == "KHOP-BIEN-SO":
                self._show_info("Biển số ra khớp với lượt vào!")
            else:
                self._show_info("Biển số ra không khớp!")
           
            # Hiển thị tiền phí
            fee = self.db.get_latest_fee_for_plate(plate_text)
            self.ed_fee.setText(f"{fee:,} VND" if fee else "")
        else:
            self._show_error(f"Camera {role} không mở hoặc không có cấu hình!")





    # === Load logo image ===
    def _show_error(self, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Lỗi", msg)




    # === Hiển thị thông báo thông tin ===
    def _show_info(self, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Thông báo", msg)




    # === Load logo image ===
    def start_cameras(self):
        import cv2
        from phanmemgiuxe.database.camera_config_db import CameraConfigDB
        import time
        self.camera_db = CameraConfigDB()
        
        # Lấy mapping camera từ DB
        mapping = self.camera_db.get_camera_mapping_configs()
        roles = ["vao_truoc", "vao_sau", "ra_truoc", "ra_sau"]
        self.cam_streams = []
        self.cam_names = []
        self.cam_error_logged = [False, False, False, False]
        for idx, role in enumerate(roles):
            cam = mapping.get(role)
            print(f"[Camera4View] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo camera {role}...")
            t_cam = time.time()
           
            # Kiểm tra cấu hình trước khi mở
            if not cam:
                self.cam_streams.append(None)
                self.cam_names.append(None)
                print(f"[Camera4View] [{time.strftime('%H:%M:%S')}] Camera {role} không có cấu hình. Bỏ qua.")
                continue
            url = cam.get("full_url") or self._build_url(cam)
            ip = cam.get("ip_address")
            port = cam.get("port")
           
            # Nếu thiếu URL, IP, port thì bỏ qua luôn
            if not url or "None" in url or url.strip() == "" or not ip or not port:
                self.cam_streams.append(None)
                self.cam_names.append(cam.get("camera_name"))
                print(f"[Camera4View] [{time.strftime('%H:%M:%S')}] Camera {role} thiếu hoặc sai URL/IP/port. Bỏ qua.")
                continue
           
            # Nếu có cấu hình hợp lệ thì mới thử mở camera
            cap = None
            try:
                t_open = time.time()
                cap = cv2.VideoCapture(url)
             
                # Thử mở trong 2 giây, nếu không mở được thì bỏ qua
                opened = False
                for _ in range(20):
                    if cap.isOpened():
                        opened = True
                        break
                    time.sleep(0.1)
                    if time.time() - t_open > 2:
                        break
                t_done = time.time()
                if opened:
                    self.cam_streams.append(cap)
                    self.cam_names.append(cam.get("camera_name"))
                    print(f"[Camera4View] [{time.strftime('%H:%M:%S')}] Mở camera {role} thành công. Mất {t_done-t_cam:.2f}s (riêng mở: {t_done-t_open:.2f}s)")
                else:
                    if cap:
                        cap.release()
                    self.cam_streams.append(None)
                    self.cam_names.append(cam.get("camera_name"))
                    print(f"[Camera4View] [{time.strftime('%H:%M:%S')}] Không mở được camera {role}. Bỏ qua.")
            except Exception as e:
                if cap:
                    cap.release()
                self.cam_streams.append(None)
                self.cam_names.append(cam.get("camera_name"))
                print(f"[Camera4View] [{time.strftime('%H:%M:%S')}] Lỗi khi mở camera {role}: {e}. Bỏ qua.")
       
        # Timer để cập nhật hình ảnh
        from PySide6.QtCore import QTimer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(40)  # ~25fps






    # === Xây dựng URL từ cấu hình camera ===
    def _build_url(self, cam):
        ip = cam.get("ip_address")
        port = cam.get("port")
        user = cam.get("username")
        pwd = cam.get("password")
        path = cam.get("url_path")
        if not ip or not port:
            return None
        if user and pwd and path:
            return f"rtsp://{user}:{pwd}@{ip}:{port}/{path}"
        if path:
            return f"rtsp://{ip}:{port}/{path}"
        return f"rtsp://{ip}:{port}"






    # === Cập nhật khung hình camera ===
    def update_frames(self):
        labels = [self.lbl_cam_in1, self.lbl_cam_in2, self.lbl_cam_out1, self.lbl_cam_out2]
        for i, cap in enumerate(getattr(self, 'cam_streams', [])):
            if cap and hasattr(cap, "isOpened") and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self._set_scaled_pixmap(labels[i], frame)
                else:
                    self._set_scaled_pixmap(labels[i], self._logo_pm)
            else:
                if self.cam_names and self.cam_names[i] and not self.cam_error_logged[i]:
                    print(f"[Camera4View] Không mở được camera '{self.cam_names[i]}' hoặc cấu hình không hợp lệ.")
                    self.cam_error_logged[i] = True
                self._set_scaled_pixmap(labels[i], self._logo_pm)






    # === Tạo QLabel cho khung camera ===
    def _set_scaled_pixmap(self, lbl: QLabel, src) -> None:
        import numpy as np
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
        from PySide6.QtCore import Qt
        scaled = pm.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        if hasattr(scaled, "setDevicePixelRatio"):
            scaled.setDevicePixelRatio(dpr)
        lbl.setPixmap(scaled)






    # === Tạo QLabel cho khung camera ===
    def closeEvent(self, event):
        for cap in getattr(self, 'cam_streams', []):
            cap.release()
        event.accept()


    
    
    
    
    # === Xây dựng UI ===
    def _build_ui(self) -> None:
        self.setObjectName("Camera4ViewRoot")
        self.setStyleSheet(
            """
            QWidget#Camera4ViewRoot {
                background-color: #f5f5f7;
            }

            /* ====== TITLE CÁC KHUNG CAMERA ====== */
            QLabel[cssClass="camTitle"] {
                font-weight: 600;
                color: #111827;
                padding-left: 8px;
                font-size: 11px;
            }

            /* ====== LABEL THÔNG TIN ====== */
            QLabel[cssClass="infoLabel"] {
                color: #111827;
                font-size: 11px;
            }

            /* ====== Ô THÔNG TIN NGÀY/GIỜ ====== */
            QLineEdit[cssClass="infoEdit"] {
                height: 22px;
                border-radius: 4px;
                border: 1px solid #e5e7eb;
                padding: 2px 6px;
                background: #ffffff;
                color: #111827;
                font-size: 11px;
            }

            /* ====== Ô BIỂN SỐ ====== */
            QLineEdit[cssClass="plateEdit"] {
                height: 26px;
                border-radius: 4px;
                border: 1px solid #fecaca;
                padding: 2px 6px;
                background: #fff7f7;
                color: #dc2626;
                font-weight: 700;
                font-size: 13px;
            }

            /* ====== Ô LOẠI XE ====== */
            QLineEdit[cssClass="groupEdit"] {
                height: 22px;
                border-radius: 4px;
                border: 1px solid #e5e7eb;
                padding: 2px 6px;
                background: #f9fafb;
                color: #111827;
                font-weight: 600;
                font-size: 11px;
            }

            /* ====== Ô TIỀN PHÍ ====== */
            QLineEdit[cssClass="feeEdit"] {
                height: 26px;
                border-radius: 4px;
                border: 1px solid #bbf7d0;
                padding: 2px 6px;
                background: #dcfce7;
                color: #047857;
                font-weight: 700;
                font-size: 13px;
            }

            /* ====== KHUNG THÔNG TIN VÀO/RA/PHÍ ====== */
            QGroupBox#InfoInBox,
            QGroupBox#InfoOutBox,
            QGroupBox#InfoFeeBox {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                margin-top: 4px;
                padding-top: 2px;
            }

            QGroupBox#InfoInBox::title,
            QGroupBox#InfoOutBox::title,
            QGroupBox#InfoFeeBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                margin-left: 4px;
                color: #111827;
                background: #f5f5f7;
                font-weight: 600;
                font-size: 11px;
            }

            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 2, 8, 4)
        root.setSpacing(4)

        # ====== KHU VỰC 4 CAM + 4 ẢNH ======
        top_frame = QFrame()
        top_layout = QVBoxLayout(top_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        # ---- 4 khung camera (hàng trên) ----
        self.lbl_cam_in1 = self._create_cam_label()
        self.lbl_cam_in2 = self._create_cam_label()
        self.lbl_cam_out1 = self._create_cam_label()
        self.lbl_cam_out2 = self._create_cam_label()

        grid.addWidget(self._wrap_cam_card("Cam vào trước", self.lbl_cam_in1), 0, 0)
        grid.addWidget(self._wrap_cam_card("Cam vào sau", self.lbl_cam_in2), 0, 1)
        grid.addWidget(self._wrap_cam_card("Cam ra trước", self.lbl_cam_out1), 0, 2)
        grid.addWidget(self._wrap_cam_card("Cam ra sau", self.lbl_cam_out2), 0, 3)

        # ---- 4 khung ảnh chụp (hàng dưới) ----
        self.lbl_img_in_face = self._create_cam_label()
        self.lbl_img_in_plate = self._create_cam_label()
        self.lbl_img_out_face = self._create_cam_label()
        self.lbl_img_out_plate = self._create_cam_label()

        grid.addWidget(
            self._wrap_cam_card("Ảnh vào - mặt", self.lbl_img_in_face), 1, 0
        )
        grid.addWidget(
            self._wrap_cam_card("Ảnh vào - biển số", self.lbl_img_in_plate), 1, 1
        )
        grid.addWidget(
            self._wrap_cam_card("Ảnh ra - mặt", self.lbl_img_out_face), 1, 2
        )
        grid.addWidget(
            self._wrap_cam_card("Ảnh ra - biển số", self.lbl_img_out_plate), 1, 3
        )

        top_layout.addLayout(grid)
        # Chiếm phần lớn màn hình
        root.addWidget(top_frame, 6)

        # ====== NÚT XỬ LÝ SỰ KIỆN ======
        btn_frame = QFrame()
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)
        from PySide6.QtWidgets import QPushButton
        btn_in = QPushButton("Xe vào")
        btn_out = QPushButton("Xe ra")
        btn_in.setFixedHeight(32)
        btn_out.setFixedHeight(32)
        btn_layout.addWidget(btn_in)
        btn_layout.addWidget(btn_out)
        btn_in.clicked.connect(lambda: self.trigger_in('vao_truoc'))
        btn_out.clicked.connect(lambda: self.trigger_out('ra_truoc'))

        # ====== THANH THÔNG TIN DƯỚI ======
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        # === THÔNG TIN XE VÀO (2x2: Ngày/BS + Giờ/Loại) ===
        gb_in = QGroupBox("THÔNG TIN XE VÀO")
        gb_in.setObjectName("InfoInBox")
        layout_in = QGridLayout(gb_in)
        layout_in.setContentsMargins(4, 2, 4, 4)
        layout_in.setHorizontalSpacing(6)
        layout_in.setVerticalSpacing(2)

        self.ed_date_in = QLineEdit()
        self.ed_time_in = QLineEdit()
        self.ed_plate_in = QLineEdit()
        self.ed_group_in = QLineEdit()

        for w in (self.ed_date_in, self.ed_time_in):
            w.setReadOnly(True)
            w.setProperty("cssClass", "infoEdit")

        plate_font = QFont("Segoe UI", 13, QFont.Weight.Bold)

        self.ed_plate_in.setReadOnly(True)
        from PySide6.QtCore import Qt
        self.ed_plate_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_plate_in.setProperty("cssClass", "plateEdit")
        self.ed_plate_in.setFont(plate_font)

        self.ed_group_in.setReadOnly(True)
        from PySide6.QtCore import Qt
        self.ed_group_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_group_in.setProperty("cssClass", "groupEdit")

        # Hàng 1: Ngày vào | Biển số vào
        layout_in.addWidget(self._make_label("Ngày vào:"), 0, 0)
        layout_in.addWidget(self.ed_date_in, 0, 1)
        layout_in.addWidget(self._make_label("Biển số vào:"), 0, 2)
        layout_in.addWidget(self.ed_plate_in, 0, 3)

        # Hàng 2: Giờ vào  | Loại xe
        layout_in.addWidget(self._make_label("Giờ vào:"), 1, 0)
        layout_in.addWidget(self.ed_time_in, 1, 1)
        layout_in.addWidget(self._make_label("Loại xe:"), 1, 2)
        layout_in.addWidget(self.ed_group_in, 1, 3)

        layout_in.setColumnStretch(1, 1)
        layout_in.setColumnStretch(3, 1)

        # === THÔNG TIN XE RA (2x2: Ngày/BS + Giờ/Loại) ===
        gb_out = QGroupBox("THÔNG TIN XE RA")
        gb_out.setObjectName("InfoOutBox")
        layout_out = QGridLayout(gb_out)
        layout_out.setContentsMargins(4, 2, 4, 4)
        layout_out.setHorizontalSpacing(6)
        layout_out.setVerticalSpacing(2)

        self.ed_date_out = QLineEdit()
        self.ed_time_out = QLineEdit()
        self.ed_plate_out = QLineEdit()
        self.ed_group_out = QLineEdit()

        for w in (self.ed_date_out, self.ed_time_out):
            w.setReadOnly(True)
            w.setProperty("cssClass", "infoEdit")

        self.ed_plate_out.setReadOnly(True)
        from PySide6.QtCore import Qt
        self.ed_plate_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_plate_out.setProperty("cssClass", "plateEdit")
        self.ed_plate_out.setFont(plate_font)

        self.ed_group_out.setReadOnly(True)
        from PySide6.QtCore import Qt
        self.ed_group_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_group_out.setProperty("cssClass", "groupEdit")

        # Hàng 1: Ngày ra | Biển số ra
        layout_out.addWidget(self._make_label("Ngày ra:"), 0, 0)
        layout_out.addWidget(self.ed_date_out, 0, 1)
        layout_out.addWidget(self._make_label("Biển số ra:"), 0, 2)
        layout_out.addWidget(self.ed_plate_out, 0, 3)

        # Hàng 2: Giờ ra  | Loại xe
        layout_out.addWidget(self._make_label("Giờ ra:"), 1, 0)
        layout_out.addWidget(self.ed_time_out, 1, 1)
        layout_out.addWidget(self._make_label("Loại xe:"), 1, 2)
        layout_out.addWidget(self.ed_group_out, 1, 3)

        layout_out.setColumnStretch(1, 1)
        layout_out.setColumnStretch(3, 1)

        # === TIỀN PHÍ (1 ô duy nhất) ===
        gb_fee = QGroupBox("TIỀN PHÍ")
        gb_fee.setObjectName("InfoFeeBox")
        layout_fee = QVBoxLayout(gb_fee)
        layout_fee.setContentsMargins(4, 6, 4, 4)
        layout_fee.setSpacing(2)

        self.ed_fee = QLineEdit()
        self.ed_fee.setReadOnly(True)
        from PySide6.QtCore import Qt
        self.ed_fee.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ed_fee.setProperty("cssClass", "feeEdit")

        layout_fee.addWidget(self.ed_fee)

        # Thêm vào thanh dưới: VÀO (2) – RA (2) – PHÍ (1)
        bottom_layout.addWidget(gb_in, 2)
        bottom_layout.addWidget(gb_out, 2)
        bottom_layout.addWidget(gb_fee, 1)

        # Thêm nút vào dưới cùng giao diện
        root.addWidget(bottom_frame, 1)
        root.addWidget(btn_frame, 0)

    
    
    
    
    
    # === Tạo QLabel hiển thị hình ảnh camera hoặc ảnh chụp ===
    def _create_cam_label(self) -> QLabel:
        """
        Tạo QLabel dùng để hiển thị hình (cam hoặc ảnh chụp).
        """
        lbl = QLabel()
        from PySide6.QtCore import Qt
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from PySide6.QtWidgets import QSizePolicy
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumHeight(220)
        lbl.setStyleSheet(
            """
            QLabel {
                background: #ffffff;
                border-radius: 14px;
                border: 1px solid #e5e7eb;
            }
            """
        )
        return lbl






    # === Bọc QLabel hình vào frame + title ===
    def _wrap_cam_card(self, title: str, img_label: QLabel) -> QFrame:
        """
        Bọc QLabel hình vào 1 frame + title phía trên cho đẹp.
        """
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        from PySide6.QtCore import Qt
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title_lbl.setProperty("cssClass", "camTitle")

        layout.addWidget(title_lbl)
        layout.addWidget(img_label, 1)
        return frame






    # === Tạo QLabel cho thông tin bên dưới ===
    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("cssClass", "infoLabel")
        return lbl

    
    
    
    
    
    # === Load logo image ===
    def _init_logo(self) -> None:
        self._logo_pm = self.qpix_logo()
        # set logo cho tất cả khung
        for lbl in [
            self.lbl_cam_in1,
            self.lbl_cam_in2,
            self.lbl_cam_out1,
            self.lbl_cam_out2,
            self.lbl_img_in_face,
            self.lbl_img_in_plate,
            self.lbl_img_out_face,
            self.lbl_img_out_plate,
        ]:
            self._set_centered_pixmap(lbl, self._logo_pm)






    # === Load logo image ===
    def qpix_logo(self) -> QPixmap:
        if os.path.exists(LOGO_PATH):
            return QPixmap(LOGO_PATH)
        # fallback: tạo ảnh rỗng từ letterbox
        return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))






    # === Tạo QLabel cho khung camera ===
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
        )
        if hasattr(scaled, "setDevicePixelRatio"):
            scaled.setDevicePixelRatio(dpr)

        lbl.setPixmap(scaled)
