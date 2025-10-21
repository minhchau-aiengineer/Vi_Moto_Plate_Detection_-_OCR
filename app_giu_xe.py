# -*- coding: utf-8 -*-
"""
        =========================================================
        = PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini) =
        =========================================================

1. Phát hiện & OCR (YOLOv8, OpenCV, Gemini AI): 	          
    Tự động phát hiện vị trí biển số (YOLO Detect) và trích xuất ký tự 
    (YOLO OCR hoặc Gemini AI). Xử lý tiền DL ảnh (CLAHE, Blur) để tăng độ chính xác OCR.

2. Giao diện - UI [PySide6 (QMainWindow, QThread, Signal/Slot)]:
    Xây dựng giao diện Desktop, hiển thị video trực tiếp, kết quả OCR, và kết nối các luồng 
    xử lý video (Worker) với giao diện chính.

3. Quản lý Dữ liệu	[SQL Server (qua pyodbc), pandas]:	
    Lưu trữ lịch sử giao dịch xe vào/ra (ParkingSessions). Tải dữ liệu lịch sử vào 
    DataFrame để hiển thị trên bảng UI và Export Excel.

4. Luồng Video (QThread, cv2.VideoCapture):
	Chạy độc lập cho hai làn xe (VÀO/RA). Chụp và xử lý ảnh khi biển số ổn định (ít nhất 1.2s), 
    sau đó gửi kết quả (ảnh, biển số, thời gian) về UI.

5. Logic Giữ Xe	(Hàm attach_out trong Class DB):
	Tự động so khớp biển số xe ra với các xe vào đang chờ (plate_out IS NULL). Cập nhật 
    trạng thái KHOP-BIEN-SO (Xanh) hoặc KHONG-KHOP-BIEN-SO (Đỏ) trong DB và trên UI.

6. Cấu hình	(.env, QSpinBox, QRadioButton):
	Cho phép người dùng chọn Index Camera, cấu hình chế độ làn xe (1 chiều/2 chiều) và 
    lựa chọn Model OCR (YOLO hoặc Gemini).

7. Xây dựng UI (_build_ui()):
	Hàm xây dựng bố cục chính, tạo các widgets như nút Bật/Tắt Cam, Điều khiển Làn, 
    các ô hiển thị thông tin xe VÀO/RA, và Bảng Lịch Sử.

8. Điều khiển Cam (start_cam_generic/stop_cam_generic):	
    Khởi động/Dừng luồng xử lý video (VideoWorker). Thiết lập chế độ Vào/Ra và 
    chế độ OCR cho Worker trước khi chạy.

9. Điều khiển Làn (on_one_way_clicked/on_two_way_clicked):
	Quản lý hướng làn xe (Vào, Ra). Cho phép chuyển đổi giữa chế độ 
    Một chiều (Cam 1 & 2 cùng hướng) và Hai chiều (Cam 1 & 2 ngược hướng).

10. Chọn OCR Model (on_ocr_mode_changed):
	Cho phép người dùng chọn Model OCR. Kiểm tra nếu thiếu API Key Gemini 
    thì buộc chuyển về YOLO và thông báo.

11. Cập nhật Real-time (on_frame/on_info/v.v):
	Các hàm @Slot nhận tín hiệu (Signal) từ luồng VideoWorker (ảnh, biển số, thời gian) và 
    cập nhật tức thời lên các ô hiển thị trên giao diện chính.

12. Quản lý Bảng (show_history_view/refresh_history):
	Chuyển đổi giữa chế độ xem Camera chính và Bảng Lịch sử. Tải và hiển thị dữ liệu 
    giao dịch từ SQL lên bảng QTableWidget.

13. Thao tác DB (on_export_excel/on_delete_history):
	Xử lý các thao tác quản lý dữ liệu: Xuất dữ liệu lịch sử ra Excel và Xóa các 
    dòng giao dịch đã chọn trong cơ sở dữ liệu. 

"""

# ==================== 1. IMPORT ====================

import os, sys, time, cv2, numpy as np, pandas as pd
from datetime import datetime

# ---- 1.1 HiDPI Cấu hình HiDPI (Độ phân giải cao) ----
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# ---- 1.2 Import PySide6 ----
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint, QUrl
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
    QFileDialog, QHeaderView, QDialog
)

# ---- 1.3 Optional SQL ----
USE_SQL = True
try:
    import pyodbc
except Exception:
    USE_SQL = False

# ---- 1.4 YOLO ----
from ultralytics import YOLO

# ---- 1.5 Gemini API (optional) ----
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_READY = False
try:
    if GEMINI_API_KEY:
        from google import generativeai as genai
        from google.api_core import exceptions as gexceptions
        from PIL import Image
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_READY = True
except Exception as _e:
    print("Gemini init failed:", _e)
    GEMINI_READY = False





# ==================== 2. CONFIG ====================

# ---- 2.1 Đường dẫn Model ----
DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")
SOUND_IN_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_vao_xin_cam_on.wav"
SOUND_OUT_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_ra_xin_cam_on.wav"

# ---- 2.2 SQL ----
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"
)

# ---- 2.3 UI ----
PANEL_W, PANEL_H = 640, 360
PANEL_BG = (255, 255, 255)

API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
           "six":"6","seven":"7","eight":"8","nine":"9"}





# ==================== 3. UTILITIES (HÀM TIỆN ÍCH) ====================

# ---- 3.1 Căn chỉnh/Điền nền ----
def letterbox(bgr, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
    if bgr is None:
        return np.full((h, w, 3), color, dtype=np.uint8)
    ih, iw = bgr.shape[:2]
    if ih == 0 or iw == 0:
        return np.full((h, w, 3), color, dtype=np.uint8)
    s = min(w/iw, h/ih); nw, nh = int(iw*s), int(ih*s)
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    top, left = (h-nh)//2, (w-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

# ---- 3.2 Chuyển đổi ảnh ----
def bgr_to_qimage(bgr):
    if bgr is None:
        bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

# ---- 3.3 Lưu ảnh ----
def save_image(img, prefix):
    ts = datetime.now().strftime("%Ym%d_%H%M%S_%f")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, img)
    return path

# ---- 3.4 OCR ----
def norm_char(x):  # Chuẩn hóa ký tự
    return OCR_MAP.get(str(x), str(x))

def plate_norm(s: str) -> str: # Chuẩn hóa biển số
    return (s or "").replace("-", "").replace(" ", "").upper()

def has_boxes(r):  # Kiểm tra có box
    try:
        return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
    except: return False

def preprocess_for_ocr(roi):  # Tiền xử lý ảnh OCR
    if roi is None: return None
    if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe,(3,3),0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)





# ==================== 4. DB LAYER ====================

class DB:
    # ---- 4.1 Khởi tạo và Kết nối ----
    def __init__(self, conn_str: str):
        self.ok = False; self.conn = None; self.cur  = None
        if not USE_SQL: return
        try:
            self.conn = pyodbc.connect(conn_str, autocommit=True)
            self.cur  = self.conn.cursor()
            self.cur.execute("""
                IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
                CREATE TABLE dbo.ParkingSessions(
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    plate_in NVARCHAR(64)  NULL,
                    date_in  NVARCHAR(16)  NULL,
                    time_in  NVARCHAR(16)  NULL,
                    image_in NVARCHAR(255) NULL,
                    plate_out NVARCHAR(64)  NULL,
                    date_out  NVARCHAR(16)  NULL,
                    time_out  NVARCHAR(16)  NULL,
                    image_out NVARCHAR(255) NULL,
                    match_status NVARCHAR(32) NULL,
                    created_at DATETIME DEFAULT GETDATE()
                );
            """)
            self.ok = True
        except Exception as e:
            print("DB connect error:", e); self.ok = False

    # ---- 4.2 Ghi nhận xe VÀO ----
    def insert_in(self, plate, d, t, img_path):
        if not self.ok: return
        try:
            self.cur.execute("""
                INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
                VALUES (?,?,?,?,?)
            """, (plate, d, t, img_path, 'PENDING'))
        except Exception as e: print("insert_in error:", e)

    # ---- 4.3 Ghi nhận xe RA và Ghép đôi ----
    def attach_out(self, plate_out, d, t, img_path) -> str:
        if not self.ok: return "Khong khop bien so"
        try:
            rows = self.cur.execute("""
                SELECT TOP 50 id, plate_in FROM dbo.ParkingSessions
                WHERE plate_out IS NULL
                ORDER BY id DESC
            """).fetchall()
            match_sid = None
            for sid, plate_in in rows:
                if plate_norm(plate_in) == plate_norm(plate_out):
                    match_sid = sid; break
            if match_sid:
                self.cur.execute("""
                    UPDATE dbo.ParkingSessions
                    SET plate_out=?, date_out=?, time_out=?, image_out=?, match_status='KHOP-BIEN-SO'
                    WHERE id=?
                """, (plate_out, d, t, img_path, match_sid))
                return "Khop bien so"
            else:
                self.cur.execute("""
                    INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
                    VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO')
                """, (plate_out, d, t, img_path))
                return "Khong khop bien so"
        except Exception as e:
            print("attach_out error:", e); return "Khong khop bien so"

    # ---- 4.4 Lấy lịch sử ----
    def fetch_history_df(self, limit=10000) -> pd.DataFrame:
        if not self.ok:
            return pd.DataFrame(columns=[
                "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
                "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
            ])
        try:
            rows = self.cur.execute(f"""
                SELECT TOP {limit}
                    id, image_in, plate_in, date_in, time_in,
                    image_out, plate_out, date_out, time_out, match_status
                FROM dbo.ParkingSessions
                ORDER BY id DESC
            """).fetchall()
            df = pd.DataFrame.from_records(
                rows,
                columns=["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
                         "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
            ).astype(object).where(pd.notnull, "")
            df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
            df.insert(0, "STT", range(1, len(df)+1))
            return df
        except Exception as e:
            print("fetch_history error:", e)
            return pd.DataFrame(columns=[
                "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
                "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
            ])

    # ---- 4.5 Xóa theo ID ----
    def delete_by_ids(self, ids):
        if not self.ok or not ids: return
        try:
            for sid in ids:
                self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
        except Exception as e: print("delete_by_ids error:", e)

    # ---- 4.6 Xóa tất cả ----
    def delete_all(self):
        if not self.ok: return
        try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
        except Exception as e: print("delete_all error:", e)





# ==================== 5. YOLO/GEMINI WRAPPERS (TRÌNH BAO BỌC MODEL) ====================
class Models:
    # ---- 5.1 Khởi tạo (Tải model) ----
    def __init__(self, det_path: str, ocr_path: str):
        self.ok = True; self.err = ""
        try:
            self.det = YOLO(det_path)
            self.ocr = YOLO(ocr_path)
        except Exception as e:
            self.ok = False; self.err = str(e)

    # ---- 5.2 YOLO phát hiện biển số ----
    def detect_plates(self, frame):
        plates, boxed = [], frame.copy()
        for r in self.det(frame):
            if not has_boxes(r): continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            for (x1,y1,x2,y2) in xyxy:
                pad=8
                x1=max(0,x1-pad); y1=max(0,y1-pad)
                x2=min(boxed.shape[1]-1,x2+pad); y2=min(boxed.shape[0]-1,y2+pad)
                roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
                plates.append(((x1,y1,x2,y2), roi))
                cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        return plates, boxed

    # ---- 5.3 OCR biển số bằng YOLO ----
    def ocr_plate_yolo(self, roi):
        roi_pre = preprocess_for_ocr(roi)
        res = self.ocr(roi_pre if roi_pre is not None else roi)
        text_raw=""
        for r in res:
            if not has_boxes(r): continue
            names = getattr(r,'names',None) or getattr(self.ocr,'names',{}) or {}
            clses = r.boxes.cls.cpu().numpy().astype(int)
            xyxys= r.boxes.xyxy.cpu().numpy()
            boxes=[]
            for i,cls in enumerate(clses):
                x1,y1,x2,y2 = xyxys[i]
                cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
                ch = norm_char(names.get(cls, str(cls)) if isinstance(names,dict) else str(cls))
                if ch.isdigit() or (ch.isalpha() and ch.isupper()):
                    boxes.append((cy,cx,ch))
            if not boxes: continue
            ys=[b[0] for b in boxes]
            if len(boxes)<=7 or (max(ys)-min(ys) < 0.2*max(ys, default=1)):
                text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
            else:
                thr=(max(ys)+min(ys))/2.0
                l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
                t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
                t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
                text_raw=f"{t1}-{t2}" if t2 else t1
        return self._format_text(text_raw)

    # ---- 5.4 OCR biển số bằng Gemini AI ----
    def ocr_plate_gemini_from_path(self, image_path: str):
        if not GEMINI_READY: return "", ""
        try:
            img = Image.open(image_path)
        except Exception as e:
            print("Gemini open image error:", e); return "", ""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash') # Dùng 1.5-flash
            prompt = ("Đây là ảnh biển số xe Việt Nam. "
                      "Hãy trích xuất CHÍNH XÁC chuỗi biển số và chỉ trả về chuỗi đó. "
                      "VD: '29-P1 123.45' hoặc '50-Z8 888.88'.")
            resp = model.generate_content([prompt, img])
            raw = (resp.text or "").strip().replace("\n", " ")
            return self._format_text(raw)
        except gexceptions.GoogleAPICallError as e:
            print("Gemini API error:", e); return "", ""
        except Exception as e:
            print("Gemini unknown error:", e); return "", ""

    # ---- 5.5 Hỗ trợ (Hàm tĩnh định dạng) ----
    @staticmethod
    def _format_text(text_raw: str):
        raw=(text_raw or '').replace('-', ' ').replace(' ', '')
        text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
        return text_fmt, (text_raw or "")





# ==================== 6. VIDEO WORKER (Luồng xử lý Video) ====================
class VideoWorker(QThread):
    frameSignal = Signal(np.ndarray, str)
    sceneSignal = Signal(str)
    roiSignal   = Signal(str, str)
    infoSignal  = Signal(dict)
    matchSignal = Signal(str)
    histSignal  = Signal()
    playSoundSignal = Signal(str)

    # ---- 6.1 Khởi tạo ----
    def __init__(self, cam_idx: int, api: int, mode: str, models: Models, db: DB,
                 stable_seconds: float = 1.2, ocr_mode: str = "yolo", title: str = "", parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.api = api
        self.mode = mode
        self.models = models
        self.db = db
        self.stable_seconds = stable_seconds
        self.ocr_mode = ocr_mode
        self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")

        self._running = False
        self.cap = None
        self.stable_start = 0.0
        self.captured = False

    # ---- 6.2 Setter/Getter ----
    def set_title(self, title: str): 
        self.title = title
    def set_ocr_mode(self, mode: str): 
        self.ocr_mode = mode
    def set_mode(self, mode: str): 
        self.mode = mode

    # ---- 6.3 Vòng lặp chính của luồng ----
    def run(self):
        self._running = True
        self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
        if not (self.cap and self.cap.isOpened()):
            self._running = False; return
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        try: self.cap.set(cv2.CAP_PROP_FPS, 30)
        except: pass

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                self.stable_start = 0.0; self.captured = False
                time.sleep(0.03); continue

            self.frameSignal.emit(frame, self.title)

            plates, boxed = self.models.detect_plates(frame)
            if not plates:
                self.stable_start = 0.0; self.captured = False
                time.sleep(0.01); continue

            best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
            roi_current = best[1]

            if self.stable_start == 0.0 or self.captured:
                self.stable_start = time.time(); self.captured = False

            if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
                scene_path = save_image(boxed if boxed is not None else frame,
                                        "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
                roi_path   = save_image(roi_current,
                                        "plate_in_roi" if self.mode=="in" else "plate_out_roi")

                if self.ocr_mode == "gemini" and GEMINI_READY:
                    text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
                else:
                    text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

                if text_fmt or text_raw:
                    now = datetime.now()
                    d = now.strftime("%d/%m/%Y")
                    t = now.strftime("%H:%M:%S")
                    plate = text_fmt or text_raw

                    self.sceneSignal.emit(scene_path)
                    self.roiSignal.emit(roi_path, self.mode)

                    if self.mode == "in":
                        self.infoSignal.emit({"date_in": d, "time_in": t, "plate_text_in": plate})
                        if self.db and self.db.ok:
                            self.db.insert_in(plate, d, t, scene_path)
                            self.histSignal.emit()
                        self.playSoundSignal.emit("in")
                    else:
                        self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
                        if self.db and self.db.ok:
                            match = self.db.attach_out(plate, d, t, scene_path)
                            self.matchSignal.emit(match)
                            self.histSignal.emit()
                        self.playSoundSignal.emit("out")
                    self.captured = True

            time.sleep(0.01)

        try:
            if self.cap: self.cap.release()
        except: pass

    # ---- 6.4 Dừng luồng ----
    def stop(self): 
        self._running = False





# ==================== 7. DELETE DIALOG (Hộp thoại Xóa) ====================

class DeleteDialog(QDialog):
    # ---- 7.1 Khởi tạo Giao diện ----
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xóa lịch sử")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background: #ffffff;
                border: 2px solid #e6e6e6;
                border-radius: 10px;
            }
        """)
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
        lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
        lay.addWidget(lab)

        row = QHBoxLayout(); row.setSpacing(12)
        self.btn_sel = QPushButton("Xóa dòng đã chọn")
        self.btn_all = QPushButton("Xóa tất cả")
        self.btn_can = QPushButton("Hủy")
        row.addWidget(self.btn_sel, 1)
        row.addWidget(self.btn_all, 1)
        row.addWidget(self.btn_can, 1)
        lay.addLayout(row)

        base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
        self.btn_sel.setStyleSheet(f"""
        QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}
        QPushButton:hover  {{ background:#d4e5ff; }}
        QPushButton:pressed{{ background:#c8deff; }}
        """)
        self.btn_all.setStyleSheet(f"""
        QPushButton {{ {base} background:#ffe0e0; border:1px solid #ffb3b3; }}
        QPushButton:hover  {{ background:#ffd1d1; }}
        QPushButton:pressed{{ background:#ffc2c2; }}
        """)
        self.btn_can.setStyleSheet(f"""
        QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}
        QPushButton:hover  {{ background:#eef0f3; }}
        QPushButton:pressed{{ background:#e7e9ed; }}
        """)

        self.btn_sel.clicked.connect(lambda: self.done(1))
        self.btn_all.clicked.connect(lambda: self.done(2))
        self.btn_can.clicked.connect(lambda: self.done(0))






# ==================== 8. MAIN WINDOW (CỬA SỔ CHÍNH ====================

class MainWindow(QMainWindow):
    # ---- 8.1 Khởi tạo Giao diện ứng dụng ----
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop App (Giữ xe)")
        self.setMinimumSize(1200, 800)
        self._init_theme()

        self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
        if not self.models.ok:
            QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
        self.db = DB(CONN_STR) if USE_SQL else DB("")

        # Khởi tạo âm thanh
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

        self.cam1_worker = None
        self.cam2_worker = None

        self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True; self.two_way_toggle = True
        self.current_ocr_mode = "yolo"

        # Lưu logo gốc để scale lại đúng ở mọi lần vẽ
        self._logo_pm = self.qpix_logo()
        self._build_ui()
        self.show_logo(1); self.show_logo(2)
        self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.refresh_history); self.hist_timer.start(5000)

    # ---- 8.2 Thiết lập Giao diện ----
    def _init_theme(self):
        self.setStyleSheet("""
        * { color: #000000; }
        QMainWindow, QWidget { background: #ffffff; }
        QWidget#SideBar { background: #ffffff; }

        QGroupBox {
            background: #ffffff;
            font-weight: 600;
            border: 2px solid #e6e6e6;
            border-radius: 12px;
            margin-top: 8px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            background: #ffffff;
        }

        QFrame[class="card-wrap"] { background: #e6e6e6; border-radius: 14px; }
        QFrame[class="card"]       { background: #ffffff; border-radius: 12px; }
        QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; }
        QLabel[class="title"] {
            font: 700 18px "Segoe UI";
            padding: 6px 10px;
            background: #ffffff;
            border-radius: 10px;
        }

        QLineEdit {
            height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px;
        }
        QTableWidget { background: #ffffff; gridline-color: #e6e6e6; }
        """)

    # ---- 8.3 Chuẩn hóa hành vi của các nút ----
    def _normalize_button(self, *btns):
        for b in btns:
            b.setAutoDefault(False); b.setDefault(False); b.setFlat(False); b.setFocusPolicy(Qt.NoFocus)
            # FIX: để nút không kéo giãn vô hạn khi phóng to
            b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

    # ---- 8.4 Sửa lỗi bo tròn: Đơn giản hóa hàm ----
    def _apply_btn_style(self, btn: QPushButton, css: str):
        btn.setStyleSheet(css)

    # ---- 8.5 Tạo khung hiển thị (Card UI) ----
    def _make_card(self, title:str, content:QWidget):
        wrap = QFrame(); wrap.setProperty("class","card-wrap")
        wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
        card = QFrame(); card.setProperty("class","card")
        v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
        title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
        hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
        title_lbl = QLabel(title); title_lbl.setProperty("class","title")
        hl.addWidget(title_lbl)
        v.addWidget(title_wrap); v.addWidget(content, 1)
        wrapL.addWidget(card)
        return wrap, title_lbl

    # ---- 8.6 Hiển thị ảnh căn giữa và giữ tỷ lệ ----
    def _set_centered_pixmap(self, lbl: QLabel, src):
        if isinstance(src, np.ndarray):
            pm = QPixmap.fromImage(bgr_to_qimage(src))
        elif isinstance(src, QImage):
            pm = QPixmap.fromImage(src)
        else:
            pm = src
        if pm is None or pm.isNull():
            lbl.clear(); return
        rect = lbl.contentsRect()
        avail = rect.size()
        dpr = lbl.devicePixelRatioF() if hasattr(lbl, "devicePixelRatioF") else 1.0
        target_w = max(1, int(avail.width()  * dpr))
        target_h = max(1, int(avail.height() * dpr))
        scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if hasattr(scaled, "setDevicePixelRatio"):
            scaled.setDevicePixelRatio(dpr)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(scaled)

    # ---- 8.7 Xây dựng cấu trúc giao diện chính (Widgets-UI) ----
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

        # LEFT
        side = QWidget(objectName="SideBar")
        side.setFixedWidth(450)
        vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

        # CAMERA CONTROL
        gb_camctl = QGroupBox("CAMERA CONTROL")
        vl_camctl = QVBoxLayout(gb_camctl)
        vl_camctl.setSpacing(10) # Khoảng cách giữa các hàng

        self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
        self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
        self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys())); self.cb_api1.hide() # Ẩn API combo box
        self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys())); self.cb_api2.hide() # Ẩn API combo box

        # Hàng 1: Index Cam 1 & Index Cam 2
        row_indices = QHBoxLayout()
        row_indices.setSpacing(10)
        row_indices.addWidget(QLabel("Index Cam 1"))
        row_indices.addWidget(self.spin_cam1, 1)
        row_indices.addWidget(QLabel("Index Cam 2"))
        row_indices.addWidget(self.spin_cam2, 1)
        vl_camctl.addLayout(row_indices)

        # Buttons
        self.btn_start1 = QPushButton("Bật Cam 1")
        self.btn_stop1  = QPushButton("Tắt Cam 1")
        self.btn_start2 = QPushButton("Bật Cam 2")
        self.btn_stop2  = QPushButton("Tắt Cam 2")
        self._normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)

        common_btn = "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;"

        self._apply_btn_style(self.btn_start1, f"""
        QPushButton {{ {common_btn} background:#d1fadf; border:1px solid #a6f4c5; }}
        QPushButton:hover  {{ background:#c3f7d6; }}
        QPushButton:pressed{{ background:#b4f3cc; }}
        QPushButton:disabled{{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }}
        """)
        self._apply_btn_style(self.btn_stop1, f"""
        QPushButton {{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }}
        QPushButton:hover  {{ background:#ffd1d1; }}
        QPushButton:pressed{{ background:#ffc2c2; }}
        QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}
        """)
        self._apply_btn_style(self.btn_start2, self.btn_start1.styleSheet())
        self._apply_btn_style(self.btn_stop2,  self.btn_stop1.styleSheet())

        # signals
        self.btn_start1.clicked.connect(self.start_cam1)
        self.btn_stop1.clicked.connect(self.stop_cam1)
        self.btn_start2.clicked.connect(self.start_cam2)
        self.btn_stop2.clicked.connect(self.stop_cam2)

        # Hàng 2: Bật/Tắt Cam 1
        row_btn1 = QHBoxLayout()
        row_btn1.setSpacing(12)
        row_btn1.addWidget(self.btn_start1)
        row_btn1.addWidget(self.btn_stop1)
        vl_camctl.addLayout(row_btn1)

        # Hàng 3: Bật/Tắt Cam 2
        row_btn2 = QHBoxLayout()
        row_btn2.setSpacing(12)
        row_btn2.addWidget(self.btn_start2)
        row_btn2.addWidget(self.btn_stop2)
        vl_camctl.addLayout(row_btn2)

        vside.addWidget(gb_camctl)

        # ĐIỀU KHIỂN LÀN
        gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
        vl_lane = QVBoxLayout(gb_lane); vl_lane.setSpacing(10)
        row_lane = QHBoxLayout(); row_lane.setSpacing(12)

        self.btn_oneway = QPushButton("1 chiều")
        self.btn_twoway = QPushButton("2 chiều")
        self.btn_reset_lane = QPushButton("Reset làn")
        self._normalize_button(self.btn_oneway, self.btn_twoway, self.btn_reset_lane)

        self._apply_btn_style(self.btn_oneway, f"""
        QPushButton {{ {common_btn} background:#dbeafe; border:1px solid #bfdbfe; }}
        QPushButton:hover  {{ background:#cfe3fd; }}
        QPushButton:pressed{{ background:#c3dcfc; }}
        QPushButton:disabled{{ background:#eef6ff; color:#777; border-color:#e3efff; }}
        """)
        self._apply_btn_style(self.btn_twoway, self.btn_oneway.styleSheet())
        self._apply_btn_style(self.btn_reset_lane, f"""
        QPushButton {{ {common_btn} background:#fff3bf; border:1px solid #ffe066; }}
        QPushButton:hover  {{ background:#ffeda3; }}
        QPushButton:pressed{{ background:#ffe788; }}
        QPushButton:disabled{{ background:#fff9dc; color:#777; border-color:#ffefb3; }}
        """)

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
        vb_ocr = QVBoxLayout(gb_ocr)
        self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)"); self.rb_yolo.setChecked(True)
        self.rb_gem  = QRadioButton("Dùng Gemini AI")
        vb_ocr.addWidget(self.rb_yolo); vb_ocr.addWidget(self.rb_gem)
        self.rb_yolo.toggled.connect(self.on_ocr_mode_changed)
        self.rb_gem.toggled.connect(self.on_ocr_mode_changed)
        if not GEMINI_READY:
            self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY (.env hoặc biến môi trường) → dùng YOLO")
        vside.addWidget(gb_ocr)

        # THÔNG TIN XE VÀO
        gb_in = QGroupBox("THÔNG TIN XE VÀO")
        gl_in = QGridLayout(gb_in)
        self.ed_date_in = QLineEdit(); 
        self.ed_time_in = QLineEdit(); 
        self.ed_plate_in = QLineEdit(); 
        self.ed_plate_in.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
        gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
        gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
        gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
        vside.addWidget(gb_in)

        # THÔNG TIN XE RA
        gb_out = QGroupBox("THÔNG TIN XE RA")
        gl_out = QGridLayout(gb_out)
        self.ed_date_out = QLineEdit(); 
        self.ed_time_out = QLineEdit(); 
        self.ed_plate_out = QLineEdit(); 
        self.ed_plate_out.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
        gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
        gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
        gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
        vside.addWidget(gb_out)

        # BẢNG LỊCH SỬ (nút)
        gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
        v_hist_btns = QVBoxLayout(gb_hist_btns)
        self.btn_show_history = QPushButton("Xem bảng lịch sử")
        self.btn_export_hist  = QPushButton("Export Excel")
        self.btn_delete_hist  = QPushButton("Xóa bảng")
        self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide()
        self._normalize_button(self.btn_show_history, self.btn_export_hist, self.btn_delete_hist, self.btn_hide_history)

        self._apply_btn_style(self.btn_show_history, f"""
        QPushButton {{ {common_btn} background:#E6F4EA; border:1px solid #cde9d6; }}
        QPushButton:hover  {{ background:#d9efe0; }}
        QPushButton:pressed{{ background:#ccead6; }}
        QPushButton:disabled{{ background:#f1faf4; color:#777; border-color:#e3f5e9; }}
        """)
        self._apply_btn_style(self.btn_hide_history, f"""
        QPushButton {{ {common_btn} background:#fff3bf; border:1px solid #f5c6c2; }}
        QPushButton:hover  {{ background:#ffeda3; }}
        QPushButton:pressed{{ background:#ffe788; }}
        QPushButton:disabled{{ background:#fff9dc; color:#777; border-color:#ffefb3; }}
        """)
        self._apply_btn_style(self.btn_export_hist, f"""
        QPushButton {{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }}
        QPushButton:hover  {{ background:#d4e5ff; }}
        QPushButton:pressed{{ background:#c8deff; }}
        QPushButton:disabled{{ background:#eef5ff; color:#777; border-color:#ddeaff; }}
        """)
        self._apply_btn_style(self.btn_delete_hist, f"""
        QPushButton {{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }}
        QPushButton:hover  {{ background:#ffd1d1; }}
        QPushButton:pressed{{ background:#ffc2c2; }}
        QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}
        """)
        row_cmd = QHBoxLayout()

        # FIX: không dùng stretch để nút không kéo dài
        row_cmd.addWidget(self.btn_export_hist)
        row_cmd.addWidget(self.btn_delete_hist)
        v_hist_btns.addWidget(self.btn_show_history)
        v_hist_btns.addLayout(row_cmd)
        v_hist_btns.addWidget(self.btn_hide_history)
        vside.addWidget(gb_hist_btns)

        vside.addStretch(1)
        root.addWidget(side)

        # RIGHT
        right_container = QVBoxLayout()
        self.main_view = QWidget()
        main_layout = QVBoxLayout(self.main_view)

        top = QHBoxLayout()
        self.lbl_cam1 = QLabel(); self.lbl_cam1.setScaledContents(False)
        self.lbl_cam2 = QLabel(); self.lbl_cam2.setScaledContents(False)
        for lbl in (self.lbl_cam1, self.lbl_cam2):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
            # FIX: không đặt minimumSize theo PANEL_W/H; chỉ đặt chiều cao gợi ý
            lbl.setMinimumHeight(220)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1)
        cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
        top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1)
        main_layout.addLayout(top)

        bottom = QHBoxLayout()
        self.lbl_scene = QLabel(); self.lbl_scene.setScaledContents(False); self.lbl_scene.setAlignment(Qt.AlignCenter); self.lbl_scene.setStyleSheet("background:#ffffff; border-radius:12px;"); self.lbl_scene.setMinimumHeight(220); self.lbl_scene.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_roi   = QLabel(); self.lbl_roi.setScaledContents(False);   self.lbl_roi.setAlignment(Qt.AlignCenter);   self.lbl_roi.setStyleSheet("background:#ffffff; border-radius:12px;");   self.lbl_roi.setMinimumHeight(220);   self.lbl_roi.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scene_card, _ = self._make_card("Image_BOX",  self.lbl_scene)
        roi_card,   _ = self._make_card("ROI_Plate", self.lbl_roi)
        bottom.addWidget(scene_card, 1); bottom.addWidget(roi_card, 1)
        main_layout.addLayout(bottom)

        self.info_group = QGroupBox("Thông tin chi tiết")
        info_layout = QGridLayout(self.info_group)
        self.txt_date_in  = QLabel("--/--/----"); self.txt_time_in  = QLabel("--:--:--")
        self.txt_plate_in = QLabel("---"); self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")
        self.txt_date_out = QLabel("--/--/----"); self.txt_time_out = QLabel("--:--:--")
        self.txt_plate_out= QLabel("---"); self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")
        # self.txt_match    = QLabel("")
        # THAY THẾ: Dùng QLineEdit thay cho QLabel để có giao diện khung
        self.txt_match = QLineEdit()
        self.txt_match.setReadOnly(True) 
        self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
        r=0
        info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
        info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
        info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
        info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
        info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
        info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
        info_layout.addWidget(QLabel("So khớp biển số:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,2)
        main_layout.addWidget(self.info_group)

        self.history_view = QWidget()
        hist_layout = QVBoxLayout(self.history_view)
        hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)")
        hist_v = QVBoxLayout(hist_group)

        self.tbl_hist = QTableWidget(0, 10)
        self.tbl_hist.setHorizontalHeaderLabels(["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"])
        header = self.tbl_hist.horizontalHeader()
        hfont = QFont(header.font()); hfont.setBold(True); header.setFont(hfont)
        self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)

        # Sửa lỗi bảng xen kẽ: Tắt
        self.tbl_hist.setAlternatingRowColors(False)

        header.setSectionResizeMode(QHeaderView.Stretch)
        hist_v.addWidget(self.tbl_hist)
        hist_layout.addWidget(hist_group)

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.main_view)
        self.stacked.addWidget(self.history_view)
        self.stacked.setCurrentIndex(0)
        right_container.addWidget(self.stacked, 1)
        root.addLayout(right_container, 1)

        self.update_titles_and_modes()

        # Kết nối các nút lịch sử sau khi tạo UI
        self.btn_show_history.clicked.connect(self.show_history_view)
        self.btn_hide_history.clicked.connect(self.show_main_view)
        self.btn_export_hist.clicked.connect(self.on_export_excel)
        self.btn_delete_hist.clicked.connect(self.on_delete_history)

    # ---- 8.8 Cập nhật hướng làn và thông báo cho worker ----
    def update_titles_and_modes(self):
        self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
        self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
        if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
        if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")

    # ---- 8.9 Đặt lại hướng làn mặc định ----
    @Slot()
    def on_reset_lanes(self):
        self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True; self.two_way_toggle = True
        self.update_titles_and_modes()
        self.show_logo(1); self.show_logo(2)

    # ---- 8.10 Chuyển đổi chế độ một chiều ----
    @Slot()
    def on_one_way_clicked(self):
        if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
        else:                       self.lane1_dir="RA";  self.lane2_dir="RA"
        self.one_way_toggle_vao = not self.one_way_toggle_vao
        self.update_titles_and_modes()

    # ---- 8.11 Chuyển đổi chế độ hai chiều ----
    @Slot()
    def on_two_way_clicked(self):
        if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
        else:                   self.lane1_dir="RA";  self.lane2_dir="VÀO"
        self.two_way_toggle = not self.two_way_toggle
        self.update_titles_and_modes()

    # ---- 8.xx Hàm này cần được viết trong class MainWindow và kết nối với worker ----
    @Slot(str)
    def update_match_status(self, status: str):
        display_status = status.replace('-', ' ').title()
        self.txt_match.setText(display_status) # <-- Cập nhật QLineEdit

        if "Khop Bien So" in display_status:
            # Xanh lá cây
            self.txt_match.setStyleSheet("color: #007700; font-weight: 700;") 
        elif "Khong Khop Bien So" in display_status:
            # Đỏ
            self.txt_match.setStyleSheet("color: #ff0000; font-weight: 700;")
        else:
            # Xanh dương (Mặc định/Chờ)
            self.txt_match.setStyleSheet("color: #007700; font-weight: 700;")

    # ---- 8.xx MỚI: Nhận tín hiệu và phát âm thanh ----
    @Slot(str)
    def on_play_sound(self, mode):
        """Phát âm thanh dựa trên chế độ (in/out)"""
        if mode == "in":
            self.sound_in.play()
        elif mode == "out":
            self.sound_out.play()
        else:
            print(f"Lỗi: Không tìm thấy file âm thanh!")

    # ---- 8.12 Xử lý thay đổi chế độ OCR ----
    @Slot()
    def on_ocr_mode_changed(self):
        self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
        if self.rb_gem.isChecked() and not GEMINI_READY:
            QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
            self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
        if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
        if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

    # ---- 8.13 Hiển thị chế độ xem Lịch sử ----
    def show_history_view(self):
        self.stacked.setCurrentIndex(1); self.btn_show_history.hide(); self.btn_hide_history.show(); self.refresh_history()
    
    # ---- 8.14 Hiển thị chế độ xem Camera chính ----
    def show_main_view(self):
        self.stacked.setCurrentIndex(0); self.btn_hide_history.hide(); self.btn_show_history.show()

    # ---- 8.15 Xuất dữ liệu lịch sử ra Excel ----
    @Slot()
    def on_export_excel(self):
        df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
        if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
        if df.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
        if not path: return
        try: df.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
        except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

    # ---- 8.16 Xóa dữ liệu lịch sử ----
    @Slot()
    def on_delete_history(self):
        if not (self.db and self.db.ok):
            QMessageBox.warning(self, "Xóa", "Chưa kết nối DB, không thể xóa."); return
        dlg = DeleteDialog(self)
        g = self.geometry(); dlg.adjustSize()
        dlg.move(self.mapToGlobal(QPoint(g.width()-dlg.width()-40, 140)))
        res = dlg.exec()
        if res == 1:
            rows = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
            if not rows: QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào."); return
            cols = [self.tbl_hist.horizontalHeaderItem(i).text() for i in range(self.tbl_hist.columnCount())]
            if "ID" not in cols: QMessageBox.warning(self, "Xóa", "Không tìm thấy cột ID."); return
            id_col = cols.index("ID"); ids = []
            for r in rows:
                item = self.tbl_hist.item(r, id_col)
                if item: ids.append(item.text())
            if not ids: QMessageBox.information(self, "Xóa", "Không lấy được ID các dòng chọn."); return
            self.db.delete_by_ids(ids); self.refresh_history()
        elif res == 2:
            self.db.delete_all(); self.refresh_history()
        else:
            return

    # ---- 8.17 image helpers ----
    def qpix_logo(self):
        if os.path.exists(LOGO_PATH):
            return QPixmap(LOGO_PATH)
        return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))

    # ---- 8.18 Hiển thị logo/ảnh mặc định trên camera ----
    def show_logo(self, which: int):
        pm = self._logo_pm
        if which == 1: self._set_centered_pixmap(self.lbl_cam1, pm)
        else:          self._set_centered_pixmap(self.lbl_cam2, pm)

    # ---- 8.19 Nhận và hiển thị khung hình video ----
    @Slot(np.ndarray, str)
    def on_frame(self, frame_bgr, title):
        sender = self.sender()
        if sender is self.cam1_worker:
            self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
        elif sender is self.cam2_worker:
            self._set_centered_pixmap(self.lbl_cam2, frame_bgr)

    # ---- 8.20 Nhận và hiển thị khung hình scene ----
    @Slot(str)
    def on_scene(self, path):
        if os.path.exists(path):
            bgr = cv2.imread(path)
            self._set_centered_pixmap(self.lbl_scene, bgr)

    # ---- 8.21 Nhận và hiển thị khung hình ROI ----
    @Slot(str, str)
    def on_roi(self, roi_path, mode):
        if os.path.exists(roi_path):
            bgr = cv2.imread(roi_path)
            self._set_centered_pixmap(self.lbl_roi, bgr)

    # ---- 8.22 Nhận và hiển thị thông tin xe ----
    @Slot(dict)
    def on_info(self, info):
        if "date_in" in info:  self.txt_date_in.setText(info["date_in"]);   self.ed_date_in.setText(info["date_in"])
        if "time_in" in info:  self.txt_time_in.setText(info["time_in"]);   self.ed_time_in.setText(info["time_in"])
        if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
        if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
        if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
        if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

    # ---- 8.23 Nhận và hiển thị thông tin so khớp ----
    @Slot(str)
    def on_match(self, txt): 
        self.txt_match.setText(txt.upper())

    # ---- 8.24 Tải và cập nhật bảng lịch sử ----
    @Slot()
    def refresh_history(self):
        df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
        if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
        if df.empty:
            self.tbl_hist.setRowCount(0)
            cols = ["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
            self.tbl_hist.setColumnCount(len(cols)); self.tbl_hist.setHorizontalHeaderLabels(cols)
            hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
            self.tbl_hist.horizontalHeader().setFont(hfont); return

        cols = list(df.columns)
        self.tbl_hist.setRowCount(len(df)); self.tbl_hist.setColumnCount(len(cols))
        self.tbl_hist.setHorizontalHeaderLabels(cols)
        hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
        self.tbl_hist.horizontalHeader().setFont(hfont)
        self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(len(df)):
            for j, col in enumerate(cols):
                val = str(df.iloc[i, j]); item = QTableWidgetItem(val)
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.tbl_hist.setItem(i, j, item)

    # ---- 8.25 camera controls ----
    def _connect_worker(self, w: VideoWorker):
        w.frameSignal.connect(self.on_frame)
        w.sceneSignal.connect(self.on_scene)
        w.roiSignal.connect(self.on_roi)
        w.infoSignal.connect(self.on_info)
        w.matchSignal.connect(self.on_match)
        w.histSignal.connect(self.refresh_history)
        w.playSoundSignal.connect(self.on_play_sound)

    # ---- 8.26 Hàm chung để khởi động camera (1 hoặc 2) ----
    def start_cam_generic(self, which: int):
        if not self.models.ok:
            QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
            return
        if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
        if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

        ocr_mode = self.current_ocr_mode
        default_api = API_MAP["DSHOW(Windows)"] 

        if which == 1:
            idx = int(self.spin_cam1.value())
            mode = "in" if self.lane1_dir=="VÀO" else "out"
            title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
            self.cam1_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
            self._connect_worker(self.cam1_worker); self.cam1_worker.start()
        else:
            idx = int(self.spin_cam2.value())
            mode = "in" if self.lane2_dir=="VÀO" else "out"
            title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
            self.cam2_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
            self._connect_worker(self.cam2_worker); self.cam2_worker.start()

    # ---- 8.27 Hàm chung để dừng camera (1 hoặc 2) ----
    def stop_cam_generic(self, which: int):
        worker = self.cam1_worker if which==1 else self.cam2_worker
        if worker and worker.isRunning():
            worker.stop(); worker.wait(1000)
        if which==1: self.cam1_worker = None; self.show_logo(1)
        else:        self.cam2_worker = None; self.show_logo(2)

    def start_cam1(self): self.start_cam_generic(1)
    def stop_cam1(self):  self.stop_cam_generic(1)
    def start_cam2(self): self.start_cam_generic(2)
    def stop_cam2(self):  self.stop_cam_generic(2)

    # ---- 8.28 Xử lý sự kiện đóng cửa sổ ----
    def closeEvent(self, event):
        try: self.stop_cam_generic(1); self.stop_cam_generic(2)
        except: pass
        super().closeEvent(event)

# ==================== 9. MAIN ====================
def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(); 
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

