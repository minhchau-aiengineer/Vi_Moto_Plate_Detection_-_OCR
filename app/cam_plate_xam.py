# # -*- coding: utf-8 -*-
# """
# PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini)
# - Lịch sử: GroupBox, tiêu đề cột đầy đủ, kéo giãn, Export Excel, Xóa (dòng chọn / tất cả) với dialog màu & bo góc.
# - Camera: căn giữa, bo góc, không in chữ lên ảnh; chỉ đổi tiêu đề card. Bật/tắt độc lập từng cam.
# - Khi tắt cam/reset -> hiện logo mặc định.
# - Điều khiển làn: 1 chiều (toggle), 2 chiều (đảo), Reset làn; đổi mode ghi nhận IN/OUT ngay cho worker.
# - OCR: YOLO mặc định; chọn "Dùng Gemini AI" -> chỉ thay bước OCR bằng Gemini, lưu/DB/hiển thị giữ nguyên.
# """


# # ==================== 1. IMPORTS ====================
# import os, sys, time, cv2 
# import numpy as np, pandas as pd
# from datetime import datetime

# # ---- 1.1 HiDPI ----
# os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
# os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint
# from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
#     QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
#     QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
#     QFileDialog, QHeaderView, QDialog, QSpacerItem
# )

# # ---- 1.2 SQL Server ---- 
# USE_SQL = True
# try:
#     import pyodbc
# except Exception:
#     USE_SQL = False

# # ---- 1.3 YOLO ----
# from ultralytics import YOLO

# # ---- 1.4 Gemini (optional) ----
# from dotenv import load_dotenv
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# GEMINI_READY = False
# try:
#     if GEMINI_API_KEY:
#         from google import generativeai as genai
#         from google.api_core import exceptions as gexceptions
#         from PIL import Image
#         genai.configure(api_key=GEMINI_API_KEY)
#         GEMINI_READY = True
# except Exception as _e:
#     print("Gemini init failed:", _e)
#     GEMINI_READY = False


# # ==================== 2. CONFIG ====================
# DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
# OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/epoch199.pt"
# SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
# LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")  

# # --- 2.1 DB Config ---
# CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=plates_db;"
#     "UID=sa;"
#     "PWD=123456"
# )

# # --- 2.2 UI Config ---
# PANEL_W, PANEL_H = 640, 360
# PANEL_BG = (232, 239, 248)  # BGR
# STABLE_SECONDS_IN  = 1.2
# STABLE_SECONDS_OUT = 1.2

# # --- 2.3 Other Config ---
# API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
# OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
#            "six":"6","seven":"7","eight":"8","nine":"9"}


# # ==================== 3. CÁC HÀM TIỆN ÍCH ====================
# # --- 3.1 XỬ LÝ ẢNH/VIDEO ---
# def letterbox(bgr, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
#     if bgr is None:
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     ih, iw = bgr.shape[:2]
#     if ih == 0 or iw == 0:
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     s = min(w/iw, h/ih); nw, nh = int(iw*s), int(ih*s)
#     resized = cv2.resize(bgr, (nw, nh))
#     canvas = np.full((h, w, 3), color, dtype=np.uint8)
#     top, left = (h-nh)//2, (w-nw)//2
#     canvas[top:top+nh, left:left+nw] = resized
#     return canvas

# # --- 3.2 ẢNH <-> QIMAGE ---
# def bgr_to_qimage(bgr):
#     if bgr is None:
#         bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     h, w, ch = rgb.shape
#     return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

# # --- 3.3 LƯU ẢNH VỚI TÊN THEO TIMESTAMP ---
# def save_image(img, prefix):
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
#     cv2.imwrite(path, img)
#     return path

# # --- 3.4 XỬ LÝ KÝ TỰ OCR ---
# def norm_char(x): 
#     return OCR_MAP.get(str(x), str(x))

# # --- 3.5 XỬ LÝ BIỂN SỐ ---
# def plate_norm(s: str) -> str: 
#     return (s or "").replace("-", "").replace(" ", "").upper()

# # --- 3.6 KIỂM TRA KẾT QUẢ YOLO CÓ BOXES ---
# def has_boxes(r):
#     try:
#         return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
#     except: return False

# # --- 3.7 TIỀN XỬ LÝ ẢNH CHO OCR ---
# def preprocess_for_ocr(roi):
#     if roi is None: return None
#     if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe,(3,3),0)
#     return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# # ==================== 4. DATABASE WRAPPER (BAO BỌC) ====================
# class DB:
#     # --- 4.1 KẾT NỐI & TẠO BẢNG ---
#     def __init__(self, conn_str: str):
#         self.ok = False; self.conn = None; self.cur  = None
#         if not USE_SQL: return
#         try:
#             self.conn = pyodbc.connect(conn_str, autocommit=True)
#             self.cur  = self.conn.cursor()
#             self.cur.execute("""
#                 IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
#                 CREATE TABLE dbo.ParkingSessions(
#                     id INT IDENTITY(1,1) PRIMARY KEY,
#                     plate_in NVARCHAR(64)  NULL,
#                     date_in  NVARCHAR(16)  NULL,
#                     time_in  NVARCHAR(16)  NULL,
#                     image_in NVARCHAR(255) NULL,
#                     plate_out NVARCHAR(64)  NULL,
#                     date_out  NVARCHAR(16)  NULL,
#                     time_out  NVARCHAR(16)  NULL,
#                     image_out NVARCHAR(255) NULL,
#                     match_status NVARCHAR(32) NULL,
#                     created_at DATETIME DEFAULT GETDATE()
#                 );
#             """)
#             self.ok = True
#         except Exception as e:
#             print("DB connect error:", e); self.ok = False

#     # --- 4.2 CHÈN BIỂN SỐ VÀO ---
#     def insert_in(self, plate, d, t, img_path):
#         if not self.ok: return
#         try:
#             self.cur.execute("""
#                 INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
#                 VALUES (?,?,?,?,?)
#             """, (plate, d, t, img_path, 'PENDING'))
#         except Exception as e: print("insert_in error:", e)

#     # --- 4.3 GẮN BIỂN SỐ RA VÀO BẢNG, TRẢ VỀ KẾT QUẢ KHỚP/KO KHỚP ---
#     def attach_out(self, plate_out, d, t, img_path) -> str:
#         if not self.ok: return "Khong khop bien so"
#         try:
#             rows = self.cur.execute("""
#                 SELECT TOP 50 id, plate_in FROM dbo.ParkingSessions
#                 WHERE plate_out IS NULL
#                 ORDER BY id DESC
#             """).fetchall()
#             match_sid = None
#             for sid, plate_in in rows:
#                 if plate_norm(plate_in) == plate_norm(plate_out):
#                     match_sid = sid; break
#             if match_sid:
#                 self.cur.execute("""
#                     UPDATE dbo.ParkingSessions
#                     SET plate_out=?, date_out=?, time_out=?, image_out=?, match_status='KHOP-BIEN-SO'
#                     WHERE id=?
#                 """, (plate_out, d, t, img_path, match_sid))
#                 return "Khop bien so"
#             else:
#                 self.cur.execute("""
#                     INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
#                     VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO')
#                 """, (plate_out, d, t, img_path))
#                 return "Khong khop bien so"
#         except Exception as e:
#             print("attach_out error:", e); return "Khong khop bien so"

#     # --- 4.4 LẤY DỮ LIỆU LỊCH SỬ DƯỚI DẠNG DATAFRAME PANDAS ---
#     def fetch_history_df(self, limit=10000) -> pd.DataFrame:
#         if not self.ok:
#             return pd.DataFrame(columns=[
#                 "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                 "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#             ])
#         try:
#             rows = self.cur.execute(f"""
#                 SELECT TOP {limit}
#                     id, image_in, plate_in, date_in, time_in,
#                     image_out, plate_out, date_out, time_out, match_status
#                 FROM dbo.ParkingSessions
#                 ORDER BY id DESC
#             """).fetchall()
#             df = pd.DataFrame.from_records(
#                 rows,
#                 columns=["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                          "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             ).astype(object).where(pd.notnull, "")
#             df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#             df.insert(0, "STT", range(1, len(df)+1))
#             return df
#         except Exception as e:
#             print("fetch_history error:", e)
#             return pd.DataFrame(columns=[
#                 "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                 "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#             ])

#     # --- 4.5 XÓA BẢN GHI ---
#     def delete_by_ids(self, ids):
#         if not self.ok or not ids: return
#         try:
#             for sid in ids:
#                 self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
#         except Exception as e: print("delete_by_ids error:", e)

#     # --- 4.6 XÓA TẤT CẢ BẢN GHI ---
#     def delete_all(self):
#         if not self.ok: return
#         try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
#         except Exception as e: print("delete_all error:", e)

# # ==================== 5. YOLO/GEMINI WRAPPERS(BAO BỌC) ====================
# class Models:
#     # --- 5.1 KHỞI TẠO MODEL ---
#     def __init__(self, det_path: str, ocr_path: str):
#         self.ok = True; self.err = ""
#         try:
#             self.det = YOLO(det_path)
#             self.ocr = YOLO(ocr_path)
#         except Exception as e:
#             self.ok = False; self.err = str(e)

#     # --- 5.2 PHÁT HIỆN BIỂN SỐ TRONG ẢNH ---
#     def detect_plates(self, frame):
#         plates, boxed = [], frame.copy()
#         for r in self.det(frame):
#             if not has_boxes(r): continue
#             xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
#             for (x1,y1,x2,y2) in xyxy:
#                 pad=8
#                 x1=max(0,x1-pad); y1=max(0,y1-pad)
#                 x2=min(boxed.shape[1]-1,x2+pad); y2=min(boxed.shape[0]-1,y2+pad)
#                 roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
#                 plates.append(((x1,y1,x2,y2), roi))
#                 cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
#                 cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
#         return plates, boxed

#     # --- 5.3 OCR BIỂN SỐ ---
#     def ocr_plate_yolo(self, roi):
#         roi_pre = preprocess_for_ocr(roi)
#         res = self.ocr(roi_pre if roi_pre is not None else roi)
#         text_raw=""
#         for r in res:
#             if not has_boxes(r): continue
#             names = getattr(r,'names',None) or getattr(self.ocr,'names',{}) or {}
#             clses = r.boxes.cls.cpu().numpy().astype(int)
#             xyxys= r.boxes.xyxy.cpu().numpy()
#             boxes=[]
#             for i,cls in enumerate(clses):
#                 x1,y1,x2,y2 = xyxys[i]
#                 cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
#                 ch = norm_char(names.get(cls, str(cls)) if isinstance(names,dict) else str(cls))
#                 if ch.isdigit() or (ch.isalpha() and ch.isupper()):
#                     boxes.append((cy,cx,ch))
#             if not boxes: continue
#             ys=[b[0] for b in boxes]
#             if len(boxes)<=7 or (max(ys)-min(ys) < 0.2*max(ys, default=1)):
#                 text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
#             else:
#                 thr=(max(ys)+min(ys))/2.0
#                 l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
#                 t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
#                 t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
#                 text_raw=f"{t1}-{t2}" if t2 else t1
#         return self._format_text(text_raw)

#     # --- 5.4 OCR BIỂN SỐ DÙNG GEMINI ---
#     def ocr_plate_gemini_from_path(self, image_path: str):
#         if not GEMINI_READY: return "", ""
#         try:
#             img = Image.open(image_path)
#         except Exception as e:
#             print("Gemini open image error:", e); return "", ""
#         try:
#             model = genai.GenerativeModel('gemini-2.5-flash')
#             prompt = ("Đây là ảnh biển số xe Việt Nam. "
#                       "Hãy trích xuất CHÍNH XÁC chuỗi biển số và chỉ trả về chuỗi đó. "
#                       "VD: '29-P1 123.45' hoặc '50-Z8 888.88'.")
#             resp = model.generate_content([prompt, img])
#             raw = (resp.text or "").strip().replace("\n", " ")
#             return self._format_text(raw)
#         except gexceptions.GoogleAPICallError as e:
#             print("Gemini API error:", e); return "", ""
#         except Exception as e:
#             print("Gemini unknown error:", e); return "", ""

#     # --- 5.5 FORMAT TEXT ---
#     @staticmethod
#     def _format_text(text_raw: str):
#         raw=(text_raw or '').replace('-', ' ').replace(' ', '')
#         text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
#         return text_fmt, (text_raw or "")

# # ==================== 6. VIDEO WORKER THREAD ====================
# class VideoWorker(QThread):
#     # --- 6.1 TÍN HIỆU ---
#     frameSignal = Signal(np.ndarray, str)
#     sceneSignal = Signal(str)
#     roiSignal   = Signal(str, str)
#     infoSignal  = Signal(dict)
#     matchSignal = Signal(str)
#     histSignal  = Signal()

#     # --- 6.2 KHỞI TẠO WORKER ---
#     def __init__(self, cam_idx: int, api: int, mode: str, models: Models, db: DB,
#                  stable_seconds: float = 1.2, ocr_mode: str = "yolo", title: str = "", parent=None):
#         super().__init__(parent)
#         self.cam_idx = cam_idx
#         self.api = api
#         self.mode = mode            # 'in' | 'out'
#         self.models = models
#         self.db = db
#         self.stable_seconds = stable_seconds
#         self.ocr_mode = ocr_mode    # 'yolo' | 'gemini'
#         self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")

#         self._running = False
#         self.cap = None
#         self.stable_start = 0.0
#         self.captured = False
#     # --- 6.3 SETTERS ---
#     def set_title(self, title: str): self.title = title
#     def set_ocr_mode(self, mode: str): self.ocr_mode = mode
#     def set_mode(self, mode: str): self.mode = mode

#     # --- 6.4 CHẠY WORKER ---
#     def run(self):
#         self._running = True
#         self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
#         if not (self.cap and self.cap.isOpened()):
#             self._running = False; return
#         try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         except: pass
#         try: self.cap.set(cv2.CAP_PROP_FPS, 30)
#         except: pass

#         while self._running:
#             ok, frame = self.cap.read()
#             if not ok:
#                 self.stable_start = 0.0; self.captured = False
#                 time.sleep(0.03); continue

#             # KHÔNG vẽ chữ lên ảnh, chỉ letterbox
#             disp = letterbox(frame)
#             self.frameSignal.emit(disp, self.title)

#             plates, boxed = self.models.detect_plates(frame)
#             if not plates:
#                 self.stable_start = 0.0; self.captured = False
#                 time.sleep(0.01); continue

#             best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
#             roi_current = best[1]

#             if self.stable_start == 0.0 or self.captured:
#                 self.stable_start = time.time(); self.captured = False

#             if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
#                 scene_path = save_image(boxed if boxed is not None else frame,
#                                         "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
#                 roi_path   = save_image(roi_current,
#                                         "plate_in_roi" if self.mode=="in" else "plate_out_roi")

#                 if self.ocr_mode == "gemini" and GEMINI_READY:
#                     text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
#                 else:
#                     text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

#                 if text_fmt or text_raw:
#                     now = datetime.now()
#                     d = now.strftime("%d/%m/%Y")
#                     t = now.strftime("%H:%M:%S")
#                     plate = text_fmt or text_raw

#                     self.sceneSignal.emit(scene_path)
#                     self.roiSignal.emit(roi_path, self.mode)

#                     if self.mode == "in":
#                         self.infoSignal.emit({"date_in": d, "time_in": t, "plate_text_in": plate})
#                         if self.db and self.db.ok:
#                             self.db.insert_in(plate, d, t, scene_path)
#                             self.histSignal.emit()
#                     else:
#                         self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
#                         if self.db and self.db.ok:
#                             match = self.db.attach_out(plate, d, t, scene_path)
#                             self.matchSignal.emit(match)
#                             self.histSignal.emit()
#                     self.captured = True

#             time.sleep(0.01)

#         try:
#             if self.cap: self.cap.release()
#         except: pass

#     # --- 6.5 DỪNG WORKER ---
#     def stop(self): 
#         self._running = False


# # ==================== 6. DELETE DIALOG(HỘI THOẠI) ====================
# class DeleteDialog(QDialog):
#     # --- 6.1 KHỞI TẠO DIALOG ---
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Xóa lịch sử")
#         self.setModal(True)
#         self.setStyleSheet("""
#             QDialog {
#                 background: #ffffff;
#                 border: 2px solid #e6e6e6;
#                 border-radius: 10px;
#             }
#             QPushButton {
#                 height: 34px; border-radius: 10px; font-weight: 600; padding: 6px 12px;
#             }
#         """)

#         # --- 6.2 XÂY DỰNG GIAO DIỆN ---
#         lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
#         lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
#         lay.addWidget(lab)

#         # --- 6.3 NÚT BẤM ---
#         row = QHBoxLayout(); row.setSpacing(12)
#         self.btn_sel = QPushButton("Xóa dòng đã chọn"); self.btn_sel.setStyleSheet("background:#dbeafe;")
#         self.btn_all = QPushButton("Xóa tất cả");       self.btn_all.setStyleSheet("background:#ffe0e0;")
#         self.btn_can = QPushButton("Hủy");              self.btn_can.setStyleSheet("background:#fff9c4;")
#         row.addWidget(self.btn_sel, 1)
#         row.addWidget(self.btn_all, 1)
#         row.addWidget(self.btn_can, 1)
#         lay.addLayout(row)

#         self.btn_sel.clicked.connect(lambda: self.done(1))
#         self.btn_all.clicked.connect(lambda: self.done(2))
#         self.btn_can.clicked.connect(lambda: self.done(0))

# # ==================== 7. MAIN WINDOW ====================
# class MainWindow(QMainWindow):
#     # --- 7.1 KHỞI TẠO MAIN WINDOW ---
#     def __init__(self):
#         super().__init__()
#         # self.setWindowIcon(QIcon(LOGO_PATH))
#         self.setWindowTitle("APP GIỮ XE (PySide6)")
#         self.setMinimumSize(1400, 900)

#         self._init_theme()

#         self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#         self.db = DB(CONN_STR) if USE_SQL else DB("")

#         self.cam1_worker = None
#         self.cam2_worker = None

#         # Làn + OCR mode
#         self.lane1_dir = "IN"; self.lane2_dir = "IN"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.current_ocr_mode = "yolo"

#         self._build_ui()
#         self.show_logo(1); self.show_logo(2)

#         self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.refresh_history); self.hist_timer.start(5000)

#     # --- 7.2 GIAO DIỆN THEME ---
#     def _init_theme(self):
#         self.setStyleSheet("""
#         * { color: #000000; }
#         QMainWindow, QWidget { background: #ffffff; }
#         QWidget#SideBar { background: #ffffff; }

#         QGroupBox {
#             background: #ffffff;
#             font-weight: 600;
#             border: 2px solid #e6e6e6;
#             border-radius: 12px;
#             margin-top: 8px;
#             padding-top: 10px;
#         }
#         QGroupBox::title {
#             subcontrol-origin: margin;
#             left: 10px;
#             padding: 0 6px;
#             background: #ffffff;
#         }

#         QFrame[class="card-wrap"] { 
#             background: #e6e6e6; 
#             border-radius: 14px; 
#         }
#         QFrame[class="card"] { 
#             background: #ffffff; 
#             border-radius: 12px; 
#         }
#         QFrame[class="title-wrap"]{ 
#             background: #e6e6e6; 
#             border-radius: 12px; 
#         }
#         QLabel[class="title"] {
#             font: 700 18px "Segoe UI";
#             padding: 6px 10px;
#             background: #ffffff;
#             border-radius: 10px;
#         }

#         QPushButton { 
#             height: 34px; 
#             border-radius: 12px; 
#             font-weight: 600; 
#             padding: 6px 12px;
#         }
#         QPushButton#btnGreen { 
#             background: #d1fadf;
#             border: 1px solid #a6f4c5; 
#         }
#         QPushButton#btnRed { 
#             background: #ffe0e0; 
#             border: 1px solid #ffb3b3; 
#         }
#         QPushButton#btnYellow { 
#             background: #fff3bf;
#             border: 1px solid #ffe066; 
#         }
#         QLineEdit {
#             height: 28px; 
#             background: #ffffff; 
#             border: 1px solid #e0e0e0; 
#             border-radius: 8px; 
#             padding: 2px 6px;
#         }
#         QTableWidget { 
#             background: #ffffff;
#             gridline-color: #e6e6e6; 
#         }
#         """)

#     # --- 7.3 TẠO CARD GIAO DIỆN ---
#     def _make_card(self, title:str, content:QWidget):
#         wrap = QFrame(); wrap.setProperty("class","card-wrap")
#         wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
#         card = QFrame(); card.setProperty("class","card")
#         v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
#         title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
#         hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
#         title_lbl = QLabel(title); title_lbl.setProperty("class","title")
#         hl.addWidget(title_lbl)
#         v.addWidget(title_wrap)
#         v.addWidget(content, 1)
#         wrapL.addWidget(card)
#         return wrap, title_lbl

#     # --- 7.4 XÂY DỰNG GIAO DIỆN CHÍNH ---
#     def _build_ui(self):
#         central = QWidget(); self.setCentralWidget(central)
#         root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

#         # ===== THANH BÊN =====
#         side = QWidget(objectName="SideBar"); side.setMinimumWidth(450)
#         vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

#         # ===== ĐIỀU KHIỂN CAMERA =====
#         # --- Camera Control GroupBox ---
#         gb_camctl = QGroupBox("CAMERA CONTROL")
#         gl_camctl = QGridLayout(gb_camctl)
#         self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
#         self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys()))
#         self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
#         self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys()))

#         # --- Nút bấm ---
#         self.btn_start1 = QPushButton("Bật Cam 1"); self.btn_start1.setObjectName("btnGreen")
#         self.btn_stop1  = QPushButton("Tắt Cam 1"); self.btn_stop1.setObjectName("btnRed")
#         self.btn_start2 = QPushButton("Bật Cam 2"); self.btn_start2.setObjectName("btnGreen")
#         self.btn_stop2  = QPushButton("Tắt Cam 2"); self.btn_stop2.setObjectName("btnRed")
#         self.btn_start1.clicked.connect(self.start_cam1)
#         self.btn_stop1.clicked.connect(self.stop_cam1)
#         self.btn_start2.clicked.connect(self.start_cam2)
#         self.btn_stop2.clicked.connect(self.stop_cam2)
#         r=0

#         # --- Bố trí lưới ---
#         gl_camctl.addWidget(QLabel("Index Cam 1"), r,0); gl_camctl.addWidget(self.spin_cam1, r,1)
#         gl_camctl.addWidget(QLabel("Backend Cam 1"), r,2); gl_camctl.addWidget(self.cb_api1, r,3); r+=1
#         gl_camctl.addWidget(QLabel("Index Cam 2"), r,0); gl_camctl.addWidget(self.spin_cam2, r,1)
#         gl_camctl.addWidget(QLabel("Backend Cam 2"), r,2); gl_camctl.addWidget(self.cb_api2, r,3); r+=1
#         gl_camctl.addWidget(self.btn_start1, r,0,1,2); gl_camctl.addWidget(self.btn_stop1, r,2,1,2); r+=1
#         gl_camctl.addWidget(self.btn_start2, r,0,1,2); gl_camctl.addWidget(self.btn_stop2, r,2,1,2)
#         vside.addWidget(gb_camctl)

#         # ===== ĐIỀU KHIỂN LÀN =====
#         # --- Làn Control GroupBox ---
#         gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
#         vl_lane = QVBoxLayout(gb_lane)
#         self.lbl_lane1 = QLabel("")
#         self.lbl_lane2 = QLabel("")
#         lane_info = QLabel("Mỗi làn: cam trước: cam1, cam sau: cam2"); lane_info.setStyleSheet("font-style: italic;")

#         # --- Nút bấm ---
#         btns_row = QHBoxLayout()
#         self.btn_oneway = QPushButton("1 chiều")
#         self.btn_twoway = QPushButton("2 chiều")
#         self.btn_reset_lane = QPushButton("Reset làn xe"); self.btn_reset_lane.setObjectName("btnYellow")
#         self.btn_oneway.setStyleSheet("background:#dbeafe; font-weight:600;")
#         self.btn_twoway.setStyleSheet("background:#dbeafe; font-weight:600;")
#         self.btn_reset_lane.setStyleSheet("font-weight:600;")
#         self.btn_oneway.clicked.connect(self.on_one_way_clicked)
#         self.btn_twoway.clicked.connect(self.on_two_way_clicked)
#         self.btn_reset_lane.clicked.connect(self.on_reset_lanes)
#         btns_row.addWidget(self.btn_oneway); btns_row.addWidget(self.btn_twoway); btns_row.addWidget(self.btn_reset_lane)

#         vl_lane.addWidget(self.lbl_lane1); vl_lane.addWidget(self.lbl_lane2); vl_lane.addWidget(lane_info); vl_lane.addLayout(btns_row)
#         vside.addWidget(gb_lane)

#         # ===== CHỌN MODEL OCR =====
#         # --- OCR Control GroupBox ---
#         gb_ocr = QGroupBox("OCR MODEL")
#         vb_ocr = QVBoxLayout(gb_ocr)

#         # --- Radio buttons ---
#         self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)"); self.rb_yolo.setChecked(True)
#         self.rb_gem  = QRadioButton("Dùng Gemini AI")
#         vb_ocr.addWidget(self.rb_yolo); vb_ocr.addWidget(self.rb_gem)
#         self.rb_yolo.toggled.connect(self.on_ocr_mode_changed)
#         self.rb_gem.toggled.connect(self.on_ocr_mode_changed)
#         if not GEMINI_READY:
#             self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY trong môi trường/.env → mặc định dùng YOLO")
#         vside.addWidget(gb_ocr)

#         # ===== THÔNG TIN XE VÀO =====
#         # --- THÔNG TIN XE VÀO GroupBox ---
#         gb_in = QGroupBox("THÔNG TIN XE VÀO")
#         gl_in = QGridLayout(gb_in)
#         self.ed_date_in = QLineEdit(); self.ed_time_in = QLineEdit(); self.ed_plate_in = QLineEdit()

#         # --- Bố trí lưới ---
#         gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
#         gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
#         gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
#         vside.addWidget(gb_in)

#         # ===== THÔNG TIN XE RA =====
#         # --- THÔNG TIN XE RA GroupBox ---
#         gb_out = QGroupBox("THÔNG TIN XE RA")
#         gl_out = QGridLayout(gb_out)
#         self.ed_date_out = QLineEdit(); self.ed_time_out = QLineEdit(); self.ed_plate_out = QLineEdit()

#         # --- Bố trí lưới ---
#         gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
#         gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
#         gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
#         vside.addWidget(gb_out)

#         # ==== BẢNG LỊCH SỬ =====
#         # --- LỊCH SỬ GroupBox ---
#         gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
#         v_hist_btns = QVBoxLayout(gb_hist_btns)

#         # --- Nút bấm ---
#         self.btn_show_history = QPushButton("Xem bảng lịch sử"); self.btn_show_history.setStyleSheet("background:#E6F4EA; font-weight:600;")
#         row_cmd = QHBoxLayout()
#         self.btn_export_hist  = QPushButton("Export Excel"); self.btn_export_hist.setStyleSheet("background:#dbeafe; font-weight:600;")
#         self.btn_delete_hist  = QPushButton("Xóa bảng");      self.btn_delete_hist.setStyleSheet("background:#dbeafe; font-weight:600;")
#         row_cmd.addWidget(self.btn_export_hist, 1); row_cmd.addWidget(self.btn_delete_hist, 1)
#         v_hist_btns.addWidget(self.btn_show_history); v_hist_btns.addLayout(row_cmd)
#         self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide(); self.btn_hide_history.setStyleSheet("background:#FCE8E6; font-weight:600;")
#         v_hist_btns.addWidget(self.btn_hide_history)
#         self.btn_show_history.clicked.connect(self.show_history_view)
#         self.btn_hide_history.clicked.connect(self.show_main_view)
#         self.btn_export_hist.clicked.connect(self.on_export_excel)
#         self.btn_delete_hist.clicked.connect(self.on_delete_history)
#         vside.addWidget(gb_hist_btns)

#         vside.addStretch(1)
#         root.addWidget(side)

#         # ===== PHẦN HIỂN THỊ CHÍNH =====
#         right_container = QVBoxLayout()

#         # ===== Main view =====
#         self.main_view = QWidget()
#         main_layout = QVBoxLayout(self.main_view)

#         # ==== Video panels =====
#         top = QHBoxLayout()
#         self.lbl_cam1 = QLabel(); self.lbl_cam1.setMinimumSize(PANEL_W, PANEL_H)
#         self.lbl_cam2 = QLabel(); self.lbl_cam2.setMinimumSize(PANEL_W, PANEL_H)
#         for lbl in (self.lbl_cam1, self.lbl_cam2):
#             lbl.setScaledContents(False)
#             lbl.setAlignment(Qt.AlignCenter)  # căn giữa pixmap
#             lbl.setStyleSheet("border-radius:12px; background:#f7efe8;")
#         cam1_card, self.cam1_title = self._make_card("Cam 1 (IN)", self.lbl_cam1)
#         cam2_card, self.cam2_title = self._make_card("Cam 2 (IN)", self.lbl_cam2)
#         top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1)
#         main_layout.addLayout(top)

#         # ==== Scene & ROI panels =====
#         bottom = QHBoxLayout()
#         self.lbl_scene = QLabel(); self.lbl_scene.setMinimumSize(PANEL_W, PANEL_H); self.lbl_scene.setScaledContents(False); self.lbl_scene.setAlignment(Qt.AlignCenter); self.lbl_scene.setStyleSheet("border-radius:12px; background:#fff;")
#         self.lbl_roi   = QLabel(); self.lbl_roi.setMinimumSize(PANEL_W, PANEL_H); self.lbl_roi.setScaledContents(False); self.lbl_roi.setAlignment(Qt.AlignCenter); self.lbl_roi.setStyleSheet("border-radius:12px; background:#fff;")
#         scene_card, _ = self._make_card("Image_BOX",  self.lbl_scene)
#         roi_card,   _ = self._make_card("ROI_Plates", self.lbl_roi)
#         bottom.addWidget(scene_card, 1); bottom.addWidget(roi_card, 1)
#         main_layout.addLayout(bottom)

#         # ==== Thông tin chi tiết =====
#         self.info_group = QGroupBox("Thông tin chi tiết")
#         info_layout = QGridLayout(self.info_group)
#         self.txt_date_in  = QLabel("--/--/----"); self.txt_time_in  = QLabel("--:--:--")
#         self.txt_plate_in = QLabel("---"); self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_date_out = QLabel("--/--/----"); self.txt_time_out = QLabel("--:--:--")
#         self.txt_plate_out= QLabel("---"); self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_match    = QLabel("")
#         r=0
        
#         # --- Bố trí lưới ---
#         info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
#         info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
#         info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
#         info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
#         info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
#         info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
#         info_layout.addWidget(QLabel("So khớp:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,5)
#         main_layout.addWidget(self.info_group)

#         # ===== History view =====
#         self.history_view = QWidget()
#         hist_layout = QVBoxLayout(self.history_view)
#         hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)")
#         hist_v = QVBoxLayout(hist_group)

#         # ---- Bảng lịch sử ----
#         self.tbl_hist = QTableWidget(0, 10)
#         self.tbl_hist.setHorizontalHeaderLabels(["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"])
#         header = self.tbl_hist.horizontalHeader()
#         hfont = QFont(header.font()); hfont.setBold(True); header.setFont(hfont)
#         self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
#         self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
#         self.tbl_hist.setAlternatingRowColors(True)
#         header.setSectionResizeMode(QHeaderView.Stretch)
#         hist_v.addWidget(self.tbl_hist)
#         hist_layout.addWidget(hist_group)

#         # ===== Stacked Widget =====
#         self.stacked = QStackedWidget()
#         self.stacked.addWidget(self.main_view)
#         self.stacked.addWidget(self.history_view)
#         self.stacked.setCurrentIndex(0)
#         right_container.addWidget(self.stacked, 1)
#         root.addLayout(right_container, 1)

#         self.update_lane_labels()
#         self.update_titles_and_modes()

#     # ---------- LANE MODE ----------
#     def update_lane_labels(self):
#         self.lbl_lane1.setText(f"Làn 1: {self.lane1_dir} (cam trước: cam1 / cam sau: cam2)")
#         self.lbl_lane2.setText(f"Làn 2: {self.lane2_dir} (cam trước: cam1 / cam sau: cam2)")

#     # --------- UPDATE TITLES AND MODES ----------
#     def update_titles_and_modes(self):
#         self.cam1_title.setText(f"Cam 1 ({'IN' if self.lane1_dir=='IN' else 'OUT'})")
#         self.cam2_title.setText(f"Cam 2 ({'OUT' if self.lane2_dir=='IN' else 'UOT'})")
#         if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="IN" else "out")
#         if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="IN" else "out")

#     # ---------- RESET LANES ----------
#     @Slot() 
#     def on_reset_lanes(self):
#         self.lane1_dir = "IN"; self.lane2_dir = "IN"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.update_lane_labels(); self.update_titles_and_modes()
#         # có yêu cầu: reset hiển thị logo
#         self.show_logo(1); self.show_logo(2)

#     # --------- ONE WAY CLICKED ----------
#     @Slot() 
#     def on_one_way_clicked(self):
#         if self.one_way_toggle_vao: self.lane1_dir="IN"; self.lane2_dir="IN"
#         else:                       self.lane1_dir="OUT";  self.lane2_dir="OUT"
#         self.one_way_toggle_vao = not self.one_way_toggle_vao
#         self.update_lane_labels(); self.update_titles_and_modes()

#     # --------- TWO WAY CLICKED ----------
#     @Slot() 
#     def on_two_way_clicked(self):
#         if self.two_way_toggle: self.lane1_dir="IN"; self.lane2_dir="OUT"
#         else:                   self.lane1_dir="OUT";  self.lane2_dir="IN"
#         self.two_way_toggle = not self.two_way_toggle
#         self.update_lane_labels(); self.update_titles_and_modes()

#     # --------- OCR MODE CHANGED ----------
#     @Slot()
#     def on_ocr_mode_changed(self):
#         self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
#         if self.rb_gem.isChecked() and not GEMINI_READY:
#             QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
#             self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
#         if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
#         if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

#     # ---------- SHOW / HIDE HISTORY VIEW ----------
#     @Slot()
#     def show_history_view(self):
#         self.stacked.setCurrentIndex(1); self.btn_show_history.hide(); self.btn_hide_history.show(); self.refresh_history()

#     # ---------- SHOW / HIDE MAIN VIEW ----------
#     @Slot()
#     def show_main_view(self):
#         self.stacked.setCurrentIndex(0); self.btn_hide_history.hide(); self.btn_show_history.show()

#     # ---------- EXPORT EXCEL ----------
#     @Slot()
#     def on_export_excel(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
#         if df.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
#         path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
#         if not path: return
#         try: df.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
#         except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

#     # ---------- DELETE HISTORY ----------
#     @Slot()
#     def on_delete_history(self):
#         if not (self.db and self.db.ok):
#             QMessageBox.warning(self, "Xóa", "Chưa kết nối DB, không thể xóa."); return
#         dlg = DeleteDialog(self)
#         # mở dialog lệch bên phải
#         g = self.geometry(); dlg.adjustSize()
#         dlg.move(self.mapToGlobal(QPoint(g.width()-dlg.width()-40, 140)))
#         res = dlg.exec()
#         if res == 1:
#             rows = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
#             if not rows: QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào."); return
#             cols = [self.tbl_hist.horizontalHeaderItem(i).text() for i in range(self.tbl_hist.columnCount())]
#             if "ID" not in cols: QMessageBox.warning(self, "Xóa", "Không tìm thấy cột ID."); return
#             id_col = cols.index("ID"); ids = []
#             for r in rows:
#                 item = self.tbl_hist.item(r, id_col)
#                 if item: ids.append(item.text())
#             if not ids: QMessageBox.information(self, "Xóa", "Không lấy được ID các dòng chọn."); return
#             self.db.delete_by_ids(ids); self.refresh_history()
#         elif res == 2:
#             self.db.delete_all(); self.refresh_history()
#         else:
#             return

#     # ---------- SHOW LOGO ----------
#     def qpix_logo(self):
#         if os.path.exists(LOGO_PATH):
#             pm = QPixmap(LOGO_PATH).scaled(PANEL_W, PANEL_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         else:
#             # fallback canvas rỗng
#             pm = QPixmap.fromImage(bgr_to_qimage(letterbox(None)))
#         return pm

#     # --------- SHOW LOGO ----------
#     @Slot(int)
#     def show_logo(self, which: int):
#         pm = self.qpix_logo()
#         if which == 1: self.lbl_cam1.setPixmap(pm)
#         else: self.lbl_cam2.setPixmap(pm)

#     # ---------- SLOT HANDLERS ----------
#     @Slot(np.ndarray, str)
#     def on_frame(self, frame_bgr, title):
#         qimg = bgr_to_qimage(letterbox(frame_bgr))
#         sender = self.sender()
#         if sender is self.cam1_worker:
#             self.lbl_cam1.setPixmap(QPixmap.fromImage(qimg))
#         elif sender is self.cam2_worker:
#             self.lbl_cam2.setPixmap(QPixmap.fromImage(qimg))

#     # --------- SCENE HANDLER ----------
#     @Slot(str)
#     def on_scene(self, path):
#         if os.path.exists(path):
#             bgr = cv2.imread(path)
#             self.lbl_scene.setPixmap(QPixmap.fromImage(bgr_to_qimage(letterbox(bgr))))

#     # --------- ROI HANDLER ----------
#     @Slot(str, str)
#     def on_roi(self, roi_path, mode):
#         if os.path.exists(roi_path):
#             bgr = cv2.imread(roi_path)
#             self.lbl_roi.setPixmap(QPixmap.fromImage(bgr_to_qimage(letterbox(bgr))))

#     # --------- INFO HANDLER ----------
#     @Slot(dict)
#     def on_info(self, info):
#         if "date_in" in info:  self.txt_date_in.setText(info["date_in"]);   self.ed_date_in.setText(info["date_in"])
#         if "time_in" in info:  self.txt_time_in.setText(info["time_in"]);   self.ed_time_in.setText(info["time_in"])
#         if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
#         if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
#         if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
#         if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

#     # --------- MATCH HANDLER ----------
#     @Slot(str)
#     def on_match(self, txt): self.txt_match.setText(txt.upper())

#     # ---------- REFRESH HISTORY ----------
#     @Slot()
#     def refresh_history(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
#         if df.empty:
#             self.tbl_hist.setRowCount(0)
#             cols = ["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             self.tbl_hist.setColumnCount(len(cols)); self.tbl_hist.setHorizontalHeaderLabels(cols)
#             hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#             self.tbl_hist.horizontalHeader().setFont(hfont); return

#         cols = list(df.columns)
#         self.tbl_hist.setRowCount(len(df)); self.tbl_hist.setColumnCount(len(cols))
#         self.tbl_hist.setHorizontalHeaderLabels(cols)
#         hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#         self.tbl_hist.horizontalHeader().setFont(hfont)
#         self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
#         for i in range(len(df)):
#             for j, col in enumerate(cols):
#                 val = str(df.iloc[i, j]); item = QTableWidgetItem(val)
#                 item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
#                 self.tbl_hist.setItem(i, j, item)

#     # ---------- CONNECT WORKER ----------
#     def _connect_worker(self, w: VideoWorker):
#         w.frameSignal.connect(self.on_frame)
#         w.sceneSignal.connect(self.on_scene)
#         w.roiSignal.connect(self.on_roi)
#         w.infoSignal.connect(self.on_info)
#         w.matchSignal.connect(self.on_match)
#         w.histSignal.connect(self.refresh_history)

#     # ---------- START / STOP CAMERAS ----------
#     def start_cam_generic(self, which: int):
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#             return
#         if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
#         if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

#         ocr_mode = self.current_ocr_mode
#         if which == 1:
#             idx = int(self.spin_cam1.value()); api = API_MAP[self.cb_api1.currentText()]
#             mode = "in" if self.lane1_dir=="IN" else "out"
#             title = f"Cam 1 ({'IN' if mode=='in' else 'OUT'})"
#             self.cam1_worker = VideoWorker(idx, api, mode, self.models, self.db, STABLE_SECONDS_IN, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam1_worker); self.cam1_worker.start()
#         else:
#             idx = int(self.spin_cam2.value()); api = API_MAP[self.cb_api2.currentText()]
#             mode = "in" if self.lane2_dir=="OUT" else "out"
#             title = f"Cam 2 ({'IN' if mode=='in' else 'OUT'})"
#             self.cam2_worker = VideoWorker(idx, api, mode, self.models, self.db, STABLE_SECONDS_OUT, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam2_worker); self.cam2_worker.start()

#     # --------- STOP CAMERAS ----------
#     def stop_cam_generic(self, which: int):
#         worker = self.cam1_worker if which==1 else self.cam2_worker
#         if worker and worker.isRunning():
#             worker.stop(); worker.wait(1000)
#         if which==1: self.cam1_worker = None; self.show_logo(1)
#         else:        self.cam2_worker = None; self.show_logo(2)

#     # --------- START / STOP CAM 1/2 ----------
#     def start_cam1(self): self.start_cam_generic(1)
#     def stop_cam1(self):  self.stop_cam_generic(1)
#     def start_cam2(self): self.start_cam_generic(2)
#     def stop_cam2(self):  self.stop_cam_generic(2)

#     # ---------- CLOSE EVENT ----------
#     def closeEvent(self, event):
#         try: self.stop_cam_generic(1); self.stop_cam_generic(2)
#         except: pass
#         super().closeEvent(event)

# # ===================== 8. MAIN ====================
# def main():
#     QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
#     app = QApplication(sys.argv)
#     w = MainWindow(); w.show()
#     sys.exit(app.exec())

# # ===================== RUN MAIN ====================
# if __name__ == "__main__":
#     main()









# # ------------------------------------------------------------------------------------------------------------------------------










# # -*- coding: utf-8 -*-
# """
# PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini)
# - Sidebar chuyển Main/History.
# - Bảng lịch sử: tiêu đề in đậm, kéo giãn, bỏ STT khi hiển thị; Export Excel, Xóa bảng (dòng chọn / tất cả).
# - Điều khiển làn: 1 chiều (toggle), 2 chiều (đảo), Reset làn; đổi **tiêu đề card** camera; KHÔNG vẽ text vào ảnh.
# - Đổi làn => đổi hướng ghi nhận (IN/OUT) cho cam 1/2.
# - OCR:
#     + "Dùng YOLO OCR (tự train)" -> như cũ.
#     + "Dùng Gemini AI" -> chỉ thay bước OCR bằng Gemini; lưu/hiển thị/DB giữ nguyên.

# Yêu cầu môi trường cho Gemini:
# - file .env chứa GEMINI_API_KEY=...  (hoặc export biến môi trường tương đương)
# """

# import os, sys, time, cv2, numpy as np, pandas as pd
# from datetime import datetime

# # ---- HiDPI ----
# os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
# os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
# from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
#     QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
#     QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
#     QFileDialog, QHeaderView
# )

# # ---- Optional SQL ----
# USE_SQL = True
# try:
#     import pyodbc
# except Exception:
#     USE_SQL = False

# # ---- YOLO ----
# from ultralytics import YOLO

# # ---- Gemini (optional) ----
# from dotenv import load_dotenv
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# GEMINI_READY = False
# try:
#     if GEMINI_API_KEY:
#         from google import generativeai as genai
#         from google.api_core import exceptions as gexceptions
#         from PIL import Image
#         genai.configure(api_key=GEMINI_API_KEY)
#         GEMINI_READY = True
# except Exception as _e:
#     print("Gemini init failed:", _e)
#     GEMINI_READY = False

# # ==================== CONFIG ====================
# DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
# OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/epoch199.pt"

# SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)

# CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=plates_db;"
#     "UID=sa;"
#     "PWD=123456"
# )

# PANEL_W, PANEL_H = 640, 360
# PANEL_BG = (232, 239, 248)  # BGR
# STABLE_SECONDS_IN  = 1.2
# STABLE_SECONDS_OUT = 1.2

# API_MAP = {
#     "DSHOW(Windows)": cv2.CAP_DSHOW,
#     "MSMF(Windows)":  cv2.CAP_MSMF,
#     "ANY":            cv2.CAP_ANY
# }

# OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
#            "six":"6","seven":"7","eight":"8","nine":"9"}

# # ==================== UTILITIES ====================
# def letterbox(bgr, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
#     if bgr is None:
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     ih, iw = bgr.shape[:2]
#     if ih == 0 or iw == 0:
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     s = min(w/iw, h/ih); nw, nh = int(iw*s), int(ih*s)
#     resized = cv2.resize(bgr, (nw, nh))
#     canvas = np.full((h, w, 3), color, dtype=np.uint8)
#     top, left = (h-nh)//2, (w-nw)//2
#     canvas[top:top+nh, left:left+nw] = resized
#     return canvas

# def bgr_to_qimage(bgr):
#     if bgr is None:
#         bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     h, w, ch = rgb.shape
#     return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

# def save_image(img, prefix):
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
#     cv2.imwrite(path, img)
#     return path

# def norm_char(x): 
#     return OCR_MAP.get(str(x), str(x))

# def plate_norm(s: str) -> str:
#     return (s or "").replace("-", "").replace(" ", "").upper()

# def has_boxes(r):
#     try:
#         return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
#     except:
#         return False

# def preprocess_for_ocr(roi):
#     if roi is None: return None
#     if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe,(3,3),0)
#     return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# # ==================== DB LAYER ====================
# class DB:
#     def __init__(self, conn_str: str):
#         self.ok = False
#         self.conn = None
#         self.cur  = None
#         if not USE_SQL:
#             return
#         try:
#             self.conn = pyodbc.connect(conn_str, autocommit=True)
#             self.cur  = self.conn.cursor()
#             self.cur.execute("""
#                 IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
#                 CREATE TABLE dbo.ParkingSessions(
#                     id INT IDENTITY(1,1) PRIMARY KEY,
#                     plate_in NVARCHAR(64)  NULL,
#                     date_in  NVARCHAR(16)  NULL,
#                     time_in  NVARCHAR(16)  NULL,
#                     image_in NVARCHAR(255) NULL,
#                     plate_out NVARCHAR(64)  NULL,
#                     date_out  NVARCHAR(16)  NULL,
#                     time_out  NVARCHAR(16)  NULL,
#                     image_out NVARCHAR(255) NULL,
#                     match_status NVARCHAR(32) NULL,
#                     created_at DATETIME DEFAULT GETDATE()
#                 );
#             """)
#             self.ok = True
#         except Exception as e:
#             print("DB connect error:", e)
#             self.ok = False

#     def insert_in(self, plate, d, t, img_path):
#         if not self.ok: return
#         try:
#             self.cur.execute("""
#                 INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
#                 VALUES (?,?,?,?,?)
#             """, (plate, d, t, img_path, 'PENDING'))
#         except Exception as e:
#             print("insert_in error:", e)

#     def attach_out(self, plate_out, d, t, img_path) -> str:
#         if not self.ok: return "Khong khop bien so"
#         try:
#             rows = self.cur.execute("""
#                 SELECT TOP 50 id, plate_in FROM dbo.ParkingSessions
#                 WHERE plate_out IS NULL
#                 ORDER BY id DESC
#             """).fetchall()
#             match_sid = None
#             for sid, plate_in in rows:
#                 if plate_norm(plate_in) == plate_norm(plate_out):
#                     match_sid = sid
#                     break
#             if match_sid:
#                 self.cur.execute("""
#                     UPDATE dbo.ParkingSessions
#                     SET plate_out=?, date_out=?, time_out=?, image_out=?, match_status='KHOP-BIEN-SO'
#                     WHERE id=?
#                 """, (plate_out, d, t, img_path, match_sid))
#                 return "Khop bien so"
#             else:
#                 self.cur.execute("""
#                     INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
#                     VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO')
#                 """, (plate_out, d, t, img_path))
#                 return "Khong khop bien so"
#         except Exception as e:
#             print("attach_out error:", e)
#             return "Khong khop bien so"

#     def fetch_history_df(self, limit=100) -> pd.DataFrame:
#         if not self.ok:
#             return pd.DataFrame(columns=[
#                 "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                 "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#             ])
#         try:
#             rows = self.cur.execute(f"""
#                 SELECT TOP {limit}
#                     id, image_in, plate_in, date_in, time_in,
#                     image_out, plate_out, date_out, time_out, match_status
#                 FROM dbo.ParkingSessions
#                 ORDER BY id DESC
#             """).fetchall()
#             df = pd.DataFrame.from_records(
#                 rows,
#                 columns=["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                          "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             ).astype(object).where(pd.notnull, "")
#             df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#             # (DB có thể không có STT; UI sẽ drop nếu có)
#             df.insert(0, "STT", range(1, len(df)+1))
#             return df
#         except Exception as e:
#             print("fetch_history error:", e)
#             return pd.DataFrame(columns=[
#                 "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                 "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#             ])

#     # ---- XÓA LỊCH SỬ ----
#     def delete_by_ids(self, ids):
#         if not self.ok or not ids: return
#         try:
#             for sid in ids:
#                 self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
#         except Exception as e:
#             print("delete_by_ids error:", e)

#     def delete_all(self):
#         if not self.ok: return
#         try:
#             self.cur.execute("DELETE FROM dbo.ParkingSessions")
#         except Exception as e:
#             print("delete_all error:", e)

# # ==================== YOLO/GEMINI WRAPPERS ====================
# class Models:
#     def __init__(self, det_path: str, ocr_path: str):
#         self.ok = True
#         self.err = ""
#         try:
#             self.det = YOLO(det_path)
#             self.ocr = YOLO(ocr_path)  # YOLO OCR (mặc định)
#         except Exception as e:
#             self.ok = False
#             self.err = str(e)

#     def detect_plates(self, frame):
#         plates, boxed = [], frame.copy()
#         for r in self.det(frame):
#             if not has_boxes(r): 
#                 continue
#             xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
#             for (x1,y1,x2,y2) in xyxy:
#                 pad=8
#                 x1=max(0,x1-pad); y1=max(0,y1-pad)
#                 x2=min(boxed.shape[1]-1,x2+pad); y2=min(boxed.shape[0]-1,y2+pad)
#                 roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
#                 plates.append(((x1,y1,x2,y2), roi))
#                 cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
#                 cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
#         return plates, boxed

#     # ---- YOLO OCR ----
#     def ocr_plate_yolo(self, roi):
#         roi_pre = preprocess_for_ocr(roi)
#         res = self.ocr(roi_pre if roi_pre is not None else roi)
#         text_raw=""
#         for r in res:
#             if not has_boxes(r): continue
#             names = getattr(r,'names',None) or getattr(self.ocr,'names',{}) or {}
#             clses = r.boxes.cls.cpu().numpy().astype(int)
#             xyxys= r.boxes.xyxy.cpu().numpy()
#             boxes=[]
#             for i,cls in enumerate(clses):
#                 x1,y1,x2,y2 = xyxys[i]
#                 cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
#                 ch = norm_char(names.get(cls, str(cls)) if isinstance(names,dict) else str(cls))
#                 if ch.isdigit() or (ch.isalpha() and ch.isupper()):
#                     boxes.append((cy,cx,ch))
#             if not boxes: 
#                 continue
#             ys=[b[0] for b in boxes]
#             if len(boxes)<=7 or (max(ys)-min(ys) < 0.2*max(ys, default=1)):
#                 text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
#             else:
#                 thr=(max(ys)+min(ys))/2.0
#                 l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
#                 t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
#                 t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
#                 text_raw=f"{t1}-{t2}" if t2 else t1
#         return self._format_text(text_raw)

#     # ---- Gemini OCR ----
#     def ocr_plate_gemini_from_path(self, image_path: str):
#         if not GEMINI_READY:
#             return "", ""
#         try:
#             img = Image.open(image_path)
#         except Exception as e:
#             print("Gemini open image error:", e)
#             return "", ""
#         try:
#             model = genai.GenerativeModel('gemini-2.5-flash')
#             prompt = (
#                 "Đây là ảnh chụp biển số xe Việt Nam. "
#                 "Nhiệm vụ: trích xuất chính xác chuỗi ký tự biển số. "
#                 "Chỉ trả lời bằng chuỗi biển số, không thêm giải thích. "
#                 "Ví dụ: '29-P1 123.45' hoặc '50-Z8 888.88'."
#             )
#             resp = model.generate_content([prompt, img])
#             raw = (resp.text or "").strip().replace("\n", " ")
#             return self._format_text(raw)
#         except gexceptions.GoogleAPICallError as e:
#             print("Gemini API error:", e)
#             return "", ""
#         except Exception as e:
#             print("Gemini unknown error:", e)
#             return "", ""

#     @staticmethod
#     def _format_text(text_raw: str):
#         raw=(text_raw or '').replace('-', ' ').replace(' ', '')
#         text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
#         return text_fmt, (text_raw or "")

# # ==================== VIDEO WORKER ====================
# class VideoWorker(QThread):
#     frameSignal = Signal(np.ndarray, str)
#     sceneSignal = Signal(str)
#     roiSignal   = Signal(str, str)
#     infoSignal  = Signal(dict)
#     matchSignal = Signal(str)
#     histSignal  = Signal()

#     def __init__(self, cam_idx: int, api: int, mode: str, models: Models, db: DB,
#                  stable_seconds: float = 1.2, ocr_mode: str = "yolo", title: str = "", parent=None):
#         super().__init__(parent)
#         self.cam_idx = cam_idx
#         self.api = api
#         self.mode = mode            # 'in' | 'out'  (CÓ THỂ ĐỔI RUNTIME)
#         self.models = models
#         self.db = db
#         self.stable_seconds = stable_seconds
#         self.ocr_mode = ocr_mode    # 'yolo' | 'gemini'
#         self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")

#         self._running = False
#         self.cap = None
#         self.stable_start = 0.0
#         self.captured = False

#     def set_title(self, title: str): self.title = title
#     def set_ocr_mode(self, mode: str): self.ocr_mode = mode
#     def set_mode(self, mode: str): self.mode = mode  # cho phép đổi IN/OUT theo làn

#     def run(self):
#         self._running = True
#         self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
#         if not (self.cap and self.cap.isOpened()):
#             self._running = False
#             return
#         try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         except: pass
#         try: self.cap.set(cv2.CAP_PROP_FPS, 30)
#         except: pass

#         while self._running:
#             ok, frame = self.cap.read()
#             if not ok:
#                 self.stable_start = 0.0
#                 self.captured = False
#                 time.sleep(0.03)
#                 continue

#             # KHÔNG vẽ tiêu đề lên ảnh nữa
#             disp = letterbox(frame)
#             self.frameSignal.emit(disp, self.title)

#             plates, boxed = self.models.detect_plates(frame)
#             if not plates:
#                 self.stable_start = 0.0
#                 self.captured = False
#                 time.sleep(0.01)
#                 continue

#             best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
#             roi_current = best[1]

#             if self.stable_start == 0.0 or self.captured:
#                 self.stable_start = time.time()
#                 self.captured = False

#             if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
#                 # Lưu hình trước (Gemini cần path)
#                 scene_path = save_image(boxed if boxed is not None else frame,
#                                         "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
#                 roi_path   = save_image(roi_current,
#                                         "plate_in_roi" if self.mode=="in" else "plate_out_roi")

#                 # OCR theo mode
#                 if self.ocr_mode == "gemini" and GEMINI_READY:
#                     text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
#                 else:
#                     text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

#                 if text_fmt or text_raw:
#                     now = datetime.now()
#                     d = now.strftime("%d/%m/%Y")
#                     t = now.strftime("%H:%M:%S")
#                     plate = text_fmt or text_raw

#                     self.sceneSignal.emit(scene_path)
#                     self.roiSignal.emit(roi_path, self.mode)

#                     if self.mode == "in":
#                         self.infoSignal.emit({"date_in": d, "time_in": t, "plate_text_in": plate})
#                         if self.db and self.db.ok:
#                             self.db.insert_in(plate, d, t, scene_path)
#                             self.histSignal.emit()
#                     else:
#                         self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
#                         if self.db and self.db.ok:
#                             match = self.db.attach_out(plate, d, t, scene_path)
#                             self.matchSignal.emit(match)
#                             self.histSignal.emit()
#                     self.captured = True

#             time.sleep(0.01)

#         try:
#             if self.cap: self.cap.release()
#         except: pass

#     def stop(self):
#         self._running = False

# # ==================== MAIN WINDOW ====================
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Phát hiện & OCR biển số xe - Desktop App (PySide6)")
#         self.setMinimumSize(1400, 900)

#         self._init_theme()

#         self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#         self.db = DB(CONN_STR) if USE_SQL else DB("")

#         self.cam1_worker = None
#         self.cam2_worker = None

#         # ===== Lane state =====
#         self.lane1_dir = "VÀO"
#         self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True
#         self.two_way_toggle = True

#         # OCR mode (yolo | gemini)
#         self.current_ocr_mode = "yolo"

#         self._build_ui()

#         self.hist_timer = QTimer(self)
#         self.hist_timer.timeout.connect(self.refresh_history)
#         self.hist_timer.start(5000)

#     def _init_theme(self):
#         self.setStyleSheet("""
#         * { color: #000000; }
#         QMainWindow, QWidget { background: #ffffff; }
#         QWidget#SideBar { background: #ffffff; }

#         QGroupBox {
#             background: #ffffff;
#             font-weight: 600;
#             border: 2px solid #e6e6e6;
#             border-radius: 8px;
#             margin-top: 8px;
#             padding-top: 10px;
#         }
#         QGroupBox::title {
#             subcontrol-origin: margin;
#             left: 10px;
#             padding: 0 6px;
#             background: #ffffff;
#         }

#         QFrame[class="card-wrap"] { background: #e6e6e6; border-radius: 12px; }
#         QFrame[class="card"]      { background: #ffffff; border-radius: 10px; }
#         QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 10px; }
#         QLabel[class="title"] {
#             font: 700 18px "Segoe UI";
#             padding: 6px 10px;
#             background: #ffffff;
#             border-radius: 8px;
#         }

#         QPushButton { height: 34px; border-radius: 8px; font-weight: 600; }
#         QPushButton#btnGreen { background: #d1fadf; border: 1px solid #a6f4c5; }
#         QPushButton#btnRed   { background: #ffe0e0; border: 1px solid #ffb3b3; }
#         QPushButton#btnYellow{ background: #fff3bf; border: 1px solid #ffe066; }

#         QLineEdit {
#             height: 28px;
#             background: #ffffff;
#             border: 1px solid #e0e0e0;
#             border-radius: 6px;
#             padding: 2px 6px;
#         }
#         QTableWidget { background: #ffffff; gridline-color: #e6e6e6; }
#         """)

#     # small helper to create a titled card and get the title QLabel back
#     def _make_card(self, title:str, content:QWidget):
#         wrap = QFrame(); wrap.setProperty("class","card-wrap")
#         wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
#         card = QFrame(); card.setProperty("class","card")
#         v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
#         title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
#         hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
#         title_lbl = QLabel(title); title_lbl.setProperty("class","title")
#         hl.addWidget(title_lbl)
#         v.addWidget(title_wrap)
#         v.addWidget(content, 1)
#         wrapL.addWidget(card)
#         return wrap, title_lbl

#     # ---------- UI ----------
#     def _build_ui(self):
#         central = QWidget(); self.setCentralWidget(central)
#         root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

#         # ---------- LEFT: SIDEBAR ----------
#         side = QWidget(objectName="SideBar"); side.setMinimumWidth(420)
#         vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

#         # CAMERA CONTROL
#         gb_camctl = QGroupBox("CAMERA CONTROL")
#         gl_camctl = QGridLayout(gb_camctl)
#         self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
#         self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys()))
#         self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
#         self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys()))
#         self.btn_start1 = QPushButton("Bật Cam 1"); self.btn_start1.setObjectName("btnGreen")
#         self.btn_stop1  = QPushButton("Tắt Cam 1"); self.btn_stop1.setObjectName("btnRed")
#         self.btn_start2 = QPushButton("Bật Cam 2"); self.btn_start2.setObjectName("btnGreen")
#         self.btn_stop2  = QPushButton("Tắt Cam 2"); self.btn_stop2.setObjectName("btnRed")
#         self.btn_start1.clicked.connect(self.start_cam1)
#         self.btn_stop1.clicked.connect(self.stop_cam1)
#         self.btn_start2.clicked.connect(self.start_cam2)
#         self.btn_stop2.clicked.connect(self.stop_cam2)
#         r=0
#         gl_camctl.addWidget(QLabel("Index Cam 1"), r,0); gl_camctl.addWidget(self.spin_cam1, r,1)
#         gl_camctl.addWidget(QLabel("Backend Cam 1"), r,2); gl_camctl.addWidget(self.cb_api1, r,3); r+=1
#         gl_camctl.addWidget(QLabel("Index Cam 2"), r,0); gl_camctl.addWidget(self.spin_cam2, r,1)
#         gl_camctl.addWidget(QLabel("Backend Cam 2"), r,2); gl_camctl.addWidget(self.cb_api2, r,3); r+=1
#         gl_camctl.addWidget(self.btn_start1, r,0,1,2); gl_camctl.addWidget(self.btn_stop1, r,2,1,2); r+=1
#         gl_camctl.addWidget(self.btn_start2, r,0,1,2); gl_camctl.addWidget(self.btn_stop2, r,2,1,2)
#         vside.addWidget(gb_camctl)

#         # ===== ĐIỀU KHIỂN LÀN =====
#         gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
#         vl_lane = QVBoxLayout(gb_lane)
#         self.lbl_lane1 = QLabel("")
#         self.lbl_lane2 = QLabel("")
#         lane_info = QLabel("Mỗi làn: cam trước: cam1, cam sau: cam2")
#         lane_info.setStyleSheet("font-style: italic;")

#         btns_row = QHBoxLayout()
#         self.btn_oneway = QPushButton("1 chiều (Toggle)")
#         self.btn_twoway = QPushButton("2 chiều (Đảo)")
#         self.btn_reset_lane = QPushButton("Reset làn xe"); self.btn_reset_lane.setObjectName("btnYellow")
#         self.btn_oneway.setStyleSheet("background:#dbeafe; font-weight:600;")
#         self.btn_twoway.setStyleSheet("background:#dbeafe; font-weight:600;")
#         self.btn_reset_lane.setStyleSheet("font-weight:600;")
#         self.btn_oneway.clicked.connect(self.on_one_way_clicked)
#         self.btn_twoway.clicked.connect(self.on_two_way_clicked)
#         self.btn_reset_lane.clicked.connect(self.on_reset_lanes)

#         btns_row.addWidget(self.btn_oneway)
#         btns_row.addWidget(self.btn_twoway)
#         btns_row.addWidget(self.btn_reset_lane)

#         vl_lane.addWidget(self.lbl_lane1)
#         vl_lane.addWidget(self.lbl_lane2)
#         vl_lane.addWidget(lane_info)
#         vl_lane.addLayout(btns_row)
#         vside.addWidget(gb_lane)

#         # OCR MODEL
#         gb_ocr = QGroupBox("OCR MODEL")
#         vb_ocr = QVBoxLayout(gb_ocr)
#         self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)"); self.rb_yolo.setChecked(True)
#         self.rb_gem  = QRadioButton("Dùng Gemini AI")
#         vb_ocr.addWidget(self.rb_yolo); vb_ocr.addWidget(self.rb_gem)
#         self.rb_yolo.toggled.connect(self.on_ocr_mode_changed)
#         self.rb_gem.toggled.connect(self.on_ocr_mode_changed)
#         if not GEMINI_READY:
#             self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY trong môi trường/.env → mặc định dùng YOLO")
#         vside.addWidget(gb_ocr)

#         # THÔNG TIN XE VÀO
#         gb_in = QGroupBox("THÔNG TIN XE VÀO")
#         gl_in = QGridLayout(gb_in)
#         self.ed_date_in = QLineEdit(); self.ed_time_in = QLineEdit(); self.ed_plate_in = QLineEdit()
#         gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
#         gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
#         gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
#         vside.addWidget(gb_in)

#         # THÔNG TIN XE RA
#         gb_out = QGroupBox("THÔNG TIN XE RA")
#         gl_out = QGridLayout(gb_out)
#         self.ed_date_out = QLineEdit(); self.ed_time_out = QLineEdit(); self.ed_plate_out = QLineEdit()
#         gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
#         gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
#         gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
#         vside.addWidget(gb_out)

#         # Nút chuyển chế độ xem & Export/Xóa
#         self.btn_show_history = QPushButton("Xem bảng lịch sử")
#         self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide()
#         self.btn_export_hist  = QPushButton("Export Excel")
#         self.btn_delete_hist  = QPushButton("Xóa bảng")
#         self.btn_show_history.setStyleSheet("background:#E6F4EA; font-weight:600;")
#         self.btn_hide_history.setStyleSheet("background:#FCE8E6; font-weight:600;")
#         self.btn_export_hist.setStyleSheet("background:#e0ecff; font-weight:600;")
#         self.btn_delete_hist.setStyleSheet("background:#ffe0e0; font-weight:600;")
#         self.btn_show_history.clicked.connect(self.show_history_view)
#         self.btn_hide_history.clicked.connect(self.show_main_view)
#         self.btn_export_hist.clicked.connect(self.on_export_excel)
#         self.btn_delete_hist.clicked.connect(self.on_delete_history)
#         vside.addWidget(self.btn_show_history)
#         vside.addWidget(self.btn_hide_history)
#         vside.addWidget(self.btn_export_hist)
#         vside.addWidget(self.btn_delete_hist)

#         vside.addStretch(1)
#         root.addWidget(side)

#         # ---------- RIGHT ----------
#         right_container = QVBoxLayout()

#         # ===== Main view =====
#         self.main_view = QWidget()
#         main_layout = QVBoxLayout(self.main_view)

#         top = QHBoxLayout()
#         self.lbl_cam1 = QLabel(); self.lbl_cam1.setMinimumSize(PANEL_W, PANEL_H); self.lbl_cam1.setScaledContents(True)
#         self.lbl_cam2 = QLabel(); self.lbl_cam2.setMinimumSize(PANEL_W, PANEL_H); self.lbl_cam2.setScaledContents(True)
#         cam1_card, self.cam1_title = self._make_card("1) Cam 1 (Vào)", self.lbl_cam1)
#         cam2_card, self.cam2_title = self._make_card("2) Cam 2 (Vào)", self.lbl_cam2)  # sẽ cập nhật theo làn 2
#         top.addWidget(cam1_card, 1)
#         top.addWidget(cam2_card, 1)
#         main_layout.addLayout(top)

#         bottom = QHBoxLayout()
#         self.lbl_scene = QLabel(); self.lbl_scene.setMinimumSize(PANEL_W, PANEL_H); self.lbl_scene.setScaledContents(True)
#         self.lbl_roi   = QLabel(); self.lbl_roi.setMinimumSize(PANEL_W, PANEL_H); self.lbl_roi.setScaledContents(True)
#         scene_card, _ = self._make_card("3) Image_BOX",  self.lbl_scene)
#         roi_card,   _ = self._make_card("4) ROI_Plates", self.lbl_roi)
#         bottom.addWidget(scene_card, 1)
#         bottom.addWidget(roi_card,   1)
#         main_layout.addLayout(bottom)

#         self.info_group = QGroupBox("Thông tin chi tiết")
#         info_layout = QGridLayout(self.info_group)
#         self.txt_date_in  = QLabel("--/--/----")
#         self.txt_time_in  = QLabel("--:--:--")
#         self.txt_plate_in = QLabel("---"); self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_date_out  = QLabel("--/--/----")
#         self.txt_time_out  = QLabel("--:--:--")
#         self.txt_plate_out = QLabel("---"); self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_match     = QLabel("")
#         r=0
#         info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
#         info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
#         info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
#         info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
#         info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
#         info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
#         info_layout.addWidget(QLabel("So khớp:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,5)
#         main_layout.addWidget(self.info_group)

#         # ===== History view =====
#         self.history_view = QWidget()
#         hist_layout = QVBoxLayout(self.history_view)
#         hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)")
#         hist_v = QVBoxLayout(hist_group)

#         self.tbl_hist = QTableWidget(0, 10)
#         self.tbl_hist.setHorizontalHeaderLabels([
#             "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#         ])
#         header = self.tbl_hist.horizontalHeader()
#         hfont = QFont(header.font()); hfont.setBold(True); header.setFont(hfont)
#         self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
#         self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
#         self.tbl_hist.setAlternatingRowColors(True)
#         header.setSectionResizeMode(QHeaderView.Stretch)  # kéo giãn full width

#         hist_v.addWidget(self.tbl_hist)
#         hist_layout.addWidget(hist_group)

#         self.stacked = QStackedWidget()
#         self.stacked.addWidget(self.main_view)
#         self.stacked.addWidget(self.history_view)
#         self.stacked.setCurrentIndex(0)
#         right_container.addWidget(self.stacked, 1)
#         root.addLayout(right_container, 1)

#         self.update_lane_labels()
#         self.update_titles_and_modes()  # set tiêu đề card & mode cam theo làn

#     # ---------- LANE LOGIC ----------
#     def update_lane_labels(self):
#         self.lbl_lane1.setText(f"Làn 1: {self.lane1_dir} (cam trước: cam1, cam sau: cam2)")
#         self.lbl_lane2.setText(f"Làn 2: {self.lane2_dir} (cam trước: cam1, cam sau: cam2)")

#     def update_titles_and_modes(self):
#         # cập nhật tiêu đề card camera theo làn
#         self.cam1_title.setText(f"1) Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
#         self.cam2_title.setText(f"2) Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
#         # cập nhật mode ghi nhận cho worker đang chạy
#         if self.cam1_worker:
#             self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
#         if self.cam2_worker:
#             self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")

#     @Slot()
#     def on_reset_lanes(self):
#         self.lane1_dir = "VÀO"
#         self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True
#         self.two_way_toggle = True
#         self.update_lane_labels()
#         self.update_titles_and_modes()

#     @Slot()
#     def on_one_way_clicked(self):
#         if self.one_way_toggle_vao:
#             self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         else:
#             self.lane1_dir = "RA";  self.lane2_dir = "RA"
#         self.one_way_toggle_vao = not self.one_way_toggle_vao
#         self.update_lane_labels()
#         self.update_titles_and_modes()

#     @Slot()
#     def on_two_way_clicked(self):
#         if self.two_way_toggle:
#             self.lane1_dir = "VÀO"; self.lane2_dir = "RA"
#         else:
#             self.lane1_dir = "RA";  self.lane2_dir = "VÀO"
#         self.two_way_toggle = not self.two_way_toggle
#         self.update_lane_labels()
#         self.update_titles_and_modes()

#     # ---------- OCR mode ----------
#     @Slot()
#     def on_ocr_mode_changed(self):
#         self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
#         if self.rb_gem.isChecked() and not GEMINI_READY:
#             QMessageBox.information(self, "Gemini",
#                 "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
#             self.rb_yolo.setChecked(True)
#             self.current_ocr_mode = "yolo"
#         if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
#         if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

#     # ---------- toggle views ----------
#     def show_history_view(self):
#         self.stacked.setCurrentIndex(1)
#         self.btn_show_history.hide()
#         self.btn_hide_history.show()
#         self.refresh_history()

#     def show_main_view(self):
#         self.stacked.setCurrentIndex(0)
#         self.btn_hide_history.hide()
#         self.btn_show_history.show()

#     # ---------- EXPORT / DELETE ----------
#     @Slot()
#     def on_export_excel(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns:
#             df = df.drop(columns=["STT"])
#         if df.empty:
#             QMessageBox.information(self, "Export", "Không có dữ liệu để export.")
#             return
#         path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
#         if not path: return
#         try:
#             df.to_excel(path, index=False)
#             QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
#         except Exception as e:
#             QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

#     @Slot()
#     def on_delete_history(self):
#         if not (self.db and self.db.ok):
#             QMessageBox.warning(self, "Xóa", "Chưa kết nối DB, không thể xóa.")
#             return

#         msg = QMessageBox(self)
#         msg.setWindowTitle("Xóa lịch sử")
#         msg.setText("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
#         btn_sel = msg.addButton("Xóa dòng đã chọn", QMessageBox.ButtonRole.ActionRole)
#         btn_all = msg.addButton("Xóa tất cả", QMessageBox.ButtonRole.ActionRole)
#         btn_cancel = msg.addButton("Hủy", QMessageBox.ButtonRole.RejectRole)
#         msg.exec()

#         clicked = msg.clickedButton()
#         if clicked == btn_sel:
#             rows = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
#             if not rows:
#                 QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào.")
#                 return
#             # Tìm cột ID
#             cols = [self.tbl_hist.horizontalHeaderItem(i).text() for i in range(self.tbl_hist.columnCount())]
#             try:
#                 id_col = cols.index("ID")
#             except ValueError:
#                 QMessageBox.warning(self, "Xóa", "Không tìm thấy cột ID.")
#                 return
#             ids = []
#             for r in rows:
#                 item = self.tbl_hist.item(r, id_col)
#                 if item: ids.append(item.text())
#             if not ids:
#                 QMessageBox.information(self, "Xóa", "Không lấy được ID các dòng chọn.")
#                 return
#             confirm = QMessageBox.question(self, "Xác nhận",
#                     f"Bạn chắc chắn xóa {len(ids)} dòng đã chọn?",
#                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
#             if confirm == QMessageBox.StandardButton.Yes:
#                 self.db.delete_by_ids(ids)
#                 self.refresh_history()

#         elif clicked == btn_all:
#             confirm = QMessageBox.question(self, "Xác nhận",
#                     "Bạn chắc chắn xóa **TẤT CẢ** lịch sử?",
#                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
#             if confirm == QMessageBox.StandardButton.Yes:
#                 self.db.delete_all()
#                 self.refresh_history()
#         else:
#             return

#     # ---------- helpers ----------
#     def show_blank(self, label: QLabel, title: str):
#         # KHÔNG vẽ chữ vào ảnh; chỉ hiển thị panel trống
#         qimg  = bgr_to_qimage(letterbox(None))
#         label.setPixmap(QPixmap.fromImage(qimg))
#         # cập nhật luôn tiêu đề card (đã làm ở update_titles_and_modes)

#     @Slot(np.ndarray, str)
#     def on_frame(self, frame_bgr, title):
#         qimg = bgr_to_qimage(letterbox(frame_bgr))
#         # xác định đến cam 1/2 dựa vào object sender/title card
#         # (title chỉ để tham khảo; ở đây cập nhật thẳng 2 label)
#         # Khi có 2 stream đồng thời, tín hiệu đến vẫn mapping theo worker đã connect
#         sender = self.sender()
#         if sender is self.cam1_worker:
#             self.lbl_cam1.setPixmap(QPixmap.fromImage(qimg))
#         elif sender is self.cam2_worker:
#             self.lbl_cam2.setPixmap(QPixmap.fromImage(qimg))

#     @Slot(str)
#     def on_scene(self, path):
#         if os.path.exists(path):
#             bgr = cv2.imread(path)
#             self.lbl_scene.setPixmap(QPixmap.fromImage(bgr_to_qimage(letterbox(bgr))))

#     @Slot(str, str)
#     def on_roi(self, roi_path, mode):
#         if os.path.exists(roi_path):
#             bgr = cv2.imread(roi_path)
#             self.lbl_roi.setPixmap(QPixmap.fromImage(bgr_to_qimage(letterbox(bgr))))

#     @Slot(dict)
#     def on_info(self, info):
#         if "date_in" in info:
#             self.txt_date_in.setText(info["date_in"]); self.ed_date_in.setText(info["date_in"])
#         if "time_in" in info:
#             self.txt_time_in.setText(info["time_in"]); self.ed_time_in.setText(info["time_in"])
#         if "plate_text_in" in info:
#             self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
#         if "date_out" in info:
#             self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
#         if "time_out" in info:
#             self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
#         if "plate_text_out" in info:
#             self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

#     @Slot(str)
#     def on_match(self, txt):
#         self.txt_match.setText(txt.upper())

#     @Slot()
#     def refresh_history(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns:
#             df = df.drop(columns=["STT"])
#         if df.empty:
#             self.tbl_hist.setRowCount(0)
#             cols = ["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                     "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             self.tbl_hist.setColumnCount(len(cols))
#             self.tbl_hist.setHorizontalHeaderLabels(cols)
#             hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#             self.tbl_hist.horizontalHeader().setFont(hfont)
#             return

#         cols = list(df.columns)
#         self.tbl_hist.setRowCount(len(df))
#         self.tbl_hist.setColumnCount(len(cols))
#         self.tbl_hist.setHorizontalHeaderLabels(cols)
#         hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#         self.tbl_hist.horizontalHeader().setFont(hfont)
#         self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

#         for i in range(len(df)):
#             for j, col in enumerate(cols):
#                 val = str(df.iloc[i, j])
#                 item = QTableWidgetItem(val)
#                 item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
#                 self.tbl_hist.setItem(i, j, item)

#     # ---------- camera controls ----------
#     def _connect_worker(self, w: VideoWorker):
#         w.frameSignal.connect(self.on_frame)
#         w.sceneSignal.connect(self.on_scene)
#         w.roiSignal.connect(self.on_roi)
#         w.infoSignal.connect(self.on_info)
#         w.matchSignal.connect(self.on_match)
#         w.histSignal.connect(self.refresh_history)

#     def start_cam_generic(self, which: int):
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#             return
#         if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
#         if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

#         self.show_main_view()

#         ocr_mode = self.current_ocr_mode
#         if which == 1:
#             idx = int(self.spin_cam1.value())
#             api = API_MAP[self.cb_api1.currentText()]
#             mode = "in" if self.lane1_dir=="VÀO" else "out"
#             title = f"Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam1_worker = VideoWorker(idx, api, mode, self.models, self.db,
#                                            STABLE_SECONDS_IN, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam1_worker)
#             self.cam1_worker.start()
#         else:
#             idx = int(self.spin_cam2.value())
#             api = API_MAP[self.cb_api2.currentText()]
#             mode = "in" if self.lane2_dir=="VÀO" else "out"
#             title = f"Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam2_worker = VideoWorker(idx, api, mode, self.models, self.db,
#                                            STABLE_SECONDS_OUT, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam2_worker)
#             self.cam2_worker.start()

#     def stop_cam_generic(self, which: int):
#         self.show_main_view()
#         worker = self.cam1_worker if which==1 else self.cam2_worker
#         if worker and worker.isRunning():
#             worker.stop(); worker.wait(1500)

#     def start_cam1(self): self.start_cam_generic(1)
#     def stop_cam1(self):  self.stop_cam_generic(1)
#     def start_cam2(self): self.start_cam_generic(2)
#     def stop_cam2(self):  self.stop_cam_generic(2)

#     def closeEvent(self, event):
#         try:
#             self.stop_cam_generic(1)
#             self.stop_cam_generic(2)
#         except: pass
#         super().closeEvent(event)

# # ==================== MAIN ====================
# def main():
#     QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
#         Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
#     )
#     app = QApplication(sys.argv)
#     w = MainWindow(); w.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()













# ----------------------------------------------------------------------------------------------------------------------------


































# # -*- coding: utf-8 -*-
# """
# PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini)
# - Lịch sử: GroupBox, tiêu đề cột đầy đủ, kéo giãn, Export Excel, Xóa (dòng chọn / tất cả) với dialog màu & bo góc.
# - Camera: căn giữa, bo góc, không in chữ lên ảnh; chỉ đổi tiêu đề card. Bật/tắt độc lập từng cam.
# - Khi tắt cam/reset -> hiện logo mặc định.
# - Điều khiển làn: 1 chiều (toggle), 2 chiều (đảo), Reset làn; đổi mode ghi nhận IN/OUT ngay cho worker.
# - OCR: YOLO mặc định; chọn "Dùng Gemini AI" -> chỉ thay bước OCR bằng Gemini, lưu/DB/hiển thị giữ nguyên.
# """

# import os, sys, time, cv2, numpy as np, pandas as pd
# from datetime import datetime

# # ---- HiDPI ----
# os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
# os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint
# from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
#     QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
#     QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
#     QFileDialog, QHeaderView, QDialog, QSpacerItem
# )

# # ---- Optional SQL ----
# USE_SQL = True
# try:
#     import pyodbc
# except Exception:
#     USE_SQL = False

# # ---- YOLO ----
# from ultralytics import YOLO

# # ---- Gemini (optional) ----
# from dotenv import load_dotenv
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# GEMINI_READY = False
# try:
#     if GEMINI_API_KEY:
#         from google import generativeai as genai
#         from google.api_core import exceptions as gexceptions
#         from PIL import Image
#         genai.configure(api_key=GEMINI_API_KEY)
#         GEMINI_READY = True
# except Exception as _e:
#     print("Gemini init failed:", _e)
#     GEMINI_READY = False

# # ==================== CONFIG ====================
# DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
# OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
# SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
# LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")  # <- đặt file logo của bạn ở đây

# CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=plates_db;"
#     "UID=sa;"
#     "PWD=123456"
# )

# PANEL_W, PANEL_H = 640, 360
# PANEL_BG = (232, 239, 248)  # BGR
# STABLE_SECONDS_IN  = 1.2
# STABLE_SECONDS_OUT = 1.2

# API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
# OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
#            "six":"6","seven":"7","eight":"8","nine":"9"}

# # ==================== UTILITIES ====================
# def letterbox(bgr, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
#     if bgr is None:
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     ih, iw = bgr.shape[:2]
#     if ih == 0 or iw == 0:
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     s = min(w/iw, h/ih); nw, nh = int(iw*s), int(ih*s)
#     resized = cv2.resize(bgr, (nw, nh))
#     canvas = np.full((h, w, 3), color, dtype=np.uint8)
#     top, left = (h-nh)//2, (w-nw)//2
#     canvas[top:top+nh, left:left+nw] = resized
#     return canvas

# def bgr_to_qimage(bgr):
#     if bgr is None:
#         bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     h, w, ch = rgb.shape
#     return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

# def save_image(img, prefix):
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
#     cv2.imwrite(path, img)
#     return path

# def norm_char(x): return OCR_MAP.get(str(x), str(x))
# def plate_norm(s: str) -> str: return (s or "").replace("-", "").replace(" ", "").upper()

# def has_boxes(r):
#     try:
#         return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
#     except: return False

# def preprocess_for_ocr(roi):
#     if roi is None: return None
#     if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe,(3,3),0)
#     return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# # ==================== DB LAYER ====================
# class DB:
#     def __init__(self, conn_str: str):
#         self.ok = False; self.conn = None; self.cur  = None
#         if not USE_SQL: return
#         try:
#             self.conn = pyodbc.connect(conn_str, autocommit=True)
#             self.cur  = self.conn.cursor()
#             self.cur.execute("""
#                 IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
#                 CREATE TABLE dbo.ParkingSessions(
#                     id INT IDENTITY(1,1) PRIMARY KEY,
#                     plate_in NVARCHAR(64)  NULL,
#                     date_in  NVARCHAR(16)  NULL,
#                     time_in  NVARCHAR(16)  NULL,
#                     image_in NVARCHAR(255) NULL,
#                     plate_out NVARCHAR(64)  NULL,
#                     date_out  NVARCHAR(16)  NULL,
#                     time_out  NVARCHAR(16)  NULL,
#                     image_out NVARCHAR(255) NULL,
#                     match_status NVARCHAR(32) NULL,
#                     created_at DATETIME DEFAULT GETDATE()
#                 );
#             """)
#             self.ok = True
#         except Exception as e:
#             print("DB connect error:", e); self.ok = False

#     def insert_in(self, plate, d, t, img_path):
#         if not self.ok: return
#         try:
#             self.cur.execute("""
#                 INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
#                 VALUES (?,?,?,?,?)
#             """, (plate, d, t, img_path, 'PENDING'))
#         except Exception as e: print("insert_in error:", e)

#     def attach_out(self, plate_out, d, t, img_path) -> str:
#         if not self.ok: return "Khong khop bien so"
#         try:
#             rows = self.cur.execute("""
#                 SELECT TOP 50 id, plate_in FROM dbo.ParkingSessions
#                 WHERE plate_out IS NULL
#                 ORDER BY id DESC
#             """).fetchall()
#             match_sid = None
#             for sid, plate_in in rows:
#                 if plate_norm(plate_in) == plate_norm(plate_out):
#                     match_sid = sid; break
#             if match_sid:
#                 self.cur.execute("""
#                     UPDATE dbo.ParkingSessions
#                     SET plate_out=?, date_out=?, time_out=?, image_out=?, match_status='KHOP-BIEN-SO'
#                     WHERE id=?
#                 """, (plate_out, d, t, img_path, match_sid))
#                 return "Khop bien so"
#             else:
#                 self.cur.execute("""
#                     INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
#                     VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO')
#                 """, (plate_out, d, t, img_path))
#                 return "Khong khop bien so"
#         except Exception as e:
#             print("attach_out error:", e); return "Khong khop bien so"

#     def fetch_history_df(self, limit=10000) -> pd.DataFrame:
#         if not self.ok:
#             return pd.DataFrame(columns=[
#                 "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                 "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#             ])
#         try:
#             rows = self.cur.execute(f"""
#                 SELECT TOP {limit}
#                     id, image_in, plate_in, date_in, time_in,
#                     image_out, plate_out, date_out, time_out, match_status
#                 FROM dbo.ParkingSessions
#                 ORDER BY id DESC
#             """).fetchall()
#             df = pd.DataFrame.from_records(
#                 rows,
#                 columns=["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                          "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             ).astype(object).where(pd.notnull, "")
#             df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#             df.insert(0, "STT", range(1, len(df)+1))
#             return df
#         except Exception as e:
#             print("fetch_history error:", e)
#             return pd.DataFrame(columns=[
#                 "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#                 "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#             ])

#     def delete_by_ids(self, ids):
#         if not self.ok or not ids: return
#         try:
#             for sid in ids:
#                 self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
#         except Exception as e: print("delete_by_ids error:", e)

#     def delete_all(self):
#         if not self.ok: return
#         try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
#         except Exception as e: print("delete_all error:", e)

# # ==================== YOLO/GEMINI WRAPPERS ====================
# class Models:
#     def __init__(self, det_path: str, ocr_path: str):
#         self.ok = True; self.err = ""
#         try:
#             self.det = YOLO(det_path)
#             self.ocr = YOLO(ocr_path)
#         except Exception as e:
#             self.ok = False; self.err = str(e)

#     def detect_plates(self, frame):
#         plates, boxed = [], frame.copy()
#         for r in self.det(frame):
#             if not has_boxes(r): continue
#             xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
#             for (x1,y1,x2,y2) in xyxy:
#                 pad=8
#                 x1=max(0,x1-pad); y1=max(0,y1-pad)
#                 x2=min(boxed.shape[1]-1,x2+pad); y2=min(boxed.shape[0]-1,y2+pad)
#                 roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)].copy()
#                 plates.append(((x1,y1,x2,y2), roi))
#                 cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
#                 cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
#         return plates, boxed

#     def ocr_plate_yolo(self, roi):
#         roi_pre = preprocess_for_ocr(roi)
#         res = self.ocr(roi_pre if roi_pre is not None else roi)
#         text_raw=""
#         for r in res:
#             if not has_boxes(r): continue
#             names = getattr(r,'names',None) or getattr(self.ocr,'names',{}) or {}
#             clses = r.boxes.cls.cpu().numpy().astype(int)
#             xyxys= r.boxes.xyxy.cpu().numpy()
#             boxes=[]
#             for i,cls in enumerate(clses):
#                 x1,y1,x2,y2 = xyxys[i]
#                 cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
#                 ch = norm_char(names.get(cls, str(cls)) if isinstance(names,dict) else str(cls))
#                 if ch.isdigit() or (ch.isalpha() and ch.isupper()):
#                     boxes.append((cy,cx,ch))
#             if not boxes: continue
#             ys=[b[0] for b in boxes]
#             if len(boxes)<=7 or (max(ys)-min(ys) < 0.2*max(ys, default=1)):
#                 text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
#             else:
#                 thr=(max(ys)+min(ys))/2.0
#                 l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
#                 t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
#                 t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
#                 text_raw=f"{t1}-{t2}" if t2 else t1
#         return self._format_text(text_raw)

#     def ocr_plate_gemini_from_path(self, image_path: str):
#         if not GEMINI_READY: return "", ""
#         try:
#             img = Image.open(image_path)
#         except Exception as e:
#             print("Gemini open image error:", e); return "", ""
#         try:
#             model = genai.GenerativeModel('gemini-2.5-flash')
#             prompt = ("Đây là ảnh biển số xe Việt Nam. "
#                       "Hãy trích xuất CHÍNH XÁC chuỗi biển số và chỉ trả về chuỗi đó. "
#                       "VD: '29-P1 123.45' hoặc '50-Z8 888.88'.")
#             resp = model.generate_content([prompt, img])
#             raw = (resp.text or "").strip().replace("\n", " ")
#             return self._format_text(raw)
#         except gexceptions.GoogleAPICallError as e:
#             print("Gemini API error:", e); return "", ""
#         except Exception as e:
#             print("Gemini unknown error:", e); return "", ""

#     @staticmethod
#     def _format_text(text_raw: str):
#         raw=(text_raw or '').replace('-', ' ').replace(' ', '')
#         text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
#         return text_fmt, (text_raw or "")

# # ==================== VIDEO WORKER ====================
# class VideoWorker(QThread):
#     frameSignal = Signal(np.ndarray, str)
#     sceneSignal = Signal(str)
#     roiSignal   = Signal(str, str)
#     infoSignal  = Signal(dict)
#     matchSignal = Signal(str)
#     histSignal  = Signal()

#     def __init__(self, cam_idx: int, api: int, mode: str, models: Models, db: DB,
#                  stable_seconds: float = 1.2, ocr_mode: str = "yolo", title: str = "", parent=None):
#         super().__init__(parent)
#         self.cam_idx = cam_idx
#         self.api = api
#         self.mode = mode            # 'in' | 'out'
#         self.models = models
#         self.db = db
#         self.stable_seconds = stable_seconds
#         self.ocr_mode = ocr_mode    # 'yolo' | 'gemini'
#         self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")

#         self._running = False
#         self.cap = None
#         self.stable_start = 0.0
#         self.captured = False

#     def set_title(self, title: str): self.title = title
#     def set_ocr_mode(self, mode: str): self.ocr_mode = mode
#     def set_mode(self, mode: str): self.mode = mode

#     def run(self):
#         self._running = True
#         self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
#         if not (self.cap and self.cap.isOpened()):
#             self._running = False; return
#         try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         except: pass
#         try: self.cap.set(cv2.CAP_PROP_FPS, 30)
#         except: pass

#         while self._running:
#             ok, frame = self.cap.read()
#             if not ok:
#                 self.stable_start = 0.0; self.captured = False
#                 time.sleep(0.03); continue

#             # KHÔNG vẽ chữ lên ảnh, chỉ letterbox
#             disp = letterbox(frame)
#             self.frameSignal.emit(disp, self.title)

#             plates, boxed = self.models.detect_plates(frame)
#             if not plates:
#                 self.stable_start = 0.0; self.captured = False
#                 time.sleep(0.01); continue

#             best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
#             roi_current = best[1]

#             if self.stable_start == 0.0 or self.captured:
#                 self.stable_start = time.time(); self.captured = False

#             if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
#                 scene_path = save_image(boxed if boxed is not None else frame,
#                                         "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
#                 roi_path   = save_image(roi_current,
#                                         "plate_in_roi" if self.mode=="in" else "plate_out_roi")

#                 if self.ocr_mode == "gemini" and GEMINI_READY:
#                     text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
#                 else:
#                     text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

#                 if text_fmt or text_raw:
#                     now = datetime.now()
#                     d = now.strftime("%d/%m/%Y")
#                     t = now.strftime("%H:%M:%S")
#                     plate = text_fmt or text_raw

#                     self.sceneSignal.emit(scene_path)
#                     self.roiSignal.emit(roi_path, self.mode)

#                     if self.mode == "in":
#                         self.infoSignal.emit({"date_in": d, "time_in": t, "plate_text_in": plate})
#                         if self.db and self.db.ok:
#                             self.db.insert_in(plate, d, t, scene_path)
#                             self.histSignal.emit()
#                     else:
#                         self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
#                         if self.db and self.db.ok:
#                             match = self.db.attach_out(plate, d, t, scene_path)
#                             self.matchSignal.emit(match)
#                             self.histSignal.emit()
#                     self.captured = True

#             time.sleep(0.01)

#         try:
#             if self.cap: self.cap.release()
#         except: pass

#     def stop(self): self._running = False

# # ==================== DELETE DIALOG ====================
# class DeleteDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Xóa lịch sử")
#         self.setModal(True)
#         self.setStyleSheet("""
#             QDialog {
#                 background: #ffffff;
#                 border: 2px solid #e6e6e6;
#                 border-radius: 10px;
#             }
#             QPushButton {
#                 height: 34px; border-radius: 10px; font-weight: 600; padding: 6px 12px;
#             }
#         """)
#         lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
#         lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
#         lay.addWidget(lab)

#         row = QHBoxLayout(); row.setSpacing(12)
#         self.btn_sel = QPushButton("Xóa dòng đã chọn"); self.btn_sel.setStyleSheet("background:#dbeafe;")
#         self.btn_all = QPushButton("Xóa tất cả");       self.btn_all.setStyleSheet("background:#ffe0e0;")
#         self.btn_can = QPushButton("Hủy");              self.btn_can.setStyleSheet("background:#fff9c4;")
#         row.addWidget(self.btn_sel, 1)
#         row.addWidget(self.btn_all, 1)
#         row.addWidget(self.btn_can, 1)
#         lay.addLayout(row)

#         self.btn_sel.clicked.connect(lambda: self.done(1))
#         self.btn_all.clicked.connect(lambda: self.done(2))
#         self.btn_can.clicked.connect(lambda: self.done(0))

# # ==================== MAIN WINDOW ====================
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Desktop App (Giữ xe)")
#         self.setMinimumSize(1400, 900)
#         self._init_theme()

#         self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#         self.db = DB(CONN_STR) if USE_SQL else DB("")

#         self.cam1_worker = None
#         self.cam2_worker = None

#         # Làn + OCR mode
#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.current_ocr_mode = "yolo"

#         self._build_ui()
#         self.show_logo(1); self.show_logo(2)

#         self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.refresh_history); self.hist_timer.start(5000)

#     def _init_theme(self):
#         self.setStyleSheet("""
#         * { color: #000000; }
#         QMainWindow, QWidget { background: #ffffff; }
#         QWidget#SideBar { background: #ffffff; }

#         QGroupBox {
#             background: #ffffff;
#             font-weight: 600;
#             border: 2px solid #e6e6e6;
#             border-radius: 12px;
#             margin-top: 8px;
#             padding-top: 10px;
#         }
#         QGroupBox::title {
#             subcontrol-origin: margin;
#             left: 10px;
#             padding: 0 6px;
#             background: #ffffff;
#         }

#         QFrame[class="card-wrap"] { background: #e6e6e6; border-radius: 14px; }
#         QFrame[class="card"]      { background: #ffffff; border-radius: 12px; }
#         QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; }
#         QLabel[class="title"] {
#             font: 700 18px "Segoe UI";
#             padding: 6px 10px;
#             background: #ffffff;
#             border-radius: 10px;
#         }

#         QPushButton { height: 34px; border-radius: 10px; font-weight: 600; }
#         QPushButton#btnGreen { background: #d1fadf; border: 1px solid #a6f4c5; border-radius: 10px;}
#         QPushButton#btnRed   { background: #ffe0e0; border: 1px solid #ffb3b3; border-radius: 10px;}
#         QPushButton#btnYellow{ background: #fff3bf; border: 1px solid #ffe066; border-radius: 10px;}
#         QLineEdit {
#             height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px;
#         }
#         QTableWidget { background: #ffffff; gridline-color: #e6e6e6; }
#         /* để bo góc nhìn thấy, luôn cho có border */
#         QPushButton { height: 34px; border-radius: 10px; font-weight: 600; border: 1px solid transparent; }

#         QPushButton#btnGreen  { background: #d1fadf; border: 1px solid #a6f4c5; }
#         QPushButton#btnRed    { background: #ffe0e0; border: 1px solid #ffb3b3; }
#         QPushButton#btnYellow { background: #fff3bf; border: 1px solid #ffe066; }

#         /* 2 nút xanh nhạt ở phần làn */
#         QPushButton#btnOneway, QPushButton#btnTwoway {
#             background: #dbeafe;
#             border: 1px solid #bfdbfe;
#         }
#         """)

#     # helper tạo card có trả về label tiêu đề
#     def _make_card(self, title:str, content:QWidget):
#         wrap = QFrame(); wrap.setProperty("class","card-wrap")
#         wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
#         card = QFrame(); card.setProperty("class","card")
#         v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
#         title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
#         hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
#         title_lbl = QLabel(title); title_lbl.setProperty("class","title")
#         hl.addWidget(title_lbl)
#         v.addWidget(title_wrap)
#         v.addWidget(content, 1)
#         wrapL.addWidget(card)
#         return wrap, title_lbl

#     # ---------- UI ----------
#     def _build_ui(self):
#         central = QWidget(); self.setCentralWidget(central)
#         root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

#         # ---------- LEFT: SIDEBAR ----------
#         side = QWidget(objectName="SideBar"); side.setMinimumWidth(450)
#         vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

#         # CAMERA CONTROL
#         gb_camctl = QGroupBox("CAMERA CONTROL")
#         gl_camctl = QGridLayout(gb_camctl)
#         self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
#         self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys()))
#         self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
#         self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys()))
#         self.btn_start1 = QPushButton("Bật Cam 1"); self.btn_start1.setObjectName("btnGreen")
#         self.btn_stop1  = QPushButton("Tắt Cam 1"); self.btn_stop1.setObjectName("btnRed")
#         self.btn_start2 = QPushButton("Bật Cam 2"); self.btn_start2.setObjectName("btnGreen")
#         self.btn_stop2  = QPushButton("Tắt Cam 2"); self.btn_stop2.setObjectName("btnRed")
#         self.btn_start1.clicked.connect(self.start_cam1)
#         self.btn_stop1.clicked.connect(self.stop_cam1)
#         self.btn_start2.clicked.connect(self.start_cam2)
#         self.btn_stop2.clicked.connect(self.stop_cam2)
#         r=0
#         gl_camctl.addWidget(QLabel("Index Cam 1"), r,0); gl_camctl.addWidget(self.spin_cam1, r,1)
#         gl_camctl.addWidget(QLabel("Backend Cam 1"), r,2); gl_camctl.addWidget(self.cb_api1, r,3); r+=1
#         gl_camctl.addWidget(QLabel("Index Cam 2"), r,0); gl_camctl.addWidget(self.spin_cam2, r,1)
#         gl_camctl.addWidget(QLabel("Backend Cam 2"), r,2); gl_camctl.addWidget(self.cb_api2, r,3); r+=1
#         gl_camctl.addWidget(self.btn_start1, r,0,1,2); gl_camctl.addWidget(self.btn_stop1, r,2,1,2); r+=1
#         gl_camctl.addWidget(self.btn_start2, r,0,1,2); gl_camctl.addWidget(self.btn_stop2, r,2,1,2)
#         vside.addWidget(gb_camctl)

#         # ===== ĐIỀU KHIỂN LÀN =====
#         gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
#         vl_lane = QVBoxLayout(gb_lane)
#         self.lbl_lane1 = QLabel("")
#         self.lbl_lane2 = QLabel("")
#         lane_info = QLabel("Mỗi làn: cam trước: cam1, cam sau: cam2"); lane_info.setStyleSheet("font-style: italic;")

#         btns_row = QHBoxLayout()
#         self.btn_oneway = QPushButton("1 chiều (Toggle)"); self.btn_oneway.setObjectName("btnOneway")
#         self.btn_twoway = QPushButton("2 chiều (Đảo)"); self.btn_twoway.setObjectName("btnTwoway")
#         self.btn_reset_lane = QPushButton("Reset làn xe"); self.btn_reset_lane.setObjectName("btnYellow")
#         # self.btn_oneway.setStyleSheet("background:#dbeafe; font-weight:600;")
#         # self.btn_twoway.setStyleSheet("background:#dbeafe; font-weight:600;")
#         # self.btn_reset_lane.setStyleSheet("font-weight:600;")
#         self.btn_oneway.clicked.connect(self.on_one_way_clicked)
#         self.btn_twoway.clicked.connect(self.on_two_way_clicked)
#         self.btn_reset_lane.clicked.connect(self.on_reset_lanes)
#         btns_row.addWidget(self.btn_oneway); btns_row.addWidget(self.btn_twoway); btns_row.addWidget(self.btn_reset_lane)

#         vl_lane.addWidget(self.lbl_lane1); vl_lane.addWidget(self.lbl_lane2); vl_lane.addWidget(lane_info); vl_lane.addLayout(btns_row)
#         vside.addWidget(gb_lane)

#         # OCR MODEL
#         gb_ocr = QGroupBox("OCR MODEL")
#         vb_ocr = QVBoxLayout(gb_ocr)
#         self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)"); self.rb_yolo.setChecked(True)
#         self.rb_gem  = QRadioButton("Dùng Gemini AI")
#         vb_ocr.addWidget(self.rb_yolo); vb_ocr.addWidget(self.rb_gem)
#         self.rb_yolo.toggled.connect(self.on_ocr_mode_changed)
#         self.rb_gem.toggled.connect(self.on_ocr_mode_changed)
#         if not GEMINI_READY:
#             self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY trong môi trường/.env → mặc định dùng YOLO")
#         vside.addWidget(gb_ocr)

#         # THÔNG TIN XE VÀO
#         gb_in = QGroupBox("THÔNG TIN XE VÀO")
#         gl_in = QGridLayout(gb_in)
#         self.ed_date_in = QLineEdit(); self.ed_time_in = QLineEdit(); self.ed_plate_in = QLineEdit()
#         gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
#         gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
#         gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
#         vside.addWidget(gb_in)

#         # THÔNG TIN XE RA
#         gb_out = QGroupBox("THÔNG TIN XE RA")
#         gl_out = QGridLayout(gb_out)
#         self.ed_date_out = QLineEdit(); self.ed_time_out = QLineEdit(); self.ed_plate_out = QLineEdit()
#         gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
#         gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
#         gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
#         vside.addWidget(gb_out)

#         # ===== BẢNG LỊCH SỬ: groupbox các nút =====
#         gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
#         v_hist_btns = QVBoxLayout(gb_hist_btns)
#         self.btn_show_history = QPushButton("Xem bảng lịch sử"); self.btn_show_history.setStyleSheet("background:#E6F4EA; font-weight:600;")
#         row_cmd = QHBoxLayout()
#         self.btn_export_hist  = QPushButton("Export Excel"); self.btn_export_hist.setStyleSheet("background:#dbeafe; font-weight:600;")
#         self.btn_delete_hist  = QPushButton("Xóa bảng");      self.btn_delete_hist.setStyleSheet("background:#dbeafe; font-weight:600;")
#         row_cmd.addWidget(self.btn_export_hist, 1); row_cmd.addWidget(self.btn_delete_hist, 1)
#         v_hist_btns.addWidget(self.btn_show_history); v_hist_btns.addLayout(row_cmd)
#         self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide(); self.btn_hide_history.setStyleSheet("background:#FCE8E6; font-weight:600;")
#         v_hist_btns.addWidget(self.btn_hide_history)
#         self.btn_show_history.clicked.connect(self.show_history_view)
#         self.btn_hide_history.clicked.connect(self.show_main_view)
#         self.btn_export_hist.clicked.connect(self.on_export_excel)
#         self.btn_delete_hist.clicked.connect(self.on_delete_history)
#         vside.addWidget(gb_hist_btns)

#         vside.addStretch(1)
#         root.addWidget(side)

#         # ---------- RIGHT ----------
#         right_container = QVBoxLayout()

#         # ===== Main view =====
#         self.main_view = QWidget()
#         main_layout = QVBoxLayout(self.main_view)

#         top = QHBoxLayout()
#         self.lbl_cam1 = QLabel(); self.lbl_cam1.setMinimumSize(PANEL_W, PANEL_H)
#         self.lbl_cam2 = QLabel(); self.lbl_cam2.setMinimumSize(PANEL_W, PANEL_H)
#         for lbl in (self.lbl_cam1, self.lbl_cam2):
#             lbl.setScaledContents(False)
#             lbl.setAlignment(Qt.AlignCenter)  # căn giữa pixmap
#             lbl.setStyleSheet("border-radius:12px; background:#f7efe8;")
#         cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1)
#         cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
#         top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1)
#         main_layout.addLayout(top)

#         bottom = QHBoxLayout()
#         self.lbl_scene = QLabel(); self.lbl_scene.setMinimumSize(PANEL_W, PANEL_H); self.lbl_scene.setScaledContents(False); self.lbl_scene.setAlignment(Qt.AlignCenter); self.lbl_scene.setStyleSheet("border-radius:12px; background:#fff;")
#         self.lbl_roi   = QLabel(); self.lbl_roi.setMinimumSize(PANEL_W, PANEL_H); self.lbl_roi.setScaledContents(False); self.lbl_roi.setAlignment(Qt.AlignCenter); self.lbl_roi.setStyleSheet("border-radius:12px; background:#fff;")
#         scene_card, _ = self._make_card("Image_BOX",  self.lbl_scene)
#         roi_card,   _ = self._make_card("ROI_Plate", self.lbl_roi)
#         bottom.addWidget(scene_card, 1); bottom.addWidget(roi_card, 1)
#         main_layout.addLayout(bottom)

#         self.info_group = QGroupBox("Thông tin chi tiết")
#         info_layout = QGridLayout(self.info_group)
#         self.txt_date_in  = QLabel("--/--/----"); self.txt_time_in  = QLabel("--:--:--")
#         self.txt_plate_in = QLabel("---"); self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_date_out = QLabel("--/--/----"); self.txt_time_out = QLabel("--:--:--")
#         self.txt_plate_out= QLabel("---"); self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_match    = QLabel("")
#         r=0
#         info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
#         info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
#         info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
#         info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
#         info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
#         info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
#         info_layout.addWidget(QLabel("So khớp:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,5)
#         main_layout.addWidget(self.info_group)

#         # ===== History view =====
#         self.history_view = QWidget()
#         hist_layout = QVBoxLayout(self.history_view)
#         hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)")
#         hist_v = QVBoxLayout(hist_group)

#         self.tbl_hist = QTableWidget(0, 10)
#         self.tbl_hist.setHorizontalHeaderLabels(["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"])
#         header = self.tbl_hist.horizontalHeader()
#         hfont = QFont(header.font()); hfont.setBold(True); header.setFont(hfont)
#         self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
#         self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
#         self.tbl_hist.setAlternatingRowColors(True)
#         header.setSectionResizeMode(QHeaderView.Stretch)
#         hist_v.addWidget(self.tbl_hist)
#         hist_layout.addWidget(hist_group)

#         self.stacked = QStackedWidget()
#         self.stacked.addWidget(self.main_view)
#         self.stacked.addWidget(self.history_view)
#         self.stacked.setCurrentIndex(0)
#         right_container.addWidget(self.stacked, 1)
#         root.addLayout(right_container, 1)

#         self.update_lane_labels()
#         self.update_titles_and_modes()

#     # ---------- LANE ----------
#     def update_lane_labels(self):
#         self.lbl_lane1.setText(f"Làn 1: {self.lane1_dir} (cam trước: cam1, cam sau: cam2)")
#         self.lbl_lane2.setText(f"Làn 2: {self.lane2_dir} (cam trước: cam1, cam sau: cam2)")

#     def update_titles_and_modes(self):
#         self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
#         self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
#         if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
#         if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")

#     @Slot() 
#     def on_reset_lanes(self):
#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.update_lane_labels(); self.update_titles_and_modes()
#         # có yêu cầu: reset hiển thị logo
#         self.show_logo(1); self.show_logo(2)

#     @Slot() 
#     def on_one_way_clicked(self):
#         if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
#         else:                       self.lane1_dir="RA";  self.lane2_dir="RA"
#         self.one_way_toggle_vao = not self.one_way_toggle_vao
#         self.update_lane_labels(); self.update_titles_and_modes()

#     @Slot() 
#     def on_two_way_clicked(self):
#         if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
#         else:                   self.lane1_dir="RA";  self.lane2_dir="VÀO"
#         self.two_way_toggle = not self.two_way_toggle
#         self.update_lane_labels(); self.update_titles_and_modes()

#     # ---------- OCR mode ----------
#     @Slot()
#     def on_ocr_mode_changed(self):
#         self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
#         if self.rb_gem.isChecked() and not GEMINI_READY:
#             QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
#             self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
#         if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
#         if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

#     # ---------- VIEW ----------
#     def show_history_view(self):
#         self.stacked.setCurrentIndex(1); self.btn_show_history.hide(); self.btn_hide_history.show(); self.refresh_history()

#     def show_main_view(self):
#         self.stacked.setCurrentIndex(0); self.btn_hide_history.hide(); self.btn_show_history.show()

#     # ---------- EXPORT / DELETE ----------
#     @Slot()
#     def on_export_excel(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
#         if df.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
#         path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
#         if not path: return
#         try: df.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
#         except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

#     @Slot()
#     def on_delete_history(self):
#         if not (self.db and self.db.ok):
#             QMessageBox.warning(self, "Xóa", "Chưa kết nối DB, không thể xóa."); return
#         dlg = DeleteDialog(self)
#         # mở dialog lệch bên phải
#         g = self.geometry(); dlg.adjustSize()
#         dlg.move(self.mapToGlobal(QPoint(g.width()-dlg.width()-40, 140)))
#         res = dlg.exec()
#         if res == 1:
#             rows = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
#             if not rows: QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào."); return
#             cols = [self.tbl_hist.horizontalHeaderItem(i).text() for i in range(self.tbl_hist.columnCount())]
#             if "ID" not in cols: QMessageBox.warning(self, "Xóa", "Không tìm thấy cột ID."); return
#             id_col = cols.index("ID"); ids = []
#             for r in rows:
#                 item = self.tbl_hist.item(r, id_col)
#                 if item: ids.append(item.text())
#             if not ids: QMessageBox.information(self, "Xóa", "Không lấy được ID các dòng chọn."); return
#             self.db.delete_by_ids(ids); self.refresh_history()
#         elif res == 2:
#             self.db.delete_all(); self.refresh_history()
#         else:
#             return

#     # ---------- helpers ----------
#     def qpix_logo(self):
#         if os.path.exists(LOGO_PATH):
#             pm = QPixmap(LOGO_PATH).scaled(PANEL_W, PANEL_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         else:
#             # fallback canvas rỗng
#             pm = QPixmap.fromImage(bgr_to_qimage(letterbox(None)))
#         return pm

#     def show_logo(self, which: int):
#         pm = self.qpix_logo()
#         if which == 1: self.lbl_cam1.setPixmap(pm)
#         else: self.lbl_cam2.setPixmap(pm)

#     @Slot(np.ndarray, str)
#     def on_frame(self, frame_bgr, title):
#         qimg = bgr_to_qimage(letterbox(frame_bgr))
#         sender = self.sender()
#         if sender is self.cam1_worker:
#             self.lbl_cam1.setPixmap(QPixmap.fromImage(qimg))
#         elif sender is self.cam2_worker:
#             self.lbl_cam2.setPixmap(QPixmap.fromImage(qimg))

#     @Slot(str)
#     def on_scene(self, path):
#         if os.path.exists(path):
#             bgr = cv2.imread(path)
#             self.lbl_scene.setPixmap(QPixmap.fromImage(bgr_to_qimage(letterbox(bgr))))

#     @Slot(str, str)
#     def on_roi(self, roi_path, mode):
#         if os.path.exists(roi_path):
#             bgr = cv2.imread(roi_path)
#             self.lbl_roi.setPixmap(QPixmap.fromImage(bgr_to_qimage(letterbox(bgr))))

#     @Slot(dict)
#     def on_info(self, info):
#         if "date_in" in info:  self.txt_date_in.setText(info["date_in"]);   self.ed_date_in.setText(info["date_in"])
#         if "time_in" in info:  self.txt_time_in.setText(info["time_in"]);   self.ed_time_in.setText(info["time_in"])
#         if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
#         if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
#         if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
#         if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

#     @Slot(str)
#     def on_match(self, txt): self.txt_match.setText(txt.upper())

#     @Slot()
#     def refresh_history(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
#         if df.empty:
#             self.tbl_hist.setRowCount(0)
#             cols = ["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             self.tbl_hist.setColumnCount(len(cols)); self.tbl_hist.setHorizontalHeaderLabels(cols)
#             hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#             self.tbl_hist.horizontalHeader().setFont(hfont); return

#         cols = list(df.columns)
#         self.tbl_hist.setRowCount(len(df)); self.tbl_hist.setColumnCount(len(cols))
#         self.tbl_hist.setHorizontalHeaderLabels(cols)
#         hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#         self.tbl_hist.horizontalHeader().setFont(hfont)
#         self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
#         for i in range(len(df)):
#             for j, col in enumerate(cols):
#                 val = str(df.iloc[i, j]); item = QTableWidgetItem(val)
#                 item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
#                 self.tbl_hist.setItem(i, j, item)

#     # ---------- camera controls ----------
#     def _connect_worker(self, w: VideoWorker):
#         w.frameSignal.connect(self.on_frame)
#         w.sceneSignal.connect(self.on_scene)
#         w.roiSignal.connect(self.on_roi)
#         w.infoSignal.connect(self.on_info)
#         w.matchSignal.connect(self.on_match)
#         w.histSignal.connect(self.refresh_history)

#     def start_cam_generic(self, which: int):
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#             return
#         if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
#         if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

#         ocr_mode = self.current_ocr_mode
#         if which == 1:
#             idx = int(self.spin_cam1.value()); api = API_MAP[self.cb_api1.currentText()]
#             mode = "in" if self.lane1_dir=="VÀO" else "out"
#             title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam1_worker = VideoWorker(idx, api, mode, self.models, self.db, STABLE_SECONDS_IN, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam1_worker); self.cam1_worker.start()
#         else:
#             idx = int(self.spin_cam2.value()); api = API_MAP[self.cb_api2.currentText()]
#             mode = "in" if self.lane2_dir=="VÀO" else "out"
#             title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam2_worker = VideoWorker(idx, api, mode, self.models, self.db, STABLE_SECONDS_OUT, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam2_worker); self.cam2_worker.start()

#     def stop_cam_generic(self, which: int):
#         worker = self.cam1_worker if which==1 else self.cam2_worker
#         if worker and worker.isRunning():
#             worker.stop(); worker.wait(1000)
#         if which==1: self.cam1_worker = None; self.show_logo(1)
#         else:        self.cam2_worker = None; self.show_logo(2)

#     def start_cam1(self): self.start_cam_generic(1)
#     def stop_cam1(self):  self.stop_cam_generic(1)
#     def start_cam2(self): self.start_cam_generic(2)
#     def stop_cam2(self):  self.stop_cam_generic(2)

#     def closeEvent(self, event):
#         try: self.stop_cam_generic(1); self.stop_cam_generic(2)
#         except: pass
#         super().closeEvent(event)

# # ==================== MAIN ====================
# def main():
#     QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
#     app = QApplication(sys.argv)
#     app.setStyle("Fusion")
#     w = MainWindow(); w.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()



































# -*- coding: utf-8 -*-
"""
PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini)
- FIXED: nút bo góc luôn đúng (mọi state), không còn “vuông”
- FIXED: ảnh/logo/camera căn GIỮA tuyệt đối (hết lệch phải trên DPI 125–150%)
- Không vẽ chữ lên ảnh; letterbox/scale chỉ làm ở GUI
"""

import os, sys, time, cv2, numpy as np, pandas as pd
from datetime import datetime

# ---- HiDPI ----
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
    QFileDialog, QHeaderView, QDialog
)

# ---- Optional SQL ----
USE_SQL = True
try:
    import pyodbc
except Exception:
    USE_SQL = False

# ---- YOLO ----
from ultralytics import YOLO

# ---- Gemini (optional) ----
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

# ==================== CONFIG ====================
DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")  # <- thay logo của bạn

CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"
)

PANEL_W, PANEL_H = 640, 360
PANEL_BG = (255, 255, 255)  # nền trắng để không viền be

API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
           "six":"6","seven":"7","eight":"8","nine":"9"}


# ==================== UTILITIES ====================
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

def bgr_to_qimage(bgr):
    """BGR -> QImage RGB888"""
    if bgr is None:
        bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

def save_image(img, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, img)
    return path

def norm_char(x): return OCR_MAP.get(str(x), str(x))
def plate_norm(s: str) -> str: return (s or "").replace("-", "").replace(" ", "").upper()

def has_boxes(r):
    try:
        return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
    except: return False

def preprocess_for_ocr(roi):
    if roi is None: return None
    if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe,(3,3),0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)


# ==================== DB LAYER ====================
class DB:
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

    def insert_in(self, plate, d, t, img_path):
        if not self.ok: return
        try:
            self.cur.execute("""
                INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
                VALUES (?,?,?,?,?)
            """, (plate, d, t, img_path, 'PENDING'))
        except Exception as e: print("insert_in error:", e)

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

    def delete_by_ids(self, ids):
        if not self.ok or not ids: return
        try:
            for sid in ids:
                self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
        except Exception as e: print("delete_by_ids error:", e)

    def delete_all(self):
        if not self.ok: return
        try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
        except Exception as e: print("delete_all error:", e)


# ==================== YOLO/GEMINI WRAPPERS ====================
class Models:
    def __init__(self, det_path: str, ocr_path: str):
        self.ok = True; self.err = ""
        try:
            self.det = YOLO(det_path)
            self.ocr = YOLO(ocr_path)
        except Exception as e:
            self.ok = False; self.err = str(e)

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

    def ocr_plate_gemini_from_path(self, image_path: str):
        if not GEMINI_READY: return "", ""
        try:
            img = Image.open(image_path)
        except Exception as e:
            print("Gemini open image error:", e); return "", ""
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
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

    @staticmethod
    def _format_text(text_raw: str):
        raw=(text_raw or '').replace('-', ' ').replace(' ', '')
        text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
        return text_fmt, (text_raw or "")


# ==================== VIDEO WORKER ====================
class VideoWorker(QThread):
    frameSignal = Signal(np.ndarray, str)
    sceneSignal = Signal(str)
    roiSignal   = Signal(str, str)
    infoSignal  = Signal(dict)
    matchSignal = Signal(str)
    histSignal  = Signal()

    def __init__(self, cam_idx: int, api: int, mode: str, models: Models, db: DB,
                 stable_seconds: float = 1.2, ocr_mode: str = "yolo", title: str = "", parent=None):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.api = api
        self.mode = mode            # 'in' | 'out'
        self.models = models
        self.db = db
        self.stable_seconds = stable_seconds
        self.ocr_mode = ocr_mode    # 'yolo' | 'gemini'
        self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")

        self._running = False
        self.cap = None
        self.stable_start = 0.0
        self.captured = False

    def set_title(self, title: str): self.title = title
    def set_ocr_mode(self, mode: str): self.ocr_mode = mode
    def set_mode(self, mode: str): self.mode = mode

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

            # Emit frame GỐC (không letterbox), GUI sẽ scale/căn giữa
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
                    else:
                        self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
                        if self.db and self.db.ok:
                            match = self.db.attach_out(plate, d, t, scene_path)
                            self.matchSignal.emit(match)
                            self.histSignal.emit()
                    self.captured = True

            time.sleep(0.01)

        try:
            if self.cap: self.cap.release()
        except: pass

    def stop(self): self._running = False



# ==================== 6.5 HISTORY LOADER WORKER (MỚI) ====================
class HistoryLoaderWorker(QThread):
    """Luồng riêng để tải dữ liệu lịch sử từ DB."""
    resultReady = Signal(pd.DataFrame) # Signal trả về DataFrame kết quả

    def __init__(self, db: DB, start_time=None, end_time=None, status_filter=None, plate_filter=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.start_time = start_time
        self.end_time = end_time
        self.status_filter = status_filter
        self.plate_filter = plate_filter

    def run(self):
        """Thực hiện truy vấn DB trong luồng này."""
        if self.db and self.db.ok:
            df = self.db.fetch_history_df(limit=10000, # Giữ nguyên limit hoặc giảm nếu cần
                                         start_time=self.start_time,
                                         end_time=self.end_time,
                                         status_filter=self.status_filter,
                                         plate_filter=self.plate_filter)
            self.resultReady.emit(df) # Gửi kết quả về luồng chính
        else:
            # Trả về DataFrame rỗng nếu DB lỗi
            self.resultReady.emit(pd.DataFrame())


# ==================== DELETE DIALOG ====================
class DeleteDialog(QDialog):
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
            QPushButton {
                height: 34px; border-radius: 10px; font-weight: 600; padding: 6px 12px;
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

        self.btn_sel.clicked.connect(lambda: self.done(1))
        self.btn_all.clicked.connect(lambda: self.done(2))
        self.btn_can.clicked.connect(lambda: self.done(0))


# ==================== MAIN WINDOW ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop App (Giữ xe)")
        self.setMinimumSize(1400, 900)
        self._init_theme()

        self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
        if not self.models.ok:
            QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
        self.db = DB(CONN_STR) if USE_SQL else DB("")

        self.cam1_worker = None
        self.cam2_worker = None

        # Làn + OCR mode
        self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True; self.two_way_toggle = True
        self.current_ocr_mode = "yolo"
        

        self._build_ui()
        self.show_logo(1); self.show_logo(2)

        self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.refresh_history); self.hist_timer.start(5000)

    # ---------- THEME ----------
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
        QFrame[class="card"]      { background: #ffffff; border-radius: 12px; }
        QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; }
        QLabel[class="title"] {
            font: 700 18px "Segoe UI";
            padding: 6px 10px;
            background: #ffffff;
            border-radius: 10px;
        }

        /* Nút chuẩn */
        QPushButton {
            height: 34px; padding: 4px 10px;
            font-weight: 600;
            border-radius: 10px;
            border: 1px solid transparent;
            background: #ffffff;
        }
        QPushButton:disabled{
            background: #f5f5f5; border: 1px solid #e5e7eb; color: #9ca3af;
        }

        /* Màu nút theo id */
        QPushButton#btnGreen  { background: #d1fadf; border: 1px solid #a6f4c5; }
        QPushButton#btnRed    { background: #ffe0e0; border: 1px solid #ffb3b3; }
        QPushButton#btnYellow { background: #fff3bf; border: 1px solid #ffe066; }
        QPushButton#btnOneway,
        QPushButton#btnTwoway { background: #dbeafe; border: 1px solid #bfdbfe; }

        /* NEW: kiểu nút bo tròn (pill) — gán property class="pill" ở code */
        QPushButton[class="pill"] {
            border-radius: 18px;              /* đổi 18 → to/nhỏ hơn tùy thích */
            padding: 8px 16px;                /* padding lớn nhìn cân */
        }

        QLineEdit {
            height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px;
        }
        QTableWidget { background: #ffffff; gridline-color: #e6e6e6; }
        """)


    # ---------- helpers (UI) ----------
    def _normalize_button(self, *btns):
        for b in btns:
            try:
                b.setAutoDefault(False)
                b.setDefault(False)
                b.setFlat(False)
                b.setFocusPolicy(Qt.NoFocus)
            except:
                pass

    def _make_card(self, title:str, content:QWidget):
        wrap = QFrame(); wrap.setProperty("class","card-wrap")
        wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
        card = QFrame(); card.setProperty("class","card")
        v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
        title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
        hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
        title_lbl = QLabel(title); title_lbl.setProperty("class","title")
        hl.addWidget(title_lbl)
        v.addWidget(title_wrap)
        v.addWidget(content, 1)
        wrapL.addWidget(card)
        return wrap, title_lbl

    def _set_centered_pixmap(self, lbl: QLabel, src):
        """Scale theo contentsRect + DPR để giữa tuyệt đối."""
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

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

        # ---------- LEFT: SIDEBAR ----------
        side = QWidget(objectName="SideBar"); side.setMinimumWidth(450)
        vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

        # CAMERA CONTROL
        gb_camctl = QGroupBox("CAMERA CONTROL")
        gl_camctl = QGridLayout(gb_camctl)
        self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
        self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys()))
        self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
        self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys()))
        self.btn_start1 = QPushButton("Bật Cam 1"); self.btn_start1.setObjectName("btnGreen")
        self.btn_stop1  = QPushButton("Tắt Cam 1"); self.btn_stop1.setObjectName("btnRed")
        self.btn_start2 = QPushButton("Bật Cam 2"); self.btn_start2.setObjectName("btnGreen")
        self.btn_stop2  = QPushButton("Tắt Cam 2"); self.btn_stop2.setObjectName("btnRed")
        self._normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)

        self.btn_start1.clicked.connect(self.start_cam1)
        self.btn_stop1.clicked.connect(self.stop_cam1)
        self.btn_start2.clicked.connect(self.start_cam2)
        self.btn_stop2.clicked.connect(self.stop_cam2)
        r=0
        gl_camctl.addWidget(QLabel("Index Cam 1"), r,0); gl_camctl.addWidget(self.spin_cam1, r,1)
        gl_camctl.addWidget(QLabel("Backend Cam 1"), r,2); gl_camctl.addWidget(self.cb_api1, r,3); r+=1
        gl_camctl.addWidget(QLabel("Index Cam 2"), r,0); gl_camctl.addWidget(self.spin_cam2, r,1)
        gl_camctl.addWidget(QLabel("Backend Cam 2"), r,2); gl_camctl.addWidget(self.cb_api2, r,3); r+=1
        gl_camctl.addWidget(self.btn_start1, r,0,1,2); gl_camctl.addWidget(self.btn_stop1, r,2,1,2); r+=1
        gl_camctl.addWidget(self.btn_start2, r,0,1,2); gl_camctl.addWidget(self.btn_stop2, r,2,1,2)
        vside.addWidget(gb_camctl)

        # ===== ĐIỀU KHIỂN LÀN =====
        gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
        vl_lane = QVBoxLayout(gb_lane)
        vl_lane.setSpacing(10)

        # Hàng nút 1 chiều + 2 chiều
        row_lane = QHBoxLayout()
        row_lane.setSpacing(12)

        self.btn_oneway = QPushButton("1 chiều")
        self.btn_oneway.setObjectName("btnOneway")
        self.btn_oneway.setProperty("class", "pill")   # NEW: bo tròn

        self.btn_twoway = QPushButton("2 chiều")
        self.btn_twoway.setObjectName("btnTwoway")
        self.btn_twoway.setProperty("class", "pill")   # NEW: bo tròn

        row_lane.addWidget(self.btn_oneway, 1)
        row_lane.addWidget(self.btn_twoway, 1)
        vl_lane.addLayout(row_lane)

        # Nút Reset nằm dưới
        self.btn_reset_lane = QPushButton("Reset làn")
        self.btn_reset_lane.setObjectName("btnYellow")
        self.btn_reset_lane.setProperty("class", "pill")  # NEW: bo tròn
        vl_lane.addWidget(self.btn_reset_lane)

        # Kết nối signal
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
        self.ed_date_in = QLineEdit(); self.ed_time_in = QLineEdit(); self.ed_plate_in = QLineEdit()
        gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
        gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
        gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
        vside.addWidget(gb_in)

        # THÔNG TIN XE RA
        gb_out = QGroupBox("THÔNG TIN XE RA")
        gl_out = QGridLayout(gb_out)
        self.ed_date_out = QLineEdit(); self.ed_time_out = QLineEdit(); self.ed_plate_out = QLineEdit()
        gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
        gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
        gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
        vside.addWidget(gb_out)

        # ===== BẢNG LỊCH SỬ: groupbox các nút =====
        gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
        v_hist_btns = QVBoxLayout(gb_hist_btns)
        self.btn_show_history = QPushButton("Xem bảng lịch sử"); self.btn_show_history.setObjectName("btnGreen")
        row_cmd = QHBoxLayout()
        self.btn_export_hist  = QPushButton("Export Excel"); self.btn_export_hist.setObjectName("btnOneway")
        self.btn_delete_hist  = QPushButton("Xóa bảng");      self.btn_delete_hist.setObjectName("btnTwoway")
        self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.setObjectName("btnRed"); self.btn_hide_history.hide()
        self._normalize_button(self.btn_show_history, self.btn_export_hist, self.btn_delete_hist, self.btn_hide_history)

        row_cmd.addWidget(self.btn_export_hist, 1); row_cmd.addWidget(self.btn_delete_hist, 1)
        v_hist_btns.addWidget(self.btn_show_history)
        v_hist_btns.addLayout(row_cmd)
        v_hist_btns.addWidget(self.btn_hide_history)

        self.btn_show_history.clicked.connect(self.on_show_all_history_clicked)
        self.btn_hide_history.clicked.connect(self.show_main_view)
        self.btn_export_hist.clicked.connect(self.on_export_excel)
        self.btn_delete_hist.clicked.connect(self.on_delete_history)

        vside.addWidget(gb_hist_btns)
        vside.addStretch(1)
        root.addWidget(side)

        # ---------- RIGHT ----------
        right_container = QVBoxLayout()

        # ===== Main view =====
        self.main_view = QWidget()
        main_layout = QVBoxLayout(self.main_view)

        top = QHBoxLayout()
        self.lbl_cam1 = QLabel(); self.lbl_cam1.setObjectName("camView")
        self.lbl_cam2 = QLabel(); self.lbl_cam2.setObjectName("camView")
        for lbl in (self.lbl_cam1, self.lbl_cam2):
            lbl.setScaledContents(False)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
            lbl.setMinimumSize(PANEL_W, PANEL_H)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1)
        cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
        top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1)
        main_layout.addLayout(top)

        bottom = QHBoxLayout()
        self.lbl_scene = QLabel(); self.lbl_scene.setScaledContents(False); self.lbl_scene.setAlignment(Qt.AlignCenter); self.lbl_scene.setStyleSheet("background:#ffffff; border-radius:12px;"); self.lbl_scene.setMinimumSize(PANEL_W, PANEL_H)
        self.lbl_roi   = QLabel(); self.lbl_roi.setScaledContents(False);   self.lbl_roi.setAlignment(Qt.AlignCenter);   self.lbl_roi.setStyleSheet("background:#ffffff; border-radius:12px;");   self.lbl_roi.setMinimumSize(PANEL_W, PANEL_H)
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
        self.txt_match    = QLabel("")
        r=0
        info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
        info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
        info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
        info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
        info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
        info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
        info_layout.addWidget(QLabel("So khớp:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,5)
        main_layout.addWidget(self.info_group)

        # ===== History view =====
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
        self.tbl_hist.setAlternatingRowColors(True)
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

    def update_titles_and_modes(self):
        self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
        self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
        if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
        if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")

    @Slot()
    def on_reset_lanes(self):
        self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True; self.two_way_toggle = True
        self.update_titles_and_modes()
        self.show_logo(1); self.show_logo(2)

    @Slot()
    def on_one_way_clicked(self):
        if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
        else:                       self.lane1_dir="RA";  self.lane2_dir="RA"
        self.one_way_toggle_vao = not self.one_way_toggle_vao
        self.update_titles_and_modes()

    @Slot()
    def on_two_way_clicked(self):
        if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
        else:                   self.lane1_dir="RA";  self.lane2_dir="VÀO"
        self.two_way_toggle = not self.two_way_toggle
        self.update_titles_and_modes()

    # ---------- OCR mode ----------
    @Slot()
    def on_ocr_mode_changed(self):
        self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
        if self.rb_gem.isChecked() and not GEMINI_READY:
            QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
            self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
        if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
        if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

    # ---------- VIEW ----------
    def show_history_view(self):
        self.stacked.setCurrentIndex(1); self.btn_show_history.hide(); self.btn_hide_history.show(); self.refresh_history()

    def show_main_view(self):
        self.stacked.setCurrentIndex(0); self.btn_hide_history.hide(); self.btn_show_history.show()

    # ---------- EXPORT / DELETE ----------
    @Slot()
    def on_export_excel(self):
        df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
        if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
        if df.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
        if not path: return
        try: df.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
        except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

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

    # ---------- image helpers ----------
    def qpix_logo(self):
        if os.path.exists(LOGO_PATH):
            # để GUI scale, lấy full-res
            pm = QPixmap(LOGO_PATH)
        else:
            pm = QPixmap.fromImage(bgr_to_qimage(letterbox(None)))
        return pm

    def show_logo(self, which: int):
        pm = self.qpix_logo()
        if which == 1: self._set_centered_pixmap(self.lbl_cam1, pm)
        else:          self._set_centered_pixmap(self.lbl_cam2, pm)

    @Slot(np.ndarray, str)
    def on_frame(self, frame_bgr, title):
        sender = self.sender()
        if sender is self.cam1_worker:
            self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
        elif sender is self.cam2_worker:
            self._set_centered_pixmap(self.lbl_cam2, frame_bgr)

    @Slot(str)
    def on_scene(self, path):
        if os.path.exists(path):
            bgr = cv2.imread(path)
            self._set_centered_pixmap(self.lbl_scene, bgr)

    @Slot(str, str)
    def on_roi(self, roi_path, mode):
        if os.path.exists(roi_path):
            bgr = cv2.imread(roi_path)
            self._set_centered_pixmap(self.lbl_roi, bgr)

    @Slot(dict)
    def on_info(self, info):
        if "date_in" in info:  self.txt_date_in.setText(info["date_in"]);   self.ed_date_in.setText(info["date_in"])
        if "time_in" in info:  self.txt_time_in.setText(info["time_in"]);   self.ed_time_in.setText(info["time_in"])
        if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
        if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
        if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
        if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

    @Slot(str)
    def on_match(self, txt): self.txt_match.setText(txt.upper())

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

    # ---------- camera controls ----------
    def _connect_worker(self, w: VideoWorker):
        w.frameSignal.connect(self.on_frame)
        w.sceneSignal.connect(self.on_scene)
        w.roiSignal.connect(self.on_roi)
        w.infoSignal.connect(self.on_info)
        w.matchSignal.connect(self.on_match)
        w.histSignal.connect(self.refresh_history)

    def start_cam_generic(self, which: int):
        if not self.models.ok:
            QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
            return
        if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
        if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

        ocr_mode = self.current_ocr_mode
        if which == 1:
            idx = int(self.spin_cam1.value()); api = API_MAP[self.cb_api1.currentText()]
            mode = "in" if self.lane1_dir=="VÀO" else "out"
            title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
            self.cam1_worker = VideoWorker(idx, api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
            self._connect_worker(self.cam1_worker); self.cam1_worker.start()
        else:
            idx = int(self.spin_cam2.value()); api = API_MAP[self.cb_api2.currentText()]
            mode = "in" if self.lane2_dir=="VÀO" else "out"
            title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
            self.cam2_worker = VideoWorker(idx, api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
            self._connect_worker(self.cam2_worker); self.cam2_worker.start()

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

    def resizeEvent(self, e):
        # tái scale khi đổi kích thước cửa sổ
        for lbl in (self.lbl_cam1, self.lbl_cam2, self.lbl_scene, self.lbl_roi):
            pm = lbl.pixmap()
            if pm: self._set_centered_pixmap(lbl, pm)
        super().resizeEvent(e)

    def closeEvent(self, event):
        try: self.stop_cam_generic(1); self.stop_cam_generic(2)
        except: pass
        super().closeEvent(event)


# ==================== MAIN ====================
def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
