# # -*- coding: utf-8 -*-
# """
# =========================================================
# = PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini) =
# =========================================================
#     # ... (Các comment mô tả giữ nguyên) ...
# """

# # ==================== 1. IMPORT ====================

# import os, sys, time, cv2, traceback
# import numpy as np, pandas as pd
# from datetime import datetime
# from ultralytics import YOLO

# # ---- 1.1 HiDPI Cấu hình HiDPI (Độ phân giải cao) ----
# os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
# os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# # ---- 1.2 Import các thư viện PySide6 ----
# from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint, QUrl, QDateTime, QDate, QTime
# from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
# from PySide6.QtMultimedia import QSoundEffect
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
#     QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
#     QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
#     QFileDialog, QHeaderView, QDialog, QDateTimeEdit,
#     QDateEdit, QTimeEdit, QCheckBox
# )

# # ---- 1.3 Optional SQL ----
# USE_SQL = True
# try:
#     import pyodbc
# except Exception:
#     USE_SQL = False

# # ---- 1.4 Gemini API (optional) ----
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
# OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
# SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
# LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")
# SOUND_IN_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_vao_xin_cam_on.wav"
# SOUND_OUT_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_ra_xin_cam_on.wav"


# CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=plates_db;"
#     "UID=sa;"
#     "PWD=123456"
# )

# PANEL_W, PANEL_H = 640, 360
# PANEL_BG = (255, 255, 255)
# API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
# OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
#            "six":"6","seven":"7","eight":"8","nine":"9"}










# # ==================== 3. UTILITIES (HÀM TIỆN ÍCH) ====================

# # 
# def save_image(img, prefix):
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#     relative_path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")  # Tạo đường dẫn tương đối trước
#     absolute_path = os.path.abspath(relative_path)                # Chuyển nó thành đường dẫn tuyệt đối
    
#     # Lưu ảnh dùng đường dẫn tuyệt đối
#     try:
#         cv2.imwrite(absolute_path, img)
#         return absolute_path
#     except Exception as e:
#         print(f"Lỗi khi lưu ảnh {absolute_path}: {e}")
#         return None 

# # 
# def letterbox(bgr, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
#     if bgr is None: 
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     ih, iw = bgr.shape[:2]
#     if ih == 0 or iw == 0: 
#         return np.full((h, w, 3), color, dtype=np.uint8)
#     s = min(w/iw, h/ih); 
#     nw, nh = int(iw*s), int(ih*s)
#     resized = cv2.resize(bgr, (nw, nh))
#     canvas = np.full((h, w, 3), color, dtype=np.uint8)
#     top, left = (h-nh)//2, (w-nw)//2
#     canvas[top:top+nh, left:left+nw] = resized

#     return canvas

# # 
# def bgr_to_qimage(bgr):
#     if bgr is None: 
#         bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     h, w, ch = rgb.shape

#     return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)


# def norm_char(x): 
#     return OCR_MAP.get(str(x), str(x))

# def plate_norm(s: str) -> str: 
#     return (s or "").replace("-", "").replace(" ", "").upper()

# def has_boxes(r):
#     try: 
#         return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
#     except: 
#         return False
    
# # 
# def preprocess_for_ocr(roi):
#     if roi is None: 
#         return None
#     if roi.shape[-1]==4: 
#         roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe,(3,3),0)

#     return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)










# # ==================== 4. DB LAYER ====================
# class DB:
#     # 
#     def __init__(self, conn_str: str):
#         self.ok = False; self.conn = None; self.cur  = None
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
#                     image_in NVARCHAR(MAX) NULL,
#                     plate_out NVARCHAR(64)  NULL,
#                     date_out  NVARCHAR(16)  NULL,
#                     time_out  NVARCHAR(16)  NULL,
#                     image_out NVARCHAR(MAX) NULL,
#                     match_status NVARCHAR(32) NULL,
#                     created_at DATETIME DEFAULT GETDATE()
#                 );
#             """)
#             self.ok = True
#         except Exception as e:
#             print("DB connect error:", e); self.ok = False

#     # 
#     def insert_in(self, plate, d, t, img_path):
#         if not self.ok or not img_path: 
#             return 
#         try:
#             self.cur.execute("""
#                 INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
#                 VALUES (?,?,?,?,?)
#             """, (plate, d, t, img_path, 'PENDING'))
#         except Exception as e: print("insert_in error:", e)

#     # 
#     def attach_out(self, plate_out, d, t, img_path) -> str:
#         if not self.ok or not img_path: 
#             return "KHONG-KHOP-BIEN-SO" 
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
#                 return "KHOP-BIEN-SO"
#             else:
#                 self.cur.execute("""
#                     INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
#                     VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO')
#                 """, (plate_out, d, t, img_path))
#                 return "KHONG-KHOP-BIEN-SO"
#         except Exception as e:
#             print("attach_out error:", e); 
#             return "KHONG-KHOP-BIEN-SO"

#     # 
#     def fetch_history_df(self, limit=10000, start_time=None, end_time=None, status_filter=None, plate_filter=None) -> pd.DataFrame:
#         """
#         Lọc theo:
#         - Khoảng thời gian VÀO/RA (dựa trên date_in+time_in và date_out+time_out, đều là NVARCHAR)
#         - Trạng thái (match_status)
#         - Biển số (plate_in/plate_out LIKE)
#         """
#         columns = [
#             "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#         ]
#         if not self.ok:
#             return pd.DataFrame(columns=["STT"] + columns)
#         try:
#             dt_in  = "TRY_CONVERT(datetime, date_in  + ' ' + time_in , 103)"
#             dt_out = "TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)"
            
#             sql = f"""
#                 SELECT TOP ({limit})
#                     id, image_in, plate_in, date_in, time_in,
#                     image_out, plate_out, date_out, time_out, match_status
#                 FROM dbo.ParkingSessions
#             """
#             where_clauses = []
#             sql_params = []

#             # ------- Lọc theo khoảng thời gian vào/ra -------
#             if start_time and end_time:
#                 where_clauses.append(f"( ({dt_in}  BETWEEN ? AND ?) OR ({dt_out} BETWEEN ? AND ?) )")
#                 sql_params += [start_time, end_time, start_time, end_time]
#             elif start_time:
#                 where_clauses.append(f"( {dt_in}  >= ? OR {dt_out} >= ? )")
#                 sql_params += [start_time, start_time]
#             elif end_time:
#                 where_clauses.append(f"( {dt_in}  <= ? OR {dt_out} <= ? )")
#                 sql_params += [end_time, end_time]

#             # ------- Lọc Trạng thái -------
#             if status_filter and len(status_filter) > 0:
#                 placeholders = ",".join("?" for _ in status_filter)
#                 where_clauses.append(f"match_status IN ({placeholders})")
#                 sql_params += status_filter

#             # ------- Lọc Biển số gần đúng ở cả vào/ra -------
#             if plate_filter and len(plate_filter.strip()) > 0:
#                 where_clauses.append("(plate_in LIKE ? OR plate_out LIKE ?)")
#                 like_term = f"%{plate_filter.strip()}%"
#                 sql_params += [like_term, like_term]

#             # 
#             if where_clauses:
#                 sql += " WHERE " + " AND ".join(where_clauses)

#             sql += f" ORDER BY COALESCE({dt_out}, {dt_in}) DESC, id DESC"
#             rows = self.cur.execute(sql, tuple(sql_params)).fetchall()

#             df = pd.DataFrame.from_records(rows, columns=columns).astype(object).where(pd.notnull, "")
#             df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#             df.insert(0, "STT", range(1, len(df) + 1))
#             return df

#         except Exception as e:
#             print(f"fetch_history_df error: {e}")
#             import traceback
#             traceback.print_exc()
#             return pd.DataFrame(columns=["STT"] + columns)


#     # 
#     def delete_by_ids(self, ids):
#         if not self.ok or not ids: 
#             return
#         try:
#             placeholders = ','.join('?' for _ in ids)
#             sql = f"DELETE FROM dbo.ParkingSessions WHERE id IN ({placeholders})"
#             self.cur.execute(sql, tuple(int(sid) for sid in ids))
#         except Exception as e: 
#             print("delete_by_ids error:", e)


#     # 
#     def delete_all(self):
#         if not self.ok: 
#             return
#         try: 
#             self.cur.execute("DELETE FROM dbo.ParkingSessions")
#         except Exception as e: 
#             print("delete_all error:", e)










# # ==================== 5. YOLO/GEMINI WRAPPERS ====================

# class Models:
#     # 
#     def __init__(self, det_path: str, ocr_path: str):
#         self.ok = True; self.err = ""
#         try:
#             self.det = YOLO(det_path)
#             self.ocr = YOLO(ocr_path)
#         except Exception as e:
#             self.ok = False; self.err = str(e)


#     # 
#     def detect_plates(self, frame):
#         plates, boxed = [], None 
#         try:
#             boxed = frame.copy() 
#             results = self.det(frame, verbose=False) 
#             for r in results:
#                 if not has_boxes(r): 
#                     continue
#                 xyxy_np = r.boxes.xyxy.cpu().numpy().astype(int)
#                 for (x1,y1,x2,y2) in xyxy_np:
#                     pad=8
#                     fh, fw = frame.shape[:2] 
#                     x1=max(0,x1-pad); y1=max(0,y1-pad)
#                     x2=min(fw-1,x2+pad); y2=min(fh-1,y2+pad)

#                     # Cắt ROI từ frame gốc
#                     roi = frame[y1:y2, x1:x2].copy()
#                     if roi.size == 0: 
#                         continue 
#                     plates.append(((x1,y1,x2,y2), roi))

#                     # Vẽ lên ảnh copy
#                     cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
#                     cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
#         except Exception as e:
#             print(f"Lỗi detect_plates: {e}")
#             return [], frame

#         return plates, boxed if boxed is not None else frame


#     # 
#     def ocr_plate_yolo(self, roi):
#         if roi is None or roi.size == 0: 
#             return "", "" 
#         try:
#             roi_pre = preprocess_for_ocr(roi)
#             input_roi = roi_pre if roi_pre is not None and roi_pre.size > 0 else roi
#             res = self.ocr(input_roi, verbose=False)
#             text_raw=""
#             for r in res:
#                 if not has_boxes(r): 
#                     continue
#                 names = getattr(r,'names',None) or getattr(self.ocr,'names',{}) or {}
#                 clses = r.boxes.cls.cpu().numpy().astype(int)
#                 xyxys= r.boxes.xyxy.cpu().numpy()
#                 boxes=[]
#                 for i,cls in enumerate(clses):
#                     x1,y1,x2,y2 = xyxys[i]
#                     cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
#                     ch = norm_char(names.get(cls, str(cls)) if isinstance(names,dict) else str(cls))
#                     if ch.isdigit() or (ch.isalpha() and ch.isupper()):
#                         boxes.append((cy,cx,ch))
#                 if not boxes: continue
#                 ys=[b[0] for b in boxes]
#                 # Sửa lỗi 'float' object cannot be interpreted as an integer
#                 h_roi = input_roi.shape[0]
#                 if len(boxes)<=7 or (max(ys)-min(ys) < 0.2 * h_roi): # So sánh với chiều cao ROI
#                     text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
#                 else:
#                     thr=(max(ys)+min(ys))/2.0
#                     l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
#                     t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
#                     t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
#                     text_raw=f"{t1}-{t2}" if t2 else t1
#             return self._format_text(text_raw)
#         except Exception as e:
#             print(f"Lỗi ocr_plate_yolo: {e}")
#             return "", "" # Trả về rỗng nếu có lỗi

#     def ocr_plate_gemini_from_path(self, image_path: str):
#         # ... (Hàm này giữ nguyên code của bạn) ...
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
#         # ... (Hàm này giữ nguyên code của bạn) ...
#         raw=(text_raw or '').replace('-', '').replace('.', '').replace(' ', '') # Bỏ luôn dấu chấm
#         # Logic định dạng lại biển số (ví dụ)
#         if len(raw) >= 7 and len(raw) <= 9:
#              # Biển 2 dòng cũ (VD: 59C112345) -> 59-C1 123.45
#              if raw[2].isalpha() and raw[3].isdigit():
#                   return f"{raw[:2]}-{raw[2:4]} {raw[4:7]}.{raw[7:]}" if len(raw) > 7 else f"{raw[:2]}-{raw[2:4]} {raw[4:]}", text_raw
#              # Biển 1 dòng mới (VD: 59C112345) -> 59C1-123.45
#              elif raw[2].isdigit() and raw[4].isalpha():
#                   return f"{raw[:4]}-{raw[4:7]}.{raw[7:]}" if len(raw) > 7 else f"{raw[:4]}-{raw[4:]}", text_raw
#         # Trả về gốc nếu không khớp định dạng mong muốn
#         return text_raw or "", text_raw or ""


# # ==================== 6. VIDEO WORKER ====================
# class VideoWorker(QThread):
#     # ... (Phần signals và __init__, setters giữ nguyên code của bạn) ...
#     frameSignal = Signal(np.ndarray, str)
#     sceneSignal = Signal(str)
#     roiSignal   = Signal(str, str)
#     infoSignal  = Signal(dict)
#     matchSignal = Signal(str)
#     histSignal  = Signal()
#     playSoundSignal = Signal(str)

#     def __init__(self, cam_idx: int, api: int, mode: str, models: Models, db: DB,
#                  stable_seconds: float = 1.2, ocr_mode: str = "yolo", title: str = "", parent=None):
#         super().__init__(parent)
#         self.cam_idx = cam_idx
#         self.api = api
#         self.mode = mode
#         self.models = models
#         self.db = db
#         self.stable_seconds = stable_seconds
#         self.ocr_mode = ocr_mode
#         self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")
#         self._running = False
#         self.cap = None
#         self.stable_start = 0.0
#         self.captured = False

#     def set_title(self, title: str): self.title = title
#     def set_ocr_mode(self, mode: str): self.ocr_mode = mode
#     def set_mode(self, mode: str): self.mode = mode

#     def run(self):
#         # ... (Phần mở camera giữ nguyên code của bạn) ...
#         self._running = True
#         try: # Thêm try-except để bắt lỗi mở camera
#              self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
#              if not (self.cap and self.cap.isOpened()):
#                   print(f"Lỗi: Không thể mở camera index {self.cam_idx} với API {self.api}")
#                   self._running = False; return
#         except Exception as e:
#              print(f"Lỗi khi khởi tạo VideoCapture: {e}")
#              self._running = False; return

#         try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#         except: pass
#         try: self.cap.set(cv2.CAP_PROP_FPS, 30)
#         except: pass


#         while self._running:
#             try: # Thêm try-except cho vòng lặp chính
#                 ok, frame = self.cap.read()
#                 if not ok or frame is None: # Kiểm tra frame hợp lệ
#                     self.stable_start = 0.0; self.captured = False
#                     time.sleep(0.05); continue # Chờ lâu hơn nếu đọc lỗi

#                 # Gửi frame gốc lên UI
#                 self.frameSignal.emit(frame, self.title)

#                 # Phát hiện biển số
#                 plates, boxed_frame = self.models.detect_plates(frame)

#                 if not plates:
#                     self.stable_start = 0.0; self.captured = False
#                     time.sleep(0.01); continue

#                 # Chọn biển số tốt nhất (ví dụ: lớn nhất)
#                 best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
#                 roi_current = best[1]
#                 if roi_current is None or roi_current.size == 0: # Kiểm tra roi hợp lệ
#                      self.stable_start = 0.0; self.captured = False
#                      time.sleep(0.01); continue

#                 # Logic ổn định
#                 if self.stable_start == 0.0: # Bắt đầu tính giờ nếu chưa tính
#                      self.stable_start = time.time()
#                 elif self.captured: # Nếu đã chụp rồi thì reset ngay
#                      self.stable_start = time.time(); self.captured = False


#                 # Đủ thời gian ổn định và chưa chụp
#                 if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
#                     # Lưu ảnh (nên dùng ảnh đã vẽ hộp)
#                     scene_img_to_save = boxed_frame if boxed_frame is not None else frame
#                     scene_path = save_image(scene_img_to_save,
#                                            "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
#                     # Lưu ROI
#                     roi_path   = save_image(roi_current,
#                                            "plate_in_roi" if self.mode=="in" else "plate_out_roi")

#                     # Kiểm tra lưu ảnh thành công
#                     if not scene_path or not roi_path:
#                         print("Lỗi: Không thể lưu ảnh scene hoặc roi.")
#                         self.captured = True # Đánh dấu đã xử lý (dù lỗi) để tránh lặp lại ngay
#                         self.stable_start = 0.0 # Reset timer
#                         continue

#                     # Thực hiện OCR
#                     text_fmt, text_raw = "", ""
#                     if self.ocr_mode == "gemini" and GEMINI_READY:
#                         text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
#                     else:
#                         text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

#                     # Có kết quả OCR
#                     if text_fmt or text_raw:
#                         now = datetime.now()
#                         d = now.strftime("%d/%m/%Y")
#                         t = now.strftime("%H:%M:%S")
#                         plate = text_fmt or text_raw

#                         # Gửi tín hiệu lên UI
#                         self.sceneSignal.emit(scene_path) # Gửi đường dẫn ảnh scene
#                         self.roiSignal.emit(roi_path, self.mode) # Gửi đường dẫn ảnh roi

#                         # Xử lý logic vào/ra và DB
#                         if self.mode == "in":
#                             self.infoSignal.emit({"date_in": d, "time_in": t, "plate_text_in": plate})
#                             if self.db and self.db.ok:
#                                 self.db.insert_in(plate, d, t, scene_path)
#                                 self.histSignal.emit()
#                             self.playSoundSignal.emit("in")
#                         else: # mode == "out"
#                             self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
#                             if self.db and self.db.ok:
#                                 match = self.db.attach_out(plate, d, t, scene_path)
#                                 self.matchSignal.emit(match)
#                                 self.histSignal.emit()
#                             self.playSoundSignal.emit("out")

#                         self.captured = True # Đánh dấu đã chụp thành công
#                         self.stable_start = 0.0 # Reset timer sau khi chụp thành công

#             except Exception as e:
#                  print(f"Lỗi trong vòng lặp VideoWorker: {e}")
#                  import traceback
#                  traceback.print_exc()
#                  self.stable_start = 0.0 # Reset nếu có lỗi
#                  self.captured = False
#                  time.sleep(0.1) # Chờ lâu hơn nếu có lỗi

#             time.sleep(0.01) # Thêm sleep nhỏ ở cuối vòng lặp

#         # Dọn dẹp khi dừng luồng
#         try:
#             if self.cap: self.cap.release()
#         except Exception as e:
#              print(f"Lỗi khi release camera: {e}")


#     def stop(self): self._running = False


# # ==================== 6.5 HISTORY LOADER WORKER (MỚI - Đặt trước MainWindow) ====================
# class HistoryLoaderWorker(QThread):
#     """Luồng riêng để tải dữ liệu lịch sử từ DB."""
#     resultReady = Signal(pd.DataFrame) # Signal trả về DataFrame kết quả

#     def __init__(self, db: DB, start_time=None, end_time=None, status_filter=None, plate_filter=None, parent=None):
#         super().__init__(parent)
#         self.db = db
#         self.start_time = start_time
#         self.end_time = end_time
#         self.status_filter = status_filter
#         self.plate_filter = plate_filter

#     def run(self):
#         """Thực hiện truy vấn DB trong luồng này."""
#         df = pd.DataFrame() # Khởi tạo df rỗng
#         print("HistoryLoaderWorker bắt đầu chạy...")
#         try:
#              if self.db and self.db.ok:
#                   df = self.db.fetch_history_df(limit=800,
#                                              start_time=self.start_time,
#                                              end_time=self.end_time,
#                                              status_filter=self.status_filter,
#                                              plate_filter=self.plate_filter)
#         except Exception as e:
#              print(f"Lỗi trong HistoryLoaderWorker.run: {e}")
#              traceback.print_exc() # In chi tiết lỗi
#         finally:
#              # Đảm bảo luôn emit DataFrame, ngay cả khi rỗng hoặc lỗi
#              self.resultReady.emit(df if df is not None else pd.DataFrame())
#              print("HistoryLoaderWorker đã chạy xong.")


# # ==================== 7. DELETE DIALOG ====================
# class DeleteDialog(QDialog):
#     # ... (Class này giữ nguyên code của bạn) ...
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Xóa lịch sử")
#         self.setModal(True)
#         self.setStyleSheet("""
#             QDialog { background: #ffffff; border-radius: 10px; }
#             QLabel { font-weight: 600; }
#         """)
#         lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
#         lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
#         lay.addWidget(lab)
#         row = QHBoxLayout(); row.setSpacing(12)
#         self.btn_sel = QPushButton("Xóa dòng đã chọn")
#         self.btn_all = QPushButton("Xóa tất cả")
#         self.btn_can = QPushButton("Hủy")
#         row.addWidget(self.btn_sel, 1); row.addWidget(self.btn_all, 1); row.addWidget(self.btn_can, 1)
#         lay.addLayout(row)
#         base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
#         self.btn_sel.setStyleSheet(f"QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}")
#         self.btn_all.setStyleSheet(f"QPushButton {{ {base} background:#ffe0e0; border:1px solid #ffb3b3; }}")
#         self.btn_can.setStyleSheet(f"QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}")
#         self.btn_sel.clicked.connect(lambda: self.done(1))
#         self.btn_all.clicked.connect(lambda: self.done(2))
#         self.btn_can.clicked.connect(lambda: self.done(0))


# # ==================== 8. MAIN WINDOW ====================
# class MainWindow(QMainWindow):
#     # ... (Hàm __init__, _init_theme, _normalize_button, _apply_btn_style, _make_card, _set_centered_pixmap giữ nguyên) ...
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Desktop App (Giữ xe)")
#         self.setMinimumSize(1200, 800)
#         self._init_theme()
#         self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
#         if not self.models.ok: QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#         self.db = DB(CONN_STR) if USE_SQL else DB("")
#         # Khởi tạo âm thanh
#         self.sound_in = QSoundEffect(self)
#         sound_in_abs = os.path.abspath(SOUND_IN_PATH)
#         if os.path.exists(sound_in_abs): self.sound_in.setSource(QUrl.fromLocalFile(sound_in_abs))
#         else: print(f"Lỗi: Không tìm thấy file âm thanh: {sound_in_abs}")
#         self.sound_out = QSoundEffect(self)
#         sound_out_abs = os.path.abspath(SOUND_OUT_PATH)
#         if os.path.exists(sound_out_abs): self.sound_out.setSource(QUrl.fromLocalFile(sound_out_abs))
#         else: print(f"Lỗi: Không tìm thấy file âm thanh: {sound_out_abs}")
#         self.cam1_worker = None
#         self.cam2_worker = None
#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.current_ocr_mode = "yolo"
#         self.history_df = pd.DataFrame() # Khởi tạo df lịch sử
#         self.current_filter_start = None
#         self.current_filter_end = None
#         self.current_filter_status = None
#         self.current_filter_plate = None
#         self.history_worker = None
#         self._hist_last_reload = 0.0
#         self._logo_pm = self.qpix_logo()
#         self._build_ui()
#         self.show_logo(1); self.show_logo(2)
#         # Kết nối timer
#         self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.on_history_signal_refresh); self.hist_timer.start(5000)

#     def _init_theme(self): self.setStyleSheet(""" * { color: #000000; } QMainWindow, QWidget { background: #ffffff; } QWidget#SideBar { background: #ffffff; } QGroupBox { background: #ffffff; font-weight: 600; border: 2px solid #e6e6e6; border-radius: 12px; margin-top: 8px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; background: #ffffff; } QFrame[class="card-wrap"] { background: #e6e6e6; border-radius: 14px; } QFrame[class="card"] { background: #ffffff; border-radius: 12px; } QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; } QLabel[class="title"] { font: 700 18px "Segoe UI"; padding: 6px 10px; background: #ffffff; border-radius: 10px; } QLineEdit { height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px; } QTableWidget { background: #ffffff; gridline-color: #e6e6e6; } """)
#     def _normalize_button(self, *btns):
#         for b in btns:
#             b.setAutoDefault(False); b.setDefault(False); b.setFlat(False); b.setFocusPolicy(Qt.NoFocus)
#             b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
#     def _apply_btn_style(self, btn: QPushButton, css: str): btn.setStyleSheet(css)
#     def _make_card(self, title:str, content:QWidget):
#         wrap = QFrame(); wrap.setProperty("class","card-wrap"); wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
#         card = QFrame(); card.setProperty("class","card"); v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
#         title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap"); hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
#         title_lbl = QLabel(title); title_lbl.setProperty("class","title"); hl.addWidget(title_lbl)
#         v.addWidget(title_wrap); v.addWidget(content, 1); wrapL.addWidget(card); return wrap, title_lbl
#     def _set_centered_pixmap(self, lbl: QLabel, src):
#         pm = None # Khởi tạo pm
#         if isinstance(src, np.ndarray): pm = QPixmap.fromImage(bgr_to_qimage(src))
#         elif isinstance(src, QImage): pm = QPixmap.fromImage(src)
#         elif isinstance(src, QPixmap): pm = src # Chấp nhận cả QPixmap
#         if pm is None or pm.isNull(): lbl.clear(); return
#         rect = lbl.contentsRect(); avail = rect.size()
#         dpr = lbl.devicePixelRatioF() if hasattr(lbl, "devicePixelRatioF") else 1.0
#         target_w = max(1, int(avail.width()  * dpr)); target_h = max(1, int(avail.height() * dpr))
#         scaled = pm.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
#         if hasattr(scaled, "setDevicePixelRatio"): scaled.setDevicePixelRatio(dpr)
#         lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setPixmap(scaled)

#     # ĐÂY LÀ HÀM QUAN TRỌNG NHẤT, THAY THẾ TOÀN BỘ HÀM CŨ
#     def _build_ui(self):
#         central = QWidget(); self.setCentralWidget(central)
#         root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)
#         # LEFT PANEL (SIDEBAR)
#         side = QWidget(objectName="SideBar"); side.setFixedWidth(450)
#         vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)
#         common_btn = "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;" # Định nghĩa sớm hơn
#         # Camera Control
#         gb_camctl = QGroupBox("CAMERA CONTROL"); vl_camctl = QVBoxLayout(gb_camctl); vl_camctl.setSpacing(10)
#         self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
#         self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
#         row_indices = QHBoxLayout(); row_indices.setSpacing(10)
#         row_indices.addWidget(QLabel("Index Cam 1")); row_indices.addWidget(self.spin_cam1, 1)
#         row_indices.addWidget(QLabel("Index Cam 2")); row_indices.addWidget(self.spin_cam2, 1)
#         vl_camctl.addLayout(row_indices)
#         self.btn_start1 = QPushButton("Bật Cam 1"); self.btn_stop1 = QPushButton("Tắt Cam 1")
#         self.btn_start2 = QPushButton("Bật Cam 2"); self.btn_stop2 = QPushButton("Tắt Cam 2")
#         self._normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)
#         self._apply_btn_style(self.btn_start1, f"QPushButton{{ {common_btn} background:#d1fadf; border:1px solid #a6f4c5; }} QPushButton:hover{{ background:#c3f7d6; }} QPushButton:disabled{{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }}")
#         self._apply_btn_style(self.btn_stop1, f"QPushButton{{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }} QPushButton:hover{{ background:#ffd1d1; }} QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}")
#         self._apply_btn_style(self.btn_start2, self.btn_start1.styleSheet()); self._apply_btn_style(self.btn_stop2, self.btn_stop1.styleSheet())
#         self.btn_start1.clicked.connect(self.start_cam1); self.btn_stop1.clicked.connect(self.stop_cam1)
#         self.btn_start2.clicked.connect(self.start_cam2); self.btn_stop2.clicked.connect(self.stop_cam2)
#         row_btn1 = QHBoxLayout(); row_btn1.setSpacing(12); row_btn1.addWidget(self.btn_start1); row_btn1.addWidget(self.btn_stop1); vl_camctl.addLayout(row_btn1)
#         row_btn2 = QHBoxLayout(); row_btn2.setSpacing(12); row_btn2.addWidget(self.btn_start2); row_btn2.addWidget(self.btn_stop2); vl_camctl.addLayout(row_btn2)
#         vside.addWidget(gb_camctl)
#         # Lane Control
#         gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN"); vl_lane = QVBoxLayout(gb_lane); vl_lane.setSpacing(10)
#         row_lane = QHBoxLayout(); row_lane.setSpacing(12)
#         self.btn_oneway = QPushButton("1 chiều"); self.btn_twoway = QPushButton("2 chiều"); self.btn_reset_lane = QPushButton("Reset làn")
#         self._normalize_button(self.btn_oneway, self.btn_twoway, self.btn_reset_lane)
#         self._apply_btn_style(self.btn_oneway, f"QPushButton{{ {common_btn} background:#dbeafe; border:1px solid #bfdbfe; }} QPushButton:hover{{ background:#cfe3fd; }}")
#         self._apply_btn_style(self.btn_twoway, self.btn_oneway.styleSheet())
#         self._apply_btn_style(self.btn_reset_lane, f"QPushButton{{ {common_btn} background:#fff3bf; border:1px solid #ffe066; }} QPushButton:hover{{ background:#ffeda3; }}")
#         row_lane.addWidget(self.btn_oneway); row_lane.addWidget(self.btn_twoway); vl_lane.addLayout(row_lane); vl_lane.addWidget(self.btn_reset_lane)
#         self.btn_oneway.clicked.connect(self.on_one_way_clicked); self.btn_twoway.clicked.connect(self.on_two_way_clicked); self.btn_reset_lane.clicked.connect(self.on_reset_lanes)
#         vside.addWidget(gb_lane)
#         # OCR Model
#         gb_ocr = QGroupBox("OCR MODEL"); vb_ocr = QVBoxLayout(gb_ocr)
#         self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)"); self.rb_yolo.setChecked(True)
#         self.rb_gem = QRadioButton("Dùng Gemini AI")
#         vb_ocr.addWidget(self.rb_yolo); vb_ocr.addWidget(self.rb_gem)
#         self.rb_yolo.toggled.connect(self.on_ocr_mode_changed); self.rb_gem.toggled.connect(self.on_ocr_mode_changed)
#         if not GEMINI_READY: self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY")
#         vside.addWidget(gb_ocr)
#         # Info IN
#         gb_in = QGroupBox("THÔNG TIN XE VÀO"); gl_in = QGridLayout(gb_in)
#         self.ed_date_in = QLineEdit(); self.ed_time_in = QLineEdit(); self.ed_plate_in = QLineEdit()
#         self.ed_plate_in.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
#         gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
#         gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
#         gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
#         vside.addWidget(gb_in)
#         # Info OUT
#         gb_out = QGroupBox("THÔNG TIN XE RA"); gl_out = QGridLayout(gb_out)
#         self.ed_date_out = QLineEdit(); self.ed_time_out = QLineEdit(); self.ed_plate_out = QLineEdit()
#         self.ed_plate_out.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
#         gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
#         gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
#         gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
#         vside.addWidget(gb_out)
#         # History Buttons
#         gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ"); v_hist_btns = QVBoxLayout(gb_hist_btns)
#         self.btn_show_history = QPushButton("Xem bảng lịch sử"); self.btn_export_hist = QPushButton("Export Excel")
#         self.btn_delete_hist = QPushButton("Xóa bảng"); self.btn_search_hist = QPushButton("Tìm kiếm")
#         self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide()
#         self._normalize_button(self.btn_show_history, self.btn_export_hist, self.btn_delete_hist, self.btn_search_hist, self.btn_hide_history)
#         self._apply_btn_style(self.btn_show_history, f"QPushButton{{ {common_btn} background:#E6F4EA; border:1px solid #cde9d6; }} QPushButton:hover{{ background:#d9efe0; }}")
#         self._apply_btn_style(self.btn_hide_history, f"QPushButton{{ {common_btn} background:#fff3bf; border:1px solid #f5c6c2; }} QPushButton:hover{{ background:#ffeda3; }}")
#         self._apply_btn_style(self.btn_export_hist, f"QPushButton{{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }} QPushButton:hover{{ background:#d4e5ff; }}")
#         self._apply_btn_style(self.btn_delete_hist, f"QPushButton{{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }} QPushButton:hover{{ background:#ffd1d1; }}")
#         self._apply_btn_style(self.btn_search_hist, f"QPushButton{{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }} QPushButton:hover{{ background:#d4e5ff; }}")
#         row_cmd = QHBoxLayout(); row_cmd.addWidget(self.btn_search_hist); row_cmd.addWidget(self.btn_export_hist); row_cmd.addWidget(self.btn_delete_hist)
#         v_hist_btns.addWidget(self.btn_show_history); v_hist_btns.addLayout(row_cmd); v_hist_btns.addWidget(self.btn_hide_history)
#         vside.addWidget(gb_hist_btns)
#         vside.addStretch(1)
#         root.addWidget(side)

#         # RIGHT PANEL (STACKED WIDGET CONTAINER)
#         right_container = QVBoxLayout()
#         # --- Page 0: Main View (Cameras) ---
#         self.main_view = QWidget(); main_layout = QVBoxLayout(self.main_view)
#         top = QHBoxLayout(); self.lbl_cam1 = QLabel(); self.lbl_cam2 = QLabel()
#         for lbl in (self.lbl_cam1, self.lbl_cam2):
#             lbl.setScaledContents(False); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("background:#ffffff; border-radius:12px;"); lbl.setMinimumHeight(220); lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1); cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
#         top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1); main_layout.addLayout(top)
#         bottom = QHBoxLayout(); self.lbl_scene = QLabel(); self.lbl_roi = QLabel()
#         for lbl in (self.lbl_scene, self.lbl_roi):
#              lbl.setScaledContents(False); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("background:#ffffff; border-radius:12px;"); lbl.setMinimumHeight(220); lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         scene_card, _ = self._make_card("Image_BOX", self.lbl_scene); roi_card, _ = self._make_card("ROI_Plate", self.lbl_roi)
#         bottom.addWidget(scene_card, 1); bottom.addWidget(roi_card, 1); main_layout.addLayout(bottom)
#         self.info_group = QGroupBox("Thông tin chi tiết"); info_layout = QGridLayout(self.info_group)
#         self.txt_date_in = QLabel("--/--/----"); self.txt_time_in = QLabel("--:--:--"); self.txt_plate_in = QLabel("---"); self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_date_out = QLabel("--/--/----"); self.txt_time_out = QLabel("--:--:--"); self.txt_plate_out= QLabel("---"); self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")
#         self.txt_match = QLineEdit(); self.txt_match.setReadOnly(True); self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
#         r=0; info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1); info_layout.addWidget(QLabel("Giờ vào:"), r,2); info_layout.addWidget(self.txt_time_in, r,3); info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
#         info_layout.addWidget(QLabel("Ngày ra:"), r,0); info_layout.addWidget(self.txt_date_out, r,1); info_layout.addWidget(QLabel("Giờ ra:"), r,2); info_layout.addWidget(self.txt_time_out, r,3); info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
#         info_layout.addWidget(QLabel("So khớp biển số:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,2); main_layout.addWidget(self.info_group)

#         # --- Page 1: History View (Table) ---
#         self.history_view = QWidget(); 
#         hist_layout = QVBoxLayout(self.history_view)
#         hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)"); 
#         hist_v = QVBoxLayout(hist_group)
#         self.tbl_hist = QTableWidget(0, 10); 
#         self.tbl_hist.setHorizontalHeaderLabels(["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"])
#         header = self.tbl_hist.horizontalHeader(); 
#         hfont = QFont(header.font()); 
#         hfont.setBold(True); 
#         header.setFont(hfont)
#         header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
#         header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Ngày vào
#         header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Giờ vào
#         header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Ngày ra
#         header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Giờ ra
#         header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Trạng thái
#         for j in range(1, 10):  # các cột còn lại
#             if header.sectionResizeMode(j) != QHeaderView.ResizeToContents:
#                 header.setSectionResizeMode(j, QHeaderView.Stretch)

#         self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); 
#         self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); 
#         self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
#         self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected); 
#         self.tbl_hist.setAlternatingRowColors(False); 
#         # header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
#         hist_v.addWidget(self.tbl_hist); 
#         hist_layout.addWidget(hist_group)

#         # --- Page 2: Detail View ---
#         self.detail_view = QWidget(); detail_layout = QVBoxLayout(self.detail_view)
#         row_btn_back = QHBoxLayout(); self.btn_back_to_history = QPushButton("⬅ Quay lại bảng lịch sử"); self._normalize_button(self.btn_back_to_history)
#         self._apply_btn_style(self.btn_back_to_history, f"QPushButton{{ {common_btn} background:#f3f4f6; border:1px solid #e5e7eb; }} QPushButton:hover{{ background:#eef0f3; }}")
#         row_btn_back.addWidget(self.btn_back_to_history); row_btn_back.addStretch(1); detail_layout.addLayout(row_btn_back)
#         row_images = QHBoxLayout(); self.lbl_detail_scene = QLabel(); self.lbl_detail_roi = QLabel()
#         for lbl in (self.lbl_detail_scene, self.lbl_detail_roi): lbl.setScaledContents(False); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("background:#ffffff; border-radius:12px;"); lbl.setMinimumHeight(320); lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
#         detail_scene_card, _ = self._make_card("Ảnh Chụp Vào (Image_IN)", self.lbl_detail_scene); detail_roi_card, _ = self._make_card("Ảnh Chụp Ra (Image_OUT)", self.lbl_detail_roi)
#         row_images.addWidget(detail_scene_card, 1); row_images.addWidget(detail_roi_card, 1); detail_layout.addLayout(row_images, 1)
#         gb_detail_info = QGroupBox("Thông tin Lượt Gửi"); gl_detail = QGridLayout(gb_detail_info)
#         self.lbl_detail_plate_in = QLineEdit(); self.lbl_detail_plate_in.setReadOnly(True); self.lbl_detail_date_in = QLineEdit(); self.lbl_detail_date_in.setReadOnly(True); self.lbl_detail_time_in = QLineEdit(); self.lbl_detail_time_in.setReadOnly(True)
#         self.lbl_detail_plate_out = QLineEdit(); self.lbl_detail_plate_out.setReadOnly(True); self.lbl_detail_date_out = QLineEdit(); self.lbl_detail_date_out.setReadOnly(True); self.lbl_detail_time_out = QLineEdit(); self.lbl_detail_time_out.setReadOnly(True); self.lbl_detail_match = QLineEdit(); self.lbl_detail_match.setReadOnly(True)
#         self.lbl_detail_plate_in.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: 700;"); self.lbl_detail_plate_out.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: 700;"); self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")
#         gl_detail.addWidget(QLabel("Biển số vào:"), 0, 0); gl_detail.addWidget(self.lbl_detail_plate_in, 0, 1); gl_detail.addWidget(QLabel("Ngày vào:"), 1, 0); gl_detail.addWidget(self.lbl_detail_date_in, 1, 1); gl_detail.addWidget(QLabel("Giờ vào:"), 2, 0); gl_detail.addWidget(self.lbl_detail_time_in, 2, 1)
#         gl_detail.addWidget(QLabel("Biển số ra:"), 0, 2); gl_detail.addWidget(self.lbl_detail_plate_out, 0, 3); gl_detail.addWidget(QLabel("Ngày ra:"), 1, 2); gl_detail.addWidget(self.lbl_detail_date_out, 1, 3); gl_detail.addWidget(QLabel("Giờ ra:"), 2, 2); gl_detail.addWidget(self.lbl_detail_time_out, 2, 3)
#         gl_detail.addWidget(QLabel("Trạng thái:"), 3, 0); gl_detail.addWidget(self.lbl_detail_match, 3, 1, 1, 3); detail_layout.addWidget(gb_detail_info)

#         # ==================== MỚI: TẠO TRANG TÌM KIẾM (SEARCH_FILTER_VIEW) (INDEX 3) - (UI HOÀN CHỈNH) ====================
#         self.search_filter_view = QWidget()
#         sfv_layout = QVBoxLayout(self.search_filter_view)
#         sfv_layout.setContentsMargins(20, 20, 20, 20)
#         sfv_layout.setSpacing(15)

#         # 1. Tiêu đề
#         sfv_title = QLabel("Bộ lọc tìm kiếm lịch sử")
#         sfv_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #333;")
#         sfv_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         sfv_layout.addWidget(sfv_title)

#         # 2. Form chứa các bộ lọc
#         sfv_form = QFrame()
#         sfv_form.setStyleSheet("QFrame { background: #f9f9f9; border: 1px solid #eee; border-radius: 10px; } QLabel { font-weight: 600; }") # Thêm style cho QLabel
#         sfv_form_layout = QVBoxLayout(sfv_form) # Dùng QVBoxLayout
#         sfv_form_layout.setContentsMargins(25, 25, 25, 25)
#         sfv_form_layout.setSpacing(18) # Tăng khoảng cách dòng

#         # ---- Hàng "Từ ngày/giờ" ----
#         row_start = QHBoxLayout(); row_start.setSpacing(10) # Giảm khoảng cách item
#         row_start.addWidget(QLabel("Từ ngày:"))
#         self.sfv_date_start = QDateEdit(QDate.currentDate().addDays(-1)) # Mặc định là hôm qua
#         self.sfv_date_start.setCalendarPopup(True); self.sfv_date_start.setDisplayFormat("dd/MM/yyyy"); self.sfv_date_start.setFixedHeight(34)
#         row_start.addWidget(self.sfv_date_start)
#         row_start.addWidget(QLabel("Giờ:"))
#         self.sfv_time_start = QTimeEdit(QTime(0, 0, 0))
#         self.sfv_time_start.setDisplayFormat("HH:mm:ss"); self.sfv_time_start.setFixedHeight(34)
#         row_start.addWidget(self.sfv_time_start)
#         row_start.addStretch(1)
#         sfv_form_layout.addLayout(row_start)

#         # ---- Hàng "Đến ngày/giờ" ----
#         row_end = QHBoxLayout(); row_end.setSpacing(10)
#         row_end.addWidget(QLabel("Đến ngày:"))
#         self.sfv_date_end = QDateEdit(QDate.currentDate())
#         self.sfv_date_end.setCalendarPopup(True); self.sfv_date_end.setDisplayFormat("dd/MM/yyyy"); self.sfv_date_end.setFixedHeight(34)
#         row_end.addWidget(self.sfv_date_end)
#         row_end.addWidget(QLabel("Giờ:"))
#         self.sfv_time_end = QTimeEdit(QTime.currentTime())
#         self.sfv_time_end.setDisplayFormat("HH:mm:ss"); self.sfv_time_end.setFixedHeight(34)
#         row_end.addWidget(self.sfv_time_end)
#         row_end.addStretch(1)
#         sfv_form_layout.addLayout(row_end)

#         # ---- Hàng "Trạng thái" (MỚI) ----
#         row_status = QHBoxLayout(); row_status.setSpacing(15)
#         row_status.addWidget(QLabel("Trạng thái:"))
#         self.sfv_chk_pending = QCheckBox("Chờ xử lý (Pending)")
#         self.sfv_chk_match = QCheckBox("Khớp biển số")
#         self.sfv_chk_mismatch = QCheckBox("Không khớp")
#         # Mặc định chọn tất cả
#         self.sfv_chk_pending.setChecked(True)
#         self.sfv_chk_match.setChecked(True)
#         self.sfv_chk_mismatch.setChecked(True)
#         row_status.addWidget(self.sfv_chk_pending)
#         row_status.addWidget(self.sfv_chk_match)
#         row_status.addWidget(self.sfv_chk_mismatch)
#         row_status.addStretch(1)
#         sfv_form_layout.addLayout(row_status)

#         # ---- Hàng "Biển số" (MỚI) ----
#         row_plate = QHBoxLayout(); row_plate.setSpacing(10)
#         row_plate.addWidget(QLabel("Biển số (tương đối):"))
#         self.sfv_txt_plate = QLineEdit()
#         self.sfv_txt_plate.setPlaceholderText("Nhập một phần hoặc toàn bộ biển số...")
#         self.sfv_txt_plate.setFixedHeight(34)
#         row_plate.addWidget(self.sfv_txt_plate)
#         sfv_form_layout.addLayout(row_plate)

#         sfv_layout.addWidget(sfv_form) # Thêm form vào layout chính

#         # ---- Hàng nút (Quay lại, Tìm kiếm) ----
#         sfv_row_btn = QHBoxLayout()
#         self.sfv_btn_back = QPushButton("Quay lại")
#         self.sfv_btn_search = QPushButton("Tìm kiếm")
#         self._normalize_button(self.sfv_btn_back, self.sfv_btn_search)
#         self._apply_btn_style(self.sfv_btn_back, f"QPushButton{{ {common_btn} background:#f3f4f6; border:1px solid #e5e7eb; }} QPushButton:hover{{ background:#eef0f3; }}")
#         self._apply_btn_style(self.sfv_btn_search, f"QPushButton{{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }} QPushButton:hover{{ background:#d4e5ff; }}")
#         sfv_row_btn.addWidget(self.sfv_btn_back); sfv_row_btn.addStretch(1); sfv_row_btn.addWidget(self.sfv_btn_search)
#         sfv_layout.addLayout(sfv_row_btn)
#         sfv_layout.addStretch(1) # Đẩy mọi thứ lên trên
#         # ==================== HẾT PHẦN SEARCH FILTER VIEW (HOÀN CHỈNH) ====================

#         # --- Stacked Widget ---
#         self.stacked = QStackedWidget()
#         self.stacked.addWidget(self.main_view)      # index 0
#         self.stacked.addWidget(self.history_view)   # index 1
#         self.stacked.addWidget(self.detail_view)    # index 2
#         self.stacked.addWidget(self.search_filter_view) # index 3
#         self.stacked.setCurrentIndex(0)
#         right_container.addWidget(self.stacked, 1)
#         root.addLayout(right_container, 1)
#         self.update_titles_and_modes()

#         # --- Connect Signals ---
#         self.btn_show_history.clicked.connect(self.on_show_all_history_clicked)
#         self.btn_hide_history.clicked.connect(self.show_main_view)
#         self.btn_export_hist.clicked.connect(self.on_export_excel)
#         self.btn_delete_hist.clicked.connect(self.on_delete_history)
#         self.btn_search_hist.clicked.connect(self.on_search_history_clicked) # Nút tìm kiếm bên trái
#         self.btn_back_to_history.clicked.connect(self.show_history_view_only) # Nút quay lại từ trang detail
#         self.sfv_btn_back.clicked.connect(self.show_main_view) # Nút quay lại từ trang search filter
#         self.sfv_btn_search.clicked.connect(self.on_run_search_from_page) # Nút tìm kiếm trên trang search filter

#     # ... (Hàm update_titles_and_modes, on_reset_lanes, on_one_way_clicked, on_two_way_clicked, update_match_status, on_play_sound, on_ocr_mode_changed giữ nguyên) ...
#     def update_titles_and_modes(self):
#         self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
#         self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
#         if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
#         if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")
#     @Slot()
#     def on_reset_lanes(self):
#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"; self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.update_titles_and_modes(); self.show_logo(1); self.show_logo(2)
#     @Slot()
#     def on_one_way_clicked(self):
#         if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
#         else: self.lane1_dir="RA"; self.lane2_dir="RA"
#         self.one_way_toggle_vao = not self.one_way_toggle_vao; self.update_titles_and_modes()
#     @Slot()
#     def on_two_way_clicked(self):
#         if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
#         else: self.lane1_dir="RA"; self.lane2_dir="VÀO"
#         self.two_way_toggle = not self.two_way_toggle; self.update_titles_and_modes()
#     @Slot(str)
#     def update_match_status(self, status: str):
#         display_status = status.replace('-', ' ').title()
#         self.txt_match.setText(display_status)
#         if "Khop Bien So" in display_status: self.txt_match.setStyleSheet("color: #007700; font-weight: 700;")
#         elif "Khong Khop Bien So" in display_status: self.txt_match.setStyleSheet("color: #ff0000; font-weight: 700;")
#         else: self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
#     @Slot(str)
#     def on_play_sound(self, mode):
#         if mode == "in": self.sound_in.play()
#         elif mode == "out": self.sound_out.play()
#     @Slot()
#     def on_ocr_mode_changed(self):
#         self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
#         if self.rb_gem.isChecked() and not GEMINI_READY:
#             QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY. Sẽ dùng YOLO OCR.")
#             self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
#         if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
#         if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

#     # ---- 8.13 Hiển thị chế độ xem Lịch sử (CHỈ GỌI HÀM PHỤ) ----
#     def show_history_view(self):
#         """Hàm này không nên được gọi trực tiếp nữa, chỉ là dự phòng."""
#         # Không reset bộ lọc ở đây
#         # Chỉ chuyển tab
#         self.show_history_view_only()
#         # Không tải lại data ở đây

#     # ---- 8.xx SỬA LẠI: Slot cho nút "Xem bảng lịch sử" ----
#     @Slot()
#     def on_show_all_history_clicked(self):
#         """Slot này được kết nối với btn_show_history. Nó CHỈ tải lại."""
#         print("\n--- DEBUG: on_show_all_history_clicked just called refresh_history_data ---\n")
#         # KHÔNG xóa bộ lọc ở đây nữa
#         # Chuyển tab nếu cần
#         if self.stacked.currentIndex() != 1:
#             self.show_history_view_only()
#         # Gọi tải lại (hàm refresh_history_data sẽ tự biết xóa bộ lọc nếu cần)
#         self.refresh_history_data(clear_filters=True) # Thêm cờ clear_filters

#     @Slot()
#     def show_history_view_only(self):
#         """Hàm phụ trợ: Chỉ chuyển tab, không tải lại dữ liệu"""
#         self.stacked.setCurrentIndex(1) # Chuyển về tab bảng (index 1)
#         self.btn_show_history.hide()
#         self.btn_hide_history.show()

#     # HÀM NÀY ĐÚNG RỒI
#     def show_main_view(self):
#         self.stacked.setCurrentIndex(0) # Về trang chính (index 0)
#         self.btn_hide_history.hide()
#         self.btn_show_history.show()

#     # ... (Hàm on_export_excel, on_delete_history giữ nguyên) ...
#     @Slot()
#     def on_export_excel(self):
#         # Lấy df hiện tại (có thể đã lọc hoặc chưa)
#         df_to_export = self.history_df.copy()
#         if not df_to_export.empty and "STT" in df_to_export.columns:
#              df_to_export = df_to_export.drop(columns=["STT"]) # Bỏ cột STT khi export
#         if df_to_export.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
#         path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
#         if not path: return
#         try: df_to_export.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
#         except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

#     @Slot()
#     def on_delete_history(self):
#         if not (self.db and self.db.ok):
#             QMessageBox.warning(self, "Xóa", "Chưa kết nối DB."); return

#         dlg = DeleteDialog(self)
#         g = self.geometry()
#         dlg.adjustSize()
#         parent_center = self.geometry().center()
#         dlg_rect = dlg.frameGeometry()
#         dlg_rect.moveCenter(self.mapToGlobal(parent_center))
#         dlg.move(dlg_rect.topLeft())
#         res = dlg.exec()

#         ids_to_delete = []
#         if res == 1:  # Xóa dòng chọn
#             rows_view = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
#             if not rows_view:
#                 QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào."); return
#             for r_view in rows_view:
#                 id_item = self.tbl_hist.item(r_view, 0)  # cột 0 là ID
#                 if id_item: ids_to_delete.append(id_item.text())
#             if not ids_to_delete:
#                 QMessageBox.warning(self, "Xóa", "Không lấy được ID các dòng chọn."); return
#             self.db.delete_by_ids(ids_to_delete)

#         elif res == 2:  # Xóa tất cả
#             confirm = QMessageBox.question(
#                 self, "Xác nhận",
#                 "Bạn chắc chắn muốn xóa TOÀN BỘ lịch sử?",
#                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
#             )
#             if confirm == QMessageBox.StandardButton.Yes:
#                 self.db.delete_all()
#             else:
#                 return
#         else:
#             return

#         # --- NEW: luôn quay về bảng lịch sử và dọn trang chi tiết ---
#         self.clear_detail_view()               # NEW
#         self.show_history_view_only()          # NEW

#         # Tải lại dữ liệu với bộ lọc hiện tại
#         self.refresh_history_data(
#             start_time=self.current_filter_start,
#             end_time=self.current_filter_end,
#             status_filter=self.current_filter_status,
#             plate_filter=self.current_filter_plate
#         )

#     def clear_detail_view(self):
#         """Xóa nội dung/ảnh ở trang chi tiết và bỏ chọn các dòng trong bảng."""
#         # Clear text fields
#         for w in (
#             self.lbl_detail_plate_in, self.lbl_detail_date_in, self.lbl_detail_time_in,
#             self.lbl_detail_plate_out, self.lbl_detail_date_out, self.lbl_detail_time_out,
#             self.lbl_detail_match
#         ):
#             w.setText("")

#         # Đổi ảnh về logo mặc định
#         self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())
#         self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

#         # Bỏ chọn các hàng trong bảng lịch sử
#         self.tbl_hist.clearSelection()



#     # ... (Hàm qpix_logo, show_logo giữ nguyên) ...
#     def qpix_logo(self):
#         if os.path.exists(LOGO_PATH): return QPixmap(LOGO_PATH)
#         return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))
#     def show_logo(self, which: int):
#         pm = self._logo_pm
#         if which == 1: self._set_centered_pixmap(self.lbl_cam1, pm)
#         else: self._set_centered_pixmap(self.lbl_cam2, pm)


#     # ... (Hàm on_frame, on_scene, on_roi, on_info, on_match giữ nguyên) ...
#     @Slot(np.ndarray, str)
#     def on_frame(self, frame_bgr, title):
#         sender = self.sender()
#         if sender is self.cam1_worker: self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
#         elif sender is self.cam2_worker: self._set_centered_pixmap(self.lbl_cam2, frame_bgr)
#     @Slot(str)
#     def on_scene(self, path):
#         # Dùng hàm kiểm tra đường dẫn an toàn
#         valid_path = self._get_valid_image_path_internal(path)
#         if valid_path: bgr = cv2.imread(valid_path); self._set_centered_pixmap(self.lbl_scene, bgr)
#         else: self._set_centered_pixmap(self.lbl_scene, self.qpix_logo()) # Hiển thị logo nếu lỗi
#     @Slot(str, str)
#     def on_roi(self, roi_path, mode):
#         valid_path = self._get_valid_image_path_internal(roi_path)
#         if valid_path: bgr = cv2.imread(valid_path); self._set_centered_pixmap(self.lbl_roi, bgr)
#         else: self._set_centered_pixmap(self.lbl_roi, self.qpix_logo())
#     @Slot(dict)
#     def on_info(self, info):
#         if "date_in" in info: self.txt_date_in.setText(info["date_in"]); self.ed_date_in.setText(info["date_in"])
#         if "time_in" in info: self.txt_time_in.setText(info["time_in"]); self.ed_time_in.setText(info["time_in"])
#         if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
#         if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
#         if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
#         if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])
#     @Slot(str)
#     def on_match(self, txt): self.txt_match.setText(txt.upper())


#     # ---- 8.24 Tải và cập nhật bảng lịch sử (SỬA LẠI LOGIC XÓA FILTER) ----
#     # Thêm tham số clear_filters=False
#     def refresh_history_data(self, start_time=None, end_time=None, status_filter=None, plate_filter=None, clear_filters=False):
#         """Khởi động luồng ngầm để tải dữ liệu lịch sử."""

#         # ***** XÓA BỘ LỌC NẾU CÓ YÊU CẦU (MỚI) *****
#         if clear_filters:
#             print("--- Clearing filters because clear_filters=True ---")
#             self.current_filter_start = None
#             self.current_filter_end = None
#             self.current_filter_status = None
#             self.current_filter_plate = None
#             # Reset các biến start_time, etc. về None để worker dùng giá trị đúng
#             start_time = None
#             end_time = None
#             status_filter = None
#             plate_filter = None

#         # Nếu worker đang chạy thì không làm gì cả
#         if self.history_worker and self.history_worker.isRunning():
#             print("History worker is already running.")
#             return

#         # Tạo và chạy worker mới với các bộ lọc (đã được xóa nếu clear_filters=True)
#         print(f"+++ Starting HistoryLoaderWorker with filters: Start={start_time}, End={end_time}, Status={status_filter}, Plate={plate_filter} +++")
#         self.history_worker = HistoryLoaderWorker(self.db, start_time, end_time, status_filter, plate_filter, self)
#         self.history_worker.resultReady.connect(self.update_history_table)
#         self.history_worker.finished.connect(self.history_worker.deleteLater)
#         self.history_worker.start()


#     # ---- 8.xx MỚI: Slot trung gian cho Timer/Worker (KHÔNG XÓA FILTER) ----
#     @Slot()
#     def on_history_signal_refresh(self):
#         """Refresh bảng lịch sử chỉ khi đang ở tab lịch sử và không quá dày"""
#         if self.stacked.currentIndex() != 1:
#             return
#         now = time.time()
#         if now - self._hist_last_reload < 5.0:   # không reload quá 1 lần / 5 giây
#             return
#         self._hist_last_reload = now
#         self.refresh_history_data(start_time=self.current_filter_start,
#                                 end_time=self.current_filter_end,
#                                 status_filter=self.current_filter_status,
#                                 plate_filter=self.current_filter_plate)


#     # ---- 8.xx MỚI: Xử lý sự kiện nhấn 'Tìm kiếm' TỪ TRANG LỌC (INDEX 3) - (KHÔNG XÓA FILTER) ----
#     @Slot()
#     def on_run_search_from_page(self):
#         # ... (Lấy start_dt, end_dt, selected_statuses, plate_text như cũ) ...
#         print(">>> Entering on_run_search_from_page")
#         qdate_start = self.sfv_date_start.date(); qtime_start = self.sfv_time_start.time(); qdate_end = self.sfv_date_end.date(); qtime_end = self.sfv_time_end.time(); start_dt = QDateTime(qdate_start, qtime_start).toPython(); end_dt = QDateTime(qdate_end, qtime_end).toPython()
#         if start_dt > end_dt: QMessageBox.warning(self, "Lỗi nhập liệu", "'Từ ngày/giờ' không được lớn hơn 'Đến ngày/giờ'.\nVui lòng kiểm tra lại."); print("<<< Exiting on_run_search_from_page (Date Error)"); return
#         selected_statuses = []; plate_text = self.sfv_txt_plate.text().strip()
#         if self.sfv_chk_pending.isChecked(): selected_statuses.append("PENDING");
#         if self.sfv_chk_match.isChecked(): selected_statuses.append("KHOP-BIEN-SO")
#         if self.sfv_chk_mismatch.isChecked(): selected_statuses.append("KHONG-KHOP-BIEN-SO")

#         # 5. LƯU LẠI BỘ LỌC HIỆN TẠI
#         self.current_filter_start = start_dt
#         self.current_filter_end = end_dt
#         self.current_filter_status = selected_statuses if selected_statuses else None
#         self.current_filter_plate = plate_text if plate_text else None
#         print(">>> Filters JUST SET in on_run_search:"); # ... (print filters) ...

#         # 6. Gọi hàm tải dữ liệu VỚI bộ lọc, KHÔNG clear_filters
#         print(">>> Calling refresh_history_data...")
#         self.refresh_history_data(start_time=self.current_filter_start,
#                                 end_time=self.current_filter_end,
#                                 status_filter=self.current_filter_status,
#                                 plate_filter=self.current_filter_plate) # Bỏ clear_filters=True
#         print(">>> Returned from refresh_history_data.")
#         # ... (print filters before switch, setCurrentIndex) ...
#         print("<<< Exiting on_run_search_from_page (Success)")
#         self.show_history_view_only()   # chuyển sang tab bảng (index 1)

#     # ---- 8.xx MỚI: Slot nhận kết quả DataFrame từ Worker ----
#     @Slot(pd.DataFrame)
#     def update_history_table(self, df: pd.DataFrame):
#         """Cập nhật QTableWidget với DataFrame nhận được (nhẹ, không block UI)."""
#         print(f"+++ update_history_table received {len(df)} rows +++")

#         # 1) Lưu df gốc (có STT để tra chi tiết)
#         self.history_df = df.copy()

#         # 2) Chuẩn bị df hiển thị (bỏ STT nếu có)
#         df_display = df.drop(columns=["STT"], errors="ignore")

#         # 3) Tắt redraw & sort để đổ nhanh
#         self.tbl_hist.setUpdatesEnabled(False)
#         self.tbl_hist.setSortingEnabled(False)

#         # 4) Cập nhật cấu trúc bảng
#         cols = list(df_display.columns)
#         self.tbl_hist.clearContents()
#         self.tbl_hist.setColumnCount(len(cols))
#         self.tbl_hist.setHorizontalHeaderLabels(cols)
#         self.tbl_hist.setSortingEnabled(False)
#         self.tbl_hist.setRowCount(len(df_display))

#         # 5) Điền dữ liệu
#         for i in range(len(df_display)):
#             for j, col in enumerate(cols):
#                 if j < self.tbl_hist.columnCount():
#                     val = df_display.iloc[i, j]
#                     item = QTableWidgetItem()
#                     # Nếu là cột ID (cột 0), dùng DisplayRole=int để Qt hiểu là số:
#                     if j == 0:
#                         try:
#                             item.setData(Qt.ItemDataRole.DisplayRole, int(val))
#                         except:
#                             item.setText(str(val))
#                     else:
#                         item.setText(str(val))
#                     item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
#                     self.tbl_hist.setItem(i, j, item)
#         self.tbl_hist.setSortingEnabled(True)
#         self.tbl_hist.sortByColumn(0, Qt.SortOrder.DescendingOrder)  # 0 = cột ID

#         # 6) Bật lại sort & redraw
#         self.tbl_hist.setSortingEnabled(True)
#         self.tbl_hist.setUpdatesEnabled(True)

#         # 7) giải phóng tham chiếu worker
#         self.history_worker = None
#         print("--- History worker reference released ---")


#     # HÀM NÀY ĐÚNG RỒI
#     @Slot()
#     def on_search_history_clicked(self):
#         """Mở trang bộ lọc tìm kiếm (index 3)"""
#         self.stacked.setCurrentIndex(3)
#         self.btn_show_history.hide(); self.btn_hide_history.show()

#     # THÊM HÀM HELPER KIỂM TRA ĐƯỜNG DẪN ẢNH
#     def _get_valid_image_path_internal(self, path_from_db):
#         if not path_from_db: return None
#         # Ưu tiên kiểm tra tuyệt đối trước
#         if os.path.exists(path_from_db): return path_from_db
#         # Thử ghép tương đối
#         maybe_path = os.path.abspath(path_from_db)
#         if os.path.exists(maybe_path): return maybe_path
#         print(f"Cảnh báo: Không tìm thấy ảnh tại '{path_from_db}' hoặc '{maybe_path}'")
#         return None

#     # HÀM NÀY ĐÚNG RỒI (Đã sửa lỗi đường dẫn)
#     @Slot()
#     def on_history_row_selected(self):
#         selected_items = self.tbl_hist.selectedItems()
#         if not selected_items or self.history_df.empty or "ID" not in self.history_df.columns: return

#         try:
#             row_index_view = selected_items[0].row()
#             id_item = self.tbl_hist.item(row_index_view, 0) # Cột 0 là ID
#             if not id_item: return
#             row_id = int(id_item.text())

#             row_data_series = self.history_df[self.history_df['ID'] == row_id]
#             if row_data_series.empty: return
#             row_data = row_data_series.iloc[0]

#             # Cập nhật thông tin trang chi tiết
#             self.lbl_detail_plate_in.setText(str(row_data.get("Biển số vào", "")))
#             self.lbl_detail_date_in.setText(str(row_data.get("Ngày vào", "")))
#             self.lbl_detail_time_in.setText(str(row_data.get("Giờ vào", "")))
#             self.lbl_detail_plate_out.setText(str(row_data.get("Biển số ra", "")))
#             self.lbl_detail_date_out.setText(str(row_data.get("Ngày ra", "")))
#             self.lbl_detail_time_out.setText(str(row_data.get("Giờ ra", "")))
#             match_status = str(row_data.get("Trạng thái", "")).replace('-', ' ').title()
#             self.lbl_detail_match.setText(match_status)
#             if "Khop Bien So" in match_status: self.lbl_detail_match.setStyleSheet("color: #007700; font-weight: 700;")
#             elif "Khong Khop Bien So" in match_status: self.lbl_detail_match.setStyleSheet("color: #ff0000; font-weight: 700;")
#             else: self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")

#             # Cập nhật ảnh (dùng hàm helper)
#             valid_in_path = self._get_valid_image_path_internal(str(row_data.get("Ảnh vào", "")))
#             valid_out_path = self._get_valid_image_path_internal(str(row_data.get("Ảnh ra", "")))

#             if valid_in_path: self._set_centered_pixmap(self.lbl_detail_scene, cv2.imread(valid_in_path))
#             else: self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())
#             if valid_out_path: self._set_centered_pixmap(self.lbl_detail_roi, cv2.imread(valid_out_path))
#             else: self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

#             # Chuyển sang trang chi tiết (index 2)
#             self.stacked.setCurrentIndex(2)

#         except Exception as e:
#             print(f"Lỗi khi chọn hàng: {e}"); import traceback; traceback.print_exc()


#     # ... (Hàm _connect_worker, start/stop_cam_generic, start/stop_cam1/2, closeEvent giữ nguyên) ...
#     def _connect_worker(self, w: VideoWorker):
#         w.frameSignal.connect(self.on_frame)
#         w.sceneSignal.connect(self.on_scene)
#         w.roiSignal.connect(self.on_roi)
#         w.infoSignal.connect(self.on_info)
#         w.matchSignal.connect(self.on_match)
#         w.histSignal.connect(self.on_history_signal_refresh)
#         w.playSoundSignal.connect(self.on_play_sound)

#     def _set_cam_buttons_state(self, which: int, running: bool):
#         if which == 1:
#             self.btn_start1.setEnabled(not running)
#             self.btn_stop1.setEnabled(running)
#         else:
#             self.btn_start2.setEnabled(not running)
#             self.btn_stop2.setEnabled(running)


#     def start_cam_generic(self, which: int):
#         if not self.models.ok: QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}"); return
#         if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
#         if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return
#         ocr_mode = self.current_ocr_mode; default_api = API_MAP["DSHOW(Windows)"]
#         if which == 1:
#             idx = int(self.spin_cam1.value()); mode = "in" if self.lane1_dir=="VÀO" else "out"; title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam1_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam1_worker); self.cam1_worker.start(); self._set_cam_buttons_state(which, True)
#         else:
#             idx = int(self.spin_cam2.value()); mode = "in" if self.lane2_dir=="VÀO" else "out"; title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam2_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam2_worker); self.cam2_worker.start(); self._set_cam_buttons_state(which, True)

#     def stop_cam_generic(self, which: int):
#         worker = self.cam1_worker if which==1 else self.cam2_worker
#         if worker and worker.isRunning(): worker.stop(); worker.wait(1000)
#         if which==1: self.cam1_worker = None; self.show_logo(1)
#         else: self.cam2_worker = None; self.show_logo(2)
#         self._set_cam_buttons_state(which, False)

#     def start_cam1(self): self.start_cam_generic(1)
#     def stop_cam1(self): self.stop_cam_generic(1)
#     def start_cam2(self): self.start_cam_generic(2)
#     def stop_cam2(self): self.stop_cam_generic(2)

#     def closeEvent(self, event):
#         try: self.stop_cam_generic(1); self.stop_cam_generic(2)
#         except: pass
#         super().closeEvent(event)

# # ==================== 9. MAIN ====================
# def main():
#     QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
#     app = QApplication(sys.argv)
#     app.setStyle("Fusion")
#     w = MainWindow()
#     w.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()



















































































# -*- coding: utf-8 -*-
"""
=========================================================
= PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini) =
=========================================================
    # ... (Các comment mô tả giữ nguyên) ...
"""

# ==================== 1. IMPORT ====================

import os, sys, time, cv2, traceback
import numpy as np, pandas as pd
from datetime import datetime

# ---- 1.1 HiDPI Cấu hình HiDPI (Độ phân giải cao) ----
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# ---- 1.2 Import PySide6 ----
# SỬA LẠI IMPORT ĐỂ THÊM QDate, QTime, QDateEdit, QTimeEdit
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint, QUrl, QDateTime, QDate, QTime
from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
    QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
    QFileDialog, QHeaderView, QDialog, QDateTimeEdit,
    QDateEdit, QTimeEdit, QCheckBox
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
# ... (Phần này giữ nguyên code của bạn) ...
DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")
SOUND_IN_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_vao_xin_cam_on.wav"
SOUND_OUT_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_ra_xin_cam_on.wav"
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"
)
PANEL_W, PANEL_H = 640, 360
PANEL_BG = (255, 255, 255)
API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
           "six":"6","seven":"7","eight":"8","nine":"9"}

# ==================== 3. UTILITIES (HÀM TIỆN ÍCH) ====================
# SỬA LẠI HÀM NÀY ĐỂ TRẢ VỀ ĐƯỜNG DẪN TUYỆT ĐỐI
def save_image(img, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Tạo đường dẫn tương đối trước
    relative_path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    # Chuyển nó thành đường dẫn tuyệt đối
    absolute_path = os.path.abspath(relative_path)
    # Lưu ảnh dùng đường dẫn tuyệt đối
    try:
        cv2.imwrite(absolute_path, img)
        # Trả về đường dẫn tuyệt đối để lưu vào DB
        return absolute_path
    except Exception as e:
        print(f"Lỗi khi lưu ảnh {absolute_path}: {e}")
        return None # Trả về None nếu có lỗi

def letterbox(bgr, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
    if bgr is None: return np.full((h, w, 3), color, dtype=np.uint8)
    ih, iw = bgr.shape[:2]
    if ih == 0 or iw == 0: return np.full((h, w, 3), color, dtype=np.uint8)
    s = min(w/iw, h/ih); nw, nh = int(iw*s), int(ih*s)
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    top, left = (h-nh)//2, (w-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def bgr_to_qimage(bgr):
    if bgr is None: bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

def norm_char(x): return OCR_MAP.get(str(x), str(x))
def plate_norm(s: str) -> str: return (s or "").replace("-", "").replace(" ", "").upper()
def has_boxes(r):
    try: return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
    except: return False
def preprocess_for_ocr(roi):
    if roi is None: return None
    if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe,(3,3),0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# ==================== 4. DB LAYER ====================
class DB:
    # ... (Hàm __init__, insert_in, attach_out giữ nguyên code của bạn) ...
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
                    image_in NVARCHAR(MAX) NULL,
                    plate_out NVARCHAR(64)  NULL,
                    date_out  NVARCHAR(16)  NULL,
                    time_out  NVARCHAR(16)  NULL,
                    image_out NVARCHAR(MAX) NULL,
                    match_status NVARCHAR(32) NULL,
                    created_at DATETIME DEFAULT GETDATE()
                );
            """)
            self.ok = True
        except Exception as e:
            print("DB connect error:", e); self.ok = False

    def insert_in(self, plate, d, t, img_path):
        if not self.ok or not img_path: return # Thêm kiểm tra img_path
        try:
            self.cur.execute("""
                INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
                VALUES (?,?,?,?,?)
            """, (plate, d, t, img_path, 'PENDING'))
        except Exception as e: print("insert_in error:", e)

    def attach_out(self, plate_out, d, t, img_path) -> str:
        if not self.ok or not img_path: return "Khong khop bien so" # Thêm kiểm tra img_path
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

    # ---- 4.4 Lấy lịch sử (ĐÃ CẬP NHẬT ĐỂ LỌC) ----
    def fetch_history_df(self, limit=10000, start_time=None, end_time=None,
                     status_filter=None, plate_filter=None) -> pd.DataFrame:
        """
        Lọc theo:
        - Khoảng thời gian VÀO/RA (dựa trên date_in+time_in và date_out+time_out, đều là NVARCHAR)
        - Trạng thái (match_status)
        - Biển số (plate_in/plate_out LIKE)
        Không dùng created_at.
        """
        columns = [
            "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
            "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
        ]
        if not self.ok:
            return pd.DataFrame(columns=["STT"] + columns)

        try:
            # Hai cột thời gian quy đổi từ chuỗi dd/MM/yyyy + HH:mm:ss
            # style 103 = dd/MM/yyyy
            dt_in  = "TRY_CONVERT(datetime, date_in  + ' ' + time_in , 103)"
            dt_out = "TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)"

            sql = f"""
                SELECT TOP ({limit})
                    id, image_in, plate_in, date_in, time_in,
                    image_out, plate_out, date_out, time_out, match_status
                FROM dbo.ParkingSessions
            """

            where_clauses = []
            sql_params = []

            # ------- Lọc theo khoảng thời gian vào/ra -------
            # Ý tưởng: nếu bản ghi có thời điểm vào/ra nằm trong khoảng thì lấy.
            # (dt_out BETWEEN ? AND ?) OR (dt_in BETWEEN ? AND ?)
            if start_time and end_time:
                where_clauses.append(f"( ({dt_in}  BETWEEN ? AND ?) OR ({dt_out} BETWEEN ? AND ?) )")
                sql_params += [start_time, end_time, start_time, end_time]
            elif start_time:
                where_clauses.append(f"( {dt_in}  >= ? OR {dt_out} >= ? )")
                sql_params += [start_time, start_time]
            elif end_time:
                where_clauses.append(f"( {dt_in}  <= ? OR {dt_out} <= ? )")
                sql_params += [end_time, end_time]

            # ------- Lọc Trạng thái -------
            if status_filter and len(status_filter) > 0:
                placeholders = ",".join("?" for _ in status_filter)
                where_clauses.append(f"match_status IN ({placeholders})")
                sql_params += status_filter

            # ------- Lọc Biển số gần đúng ở cả vào/ra -------
            if plate_filter and len(plate_filter.strip()) > 0:
                where_clauses.append("(plate_in LIKE ? OR plate_out LIKE ?)")
                like_term = f"%{plate_filter.strip()}%"
                sql_params += [like_term, like_term]

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            # ------- Sắp xếp mới: ưu tiên thời điểm gần nhất trong hai mốc, rồi theo id -------
            # COALESCE: nếu dt_out có thì ưu tiên dt_out; nếu chưa có (PENDING) dùng dt_in
            sql += f" ORDER BY COALESCE({dt_out}, {dt_in}) DESC, id DESC"

            rows = self.cur.execute(sql, tuple(sql_params)).fetchall()

            df = pd.DataFrame.from_records(rows, columns=columns).astype(object).where(pd.notnull, "")
            df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
            df.insert(0, "STT", range(1, len(df) + 1))
            return df

        except Exception as e:
            print(f"fetch_history_df error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=["STT"] + columns)


    # ... (Hàm delete_by_ids, delete_all giữ nguyên code của bạn) ...
    def delete_by_ids(self, ids):
        if not self.ok or not ids: return
        try:
            placeholders = ','.join('?' for _ in ids)
            sql = f"DELETE FROM dbo.ParkingSessions WHERE id IN ({placeholders})"
            self.cur.execute(sql, tuple(int(sid) for sid in ids))
        except Exception as e: print("delete_by_ids error:", e)

    def delete_all(self):
        if not self.ok: return
        try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
        except Exception as e: print("delete_all error:", e)


# ==================== 5. YOLO/GEMINI WRAPPERS ====================
class Models:
    # ... (Toàn bộ class này giữ nguyên code của bạn) ...
    def __init__(self, det_path: str, ocr_path: str):
        self.ok = True; self.err = ""
        try:
            self.det = YOLO(det_path)
            self.ocr = YOLO(ocr_path)
        except Exception as e:
            self.ok = False; self.err = str(e)

    def detect_plates(self, frame):
        plates, boxed = [], None # Khởi tạo boxed = None
        try:
            boxed = frame.copy() # Copy frame gốc
            results = self.det(frame, verbose=False) # Tắt verbose
            for r in results:
                if not has_boxes(r): continue
                xyxy_np = r.boxes.xyxy.cpu().numpy().astype(int)
                for (x1,y1,x2,y2) in xyxy_np:
                    pad=8
                    fh, fw = frame.shape[:2] # Lấy kích thước frame gốc
                    x1=max(0,x1-pad); y1=max(0,y1-pad)
                    x2=min(fw-1,x2+pad); y2=min(fh-1,y2+pad)
                    # Cắt ROI từ frame gốc
                    roi = frame[y1:y2, x1:x2].copy()
                    if roi.size == 0: continue # Bỏ qua ROI rỗng
                    plates.append(((x1,y1,x2,y2), roi))
                    # Vẽ lên ảnh copy
                    cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        except Exception as e:
            print(f"Lỗi detect_plates: {e}")
            # Nếu có lỗi, trả về frame gốc và không có plates
            return [], frame
        # Trả về ảnh đã vẽ và danh sách plates
        return plates, boxed if boxed is not None else frame


    def ocr_plate_yolo(self, roi):
        if roi is None or roi.size == 0: return "", "" # Kiểm tra roi rỗng
        try:
            roi_pre = preprocess_for_ocr(roi)
            # Nếu preprocess lỗi, dùng roi gốc
            input_roi = roi_pre if roi_pre is not None and roi_pre.size > 0 else roi
            res = self.ocr(input_roi, verbose=False) # Tắt verbose
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
                # Sửa lỗi 'float' object cannot be interpreted as an integer
                h_roi = input_roi.shape[0]
                if len(boxes)<=7 or (max(ys)-min(ys) < 0.2 * h_roi): # So sánh với chiều cao ROI
                    text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
                else:
                    thr=(max(ys)+min(ys))/2.0
                    l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
                    t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
                    t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
                    text_raw=f"{t1}-{t2}" if t2 else t1
            return self._format_text(text_raw)
        except Exception as e:
            print(f"Lỗi ocr_plate_yolo: {e}")
            return "", "" # Trả về rỗng nếu có lỗi

    def ocr_plate_gemini_from_path(self, image_path: str):
        # ... (Hàm này giữ nguyên code của bạn) ...
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
        # ... (Hàm này giữ nguyên code của bạn) ...
        raw=(text_raw or '').replace('-', '').replace('.', '').replace(' ', '') # Bỏ luôn dấu chấm
        # Logic định dạng lại biển số (ví dụ)
        if len(raw) >= 7 and len(raw) <= 9:
             # Biển 2 dòng cũ (VD: 59C112345) -> 59-C1 123.45
             if raw[2].isalpha() and raw[3].isdigit():
                  return f"{raw[:2]}-{raw[2:4]} {raw[4:7]}.{raw[7:]}" if len(raw) > 7 else f"{raw[:2]}-{raw[2:4]} {raw[4:]}", text_raw
             # Biển 1 dòng mới (VD: 59C112345) -> 59C1-123.45
             elif raw[2].isdigit() and raw[4].isalpha():
                  return f"{raw[:4]}-{raw[4:7]}.{raw[7:]}" if len(raw) > 7 else f"{raw[:4]}-{raw[4:]}", text_raw
        # Trả về gốc nếu không khớp định dạng mong muốn
        return text_raw or "", text_raw or ""


# ==================== 6. VIDEO WORKER ====================
class VideoWorker(QThread):
    # ... (Phần signals và __init__, setters giữ nguyên code của bạn) ...
    frameSignal = Signal(np.ndarray, str)
    sceneSignal = Signal(str)
    roiSignal   = Signal(str, str)
    infoSignal  = Signal(dict)
    matchSignal = Signal(str)
    histSignal  = Signal()
    playSoundSignal = Signal(str)

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

    def set_title(self, title: str): self.title = title
    def set_ocr_mode(self, mode: str): self.ocr_mode = mode
    def set_mode(self, mode: str): self.mode = mode

    def run(self):
        # ... (Phần mở camera giữ nguyên code của bạn) ...
        self._running = True
        try: # Thêm try-except để bắt lỗi mở camera
             self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
             if not (self.cap and self.cap.isOpened()):
                  print(f"Lỗi: Không thể mở camera index {self.cam_idx} với API {self.api}")
                  self._running = False; return
        except Exception as e:
             print(f"Lỗi khi khởi tạo VideoCapture: {e}")
             self._running = False; return

        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        try: self.cap.set(cv2.CAP_PROP_FPS, 30)
        except: pass


        while self._running:
            try: # Thêm try-except cho vòng lặp chính
                ok, frame = self.cap.read()
                if not ok or frame is None: # Kiểm tra frame hợp lệ
                    self.stable_start = 0.0; self.captured = False
                    time.sleep(0.05); continue # Chờ lâu hơn nếu đọc lỗi

                # Gửi frame gốc lên UI
                self.frameSignal.emit(frame, self.title)

                # Phát hiện biển số
                plates, boxed_frame = self.models.detect_plates(frame)

                if not plates:
                    self.stable_start = 0.0; self.captured = False
                    time.sleep(0.01); continue

                # Chọn biển số tốt nhất (ví dụ: lớn nhất)
                best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
                roi_current = best[1]
                if roi_current is None or roi_current.size == 0: # Kiểm tra roi hợp lệ
                     self.stable_start = 0.0; self.captured = False
                     time.sleep(0.01); continue

                # Logic ổn định
                if self.stable_start == 0.0: # Bắt đầu tính giờ nếu chưa tính
                     self.stable_start = time.time()
                elif self.captured: # Nếu đã chụp rồi thì reset ngay
                     self.stable_start = time.time(); self.captured = False


                # Đủ thời gian ổn định và chưa chụp
                if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
                    # Lưu ảnh (nên dùng ảnh đã vẽ hộp)
                    scene_img_to_save = boxed_frame if boxed_frame is not None else frame
                    scene_path = save_image(scene_img_to_save,
                                           "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
                    # Lưu ROI
                    roi_path   = save_image(roi_current,
                                           "plate_in_roi" if self.mode=="in" else "plate_out_roi")

                    # Kiểm tra lưu ảnh thành công
                    if not scene_path or not roi_path:
                        print("Lỗi: Không thể lưu ảnh scene hoặc roi.")
                        self.captured = True # Đánh dấu đã xử lý (dù lỗi) để tránh lặp lại ngay
                        self.stable_start = 0.0 # Reset timer
                        continue

                    # Thực hiện OCR
                    text_fmt, text_raw = "", ""
                    if self.ocr_mode == "gemini" and GEMINI_READY:
                        text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
                    else:
                        text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

                    # Có kết quả OCR
                    if text_fmt or text_raw:
                        now = datetime.now()
                        d = now.strftime("%d/%m/%Y")
                        t = now.strftime("%H:%M:%S")
                        plate = text_fmt or text_raw

                        # Gửi tín hiệu lên UI
                        self.sceneSignal.emit(scene_path) # Gửi đường dẫn ảnh scene
                        self.roiSignal.emit(roi_path, self.mode) # Gửi đường dẫn ảnh roi

                        # Xử lý logic vào/ra và DB
                        if self.mode == "in":
                            self.infoSignal.emit({"date_in": d, "time_in": t, "plate_text_in": plate})
                            if self.db and self.db.ok:
                                self.db.insert_in(plate, d, t, scene_path)
                                self.histSignal.emit()
                            self.playSoundSignal.emit("in")
                        else: # mode == "out"
                            self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
                            if self.db and self.db.ok:
                                match = self.db.attach_out(plate, d, t, scene_path)
                                self.matchSignal.emit(match)
                                self.histSignal.emit()
                            self.playSoundSignal.emit("out")

                        self.captured = True # Đánh dấu đã chụp thành công
                        self.stable_start = 0.0 # Reset timer sau khi chụp thành công

            except Exception as e:
                 print(f"Lỗi trong vòng lặp VideoWorker: {e}")
                 import traceback
                 traceback.print_exc()
                 self.stable_start = 0.0 # Reset nếu có lỗi
                 self.captured = False
                 time.sleep(0.1) # Chờ lâu hơn nếu có lỗi

            time.sleep(0.01) # Thêm sleep nhỏ ở cuối vòng lặp

        # Dọn dẹp khi dừng luồng
        try:
            if self.cap: self.cap.release()
        except Exception as e:
             print(f"Lỗi khi release camera: {e}")


    def stop(self): self._running = False


# ==================== 6.5 HISTORY LOADER WORKER (MỚI - Đặt trước MainWindow) ====================
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
        df = pd.DataFrame() # Khởi tạo df rỗng
        print("HistoryLoaderWorker bắt đầu chạy...")
        try:
             if self.db and self.db.ok:
                  df = self.db.fetch_history_df(limit=800,
                                             start_time=self.start_time,
                                             end_time=self.end_time,
                                             status_filter=self.status_filter,
                                             plate_filter=self.plate_filter)
        except Exception as e:
             print(f"Lỗi trong HistoryLoaderWorker.run: {e}")
             traceback.print_exc() # In chi tiết lỗi
        finally:
             # Đảm bảo luôn emit DataFrame, ngay cả khi rỗng hoặc lỗi
             self.resultReady.emit(df if df is not None else pd.DataFrame())
             print("HistoryLoaderWorker đã chạy xong.")


# ==================== 7. DELETE DIALOG ====================
class DeleteDialog(QDialog):
    # ... (Class này giữ nguyên code của bạn) ...
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xóa lịch sử")
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background: #ffffff; border-radius: 10px; }
            QLabel { font-weight: 600; }
        """)
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
        lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
        lay.addWidget(lab)
        row = QHBoxLayout(); row.setSpacing(12)
        self.btn_sel = QPushButton("Xóa dòng đã chọn")
        self.btn_all = QPushButton("Xóa tất cả")
        self.btn_can = QPushButton("Hủy")
        row.addWidget(self.btn_sel, 1); row.addWidget(self.btn_all, 1); row.addWidget(self.btn_can, 1)
        lay.addLayout(row)
        base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
        self.btn_sel.setStyleSheet(f"QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}")
        self.btn_all.setStyleSheet(f"QPushButton {{ {base} background:#ffe0e0; border:1px solid #ffb3b3; }}")
        self.btn_can.setStyleSheet(f"QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}")
        self.btn_sel.clicked.connect(lambda: self.done(1))
        self.btn_all.clicked.connect(lambda: self.done(2))
        self.btn_can.clicked.connect(lambda: self.done(0))


# ==================== 8. MAIN WINDOW ====================
class MainWindow(QMainWindow):
    # ... (Hàm __init__, _init_theme, _normalize_button, _apply_btn_style, _make_card, _set_centered_pixmap giữ nguyên) ...
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop App (Giữ xe)")
        self.setMinimumSize(1200, 800)
        self._init_theme()
        self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
        if not self.models.ok: QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
        self.db = DB(CONN_STR) if USE_SQL else DB("")
        # Khởi tạo âm thanh
        self.sound_in = QSoundEffect(self)
        sound_in_abs = os.path.abspath(SOUND_IN_PATH)
        if os.path.exists(sound_in_abs): self.sound_in.setSource(QUrl.fromLocalFile(sound_in_abs))
        else: print(f"Lỗi: Không tìm thấy file âm thanh: {sound_in_abs}")
        self.sound_out = QSoundEffect(self)
        sound_out_abs = os.path.abspath(SOUND_OUT_PATH)
        if os.path.exists(sound_out_abs): self.sound_out.setSource(QUrl.fromLocalFile(sound_out_abs))
        else: print(f"Lỗi: Không tìm thấy file âm thanh: {sound_out_abs}")
        self.cam1_worker = None
        self.cam2_worker = None
        self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
        self.one_way_toggle_vao = True; self.two_way_toggle = True
        self.current_ocr_mode = "yolo"
        self.history_df = pd.DataFrame() # Khởi tạo df lịch sử
        self.current_filter_start = None
        self.current_filter_end = None
        self.current_filter_status = None
        self.current_filter_plate = None
        self.history_worker = None
        self._hist_last_reload = 0.0
        self._logo_pm = self.qpix_logo()
        self._build_ui()
        self.show_logo(1); self.show_logo(2)
        # Kết nối timer
        self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.on_history_signal_refresh); self.hist_timer.start(5000)

    def _init_theme(self): self.setStyleSheet(""" * { color: #000000; } QMainWindow, QWidget { background: #ffffff; } QWidget#SideBar { background: #ffffff; } QGroupBox { background: #ffffff; font-weight: 600; border: 2px solid #e6e6e6; border-radius: 12px; margin-top: 8px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; background: #ffffff; } QFrame[class="card-wrap"] { background: #e6e6e6; border-radius: 14px; } QFrame[class="card"] { background: #ffffff; border-radius: 12px; } QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; } QLabel[class="title"] { font: 700 18px "Segoe UI"; padding: 6px 10px; background: #ffffff; border-radius: 10px; } QLineEdit { height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px; } QTableWidget { background: #ffffff; gridline-color: #e6e6e6; } """)
    def _normalize_button(self, *btns):
        for b in btns:
            b.setAutoDefault(False); b.setDefault(False); b.setFlat(False); b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    def _apply_btn_style(self, btn: QPushButton, css: str): btn.setStyleSheet(css)
    def _make_card(self, title:str, content:QWidget):
        wrap = QFrame(); wrap.setProperty("class","card-wrap"); wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
        card = QFrame(); card.setProperty("class","card"); v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
        title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap"); hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
        title_lbl = QLabel(title); title_lbl.setProperty("class","title"); hl.addWidget(title_lbl)
        v.addWidget(title_wrap); v.addWidget(content, 1); wrapL.addWidget(card); return wrap, title_lbl
    def _set_centered_pixmap(self, lbl: QLabel, src):
        pm = None # Khởi tạo pm
        if isinstance(src, np.ndarray): pm = QPixmap.fromImage(bgr_to_qimage(src))
        elif isinstance(src, QImage): pm = QPixmap.fromImage(src)
        elif isinstance(src, QPixmap): pm = src # Chấp nhận cả QPixmap
        if pm is None or pm.isNull(): lbl.clear(); return
        rect = lbl.contentsRect(); avail = rect.size()
        dpr = lbl.devicePixelRatioF() if hasattr(lbl, "devicePixelRatioF") else 1.0
        target_w = max(1, int(avail.width()  * dpr)); target_h = max(1, int(avail.height() * dpr))
        scaled = pm.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        if hasattr(scaled, "setDevicePixelRatio"): scaled.setDevicePixelRatio(dpr)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setPixmap(scaled)

    # ĐÂY LÀ HÀM QUAN TRỌNG NHẤT, THAY THẾ TOÀN BỘ HÀM CŨ
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)
        # LEFT PANEL (SIDEBAR)
        side = QWidget(objectName="SideBar"); side.setFixedWidth(450)
        vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)
        common_btn = "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;" # Định nghĩa sớm hơn
        # Camera Control
        gb_camctl = QGroupBox("CAMERA CONTROL"); vl_camctl = QVBoxLayout(gb_camctl); vl_camctl.setSpacing(10)
        self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
        self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
        row_indices = QHBoxLayout(); row_indices.setSpacing(10)
        row_indices.addWidget(QLabel("Index Cam 1")); row_indices.addWidget(self.spin_cam1, 1)
        row_indices.addWidget(QLabel("Index Cam 2")); row_indices.addWidget(self.spin_cam2, 1)
        vl_camctl.addLayout(row_indices)
        self.btn_start1 = QPushButton("Bật Cam 1"); self.btn_stop1 = QPushButton("Tắt Cam 1")
        self.btn_start2 = QPushButton("Bật Cam 2"); self.btn_stop2 = QPushButton("Tắt Cam 2")
        self._normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)
        self._apply_btn_style(self.btn_start1, f"QPushButton{{ {common_btn} background:#d1fadf; border:1px solid #a6f4c5; }} QPushButton:hover{{ background:#c3f7d6; }} QPushButton:disabled{{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }}")
        self._apply_btn_style(self.btn_stop1, f"QPushButton{{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }} QPushButton:hover{{ background:#ffd1d1; }} QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}")
        self._apply_btn_style(self.btn_start2, self.btn_start1.styleSheet()); self._apply_btn_style(self.btn_stop2, self.btn_stop1.styleSheet())
        self.btn_start1.clicked.connect(self.start_cam1); self.btn_stop1.clicked.connect(self.stop_cam1)
        self.btn_start2.clicked.connect(self.start_cam2); self.btn_stop2.clicked.connect(self.stop_cam2)
        row_btn1 = QHBoxLayout(); row_btn1.setSpacing(12); row_btn1.addWidget(self.btn_start1); row_btn1.addWidget(self.btn_stop1); vl_camctl.addLayout(row_btn1)
        row_btn2 = QHBoxLayout(); row_btn2.setSpacing(12); row_btn2.addWidget(self.btn_start2); row_btn2.addWidget(self.btn_stop2); vl_camctl.addLayout(row_btn2)
        vside.addWidget(gb_camctl)
        # Lane Control
        gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN"); vl_lane = QVBoxLayout(gb_lane); vl_lane.setSpacing(10)
        row_lane = QHBoxLayout(); row_lane.setSpacing(12)
        self.btn_oneway = QPushButton("1 chiều"); self.btn_twoway = QPushButton("2 chiều"); self.btn_reset_lane = QPushButton("Reset làn")
        self._normalize_button(self.btn_oneway, self.btn_twoway, self.btn_reset_lane)
        self._apply_btn_style(self.btn_oneway, f"QPushButton{{ {common_btn} background:#dbeafe; border:1px solid #bfdbfe; }} QPushButton:hover{{ background:#cfe3fd; }}")
        self._apply_btn_style(self.btn_twoway, self.btn_oneway.styleSheet())
        self._apply_btn_style(self.btn_reset_lane, f"QPushButton{{ {common_btn} background:#fff3bf; border:1px solid #ffe066; }} QPushButton:hover{{ background:#ffeda3; }}")
        row_lane.addWidget(self.btn_oneway); row_lane.addWidget(self.btn_twoway); vl_lane.addLayout(row_lane); vl_lane.addWidget(self.btn_reset_lane)
        self.btn_oneway.clicked.connect(self.on_one_way_clicked); self.btn_twoway.clicked.connect(self.on_two_way_clicked); self.btn_reset_lane.clicked.connect(self.on_reset_lanes)
        vside.addWidget(gb_lane)
        # OCR Model
        gb_ocr = QGroupBox("OCR MODEL"); vb_ocr = QVBoxLayout(gb_ocr)
        self.rb_yolo = QRadioButton("Dùng YOLO OCR (tự train)"); self.rb_yolo.setChecked(True)
        self.rb_gem = QRadioButton("Dùng Gemini AI")
        vb_ocr.addWidget(self.rb_yolo); vb_ocr.addWidget(self.rb_gem)
        self.rb_yolo.toggled.connect(self.on_ocr_mode_changed); self.rb_gem.toggled.connect(self.on_ocr_mode_changed)
        if not GEMINI_READY: self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY")
        vside.addWidget(gb_ocr)
        # Info IN
        gb_in = QGroupBox("THÔNG TIN XE VÀO"); gl_in = QGridLayout(gb_in)
        self.ed_date_in = QLineEdit(); self.ed_time_in = QLineEdit(); self.ed_plate_in = QLineEdit()
        self.ed_plate_in.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
        gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
        gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
        gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
        vside.addWidget(gb_in)
        # Info OUT
        gb_out = QGroupBox("THÔNG TIN XE RA"); gl_out = QGridLayout(gb_out)
        self.ed_date_out = QLineEdit(); self.ed_time_out = QLineEdit(); self.ed_plate_out = QLineEdit()
        self.ed_plate_out.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
        gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
        gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
        gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
        vside.addWidget(gb_out)
        # History Buttons
        gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ"); v_hist_btns = QVBoxLayout(gb_hist_btns)
        self.btn_show_history = QPushButton("Xem bảng lịch sử"); self.btn_export_hist = QPushButton("Export Excel")
        self.btn_delete_hist = QPushButton("Xóa bảng"); self.btn_search_hist = QPushButton("Tìm kiếm")
        self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide()
        self._normalize_button(self.btn_show_history, self.btn_export_hist, self.btn_delete_hist, self.btn_search_hist, self.btn_hide_history)
        self._apply_btn_style(self.btn_show_history, f"QPushButton{{ {common_btn} background:#E6F4EA; border:1px solid #cde9d6; }} QPushButton:hover{{ background:#d9efe0; }}")
        self._apply_btn_style(self.btn_hide_history, f"QPushButton{{ {common_btn} background:#fff3bf; border:1px solid #f5c6c2; }} QPushButton:hover{{ background:#ffeda3; }}")
        self._apply_btn_style(self.btn_export_hist, f"QPushButton{{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }} QPushButton:hover{{ background:#d4e5ff; }}")
        self._apply_btn_style(self.btn_delete_hist, f"QPushButton{{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }} QPushButton:hover{{ background:#ffd1d1; }}")
        self._apply_btn_style(self.btn_search_hist, f"QPushButton{{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }} QPushButton:hover{{ background:#d4e5ff; }}")
        row_cmd = QHBoxLayout(); row_cmd.addWidget(self.btn_search_hist); row_cmd.addWidget(self.btn_export_hist); row_cmd.addWidget(self.btn_delete_hist)
        v_hist_btns.addWidget(self.btn_show_history); v_hist_btns.addLayout(row_cmd); v_hist_btns.addWidget(self.btn_hide_history)
        vside.addWidget(gb_hist_btns)
        vside.addStretch(1)
        root.addWidget(side)

        # RIGHT PANEL (STACKED WIDGET CONTAINER)
        right_container = QVBoxLayout()
        # --- Page 0: Main View (Cameras) ---
        self.main_view = QWidget(); main_layout = QVBoxLayout(self.main_view)
        top = QHBoxLayout(); self.lbl_cam1 = QLabel(); self.lbl_cam2 = QLabel()
        for lbl in (self.lbl_cam1, self.lbl_cam2):
            lbl.setScaledContents(False); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("background:#ffffff; border-radius:12px;"); lbl.setMinimumHeight(220); lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1); cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
        top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1); main_layout.addLayout(top)
        bottom = QHBoxLayout(); self.lbl_scene = QLabel(); self.lbl_roi = QLabel()
        for lbl in (self.lbl_scene, self.lbl_roi):
             lbl.setScaledContents(False); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("background:#ffffff; border-radius:12px;"); lbl.setMinimumHeight(220); lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scene_card, _ = self._make_card("Image_BOX", self.lbl_scene); roi_card, _ = self._make_card("ROI_Plate", self.lbl_roi)
        bottom.addWidget(scene_card, 1); bottom.addWidget(roi_card, 1); main_layout.addLayout(bottom)
        self.info_group = QGroupBox("Thông tin chi tiết"); info_layout = QGridLayout(self.info_group)
        self.txt_date_in = QLabel("--/--/----"); self.txt_time_in = QLabel("--:--:--"); self.txt_plate_in = QLabel("---"); self.txt_plate_in.setStyleSheet("color:#c1121f; font-weight:700")
        self.txt_date_out = QLabel("--/--/----"); self.txt_time_out = QLabel("--:--:--"); self.txt_plate_out= QLabel("---"); self.txt_plate_out.setStyleSheet("color:#c1121f; font-weight:700")
        self.txt_match = QLineEdit(); self.txt_match.setReadOnly(True); self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
        r=0; info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1); info_layout.addWidget(QLabel("Giờ vào:"), r,2); info_layout.addWidget(self.txt_time_in, r,3); info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
        info_layout.addWidget(QLabel("Ngày ra:"), r,0); info_layout.addWidget(self.txt_date_out, r,1); info_layout.addWidget(QLabel("Giờ ra:"), r,2); info_layout.addWidget(self.txt_time_out, r,3); info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
        info_layout.addWidget(QLabel("So khớp biển số:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,2); main_layout.addWidget(self.info_group)

        # --- Page 1: History View (Table) ---
        self.history_view = QWidget(); 
        hist_layout = QVBoxLayout(self.history_view)
        hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)"); 
        hist_v = QVBoxLayout(hist_group)
        self.tbl_hist = QTableWidget(0, 10); 
        self.tbl_hist.setHorizontalHeaderLabels(["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"])
        header = self.tbl_hist.horizontalHeader(); 
        hfont = QFont(header.font()); 
        hfont.setBold(True); 
        header.setFont(hfont)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Ngày vào
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Giờ vào
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Ngày ra
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)  # Giờ ra
        header.setSectionResizeMode(9, QHeaderView.ResizeToContents)  # Trạng thái
        for j in range(1, 10):  # các cột còn lại
            if header.sectionResizeMode(j) != QHeaderView.ResizeToContents:
                header.setSectionResizeMode(j, QHeaderView.Stretch)

        self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); 
        self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); 
        self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected); 
        self.tbl_hist.setAlternatingRowColors(False); 
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hist_v.addWidget(self.tbl_hist); 
        hist_layout.addWidget(hist_group)

        # --- Page 2: Detail View ---
        self.detail_view = QWidget(); detail_layout = QVBoxLayout(self.detail_view)
        row_btn_back = QHBoxLayout(); self.btn_back_to_history = QPushButton("⬅ Quay lại bảng lịch sử"); self._normalize_button(self.btn_back_to_history)
        self._apply_btn_style(self.btn_back_to_history, f"QPushButton{{ {common_btn} background:#f3f4f6; border:1px solid #e5e7eb; }} QPushButton:hover{{ background:#eef0f3; }}")
        row_btn_back.addWidget(self.btn_back_to_history); row_btn_back.addStretch(1); detail_layout.addLayout(row_btn_back)
        row_images = QHBoxLayout(); self.lbl_detail_scene = QLabel(); self.lbl_detail_roi = QLabel()
        for lbl in (self.lbl_detail_scene, self.lbl_detail_roi): lbl.setScaledContents(False); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setStyleSheet("background:#ffffff; border-radius:12px;"); lbl.setMinimumHeight(320); lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        detail_scene_card, _ = self._make_card("Ảnh Chụp Vào (Image_IN)", self.lbl_detail_scene); detail_roi_card, _ = self._make_card("Ảnh Chụp Ra (Image_OUT)", self.lbl_detail_roi)
        row_images.addWidget(detail_scene_card, 1); row_images.addWidget(detail_roi_card, 1); detail_layout.addLayout(row_images, 1)
        gb_detail_info = QGroupBox("Thông tin Lượt Gửi"); gl_detail = QGridLayout(gb_detail_info)
        self.lbl_detail_plate_in = QLineEdit(); self.lbl_detail_plate_in.setReadOnly(True); self.lbl_detail_date_in = QLineEdit(); self.lbl_detail_date_in.setReadOnly(True); self.lbl_detail_time_in = QLineEdit(); self.lbl_detail_time_in.setReadOnly(True)
        self.lbl_detail_plate_out = QLineEdit(); self.lbl_detail_plate_out.setReadOnly(True); self.lbl_detail_date_out = QLineEdit(); self.lbl_detail_date_out.setReadOnly(True); self.lbl_detail_time_out = QLineEdit(); self.lbl_detail_time_out.setReadOnly(True); self.lbl_detail_match = QLineEdit(); self.lbl_detail_match.setReadOnly(True)
        self.lbl_detail_plate_in.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: 700;"); self.lbl_detail_plate_out.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: 700;"); self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")
        gl_detail.addWidget(QLabel("Biển số vào:"), 0, 0); gl_detail.addWidget(self.lbl_detail_plate_in, 0, 1); gl_detail.addWidget(QLabel("Ngày vào:"), 1, 0); gl_detail.addWidget(self.lbl_detail_date_in, 1, 1); gl_detail.addWidget(QLabel("Giờ vào:"), 2, 0); gl_detail.addWidget(self.lbl_detail_time_in, 2, 1)
        gl_detail.addWidget(QLabel("Biển số ra:"), 0, 2); gl_detail.addWidget(self.lbl_detail_plate_out, 0, 3); gl_detail.addWidget(QLabel("Ngày ra:"), 1, 2); gl_detail.addWidget(self.lbl_detail_date_out, 1, 3); gl_detail.addWidget(QLabel("Giờ ra:"), 2, 2); gl_detail.addWidget(self.lbl_detail_time_out, 2, 3)
        gl_detail.addWidget(QLabel("Trạng thái:"), 3, 0); gl_detail.addWidget(self.lbl_detail_match, 3, 1, 1, 3); detail_layout.addWidget(gb_detail_info)

        # ==================== MỚI: TẠO TRANG TÌM KIẾM (SEARCH_FILTER_VIEW) (INDEX 3) - (UI HOÀN CHỈNH) ====================
        self.search_filter_view = QWidget()
        sfv_layout = QVBoxLayout(self.search_filter_view)
        sfv_layout.setContentsMargins(20, 20, 20, 20)
        sfv_layout.setSpacing(15)

        # 1. Tiêu đề
        sfv_title = QLabel("Bộ lọc tìm kiếm lịch sử")
        sfv_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #333;")
        sfv_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sfv_layout.addWidget(sfv_title)

        # 2. Form chứa các bộ lọc
        sfv_form = QFrame()
        sfv_form.setStyleSheet("QFrame { background: #f9f9f9; border: 1px solid #eee; border-radius: 10px; } QLabel { font-weight: 600; }") # Thêm style cho QLabel
        sfv_form_layout = QVBoxLayout(sfv_form) # Dùng QVBoxLayout
        sfv_form_layout.setContentsMargins(25, 25, 25, 25)
        sfv_form_layout.setSpacing(18) # Tăng khoảng cách dòng

        # ---- Hàng "Từ ngày/giờ" ----
        row_start = QHBoxLayout(); row_start.setSpacing(10) # Giảm khoảng cách item
        row_start.addWidget(QLabel("Từ ngày:"))
        self.sfv_date_start = QDateEdit(QDate.currentDate().addDays(-1)) # Mặc định là hôm qua
        self.sfv_date_start.setCalendarPopup(True); self.sfv_date_start.setDisplayFormat("dd/MM/yyyy"); self.sfv_date_start.setFixedHeight(34)
        row_start.addWidget(self.sfv_date_start)
        row_start.addWidget(QLabel("Giờ:"))
        self.sfv_time_start = QTimeEdit(QTime(0, 0, 0))
        self.sfv_time_start.setDisplayFormat("HH:mm:ss"); self.sfv_time_start.setFixedHeight(34)
        row_start.addWidget(self.sfv_time_start)
        row_start.addStretch(1)
        sfv_form_layout.addLayout(row_start)

        # ---- Hàng "Đến ngày/giờ" ----
        row_end = QHBoxLayout(); row_end.setSpacing(10)
        row_end.addWidget(QLabel("Đến ngày:"))
        self.sfv_date_end = QDateEdit(QDate.currentDate())
        self.sfv_date_end.setCalendarPopup(True); self.sfv_date_end.setDisplayFormat("dd/MM/yyyy"); self.sfv_date_end.setFixedHeight(34)
        row_end.addWidget(self.sfv_date_end)
        row_end.addWidget(QLabel("Giờ:"))
        self.sfv_time_end = QTimeEdit(QTime.currentTime())
        self.sfv_time_end.setDisplayFormat("HH:mm:ss"); self.sfv_time_end.setFixedHeight(34)
        row_end.addWidget(self.sfv_time_end)
        row_end.addStretch(1)
        sfv_form_layout.addLayout(row_end)

        # ---- Hàng "Trạng thái" (MỚI) ----
        row_status = QHBoxLayout(); row_status.setSpacing(15)
        row_status.addWidget(QLabel("Trạng thái:"))
        self.sfv_chk_pending = QCheckBox("Chờ xử lý (Pending)")
        self.sfv_chk_match = QCheckBox("Khớp biển số")
        self.sfv_chk_mismatch = QCheckBox("Không khớp")
        # Mặc định chọn tất cả
        self.sfv_chk_pending.setChecked(True)
        self.sfv_chk_match.setChecked(True)
        self.sfv_chk_mismatch.setChecked(True)
        row_status.addWidget(self.sfv_chk_pending)
        row_status.addWidget(self.sfv_chk_match)
        row_status.addWidget(self.sfv_chk_mismatch)
        row_status.addStretch(1)
        sfv_form_layout.addLayout(row_status)

        # ---- Hàng "Biển số" (MỚI) ----
        row_plate = QHBoxLayout(); row_plate.setSpacing(10)
        row_plate.addWidget(QLabel("Biển số (tương đối):"))
        self.sfv_txt_plate = QLineEdit()
        self.sfv_txt_plate.setPlaceholderText("Nhập một phần hoặc toàn bộ biển số...")
        self.sfv_txt_plate.setFixedHeight(34)
        row_plate.addWidget(self.sfv_txt_plate)
        sfv_form_layout.addLayout(row_plate)

        sfv_layout.addWidget(sfv_form) # Thêm form vào layout chính

        # ---- Hàng nút (Quay lại, Tìm kiếm) ----
        sfv_row_btn = QHBoxLayout()
        self.sfv_btn_back = QPushButton("Quay lại")
        self.sfv_btn_search = QPushButton("Tìm kiếm")
        self._normalize_button(self.sfv_btn_back, self.sfv_btn_search)
        self._apply_btn_style(self.sfv_btn_back, f"QPushButton{{ {common_btn} background:#f3f4f6; border:1px solid #e5e7eb; }} QPushButton:hover{{ background:#eef0f3; }}")
        self._apply_btn_style(self.sfv_btn_search, f"QPushButton{{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }} QPushButton:hover{{ background:#d4e5ff; }}")
        sfv_row_btn.addWidget(self.sfv_btn_back); sfv_row_btn.addStretch(1); sfv_row_btn.addWidget(self.sfv_btn_search)
        sfv_layout.addLayout(sfv_row_btn)
        sfv_layout.addStretch(1) # Đẩy mọi thứ lên trên
        # ==================== HẾT PHẦN SEARCH FILTER VIEW (HOÀN CHỈNH) ====================

        # --- Stacked Widget ---
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.main_view)      # index 0
        self.stacked.addWidget(self.history_view)   # index 1
        self.stacked.addWidget(self.detail_view)    # index 2
        self.stacked.addWidget(self.search_filter_view) # index 3
        self.stacked.setCurrentIndex(0)
        right_container.addWidget(self.stacked, 1)
        root.addLayout(right_container, 1)
        self.update_titles_and_modes()

        # --- Connect Signals ---
        self.btn_show_history.clicked.connect(self.on_show_all_history_clicked)
        self.btn_hide_history.clicked.connect(self.show_main_view)
        self.btn_export_hist.clicked.connect(self.on_export_excel)
        self.btn_delete_hist.clicked.connect(self.on_delete_history)
        self.btn_search_hist.clicked.connect(self.on_search_history_clicked) # Nút tìm kiếm bên trái
        self.btn_back_to_history.clicked.connect(self.show_history_view_only) # Nút quay lại từ trang detail
        self.sfv_btn_back.clicked.connect(self.show_main_view) # Nút quay lại từ trang search filter
        self.sfv_btn_search.clicked.connect(self.on_run_search_from_page) # Nút tìm kiếm trên trang search filter

    # ... (Hàm update_titles_and_modes, on_reset_lanes, on_one_way_clicked, on_two_way_clicked, update_match_status, on_play_sound, on_ocr_mode_changed giữ nguyên) ...
    def update_titles_and_modes(self):
        self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
        self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
        if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
        if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")
    @Slot()
    def on_reset_lanes(self):
        self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"; self.one_way_toggle_vao = True; self.two_way_toggle = True
        self.update_titles_and_modes(); self.show_logo(1); self.show_logo(2)
    @Slot()
    def on_one_way_clicked(self):
        if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
        else: self.lane1_dir="RA"; self.lane2_dir="RA"
        self.one_way_toggle_vao = not self.one_way_toggle_vao; self.update_titles_and_modes()
    @Slot()
    def on_two_way_clicked(self):
        if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
        else: self.lane1_dir="RA"; self.lane2_dir="VÀO"
        self.two_way_toggle = not self.two_way_toggle; self.update_titles_and_modes()
    @Slot(str)
    def update_match_status(self, status: str):
        display_status = status.replace('-', ' ').title()
        self.txt_match.setText(display_status)
        if "Khop Bien So" in display_status: self.txt_match.setStyleSheet("color: #007700; font-weight: 700;")
        elif "Khong Khop Bien So" in display_status: self.txt_match.setStyleSheet("color: #ff0000; font-weight: 700;")
        else: self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
    @Slot(str)
    def on_play_sound(self, mode):
        if mode == "in": self.sound_in.play()
        elif mode == "out": self.sound_out.play()
    @Slot()
    def on_ocr_mode_changed(self):
        self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
        if self.rb_gem.isChecked() and not GEMINI_READY:
            QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY. Sẽ dùng YOLO OCR.")
            self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
        if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
        if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

    # ---- 8.13 Hiển thị chế độ xem Lịch sử (CHỈ GỌI HÀM PHỤ) ----
    def show_history_view(self):
        """Hàm này không nên được gọi trực tiếp nữa, chỉ là dự phòng."""
        # Không reset bộ lọc ở đây
        # Chỉ chuyển tab
        self.show_history_view_only()
        # Không tải lại data ở đây

    # ---- 8.xx SỬA LẠI: Slot cho nút "Xem bảng lịch sử" ----
    @Slot()
    def on_show_all_history_clicked(self):
        """Slot này được kết nối với btn_show_history. Nó CHỈ tải lại."""
        print("\n--- DEBUG: on_show_all_history_clicked just called refresh_history_data ---\n")
        # KHÔNG xóa bộ lọc ở đây nữa
        # Chuyển tab nếu cần
        if self.stacked.currentIndex() != 1:
            self.show_history_view_only()
        # Gọi tải lại (hàm refresh_history_data sẽ tự biết xóa bộ lọc nếu cần)
        self.refresh_history_data(clear_filters=True) # Thêm cờ clear_filters

    @Slot()
    def show_history_view_only(self):
        """Hàm phụ trợ: Chỉ chuyển tab, không tải lại dữ liệu"""
        self.stacked.setCurrentIndex(1) # Chuyển về tab bảng (index 1)
        self.btn_show_history.hide()
        self.btn_hide_history.show()

    # HÀM NÀY ĐÚNG RỒI
    def show_main_view(self):
        self.stacked.setCurrentIndex(0) # Về trang chính (index 0)
        self.btn_hide_history.hide()
        self.btn_show_history.show()

    # ... (Hàm on_export_excel, on_delete_history giữ nguyên) ...
    @Slot()
    def on_export_excel(self):
        # Lấy df hiện tại (có thể đã lọc hoặc chưa)
        df_to_export = self.history_df.copy()
        if not df_to_export.empty and "STT" in df_to_export.columns:
             df_to_export = df_to_export.drop(columns=["STT"]) # Bỏ cột STT khi export
        if df_to_export.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
        path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
        if not path: return
        try: df_to_export.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
        except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

    @Slot()
    def on_delete_history(self):
        if not (self.db and self.db.ok):
            QMessageBox.warning(self, "Xóa", "Chưa kết nối DB."); return

        dlg = DeleteDialog(self)
        g = self.geometry(); dlg.adjustSize()
        dlg.move(self.mapToGlobal(QPoint(g.width()-dlg.width()-40, 140)))
        res = dlg.exec()

        ids_to_delete = []
        if res == 1:  # Xóa dòng chọn
            rows_view = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
            if not rows_view:
                QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào."); return
            for r_view in rows_view:
                id_item = self.tbl_hist.item(r_view, 0)  # cột 0 là ID
                if id_item: ids_to_delete.append(id_item.text())
            if not ids_to_delete:
                QMessageBox.warning(self, "Xóa", "Không lấy được ID các dòng chọn."); return
            self.db.delete_by_ids(ids_to_delete)

        elif res == 2:  # Xóa tất cả
            confirm = QMessageBox.question(
                self, "Xác nhận",
                "Bạn chắc chắn muốn xóa TOÀN BỘ lịch sử?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self.db.delete_all()
            else:
                return
        else:
            return

        # --- NEW: luôn quay về bảng lịch sử và dọn trang chi tiết ---
        self.clear_detail_view()               # NEW
        self.show_history_view_only()          # NEW

        # Tải lại dữ liệu với bộ lọc hiện tại
        self.refresh_history_data(
            start_time=self.current_filter_start,
            end_time=self.current_filter_end,
            status_filter=self.current_filter_status,
            plate_filter=self.current_filter_plate
        )

    def clear_detail_view(self):
        """Xóa nội dung/ảnh ở trang chi tiết và bỏ chọn các dòng trong bảng."""
        # Clear text fields
        for w in (
            self.lbl_detail_plate_in, self.lbl_detail_date_in, self.lbl_detail_time_in,
            self.lbl_detail_plate_out, self.lbl_detail_date_out, self.lbl_detail_time_out,
            self.lbl_detail_match
        ):
            w.setText("")

        # Đổi ảnh về logo mặc định
        self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())
        self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

        # Bỏ chọn các hàng trong bảng lịch sử
        self.tbl_hist.clearSelection()



    # ... (Hàm qpix_logo, show_logo giữ nguyên) ...
    def qpix_logo(self):
        if os.path.exists(LOGO_PATH): return QPixmap(LOGO_PATH)
        return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))
    def show_logo(self, which: int):
        pm = self._logo_pm
        if which == 1: self._set_centered_pixmap(self.lbl_cam1, pm)
        else: self._set_centered_pixmap(self.lbl_cam2, pm)


    # ... (Hàm on_frame, on_scene, on_roi, on_info, on_match giữ nguyên) ...
    @Slot(np.ndarray, str)
    def on_frame(self, frame_bgr, title):
        sender = self.sender()
        if sender is self.cam1_worker: self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
        elif sender is self.cam2_worker: self._set_centered_pixmap(self.lbl_cam2, frame_bgr)
    @Slot(str)
    def on_scene(self, path):
        # Dùng hàm kiểm tra đường dẫn an toàn
        valid_path = self._get_valid_image_path_internal(path)
        if valid_path: bgr = cv2.imread(valid_path); self._set_centered_pixmap(self.lbl_scene, bgr)
        else: self._set_centered_pixmap(self.lbl_scene, self.qpix_logo()) # Hiển thị logo nếu lỗi
    @Slot(str, str)
    def on_roi(self, roi_path, mode):
        valid_path = self._get_valid_image_path_internal(roi_path)
        if valid_path: bgr = cv2.imread(valid_path); self._set_centered_pixmap(self.lbl_roi, bgr)
        else: self._set_centered_pixmap(self.lbl_roi, self.qpix_logo())
    @Slot(dict)
    def on_info(self, info):
        if "date_in" in info: self.txt_date_in.setText(info["date_in"]); self.ed_date_in.setText(info["date_in"])
        if "time_in" in info: self.txt_time_in.setText(info["time_in"]); self.ed_time_in.setText(info["time_in"])
        if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
        if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
        if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
        if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])
    @Slot(str)
    def on_match(self, txt): self.txt_match.setText(txt.upper())


    # ---- 8.24 Tải và cập nhật bảng lịch sử (SỬA LẠI LOGIC XÓA FILTER) ----
    # Thêm tham số clear_filters=False
    def refresh_history_data(self, start_time=None, end_time=None, status_filter=None, plate_filter=None, clear_filters=False):
        """Khởi động luồng ngầm để tải dữ liệu lịch sử."""

        # ***** XÓA BỘ LỌC NẾU CÓ YÊU CẦU (MỚI) *****
        if clear_filters:
            print("--- Clearing filters because clear_filters=True ---")
            self.current_filter_start = None
            self.current_filter_end = None
            self.current_filter_status = None
            self.current_filter_plate = None
            # Reset các biến start_time, etc. về None để worker dùng giá trị đúng
            start_time = None
            end_time = None
            status_filter = None
            plate_filter = None

        # Nếu worker đang chạy thì không làm gì cả
        if self.history_worker and self.history_worker.isRunning():
            print("History worker is already running.")
            return

        # Tạo và chạy worker mới với các bộ lọc (đã được xóa nếu clear_filters=True)
        print(f"+++ Starting HistoryLoaderWorker with filters: Start={start_time}, End={end_time}, Status={status_filter}, Plate={plate_filter} +++")
        self.history_worker = HistoryLoaderWorker(self.db, start_time, end_time, status_filter, plate_filter, self)
        self.history_worker.resultReady.connect(self.update_history_table)
        self.history_worker.finished.connect(self.history_worker.deleteLater)
        self.history_worker.start()


    # ---- 8.xx MỚI: Slot trung gian cho Timer/Worker (KHÔNG XÓA FILTER) ----
    @Slot()
    def on_history_signal_refresh(self):
        """Refresh bảng lịch sử chỉ khi đang ở tab lịch sử và không quá dày"""
        if self.stacked.currentIndex() != 1:
            return
        now = time.time()
        if now - self._hist_last_reload < 5.0:   # không reload quá 1 lần / 5 giây
            return
        self._hist_last_reload = now
        self.refresh_history_data(start_time=self.current_filter_start,
                                end_time=self.current_filter_end,
                                status_filter=self.current_filter_status,
                                plate_filter=self.current_filter_plate)


    # ---- 8.xx MỚI: Xử lý sự kiện nhấn 'Tìm kiếm' TỪ TRANG LỌC (INDEX 3) - (KHÔNG XÓA FILTER) ----
    @Slot()
    def on_run_search_from_page(self):
        # ... (Lấy start_dt, end_dt, selected_statuses, plate_text như cũ) ...
        print(">>> Entering on_run_search_from_page")
        qdate_start = self.sfv_date_start.date(); qtime_start = self.sfv_time_start.time(); qdate_end = self.sfv_date_end.date(); qtime_end = self.sfv_time_end.time(); start_dt = QDateTime(qdate_start, qtime_start).toPython(); end_dt = QDateTime(qdate_end, qtime_end).toPython()
        if start_dt > end_dt: QMessageBox.warning(self, "Lỗi nhập liệu", "'Từ ngày/giờ' không được lớn hơn 'Đến ngày/giờ'.\nVui lòng kiểm tra lại."); print("<<< Exiting on_run_search_from_page (Date Error)"); return
        selected_statuses = []; plate_text = self.sfv_txt_plate.text().strip()
        if self.sfv_chk_pending.isChecked(): selected_statuses.append("PENDING");
        if self.sfv_chk_match.isChecked(): selected_statuses.append("KHOP-BIEN-SO")
        if self.sfv_chk_mismatch.isChecked(): selected_statuses.append("KHONG-KHOP-BIEN-SO")

        # 5. LƯU LẠI BỘ LỌC HIỆN TẠI
        self.current_filter_start = start_dt
        self.current_filter_end = end_dt
        self.current_filter_status = selected_statuses if selected_statuses else None
        self.current_filter_plate = plate_text if plate_text else None
        print(">>> Filters JUST SET in on_run_search:"); # ... (print filters) ...

        # 6. Gọi hàm tải dữ liệu VỚI bộ lọc, KHÔNG clear_filters
        print(">>> Calling refresh_history_data...")
        self.refresh_history_data(start_time=self.current_filter_start,
                                end_time=self.current_filter_end,
                                status_filter=self.current_filter_status,
                                plate_filter=self.current_filter_plate) # Bỏ clear_filters=True
        print(">>> Returned from refresh_history_data.")
        # ... (print filters before switch, setCurrentIndex) ...
        print("<<< Exiting on_run_search_from_page (Success)")
        self.show_history_view_only()   # chuyển sang tab bảng (index 1)

    # ---- 8.xx MỚI: Slot nhận kết quả DataFrame từ Worker ----
    @Slot(pd.DataFrame)
    def update_history_table(self, df: pd.DataFrame):
        """Cập nhật QTableWidget với DataFrame nhận được (nhẹ, không block UI)."""
        print(f"+++ update_history_table received {len(df)} rows +++")

        # 1) Lưu df gốc (có STT để tra chi tiết)
        self.history_df = df.copy()

        # 2) Chuẩn bị df hiển thị (bỏ STT nếu có)
        df_display = df.drop(columns=["STT"], errors="ignore")

        # 3) Tắt redraw & sort để đổ nhanh
        self.tbl_hist.setUpdatesEnabled(False)
        self.tbl_hist.setSortingEnabled(False)

        # 4) Cập nhật cấu trúc bảng
        cols = list(df_display.columns)
        self.tbl_hist.clearContents()
        self.tbl_hist.setColumnCount(len(cols))
        self.tbl_hist.setHorizontalHeaderLabels(cols)
        self.tbl_hist.setSortingEnabled(False)
        self.tbl_hist.setRowCount(len(df_display))

        # 5) Điền dữ liệu
        for i in range(len(df_display)):
            for j, col in enumerate(cols):
                if j < self.tbl_hist.columnCount():
                    val = df_display.iloc[i, j]
                    item = QTableWidgetItem()
                    # Nếu là cột ID (cột 0), dùng DisplayRole=int để Qt hiểu là số:
                    if j == 0:
                        try:
                            item.setData(Qt.ItemDataRole.DisplayRole, int(val))
                        except:
                            item.setText(str(val))
                    else:
                        item.setText(str(val))
                    item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                    self.tbl_hist.setItem(i, j, item)
        self.tbl_hist.setSortingEnabled(True)
        self.tbl_hist.sortByColumn(0, Qt.SortOrder.DescendingOrder)  # 0 = cột ID

        # 6) Bật lại sort & redraw
        self.tbl_hist.setSortingEnabled(True)
        self.tbl_hist.setUpdatesEnabled(True)

        # 7) giải phóng tham chiếu worker
        self.history_worker = None
        print("--- History worker reference released ---")


    # HÀM NÀY ĐÚNG RỒI
    @Slot()
    def on_search_history_clicked(self):
        """Mở trang bộ lọc tìm kiếm (index 3)"""
        self.stacked.setCurrentIndex(3)
        self.btn_show_history.hide(); self.btn_hide_history.show()

    # THÊM HÀM HELPER KIỂM TRA ĐƯỜNG DẪN ẢNH
    def _get_valid_image_path_internal(self, path_from_db):
        if not path_from_db: return None
        # Ưu tiên kiểm tra tuyệt đối trước
        if os.path.exists(path_from_db): return path_from_db
        # Thử ghép tương đối
        maybe_path = os.path.abspath(path_from_db)
        if os.path.exists(maybe_path): return maybe_path
        print(f"Cảnh báo: Không tìm thấy ảnh tại '{path_from_db}' hoặc '{maybe_path}'")
        return None

    # HÀM NÀY ĐÚNG RỒI (Đã sửa lỗi đường dẫn)
    @Slot()
    def on_history_row_selected(self):
        selected_items = self.tbl_hist.selectedItems()
        if not selected_items or self.history_df.empty or "ID" not in self.history_df.columns: return

        try:
            row_index_view = selected_items[0].row()
            id_item = self.tbl_hist.item(row_index_view, 0) # Cột 0 là ID
            if not id_item: return
            row_id = int(id_item.text())

            row_data_series = self.history_df[self.history_df['ID'] == row_id]
            if row_data_series.empty: return
            row_data = row_data_series.iloc[0]

            # Cập nhật thông tin trang chi tiết
            self.lbl_detail_plate_in.setText(str(row_data.get("Biển số vào", "")))
            self.lbl_detail_date_in.setText(str(row_data.get("Ngày vào", "")))
            self.lbl_detail_time_in.setText(str(row_data.get("Giờ vào", "")))
            self.lbl_detail_plate_out.setText(str(row_data.get("Biển số ra", "")))
            self.lbl_detail_date_out.setText(str(row_data.get("Ngày ra", "")))
            self.lbl_detail_time_out.setText(str(row_data.get("Giờ ra", "")))
            match_status = str(row_data.get("Trạng thái", "")).replace('-', ' ').title()
            self.lbl_detail_match.setText(match_status)
            if "Khop Bien So" in match_status: self.lbl_detail_match.setStyleSheet("color: #007700; font-weight: 700;")
            elif "Khong Khop Bien So" in match_status: self.lbl_detail_match.setStyleSheet("color: #ff0000; font-weight: 700;")
            else: self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")

            # Cập nhật ảnh (dùng hàm helper)
            valid_in_path = self._get_valid_image_path_internal(str(row_data.get("Ảnh vào", "")))
            valid_out_path = self._get_valid_image_path_internal(str(row_data.get("Ảnh ra", "")))

            if valid_in_path: self._set_centered_pixmap(self.lbl_detail_scene, cv2.imread(valid_in_path))
            else: self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())
            if valid_out_path: self._set_centered_pixmap(self.lbl_detail_roi, cv2.imread(valid_out_path))
            else: self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())

            # Chuyển sang trang chi tiết (index 2)
            self.stacked.setCurrentIndex(2)

        except Exception as e:
            print(f"Lỗi khi chọn hàng: {e}"); import traceback; traceback.print_exc()


    # ... (Hàm _connect_worker, start/stop_cam_generic, start/stop_cam1/2, closeEvent giữ nguyên) ...
    def _connect_worker(self, w: VideoWorker):
        w.frameSignal.connect(self.on_frame)
        w.sceneSignal.connect(self.on_scene)
        w.roiSignal.connect(self.on_roi)
        w.infoSignal.connect(self.on_info)
        w.matchSignal.connect(self.on_match)
        w.histSignal.connect(self.on_history_signal_refresh)
        w.playSoundSignal.connect(self.on_play_sound)

    def start_cam_generic(self, which: int):
        if not self.models.ok: QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}"); return
        if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
        if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return
        ocr_mode = self.current_ocr_mode; default_api = API_MAP["DSHOW(Windows)"]
        if which == 1:
            idx = int(self.spin_cam1.value()); mode = "in" if self.lane1_dir=="VÀO" else "out"; title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
            self.cam1_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
            self._connect_worker(self.cam1_worker); self.cam1_worker.start()
        else:
            idx = int(self.spin_cam2.value()); mode = "in" if self.lane2_dir=="VÀO" else "out"; title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
            self.cam2_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
            self._connect_worker(self.cam2_worker); self.cam2_worker.start()

    def stop_cam_generic(self, which: int):
        worker = self.cam1_worker if which==1 else self.cam2_worker
        if worker and worker.isRunning(): worker.stop(); worker.wait(1000)
        if which==1: self.cam1_worker = None; self.show_logo(1)
        else: self.cam2_worker = None; self.show_logo(2)

    def start_cam1(self): self.start_cam_generic(1)
    def stop_cam1(self): self.stop_cam_generic(1)
    def start_cam2(self): self.start_cam_generic(2)
    def stop_cam2(self): self.stop_cam_generic(2)

    def closeEvent(self, event):
        try: self.stop_cam_generic(1); self.stop_cam_generic(2)
        except: pass
        super().closeEvent(event)

# ==================== 9. MAIN ====================
def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()




