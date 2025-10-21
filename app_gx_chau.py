# # -*- coding: utf-8 -*-
# """
#         =========================================================
#         = PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini) =
#         =========================================================

# 1. Phát hiện & OCR (YOLOv8, OpenCV, Gemini AI): 	          
#     Tự động phát hiện vị trí biển số (YOLO Detect) và trích xuất ký tự 
#     (YOLO OCR hoặc Gemini AI). Xử lý tiền DL ảnh (CLAHE, Blur) để tăng độ chính xác OCR.

# 2. Giao diện - UI [PySide6 (QMainWindow, QThread, Signal/Slot)]:
#     Xây dựng giao diện Desktop, hiển thị video trực tiếp, kết quả OCR, và kết nối các luồng 
#     xử lý video (Worker) với giao diện chính.

# 3. Quản lý Dữ liệu	[SQL Server (qua pyodbc), pandas]:	
#     Lưu trữ lịch sử giao dịch xe vào/ra (ParkingSessions). Tải dữ liệu lịch sử vào 
#     DataFrame để hiển thị trên bảng UI và Export Excel.

# 4. Luồng Video (QThread, cv2.VideoCapture):
# 	Chạy độc lập cho hai làn xe (VÀO/RA). Chụp và xử lý ảnh khi biển số ổn định (ít nhất 1.2s), 
#     sau đó gửi kết quả (ảnh, biển số, thời gian) về UI.

# 5. Logic Giữ Xe	(Hàm attach_out trong Class DB):
# 	Tự động so khớp biển số xe ra với các xe vào đang chờ (plate_out IS NULL). Cập nhật 
#     trạng thái KHOP-BIEN-SO (Xanh) hoặc KHONG-KHOP-BIEN-SO (Đỏ) trong DB và trên UI.

# 6. Cấu hình	(.env, QSpinBox, QRadioButton):
# 	Cho phép người dùng chọn Index Camera, cấu hình chế độ làn xe (1 chiều/2 chiều) và 
#     lựa chọn Model OCR (YOLO hoặc Gemini).

# 7. Xây dựng UI (_build_ui()):
# 	Hàm xây dựng bố cục chính, tạo các widgets như nút Bật/Tắt Cam, Điều khiển Làn, 
#     các ô hiển thị thông tin xe VÀO/RA, và Bảng Lịch Sử.

# 8. Điều khiển Cam (start_cam_generic/stop_cam_generic):	
#     Khởi động/Dừng luồng xử lý video (VideoWorker). Thiết lập chế độ Vào/Ra và 
#     chế độ OCR cho Worker trước khi chạy.

# 9. Điều khiển Làn (on_one_way_clicked/on_two_way_clicked):
# 	Quản lý hướng làn xe (Vào, Ra). Cho phép chuyển đổi giữa chế độ 
#     Một chiều (Cam 1 & 2 cùng hướng) và Hai chiều (Cam 1 & 2 ngược hướng).

# 10. Chọn OCR Model (on_ocr_mode_changed):
# 	Cho phép người dùng chọn Model OCR. Kiểm tra nếu thiếu API Key Gemini 
#     thì buộc chuyển về YOLO và thông báo.

# 11. Cập nhật Real-time (on_frame/on_info/v.v):
# 	Các hàm @Slot nhận tín hiệu (Signal) từ luồng VideoWorker (ảnh, biển số, thời gian) và 
#     cập nhật tức thời lên các ô hiển thị trên giao diện chính.

# 12. Quản lý Bảng (show_history_view/refresh_history):
# 	Chuyển đổi giữa chế độ xem Camera chính và Bảng Lịch sử. Tải và hiển thị dữ liệu 
#     giao dịch từ SQL lên bảng QTableWidget.

# 13. Thao tác DB (on_export_excel/on_delete_history):
# 	Xử lý các thao tác quản lý dữ liệu: Xuất dữ liệu lịch sử ra Excel và Xóa các 
#     dòng giao dịch đã chọn trong cơ sở dữ liệu. 

# """

# # ==================== 1. IMPORT ====================

# import os, sys, time, cv2, numpy as np, pandas as pd
# from datetime import datetime

# # ---- 1.1 HiDPI Cấu hình HiDPI (Độ phân giải cao) ----
# os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
# os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# # ---- 1.2 Import PySide6 ----
# from PySide6.QtCore import QDateTime
# from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint, QUrl
# from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
# from PySide6.QtMultimedia import QSoundEffect
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
#     QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
#     QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
#     QFileDialog, QHeaderView, QDialog, QDateTimeEdit
# )

# # ---- 1.3 Optional SQL ----
# USE_SQL = True
# try:
#     import pyodbc
# except Exception:
#     USE_SQL = False

# # ---- 1.4 YOLO ----
# from ultralytics import YOLO

# # ---- 1.5 Gemini API (optional) ----
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

# # ---- 2.1 Đường dẫn Model ----
# DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
# OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
# SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
# LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")
# SOUND_IN_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_vao_xin_cam_on.wav"
# SOUND_OUT_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_ra_xin_cam_on.wav"

# # ---- 2.2 SQL ----
# CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=plates_db;"
#     "UID=sa;"
#     "PWD=123456"
# )

# # ---- 2.3 UI ----
# PANEL_W, PANEL_H = 640, 360
# PANEL_BG = (255, 255, 255)

# API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
# OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
#            "six":"6","seven":"7","eight":"8","nine":"9"}





# # ==================== 3. UTILITIES (HÀM TIỆN ÍCH) ====================

# # ---- 3.1 Căn chỉnh/Điền nền ----
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

# # ---- 3.2 Chuyển đổi ảnh ----
# def bgr_to_qimage(bgr):
#     if bgr is None:
#         bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     h, w, ch = rgb.shape
#     return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

# # ---- 3.3 Lưu ảnh ----
# def save_image(img, prefix):
#     ts = datetime.now().strftime("%Ym%d_%H%M%S_%f")
#     path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
#     cv2.imwrite(path, img)
#     return path

# # ---- 3.4 OCR ----
# def norm_char(x):  # Chuẩn hóa ký tự
#     return OCR_MAP.get(str(x), str(x))

# def plate_norm(s: str) -> str: # Chuẩn hóa biển số
#     return (s or "").replace("-", "").replace(" ", "").upper()

# def has_boxes(r):  # Kiểm tra có box
#     try:
#         return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
#     except: return False

# def preprocess_for_ocr(roi):  # Tiền xử lý ảnh OCR
#     if roi is None: return None
#     if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe,(3,3),0)
#     return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)





# # ==================== 4. DB LAYER ====================

# class DB:
#     # ---- 4.1 Khởi tạo và Kết nối ----
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

#     # ---- 4.2 Ghi nhận xe VÀO ----
#     def insert_in(self, plate, d, t, img_path):
#         if not self.ok: return
#         try:
#             self.cur.execute("""
#                 INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
#                 VALUES (?,?,?,?,?)
#             """, (plate, d, t, img_path, 'PENDING'))
#         except Exception as e: print("insert_in error:", e)

#     # ---- 4.3 Ghi nhận xe RA và Ghép đôi ----
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

#     # # ---- 4.4 Lấy lịch sử ----
#     # def fetch_history_df(self, limit=10000) -> pd.DataFrame:
#     #     if not self.ok:
#     #         return pd.DataFrame(columns=[
#     #             "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#     #             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#     #         ])
#     #     try:
#     #         rows = self.cur.execute(f"""
#     #             SELECT TOP {limit}
#     #                 id, image_in, plate_in, date_in, time_in,
#     #                 image_out, plate_out, date_out, time_out, match_status
#     #             FROM dbo.ParkingSessions
#     #             ORDER BY id DESC
#     #         """).fetchall()
#     #         df = pd.DataFrame.from_records(
#     #             rows,
#     #             columns=["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#     #                      "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#     #         ).astype(object).where(pd.notnull, "")
#     #         df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#     #         df.insert(0, "STT", range(1, len(df)+1))
#     #         return df
#     #     except Exception as e:
#     #         print("fetch_history error:", e)
#     #         return pd.DataFrame(columns=[
#     #             "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#     #             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#     #         ])

#     # ---- 4.4 Lấy lịch sử (ĐÃ CẬP NHẬT) ----
#     def fetch_history_df(self, limit=10000, start_time=None, end_time=None) -> pd.DataFrame:
#         columns = [
#             "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#         ]
#         if not self.ok:
#             # Sửa cột: thêm STT để khớp với logic mới
#             return pd.DataFrame(columns=["STT"] + columns) 
        
#         try:
#             # Xây dựng câu lệnh SQL động
#             sql = f"""
#                 SELECT TOP ({limit})
#                     id, image_in, plate_in, date_in, time_in,
#                     image_out, plate_out, date_out, time_out, match_status
#                 FROM dbo.ParkingSessions
#             """
            
#             where_clauses = []
#             sql_params = [] # Tham số cho WHERE

#             if start_time:
#                 # Dùng created_at để lọc vì đây là cột DATETIME
#                 where_clauses.append("created_at >= ?") 
#                 sql_params.append(start_time)
            
#             if end_time:
#                 where_clauses.append("created_at <= ?")
#                 sql_params.append(end_time)
            
#             if where_clauses:
#                 sql += " WHERE " + " AND ".join(where_clauses)
            
#             sql += " ORDER BY id DESC"
            
#             rows = self.cur.execute(sql, tuple(sql_params)).fetchall()
            
#             df = pd.DataFrame.from_records(
#                 rows,
#                 columns=columns
#             ).astype(object).where(pd.notnull, "")
            
#             df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#             df.insert(0, "STT", range(1, len(df)+1))
#             return df
        
#         except Exception as e:
#             print("fetch_history error:", e)
#             return pd.DataFrame(columns=["STT"] + columns)

#     # ---- 4.5 Xóa theo ID ----
#     def delete_by_ids(self, ids):
#         if not self.ok or not ids: return
#         try:
#             for sid in ids:
#                 self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
#         except Exception as e: print("delete_by_ids error:", e)

#     # ---- 4.6 Xóa tất cả ----
#     def delete_all(self):
#         if not self.ok: return
#         try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
#         except Exception as e: print("delete_all error:", e)





# # ==================== 5. YOLO/GEMINI WRAPPERS (TRÌNH BAO BỌC MODEL) ====================
# class Models:
#     # ---- 5.1 Khởi tạo (Tải model) ----
#     def __init__(self, det_path: str, ocr_path: str):
#         self.ok = True; self.err = ""
#         try:
#             self.det = YOLO(det_path)
#             self.ocr = YOLO(ocr_path)
#         except Exception as e:
#             self.ok = False; self.err = str(e)

#     # ---- 5.2 YOLO phát hiện biển số ----
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

#     # ---- 5.3 OCR biển số bằng YOLO ----
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

#     # ---- 5.4 OCR biển số bằng Gemini AI ----
#     def ocr_plate_gemini_from_path(self, image_path: str):
#         if not GEMINI_READY: return "", ""
#         try:
#             img = Image.open(image_path)
#         except Exception as e:
#             print("Gemini open image error:", e); return "", ""
#         try:
#             model = genai.GenerativeModel('gemini-1.5-flash') # Dùng 1.5-flash
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

#     # ---- 5.5 Hỗ trợ (Hàm tĩnh định dạng) ----
#     @staticmethod
#     def _format_text(text_raw: str):
#         raw=(text_raw or '').replace('-', ' ').replace(' ', '')
#         text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
#         return text_fmt, (text_raw or "")





# # ==================== 6. VIDEO WORKER (Luồng xử lý Video) ====================
# class VideoWorker(QThread):
#     frameSignal = Signal(np.ndarray, str)
#     sceneSignal = Signal(str)
#     roiSignal   = Signal(str, str)
#     infoSignal  = Signal(dict)
#     matchSignal = Signal(str)
#     histSignal  = Signal()
#     playSoundSignal = Signal(str)

#     # ---- 6.1 Khởi tạo ----
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

#     # ---- 6.2 Setter/Getter ----
#     def set_title(self, title: str): 
#         self.title = title
#     def set_ocr_mode(self, mode: str): 
#         self.ocr_mode = mode
#     def set_mode(self, mode: str): 
#         self.mode = mode

#     # ---- 6.3 Vòng lặp chính của luồng ----
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

#             self.frameSignal.emit(frame, self.title)

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
#                         self.playSoundSignal.emit("in")
#                     else:
#                         self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
#                         if self.db and self.db.ok:
#                             match = self.db.attach_out(plate, d, t, scene_path)
#                             self.matchSignal.emit(match)
#                             self.histSignal.emit()
#                         self.playSoundSignal.emit("out")
#                     self.captured = True

#             time.sleep(0.01)

#         try:
#             if self.cap: self.cap.release()
#         except: pass

#     # ---- 6.4 Dừng luồng ----
#     def stop(self): 
#         self._running = False





# # ==================== 7. DELETE DIALOG (Hộp thoại Xóa) ====================

# class DeleteDialog(QDialog):
#     # ---- 7.1 Khởi tạo Giao diện ----
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
#         """)
#         lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
#         lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
#         lay.addWidget(lab)

#         row = QHBoxLayout(); row.setSpacing(12)
#         self.btn_sel = QPushButton("Xóa dòng đã chọn")
#         self.btn_all = QPushButton("Xóa tất cả")
#         self.btn_can = QPushButton("Hủy")
#         row.addWidget(self.btn_sel, 1)
#         row.addWidget(self.btn_all, 1)
#         row.addWidget(self.btn_can, 1)
#         lay.addLayout(row)

#         base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
#         self.btn_sel.setStyleSheet(f"""
#         QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         QPushButton:pressed{{ background:#c8deff; }}
#         """)
#         self.btn_all.setStyleSheet(f"""
#         QPushButton {{ {base} background:#ffe0e0; border:1px solid #ffb3b3; }}
#         QPushButton:hover  {{ background:#ffd1d1; }}
#         QPushButton:pressed{{ background:#ffc2c2; }}
#         """)
#         self.btn_can.setStyleSheet(f"""
#         QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}
#         QPushButton:hover  {{ background:#eef0f3; }}
#         QPushButton:pressed{{ background:#e7e9ed; }}
#         """)

#         self.btn_sel.clicked.connect(lambda: self.done(1))
#         self.btn_all.clicked.connect(lambda: self.done(2))
#         self.btn_can.clicked.connect(lambda: self.done(0))





# # ==================== 7.5 SEARCH DIALOG (HỘP THOẠI TÌM KIẾM) - MỚI ====================
# class SearchDialog(QDialog):
#     """Hộp thoại cho phép chọn khoảng thời gian tìm kiếm."""
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Tìm kiếm theo thời gian")
#         self.setModal(True)
#         self.setStyleSheet("""
#             QDialog { background: #ffffff; border-radius: 10px; }
#             QLabel { font-weight: 600; }
#         """)
        
#         lay = QGridLayout(self)
#         lay.setContentsMargins(16, 16, 16, 16)
#         lay.setSpacing(12)

#         lay.addWidget(QLabel("Từ ngày:"), 0, 0)
#         self.dt_start = QDateTimeEdit(self)
#         self.dt_start.setDateTime(datetime.now().replace(hour=0, minute=0, second=0))
#         self.dt_start.setCalendarPopup(True)
#         self.dt_start.setDisplayFormat("dd/MM/yyyy HH:mm:ss")
#         lay.addWidget(self.dt_start, 0, 1)

#         lay.addWidget(QLabel("Đến ngày:"), 1, 0)
#         self.dt_end = QDateTimeEdit(self)
#         self.dt_end.setDateTime(datetime.now())
#         self.dt_end.setCalendarPopup(True)
#         self.dt_end.setDisplayFormat("dd/MM/yyyy HH:mm:ss")
#         lay.addWidget(self.dt_end, 1, 1)

#         row_btn = QHBoxLayout()
#         self.btn_search = QPushButton("Tìm kiếm")
#         self.btn_cancel = QPushButton("Hủy")
#         row_btn.addStretch(1)
#         row_btn.addWidget(self.btn_cancel)
#         row_btn.addWidget(self.btn_search)
        
#         # Áp style cho nút (tùy chọn)
#         base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
#         self.btn_search.setStyleSheet(f"QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}")
#         self.btn_cancel.setStyleSheet(f"QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}")

#         lay.addLayout(row_btn, 2, 0, 1, 2)

#         self.btn_search.clicked.connect(self.accept) # self.accept() sẽ đóng dialog và trả về 1
#         self.btn_cancel.clicked.connect(self.reject) # self.reject() sẽ đóng dialog và trả về 0

#     def get_date_range(self):
#         """Trả về (start_datetime, end_datetime)"""
#         return self.dt_start.dateTime().toPython(), self.dt_end.dateTime().toPython()

#     @staticmethod
#     def get_range(parent=None):
#         """Hàm helper để gọi dialog và lấy kết quả."""
#         dialog = SearchDialog(parent)
#         if dialog.exec() == QDialog.DialogCode.Accepted:
#             return dialog.get_date_range()
#         return None, None





# # ==================== 8. MAIN WINDOW (CỬA SỔ CHÍNH ====================

# class MainWindow(QMainWindow):
#     # ---- 8.1 Khởi tạo Giao diện ứng dụng ----
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Desktop App (Giữ xe)")
#         self.setMinimumSize(1200, 800)
#         self._init_theme()

#         self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#         self.db = DB(CONN_STR) if USE_SQL else DB("")

#         # Khởi tạo âm thanh
#         self.sound_in = QSoundEffect(self)
#         sound_in_abs = os.path.abspath(SOUND_IN_PATH)
#         if os.path.exists(sound_in_abs):
#             self.sound_in.setSource(QUrl.fromLocalFile(sound_in_abs))
#         else:
#             print(f"Lỗi: Không tìm thấy file âm thanh: {sound_in_abs}")

#         self.sound_out = QSoundEffect(self)
#         sound_out_abs = os.path.abspath(SOUND_OUT_PATH)
#         if os.path.exists(sound_out_abs):
#             self.sound_out.setSource(QUrl.fromLocalFile(sound_out_abs))
#         else:
#             print(f"Lỗi: Không tìm thấy file âm thanh: {sound_out_abs}")

#         self.cam1_worker = None
#         self.cam2_worker = None

#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.current_ocr_mode = "yolo"
#         self.history_df = pd.DataFrame()

#         # Lưu logo gốc để scale lại đúng ở mọi lần vẽ
#         self._logo_pm = self.qpix_logo()
#         self._build_ui()
#         self.show_logo(1); self.show_logo(2)
#         self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.on_history_signal_refresh); self.hist_timer.start(5000)

#     # ---- 8.2 Thiết lập Giao diện ----
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
#         QFrame[class="card"]       { background: #ffffff; border-radius: 12px; }
#         QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; }
#         QLabel[class="title"] {
#             font: 700 18px "Segoe UI";
#             padding: 6px 10px;
#             background: #ffffff;
#             border-radius: 10px;
#         }

#         QLineEdit {
#             height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px;
#         }
#         QTableWidget { background: #ffffff; gridline-color: #e6e6e6; }
#         """)

#     # ---- 8.3 Chuẩn hóa hành vi của các nút ----
#     def _normalize_button(self, *btns):
#         for b in btns:
#             b.setAutoDefault(False); b.setDefault(False); b.setFlat(False); b.setFocusPolicy(Qt.NoFocus)
#             # FIX: để nút không kéo giãn vô hạn khi phóng to
#             b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

#     # ---- 8.4 Sửa lỗi bo tròn: Đơn giản hóa hàm ----
#     def _apply_btn_style(self, btn: QPushButton, css: str):
#         btn.setStyleSheet(css)

#     # ---- 8.5 Tạo khung hiển thị (Card UI) ----
#     def _make_card(self, title:str, content:QWidget):
#         wrap = QFrame(); wrap.setProperty("class","card-wrap")
#         wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
#         card = QFrame(); card.setProperty("class","card")
#         v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
#         title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
#         hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
#         title_lbl = QLabel(title); title_lbl.setProperty("class","title")
#         hl.addWidget(title_lbl)
#         v.addWidget(title_wrap); v.addWidget(content, 1)
#         wrapL.addWidget(card)
#         return wrap, title_lbl

#     # ---- 8.6 Hiển thị ảnh căn giữa và giữ tỷ lệ ----
#     def _set_centered_pixmap(self, lbl: QLabel, src):
#         if isinstance(src, np.ndarray):
#             pm = QPixmap.fromImage(bgr_to_qimage(src))
#         elif isinstance(src, QImage):
#             pm = QPixmap.fromImage(src)
#         else:
#             pm = src
#         if pm is None or pm.isNull():
#             lbl.clear(); return
#         rect = lbl.contentsRect()
#         avail = rect.size()
#         dpr = lbl.devicePixelRatioF() if hasattr(lbl, "devicePixelRatioF") else 1.0
#         target_w = max(1, int(avail.width()  * dpr))
#         target_h = max(1, int(avail.height() * dpr))
#         scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         if hasattr(scaled, "setDevicePixelRatio"):
#             scaled.setDevicePixelRatio(dpr)
#         lbl.setAlignment(Qt.AlignCenter)
#         lbl.setPixmap(scaled)

#     # ---- 8.7 Xây dựng cấu trúc giao diện chính (Widgets-UI) ----
#     def _build_ui(self):
#         central = QWidget(); self.setCentralWidget(central)
#         root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

#         # LEFT
#         side = QWidget(objectName="SideBar")
#         side.setFixedWidth(450)
#         vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

#         # CAMERA CONTROL
#         gb_camctl = QGroupBox("CAMERA CONTROL")
#         vl_camctl = QVBoxLayout(gb_camctl)
#         vl_camctl.setSpacing(10) # Khoảng cách giữa các hàng

#         self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
#         self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
#         self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys())); self.cb_api1.hide() # Ẩn API combo box
#         self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys())); self.cb_api2.hide() # Ẩn API combo box

#         # Hàng 1: Index Cam 1 & Index Cam 2
#         row_indices = QHBoxLayout()
#         row_indices.setSpacing(10)
#         row_indices.addWidget(QLabel("Index Cam 1"))
#         row_indices.addWidget(self.spin_cam1, 1)
#         row_indices.addWidget(QLabel("Index Cam 2"))
#         row_indices.addWidget(self.spin_cam2, 1)
#         vl_camctl.addLayout(row_indices)

#         # Buttons
#         self.btn_start1 = QPushButton("Bật Cam 1")
#         self.btn_stop1  = QPushButton("Tắt Cam 1")
#         self.btn_start2 = QPushButton("Bật Cam 2")
#         self.btn_stop2  = QPushButton("Tắt Cam 2")
#         self._normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)

#         common_btn = "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;"

#         self._apply_btn_style(self.btn_start1, f"""
#         QPushButton {{ {common_btn} background:#d1fadf; border:1px solid #a6f4c5; }}
#         QPushButton:hover  {{ background:#c3f7d6; }}
#         QPushButton:pressed{{ background:#b4f3cc; }}
#         QPushButton:disabled{{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }}
#         """)
#         self._apply_btn_style(self.btn_stop1, f"""
#         QPushButton {{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }}
#         QPushButton:hover  {{ background:#ffd1d1; }}
#         QPushButton:pressed{{ background:#ffc2c2; }}
#         QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}
#         """)
#         self._apply_btn_style(self.btn_start2, self.btn_start1.styleSheet())
#         self._apply_btn_style(self.btn_stop2,  self.btn_stop1.styleSheet())

#         # signals
#         self.btn_start1.clicked.connect(self.start_cam1)
#         self.btn_stop1.clicked.connect(self.stop_cam1)
#         self.btn_start2.clicked.connect(self.start_cam2)
#         self.btn_stop2.clicked.connect(self.stop_cam2)

#         # Hàng 2: Bật/Tắt Cam 1
#         row_btn1 = QHBoxLayout()
#         row_btn1.setSpacing(12)
#         row_btn1.addWidget(self.btn_start1)
#         row_btn1.addWidget(self.btn_stop1)
#         vl_camctl.addLayout(row_btn1)

#         # Hàng 3: Bật/Tắt Cam 2
#         row_btn2 = QHBoxLayout()
#         row_btn2.setSpacing(12)
#         row_btn2.addWidget(self.btn_start2)
#         row_btn2.addWidget(self.btn_stop2)
#         vl_camctl.addLayout(row_btn2)

#         vside.addWidget(gb_camctl)

#         # ĐIỀU KHIỂN LÀN
#         gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
#         vl_lane = QVBoxLayout(gb_lane); vl_lane.setSpacing(10)
#         row_lane = QHBoxLayout(); row_lane.setSpacing(12)

#         self.btn_oneway = QPushButton("1 chiều")
#         self.btn_twoway = QPushButton("2 chiều")
#         self.btn_reset_lane = QPushButton("Reset làn")
#         self._normalize_button(self.btn_oneway, self.btn_twoway, self.btn_reset_lane)

#         self._apply_btn_style(self.btn_oneway, f"""
#         QPushButton {{ {common_btn} background:#dbeafe; border:1px solid #bfdbfe; }}
#         QPushButton:hover  {{ background:#cfe3fd; }}
#         QPushButton:pressed{{ background:#c3dcfc; }}
#         QPushButton:disabled{{ background:#eef6ff; color:#777; border-color:#e3efff; }}
#         """)
#         self._apply_btn_style(self.btn_twoway, self.btn_oneway.styleSheet())
#         self._apply_btn_style(self.btn_reset_lane, f"""
#         QPushButton {{ {common_btn} background:#fff3bf; border:1px solid #ffe066; }}
#         QPushButton:hover  {{ background:#ffeda3; }}
#         QPushButton:pressed{{ background:#ffe788; }}
#         QPushButton:disabled{{ background:#fff9dc; color:#777; border-color:#ffefb3; }}
#         """)

#         row_lane.addWidget(self.btn_oneway)
#         row_lane.addWidget(self.btn_twoway)
#         vl_lane.addLayout(row_lane)
#         vl_lane.addWidget(self.btn_reset_lane)

#         self.btn_oneway.clicked.connect(self.on_one_way_clicked)
#         self.btn_twoway.clicked.connect(self.on_two_way_clicked)
#         self.btn_reset_lane.clicked.connect(self.on_reset_lanes)
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
#             self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY (.env hoặc biến môi trường) → dùng YOLO")
#         vside.addWidget(gb_ocr)

#         # THÔNG TIN XE VÀO
#         gb_in = QGroupBox("THÔNG TIN XE VÀO")
#         gl_in = QGridLayout(gb_in)
#         self.ed_date_in = QLineEdit(); 
#         self.ed_time_in = QLineEdit(); 
#         self.ed_plate_in = QLineEdit(); 
#         self.ed_plate_in.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
#         gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
#         gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
#         gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
#         vside.addWidget(gb_in)

#         # THÔNG TIN XE RA
#         gb_out = QGroupBox("THÔNG TIN XE RA")
#         gl_out = QGridLayout(gb_out)
#         self.ed_date_out = QLineEdit(); 
#         self.ed_time_out = QLineEdit(); 
#         self.ed_plate_out = QLineEdit(); 
#         self.ed_plate_out.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
#         gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
#         gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
#         gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
#         vside.addWidget(gb_out)

#         # BẢNG LỊCH SỬ (nút)
#         gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
#         v_hist_btns = QVBoxLayout(gb_hist_btns)
#         self.btn_show_history = QPushButton("Xem bảng lịch sử")
#         self.btn_export_hist  = QPushButton("Export Excel")
#         self.btn_delete_hist  = QPushButton("Xóa bảng")
#         self.btn_search_hist  = QPushButton("Tìm kiếm")
#         self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide()
#         self._normalize_button(self.btn_show_history, self.btn_export_hist, self.btn_delete_hist, self.btn_search_hist, self.btn_hide_history)

#         self._apply_btn_style(self.btn_show_history, f"""
#         QPushButton {{ {common_btn} background:#E6F4EA; border:1px solid #cde9d6; }}
#         QPushButton:hover  {{ background:#d9efe0; }}
#         QPushButton:pressed{{ background:#ccead6; }}
#         QPushButton:disabled{{ background:#f1faf4; color:#777; border-color:#e3f5e9; }}
#         """)
#         self._apply_btn_style(self.btn_hide_history, f"""
#         QPushButton {{ {common_btn} background:#fff3bf; border:1px solid #f5c6c2; }}
#         QPushButton:hover  {{ background:#ffeda3; }}
#         QPushButton:pressed{{ background:#ffe788; }}
#         QPushButton:disabled{{ background:#fff9dc; color:#777; border-color:#ffefb3; }}
#         """)
#         self._apply_btn_style(self.btn_export_hist, f"""
#         QPushButton {{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         QPushButton:pressed{{ background:#c8deff; }}
#         QPushButton:disabled{{ background:#eef5ff; color:#777; border-color:#ddeaff; }}
#         """)
#         self._apply_btn_style(self.btn_delete_hist, f"""
#         QPushButton {{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }}
#         QPushButton:hover  {{ background:#ffd1d1; }}
#         QPushButton:pressed{{ background:#ffc2c2; }}
#         QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}
#         """)
#         self._apply_btn_style(self.btn_search_hist, f"""
#         QPushButton {{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         QPushButton:pressed{{ background:#c8deff; }}
#         QPushButton:disabled{{ background:#eef5ff; color:#777; border-color:#ddeaff; }}
#         """)
#         row_cmd = QHBoxLayout()

#         # FIX: không dùng stretch để nút không kéo dài
#         row_cmd.addWidget(self.btn_search_hist)
#         row_cmd.addWidget(self.btn_export_hist)
#         row_cmd.addWidget(self.btn_delete_hist)
#         v_hist_btns.addWidget(self.btn_show_history)
#         v_hist_btns.addLayout(row_cmd)
#         v_hist_btns.addWidget(self.btn_hide_history)
#         vside.addWidget(gb_hist_btns)

#         vside.addStretch(1)
#         root.addWidget(side)

#         # RIGHT
#         right_container = QVBoxLayout()
#         self.main_view = QWidget()
#         main_layout = QVBoxLayout(self.main_view)

#         top = QHBoxLayout()
#         self.lbl_cam1 = QLabel(); self.lbl_cam1.setScaledContents(False)
#         self.lbl_cam2 = QLabel(); self.lbl_cam2.setScaledContents(False)
#         for lbl in (self.lbl_cam1, self.lbl_cam2):
#             lbl.setAlignment(Qt.AlignCenter)
#             lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
#             # FIX: không đặt minimumSize theo PANEL_W/H; chỉ đặt chiều cao gợi ý
#             lbl.setMinimumHeight(220)
#             lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1)
#         cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
#         top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1)
#         main_layout.addLayout(top)

#         bottom = QHBoxLayout()
#         self.lbl_scene = QLabel(); self.lbl_scene.setScaledContents(False); self.lbl_scene.setAlignment(Qt.AlignCenter); self.lbl_scene.setStyleSheet("background:#ffffff; border-radius:12px;"); self.lbl_scene.setMinimumHeight(220); self.lbl_scene.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.lbl_roi   = QLabel(); self.lbl_roi.setScaledContents(False);   self.lbl_roi.setAlignment(Qt.AlignCenter);   self.lbl_roi.setStyleSheet("background:#ffffff; border-radius:12px;");   self.lbl_roi.setMinimumHeight(220);   self.lbl_roi.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
#         # self.txt_match    = QLabel("")
#         # THAY THẾ: Dùng QLineEdit thay cho QLabel để có giao diện khung
#         self.txt_match = QLineEdit()
#         self.txt_match.setReadOnly(True) 
#         self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
#         r=0
#         info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
#         info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
#         info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
#         info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
#         info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
#         info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
#         info_layout.addWidget(QLabel("So khớp biển số:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,2)
#         main_layout.addWidget(self.info_group)

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
#         self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected)

#         # Sửa lỗi bảng xen kẽ: Tắt
#         self.tbl_hist.setAlternatingRowColors(False)

#         header.setSectionResizeMode(QHeaderView.Stretch)
#         hist_v.addWidget(self.tbl_hist)
#         hist_layout.addWidget(hist_group)

#         self.stacked = QStackedWidget()
#         self.stacked.addWidget(self.main_view)
#         self.stacked.addWidget(self.history_view)
#         self.stacked.setCurrentIndex(0)
#         right_container.addWidget(self.stacked, 1)
#         root.addLayout(right_container, 1)

#         self.update_titles_and_modes()

#         # Kết nối các nút lịch sử sau khi tạo UI
#         self.btn_show_history.clicked.connect(self.show_history_view)
#         self.btn_hide_history.clicked.connect(self.show_main_view)
#         self.btn_export_hist.clicked.connect(self.on_export_excel)
#         self.btn_delete_hist.clicked.connect(self.on_delete_history)
#         self.btn_search_hist.clicked.connect(self.on_search_history_clicked)

#     # ---- 8.8 Cập nhật hướng làn và thông báo cho worker ----
#     def update_titles_and_modes(self):
#         self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
#         self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
#         if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
#         if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")

#     # ---- 8.9 Đặt lại hướng làn mặc định ----
#     @Slot()
#     def on_reset_lanes(self):
#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.update_titles_and_modes()
#         self.show_logo(1); self.show_logo(2)

#     # ---- 8.10 Chuyển đổi chế độ một chiều ----
#     @Slot()
#     def on_one_way_clicked(self):
#         if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
#         else:                       self.lane1_dir="RA";  self.lane2_dir="RA"
#         self.one_way_toggle_vao = not self.one_way_toggle_vao
#         self.update_titles_and_modes()

#     # ---- 8.11 Chuyển đổi chế độ hai chiều ----
#     @Slot()
#     def on_two_way_clicked(self):
#         if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
#         else:                   self.lane1_dir="RA";  self.lane2_dir="VÀO"
#         self.two_way_toggle = not self.two_way_toggle
#         self.update_titles_and_modes()

#     # ---- 8.xx Hàm này cần được viết trong class MainWindow và kết nối với worker ----
#     @Slot(str)
#     def update_match_status(self, status: str):
#         display_status = status.replace('-', ' ').title()
#         self.txt_match.setText(display_status) # <-- Cập nhật QLineEdit

#         if "Khop Bien So" in display_status:
#             # Xanh lá cây
#             self.txt_match.setStyleSheet("color: #007700; font-weight: 700;") 
#         elif "Khong Khop Bien So" in display_status:
#             # Đỏ
#             self.txt_match.setStyleSheet("color: #ff0000; font-weight: 700;")
#         else:
#             # Xanh dương (Mặc định/Chờ)
#             self.txt_match.setStyleSheet("color: #007700; font-weight: 700;")

#     # ---- 8.xx MỚI: Nhận tín hiệu và phát âm thanh ----
#     @Slot(str)
#     def on_play_sound(self, mode):
#         """Phát âm thanh dựa trên chế độ (in/out)"""
#         if mode == "in":
#             self.sound_in.play()
#         elif mode == "out":
#             self.sound_out.play()
#         else:
#             print(f"Lỗi: Không tìm thấy file âm thanh!")

#     # ---- 8.12 Xử lý thay đổi chế độ OCR ----
#     @Slot()
#     def on_ocr_mode_changed(self):
#         self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
#         if self.rb_gem.isChecked() and not GEMINI_READY:
#             QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
#             self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
#         if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
#         if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

#     # ---- 8.13 Hiển thị chế độ xem Lịch sử ----
#     def show_history_view(self):
#         self.stacked.setCurrentIndex(1); self.btn_show_history.hide(); self.btn_hide_history.show(); self.refresh_history_data()
    
#     # ---- 8.14 Hiển thị chế độ xem Camera chính ----
#     def show_main_view(self):
#         self.stacked.setCurrentIndex(0); self.btn_hide_history.hide(); self.btn_show_history.show()

#     # ---- 8.15 Xuất dữ liệu lịch sử ra Excel ----
#     @Slot()
#     def on_export_excel(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
#         if df.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
#         path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
#         if not path: return
#         try: df.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
#         except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

#     # ---- 8.16 Xóa dữ liệu lịch sử ----
#     @Slot()
#     def on_delete_history(self):
#         if not (self.db and self.db.ok):
#             QMessageBox.warning(self, "Xóa", "Chưa kết nối DB, không thể xóa."); return
#         dlg = DeleteDialog(self)
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
#             self.db.delete_by_ids(ids); self.refresh_history_data()
#         elif res == 2:
#             self.db.delete_all(); self.refresh_history_data()
#         else:
#             return

#     # ---- 8.17 image helpers ----
#     def qpix_logo(self):
#         if os.path.exists(LOGO_PATH):
#             return QPixmap(LOGO_PATH)
#         return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))

#     # ---- 8.18 Hiển thị logo/ảnh mặc định trên camera ----
#     def show_logo(self, which: int):
#         pm = self._logo_pm
#         if which == 1: self._set_centered_pixmap(self.lbl_cam1, pm)
#         else:          self._set_centered_pixmap(self.lbl_cam2, pm)

#     # ---- 8.19 Nhận và hiển thị khung hình video ----
#     @Slot(np.ndarray, str)
#     def on_frame(self, frame_bgr, title):
#         sender = self.sender()
#         if sender is self.cam1_worker:
#             self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
#         elif sender is self.cam2_worker:
#             self._set_centered_pixmap(self.lbl_cam2, frame_bgr)

#     # ---- 8.20 Nhận và hiển thị khung hình scene ----
#     @Slot(str)
#     def on_scene(self, path):
#         if os.path.exists(path):
#             bgr = cv2.imread(path)
#             self._set_centered_pixmap(self.lbl_scene, bgr)

#     # ---- 8.21 Nhận và hiển thị khung hình ROI ----
#     @Slot(str, str)
#     def on_roi(self, roi_path, mode):
#         if os.path.exists(roi_path):
#             bgr = cv2.imread(roi_path)
#             self._set_centered_pixmap(self.lbl_roi, bgr)

#     # ---- 8.22 Nhận và hiển thị thông tin xe ----
#     @Slot(dict)
#     def on_info(self, info):
#         if "date_in" in info:  self.txt_date_in.setText(info["date_in"]);   self.ed_date_in.setText(info["date_in"])
#         if "time_in" in info:  self.txt_time_in.setText(info["time_in"]);   self.ed_time_in.setText(info["time_in"])
#         if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
#         if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
#         if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
#         if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

#     # ---- 8.23 Nhận và hiển thị thông tin so khớp ----
#     @Slot(str)
#     def on_match(self, txt): 
#         self.txt_match.setText(txt.upper())

#     # ---- 8.24 Tải và cập nhật bảng lịch sử ----
#     @Slot()
#     def refresh_history_data(self, start_time=None, end_time=None):
#         """Hàm chính tải dữ liệu từ DB và cập nhật bảng"""
#         df = self.db.fetch_history_df(limit=10000, start_time=start_time, end_time=end_time) if (self.db and self.db.ok) else pd.DataFrame()

#         # Lưu df để dùng cho việc nhấp chuột
#         self.history_df = df.copy() 

#         if not df.empty and "STT" in df.columns: 
#             df = df.drop(columns=["STT"])
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

#     # ---- 8.25 camera controls ----
#     def _connect_worker(self, w: VideoWorker):
#         w.frameSignal.connect(self.on_frame)
#         w.sceneSignal.connect(self.on_scene)
#         w.roiSignal.connect(self.on_roi)
#         w.infoSignal.connect(self.on_info)
#         w.matchSignal.connect(self.on_match)
#         w.histSignal.connect(self.on_history_signal_refresh)
#         w.playSoundSignal.connect(self.on_play_sound)

#     # ---- 8.xx MỚI: Slot trung gian cho Timer/Worker ----
#     @Slot()
#     def on_history_signal_refresh(self):
#         """Slot này nhận tín hiệu từ worker/timer và gọi hàm tải dữ liệu chính"""
#         # Chỉ làm mới nếu tab lịch sử đang được xem
#         if self.stacked.currentIndex() == 1:
#             # Tải mà không có bộ lọc thời gian
#             self.refresh_history_data() 

#     # ---- 8.xx MỚI: Xử lý sự kiện nhấp vào nút Tìm kiếm ----
#     @Slot()
#     def on_search_history_clicked(self):
#         start_dt, end_dt = SearchDialog.get_range(self)
        
#         if start_dt and end_dt:
#             # Gọi hàm tải dữ liệu với khoảng thời gian
#             self.refresh_history_data(start_time=start_dt, end_time=end_dt)
#             # Chuyển sang tab lịch sử nếu chưa ở đó
#             if self.stacked.currentIndex() != 1:
#                 self.show_history_view()

#     # ---- 8.xx MỚI: Xử lý sự kiện nhấp vào hàng trong bảng ----
#     @Slot()
#     def on_history_row_selected(self):
#         selected_items = self.tbl_hist.selectedItems()
#         # history_df phải có cột "STT"
#         if not selected_items or self.history_df.empty or "STT" not in self.history_df.columns:
#             return

#         try:
#             # Lấy hàng được chọn
#             row_index_view = selected_items[0].row() # Đây là chỉ số của view
            
#             # Lấy STT từ view
#             stt_item = self.tbl_hist.item(row_index_view, self.history_df.columns.get_loc("STT"))
#             if not stt_item: return
            
#             stt = int(stt_item.text())

#             # Tìm hàng trong DataFrame gốc dựa trên STT
#             row_data_series = self.history_df[self.history_df['STT'] == stt]
#             if row_data_series.empty: return
            
#             row_data = row_data_series.iloc[0] # Lấy dòng đầu tiên khớp
            
#             # 1. Cập nhật thông tin bên trái
#             self.ed_date_in.setText(str(row_data.get("Ngày vào", "")))
#             self.ed_time_in.setText(str(row_data.get("Giờ vào", "")))
#             self.ed_plate_in.setText(str(row_data.get("Biển số vào", "")))
            
#             self.ed_date_out.setText(str(row_data.get("Ngày ra", "")))
#             self.ed_time_out.setText(str(row_data.get("Giờ ra", "")))
#             self.ed_plate_out.setText(str(row_data.get("Biển số ra", "")))

#             # 2. Cập nhật hình ảnh (Image_BOX và ROI_Plate)
#             img_in_path = str(row_data.get("Ảnh vào", ""))
            
#             # Ưu tiên hiển thị ảnh ra (nếu có) trên ROI
#             img_out_path = str(row_data.get("Ảnh ra", "")) 
#             img_roi_to_show = img_out_path if (img_out_path and os.path.exists(img_out_path)) else img_in_path

#             if img_in_path and os.path.exists(img_in_path):
#                 bgr_in = cv2.imread(img_in_path)
#                 self._set_centered_pixmap(self.lbl_scene, bgr_in) # Ảnh vào cho Image_BOX
#             else:
#                 self._set_centered_pixmap(self.lbl_scene, self.qpix_logo()) # Hiển thị logo nếu không có ảnh

#             if img_roi_to_show and os.path.exists(img_roi_to_show):
#                 bgr_roi = cv2.imread(img_roi_to_show)
#                 self._set_centered_pixmap(self.lbl_roi, bgr_roi) # Ảnh ra (hoặc vào) cho ROI_Plate
#             else:
#                 self._set_centered_pixmap(self.lbl_roi, self.qpix_logo())
                
#             # 3. Chuyển về màn hình chính để xem ảnh
#             self.show_main_view()
            
#         except Exception as e:
#             print(f"Lỗi khi chọn hàng: {e}")

#     # ---- 8.26 Hàm chung để khởi động camera (1 hoặc 2) ----
#     def start_cam_generic(self, which: int):
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#             return
#         if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
#         if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

#         ocr_mode = self.current_ocr_mode
#         default_api = API_MAP["DSHOW(Windows)"] 

#         if which == 1:
#             idx = int(self.spin_cam1.value())
#             mode = "in" if self.lane1_dir=="VÀO" else "out"
#             title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam1_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam1_worker); self.cam1_worker.start()
#         else:
#             idx = int(self.spin_cam2.value())
#             mode = "in" if self.lane2_dir=="VÀO" else "out"
#             title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam2_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam2_worker); self.cam2_worker.start()

#     # ---- 8.27 Hàm chung để dừng camera (1 hoặc 2) ----
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

#     # ---- 8.28 Xử lý sự kiện đóng cửa sổ ----
#     def closeEvent(self, event):
#         try: self.stop_cam_generic(1); self.stop_cam_generic(2)
#         except: pass
#         super().closeEvent(event)

# # ==================== 9. MAIN ====================
# def main():
#     QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
#     app = QApplication(sys.argv)
#     app.setStyle("Fusion")
#     w = MainWindow(); 
#     w.show()
#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()


























































































# # -*- coding: utf-8 -*-
# """
#         =========================================================
#         = PySide6 app: Phát hiện & OCR biển số xe (YOLO/Gemini) =
#         =========================================================

# 1. Phát hiện & OCR (YOLOv8, OpenCV, Gemini AI): 	          
#     Tự động phát hiện vị trí biển số (YOLO Detect) và trích xuất ký tự 
#     (YOLO OCR hoặc Gemini AI). Xử lý tiền DL ảnh (CLAHE, Blur) để tăng độ chính xác OCR.

# 2. Giao diện - UI [PySide6 (QMainWindow, QThread, Signal/Slot)]:
#     Xây dựng giao diện Desktop, hiển thị video trực tiếp, kết quả OCR, và kết nối các luồng 
#     xử lý video (Worker) với giao diện chính.

# 3. Quản lý Dữ liệu	[SQL Server (qua pyodbc), pandas]:	
#     Lưu trữ lịch sử giao dịch xe vào/ra (ParkingSessions). Tải dữ liệu lịch sử vào 
#     DataFrame để hiển thị trên bảng UI và Export Excel.

# 4. Luồng Video (QThread, cv2.VideoCapture):
# 	Chạy độc lập cho hai làn xe (VÀO/RA). Chụp và xử lý ảnh khi biển số ổn định (ít nhất 1.2s), 
#     sau đó gửi kết quả (ảnh, biển số, thời gian) về UI.

# 5. Logic Giữ Xe	(Hàm attach_out trong Class DB):
# 	Tự động so khớp biển số xe ra với các xe vào đang chờ (plate_out IS NULL). Cập nhật 
#     trạng thái KHOP-BIEN-SO (Xanh) hoặc KHONG-KHOP-BIEN-SO (Đỏ) trong DB và trên UI.

# 6. Cấu hình	(.env, QSpinBox, QRadioButton):
# 	Cho phép người dùng chọn Index Camera, cấu hình chế độ làn xe (1 chiều/2 chiều) và 
#     lựa chọn Model OCR (YOLO hoặc Gemini).

# 7. Xây dựng UI (_build_ui()):
# 	Hàm xây dựng bố cục chính, tạo các widgets như nút Bật/Tắt Cam, Điều khiển Làn, 
#     các ô hiển thị thông tin xe VÀO/RA, và Bảng Lịch Sử.

# 8. Điều khiển Cam (start_cam_generic/stop_cam_generic):	
#     Khởi động/Dừng luồng xử lý video (VideoWorker). Thiết lập chế độ Vào/Ra và 
#     chế độ OCR cho Worker trước khi chạy.

# 9. Điều khiển Làn (on_one_way_clicked/on_two_way_clicked):
# 	Quản lý hướng làn xe (Vào, Ra). Cho phép chuyển đổi giữa chế độ 
#     Một chiều (Cam 1 & 2 cùng hướng) và Hai chiều (Cam 1 & 2 ngược hướng).

# 10. Chọn OCR Model (on_ocr_mode_changed):
# 	Cho phép người dùng chọn Model OCR. Kiểm tra nếu thiếu API Key Gemini 
#     thì buộc chuyển về YOLO và thông báo.

# 11. Cập nhật Real-time (on_frame/on_info/v.v):
# 	Các hàm @Slot nhận tín hiệu (Signal) từ luồng VideoWorker (ảnh, biển số, thời gian) và 
#     cập nhật tức thời lên các ô hiển thị trên giao diện chính.

# 12. Quản lý Bảng (show_history_view/refresh_history):
# 	Chuyển đổi giữa chế độ xem Camera chính và Bảng Lịch sử. Tải và hiển thị dữ liệu 
#     giao dịch từ SQL lên bảng QTableWidget.

# 13. Thao tác DB (on_export_excel/on_delete_history):
# 	Xử lý các thao tác quản lý dữ liệu: Xuất dữ liệu lịch sử ra Excel và Xóa các 
#     dòng giao dịch đã chọn trong cơ sở dữ liệu. 

# """

# # ==================== 1. IMPORT ====================

# import os, sys, time, cv2, numpy as np, pandas as pd
# from datetime import datetime

# # ---- 1.1 HiDPI Cấu hình HiDPI (Độ phân giải cao) ----
# os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
# os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# # ---- 1.2 Import PySide6 ----
# from PySide6.QtCore import QDateTime
# from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint, QUrl
# from PySide6.QtGui import QImage, QPixmap, QGuiApplication, QFont
# from PySide6.QtMultimedia import QSoundEffect
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox, QComboBox,
#     QGridLayout, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem,
#     QSizePolicy, QMessageBox, QLineEdit, QRadioButton, QFrame, QStackedWidget,
#     QFileDialog, QHeaderView, QDialog, QDateTimeEdit
# )

# # ---- 1.3 Optional SQL ----
# USE_SQL = True
# try:
#     import pyodbc
# except Exception:
#     USE_SQL = False

# # ---- 1.4 YOLO ----
# from ultralytics import YOLO

# # ---- 1.5 Gemini API (optional) ----
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

# # ---- 2.1 Đường dẫn Model ----
# DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
# OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/Tong_Hop_4_Dataset.pt"
# SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)
# LOGO_PATH = os.path.join("D:/Documents/IUH_Student/OCR/logo", "logo_cholimex.jpg")
# SOUND_IN_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_vao_xin_cam_on.wav"
# SOUND_OUT_PATH = "D:/Documents/IUH_Student/OCR/audio/moi_ra_xin_cam_on.wav"

# # ---- 2.2 SQL ----
# CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=plates_db;"
#     "UID=sa;"
#     "PWD=123456"
# )

# # ---- 2.3 UI ----
# PANEL_W, PANEL_H = 640, 360
# PANEL_BG = (255, 255, 255)

# API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
# OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
#            "six":"6","seven":"7","eight":"8","nine":"9"}





# # ==================== 3. UTILITIES (HÀM TIỆN ÍCH) ====================

# # ---- 3.1 Căn chỉnh/Điền nền ----
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

# # ---- 3.2 Chuyển đổi ảnh ----
# def bgr_to_qimage(bgr):
#     if bgr is None:
#         bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
#     rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#     h, w, ch = rgb.shape
#     return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)

# # ---- 3.3 Lưu ảnh ----
# def save_image(img, prefix):
#     ts = datetime.now().strftime("%Ym%d_%H%M%S_%f")
#     path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
#     cv2.imwrite(path, img)
#     return path

# # ---- 3.4 OCR ----
# def norm_char(x):  # Chuẩn hóa ký tự
#     return OCR_MAP.get(str(x), str(x))

# def plate_norm(s: str) -> str: # Chuẩn hóa biển số
#     return (s or "").replace("-", "").replace(" ", "").upper()

# def has_boxes(r):  # Kiểm tra có box
#     try:
#         return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
#     except: return False

# def preprocess_for_ocr(roi):  # Tiền xử lý ảnh OCR
#     if roi is None: return None
#     if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe,(3,3),0)
#     return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)





# # ==================== 4. DB LAYER ====================

# class DB:
#     # ---- 4.1 Khởi tạo và Kết nối ----
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

#     # ---- 4.2 Ghi nhận xe VÀO ----
#     def insert_in(self, plate, d, t, img_path):
#         if not self.ok: return
#         try:
#             self.cur.execute("""
#                 INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
#                 VALUES (?,?,?,?,?)
#             """, (plate, d, t, img_path, 'PENDING'))
#         except Exception as e: print("insert_in error:", e)

#     # ---- 4.3 Ghi nhận xe RA và Ghép đôi ----
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

#     # # ---- 4.4 Lấy lịch sử ----
#     # def fetch_history_df(self, limit=10000) -> pd.DataFrame:
#     #     if not self.ok:
#     #         return pd.DataFrame(columns=[
#     #             "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#     #             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#     #         ])
#     #     try:
#     #         rows = self.cur.execute(f"""
#     #             SELECT TOP {limit}
#     #                 id, image_in, plate_in, date_in, time_in,
#     #                 image_out, plate_out, date_out, time_out, match_status
#     #             FROM dbo.ParkingSessions
#     #             ORDER BY id DESC
#     #         """).fetchall()
#     #         df = pd.DataFrame.from_records(
#     #             rows,
#     #             columns=["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#     #                      "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#     #         ).astype(object).where(pd.notnull, "")
#     #         df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#     #         df.insert(0, "STT", range(1, len(df)+1))
#     #         return df
#     #     except Exception as e:
#     #         print("fetch_history error:", e)
#     #         return pd.DataFrame(columns=[
#     #             "STT","ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#     #             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#     #         ])

#     # ---- 4.4 Lấy lịch sử (ĐÃ CẬP NHẬT) ----
#     def fetch_history_df(self, limit=10000, start_time=None, end_time=None) -> pd.DataFrame:
#         columns = [
#             "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
#             "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
#         ]
#         if not self.ok:
#             # Sửa cột: thêm STT để khớp với logic mới
#             return pd.DataFrame(columns=["STT"] + columns) 
        
#         try:
#             # Xây dựng câu lệnh SQL động
#             sql = f"""
#                 SELECT TOP ({limit})
#                     id, image_in, plate_in, date_in, time_in,
#                     image_out, plate_out, date_out, time_out, match_status
#                 FROM dbo.ParkingSessions
#             """
            
#             where_clauses = []
#             sql_params = [] # Tham số cho WHERE

#             if start_time:
#                 # Dùng created_at để lọc vì đây là cột DATETIME
#                 where_clauses.append("created_at >= ?") 
#                 sql_params.append(start_time)
            
#             if end_time:
#                 where_clauses.append("created_at <= ?")
#                 sql_params.append(end_time)
            
#             if where_clauses:
#                 sql += " WHERE " + " AND ".join(where_clauses)
            
#             sql += " ORDER BY id DESC"
            
#             rows = self.cur.execute(sql, tuple(sql_params)).fetchall()
            
#             df = pd.DataFrame.from_records(
#                 rows,
#                 columns=columns
#             ).astype(object).where(pd.notnull, "")
            
#             df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
#             df.insert(0, "STT", range(1, len(df)+1))
#             return df
        
#         except Exception as e:
#             print("fetch_history error:", e)
#             return pd.DataFrame(columns=["STT"] + columns)

#     # ---- 4.5 Xóa theo ID ----
#     def delete_by_ids(self, ids):
#         if not self.ok or not ids: return
#         try:
#             for sid in ids:
#                 self.cur.execute("DELETE FROM dbo.ParkingSessions WHERE id=?", (int(sid),))
#         except Exception as e: print("delete_by_ids error:", e)

#     # ---- 4.6 Xóa tất cả ----
#     def delete_all(self):
#         if not self.ok: return
#         try: self.cur.execute("DELETE FROM dbo.ParkingSessions")
#         except Exception as e: print("delete_all error:", e)





# # ==================== 5. YOLO/GEMINI WRAPPERS (TRÌNH BAO BỌC MODEL) ====================
# class Models:
#     # ---- 5.1 Khởi tạo (Tải model) ----
#     def __init__(self, det_path: str, ocr_path: str):
#         self.ok = True; self.err = ""
#         try:
#             self.det = YOLO(det_path)
#             self.ocr = YOLO(ocr_path)
#         except Exception as e:
#             self.ok = False; self.err = str(e)

#     # ---- 5.2 YOLO phát hiện biển số ----
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

#     # ---- 5.3 OCR biển số bằng YOLO ----
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

#     # ---- 5.4 OCR biển số bằng Gemini AI ----
#     def ocr_plate_gemini_from_path(self, image_path: str):
#         if not GEMINI_READY: return "", ""
#         try:
#             img = Image.open(image_path)
#         except Exception as e:
#             print("Gemini open image error:", e); return "", ""
#         try:
#             model = genai.GenerativeModel('gemini-1.5-flash') # Dùng 1.5-flash
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

#     # ---- 5.5 Hỗ trợ (Hàm tĩnh định dạng) ----
#     @staticmethod
#     def _format_text(text_raw: str):
#         raw=(text_raw or '').replace('-', ' ').replace(' ', '')
#         text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
#         return text_fmt, (text_raw or "")





# # ==================== 6. VIDEO WORKER (Luồng xử lý Video) ====================
# class VideoWorker(QThread):
#     frameSignal = Signal(np.ndarray, str)
#     sceneSignal = Signal(str)
#     roiSignal   = Signal(str, str)
#     infoSignal  = Signal(dict)
#     matchSignal = Signal(str)
#     histSignal  = Signal()
#     playSoundSignal = Signal(str)

#     # ---- 6.1 Khởi tạo ----
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

#     # ---- 6.2 Setter/Getter ----
#     def set_title(self, title: str): 
#         self.title = title
#     def set_ocr_mode(self, mode: str): 
#         self.ocr_mode = mode
#     def set_mode(self, mode: str): 
#         self.mode = mode

#     # ---- 6.3 Vòng lặp chính của luồng ----
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

#             self.frameSignal.emit(frame, self.title)

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
#                         self.playSoundSignal.emit("in")
#                     else:
#                         self.infoSignal.emit({"date_out": d, "time_out": t, "plate_text_out": plate})
#                         if self.db and self.db.ok:
#                             match = self.db.attach_out(plate, d, t, scene_path)
#                             self.matchSignal.emit(match)
#                             self.histSignal.emit()
#                         self.playSoundSignal.emit("out")
#                     self.captured = True

#             time.sleep(0.01)

#         try:
#             if self.cap: self.cap.release()
#         except: pass

#     # ---- 6.4 Dừng luồng ----
#     def stop(self): 
#         self._running = False





# # ==================== 7. DELETE DIALOG (Hộp thoại Xóa) ====================

# class DeleteDialog(QDialog):
#     # ---- 7.1 Khởi tạo Giao diện ----
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
#         """)
#         lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
#         lab = QLabel("Bạn muốn xóa dữ liệu lịch sử như thế nào?")
#         lay.addWidget(lab)

#         row = QHBoxLayout(); row.setSpacing(12)
#         self.btn_sel = QPushButton("Xóa dòng đã chọn")
#         self.btn_all = QPushButton("Xóa tất cả")
#         self.btn_can = QPushButton("Hủy")
#         row.addWidget(self.btn_sel, 1)
#         row.addWidget(self.btn_all, 1)
#         row.addWidget(self.btn_can, 1)
#         lay.addLayout(row)

#         base = "height:34px; font-weight:600; border-radius:10px; padding:6px 12px;"
#         self.btn_sel.setStyleSheet(f"""
#         QPushButton {{ {base} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         QPushButton:pressed{{ background:#c8deff; }}
#         """)
#         self.btn_all.setStyleSheet(f"""
#         QPushButton {{ {base} background:#ffe0e0; border:1px solid #ffb3b3; }}
#         QPushButton:hover  {{ background:#ffd1d1; }}
#         QPushButton:pressed{{ background:#ffc2c2; }}
#         """)
#         self.btn_can.setStyleSheet(f"""
#         QPushButton {{ {base} background:#f3f4f6; border:1px solid #e5e7eb; }}
#         QPushButton:hover  {{ background:#eef0f3; }}
#         QPushButton:pressed{{ background:#e7e9ed; }}
#         """)

#         self.btn_sel.clicked.connect(lambda: self.done(1))
#         self.btn_all.clicked.connect(lambda: self.done(2))
#         self.btn_can.clicked.connect(lambda: self.done(0))



# # ==================== 8. MAIN WINDOW (CỬA SỔ CHÍNH ====================

# class MainWindow(QMainWindow):
#     # ---- 8.1 Khởi tạo Giao diện ứng dụng ----
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Desktop App (Giữ xe)")
#         self.setMinimumSize(1200, 800)
#         self._init_theme()

#         self.models = Models(DETECT_MODEL_PATH, OCR_MODEL_PATH)
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#         self.db = DB(CONN_STR) if USE_SQL else DB("")

#         # Khởi tạo âm thanh
#         self.sound_in = QSoundEffect(self)
#         sound_in_abs = os.path.abspath(SOUND_IN_PATH)
#         if os.path.exists(sound_in_abs):
#             self.sound_in.setSource(QUrl.fromLocalFile(sound_in_abs))
#         else:
#             print(f"Lỗi: Không tìm thấy file âm thanh: {sound_in_abs}")

#         self.sound_out = QSoundEffect(self)
#         sound_out_abs = os.path.abspath(SOUND_OUT_PATH)
#         if os.path.exists(sound_out_abs):
#             self.sound_out.setSource(QUrl.fromLocalFile(sound_out_abs))
#         else:
#             print(f"Lỗi: Không tìm thấy file âm thanh: {sound_out_abs}")

#         self.cam1_worker = None
#         self.cam2_worker = None

#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.current_ocr_mode = "yolo"
#         self.history_df = pd.DataFrame()

#         # Lưu logo gốc để scale lại đúng ở mọi lần vẽ
#         self._logo_pm = self.qpix_logo()
#         self._build_ui()
#         self.show_logo(1); self.show_logo(2)
#         self.hist_timer = QTimer(self); self.hist_timer.timeout.connect(self.on_history_signal_refresh); self.hist_timer.start(5000)

#     # ---- 8.2 Thiết lập Giao diện ----
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
#         QFrame[class="card"]       { background: #ffffff; border-radius: 12px; }
#         QFrame[class="title-wrap"]{ background: #e6e6e6; border-radius: 12px; }
#         QLabel[class="title"] {
#             font: 700 18px "Segoe UI";
#             padding: 6px 10px;
#             background: #ffffff;
#             border-radius: 10px;
#         }

#         QLineEdit {
#             height: 28px; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 2px 6px;
#         }
#         QTableWidget { background: #ffffff; gridline-color: #e6e6e6; }
#         """)

#     # ---- 8.3 Chuẩn hóa hành vi của các nút ----
#     def _normalize_button(self, *btns):
#         for b in btns:
#             b.setAutoDefault(False); b.setDefault(False); b.setFlat(False); b.setFocusPolicy(Qt.NoFocus)
#             # FIX: để nút không kéo giãn vô hạn khi phóng to
#             b.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

#     # ---- 8.4 Sửa lỗi bo tròn: Đơn giản hóa hàm ----
#     def _apply_btn_style(self, btn: QPushButton, css: str):
#         btn.setStyleSheet(css)

#     # ---- 8.5 Tạo khung hiển thị (Card UI) ----
#     def _make_card(self, title:str, content:QWidget):
#         wrap = QFrame(); wrap.setProperty("class","card-wrap")
#         wrapL = QVBoxLayout(wrap); wrapL.setContentsMargins(2,2,2,2)
#         card = QFrame(); card.setProperty("class","card")
#         v = QVBoxLayout(card); v.setContentsMargins(8,8,8,8)
#         title_wrap = QFrame(); title_wrap.setProperty("class","title-wrap")
#         hl = QHBoxLayout(title_wrap); hl.setContentsMargins(2,2,2,2)
#         title_lbl = QLabel(title); title_lbl.setProperty("class","title")
#         hl.addWidget(title_lbl)
#         v.addWidget(title_wrap); v.addWidget(content, 1)
#         wrapL.addWidget(card)
#         return wrap, title_lbl

#     # ---- 8.6 Hiển thị ảnh căn giữa và giữ tỷ lệ ----
#     def _set_centered_pixmap(self, lbl: QLabel, src):
#         if isinstance(src, np.ndarray):
#             pm = QPixmap.fromImage(bgr_to_qimage(src))
#         elif isinstance(src, QImage):
#             pm = QPixmap.fromImage(src)
#         else:
#             pm = src
#         if pm is None or pm.isNull():
#             lbl.clear(); return
#         rect = lbl.contentsRect()
#         avail = rect.size()
#         dpr = lbl.devicePixelRatioF() if hasattr(lbl, "devicePixelRatioF") else 1.0
#         target_w = max(1, int(avail.width()  * dpr))
#         target_h = max(1, int(avail.height() * dpr))
#         scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         if hasattr(scaled, "setDevicePixelRatio"):
#             scaled.setDevicePixelRatio(dpr)
#         lbl.setAlignment(Qt.AlignCenter)
#         lbl.setPixmap(scaled)

#     # ---- 8.7 Xây dựng cấu trúc giao diện chính (Widgets-UI) ----
#     def _build_ui(self):
#         central = QWidget(); self.setCentralWidget(central)
#         root = QHBoxLayout(central); root.setContentsMargins(12,12,12,12)

#         # LEFT
#         side = QWidget(objectName="SideBar")
#         side.setFixedWidth(450)
#         vside = QVBoxLayout(side); vside.setContentsMargins(10,10,10,10); vside.setSpacing(12)

#         # CAMERA CONTROL
#         gb_camctl = QGroupBox("CAMERA CONTROL")
#         vl_camctl = QVBoxLayout(gb_camctl)
#         vl_camctl.setSpacing(10) # Khoảng cách giữa các hàng

#         self.spin_cam1 = QSpinBox(); self.spin_cam1.setRange(0,9); self.spin_cam1.setValue(0)
#         self.spin_cam2 = QSpinBox(); self.spin_cam2.setRange(0,9); self.spin_cam2.setValue(0)
#         self.cb_api1  = QComboBox(); self.cb_api1.addItems(list(API_MAP.keys())); self.cb_api1.hide() # Ẩn API combo box
#         self.cb_api2  = QComboBox(); self.cb_api2.addItems(list(API_MAP.keys())); self.cb_api2.hide() # Ẩn API combo box

#         # Hàng 1: Index Cam 1 & Index Cam 2
#         row_indices = QHBoxLayout()
#         row_indices.setSpacing(10)
#         row_indices.addWidget(QLabel("Index Cam 1"))
#         row_indices.addWidget(self.spin_cam1, 1)
#         row_indices.addWidget(QLabel("Index Cam 2"))
#         row_indices.addWidget(self.spin_cam2, 1)
#         vl_camctl.addLayout(row_indices)

#         # Buttons
#         self.btn_start1 = QPushButton("Bật Cam 1")
#         self.btn_stop1  = QPushButton("Tắt Cam 1")
#         self.btn_start2 = QPushButton("Bật Cam 2")
#         self.btn_stop2  = QPushButton("Tắt Cam 2")
#         self._normalize_button(self.btn_start1, self.btn_stop1, self.btn_start2, self.btn_stop2)

#         common_btn = "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;"

#         self._apply_btn_style(self.btn_start1, f"""
#         QPushButton {{ {common_btn} background:#d1fadf; border:1px solid #a6f4c5; }}
#         QPushButton:hover  {{ background:#c3f7d6; }}
#         QPushButton:pressed{{ background:#b4f3cc; }}
#         QPushButton:disabled{{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }}
#         """)
#         self._apply_btn_style(self.btn_stop1, f"""
#         QPushButton {{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }}
#         QPushButton:hover  {{ background:#ffd1d1; }}
#         QPushButton:pressed{{ background:#ffc2c2; }}
#         QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}
#         """)
#         self._apply_btn_style(self.btn_start2, self.btn_start1.styleSheet())
#         self._apply_btn_style(self.btn_stop2,  self.btn_stop1.styleSheet())

#         # signals
#         self.btn_start1.clicked.connect(self.start_cam1)
#         self.btn_stop1.clicked.connect(self.stop_cam1)
#         self.btn_start2.clicked.connect(self.start_cam2)
#         self.btn_stop2.clicked.connect(self.stop_cam2)

#         # Hàng 2: Bật/Tắt Cam 1
#         row_btn1 = QHBoxLayout()
#         row_btn1.setSpacing(12)
#         row_btn1.addWidget(self.btn_start1)
#         row_btn1.addWidget(self.btn_stop1)
#         vl_camctl.addLayout(row_btn1)

#         # Hàng 3: Bật/Tắt Cam 2
#         row_btn2 = QHBoxLayout()
#         row_btn2.setSpacing(12)
#         row_btn2.addWidget(self.btn_start2)
#         row_btn2.addWidget(self.btn_stop2)
#         vl_camctl.addLayout(row_btn2)

#         vside.addWidget(gb_camctl)

#         # ĐIỀU KHIỂN LÀN
#         gb_lane = QGroupBox("ĐIỀU KHIỂN LÀN")
#         vl_lane = QVBoxLayout(gb_lane); vl_lane.setSpacing(10)
#         row_lane = QHBoxLayout(); row_lane.setSpacing(12)

#         self.btn_oneway = QPushButton("1 chiều")
#         self.btn_twoway = QPushButton("2 chiều")
#         self.btn_reset_lane = QPushButton("Reset làn")
#         self._normalize_button(self.btn_oneway, self.btn_twoway, self.btn_reset_lane)

#         self._apply_btn_style(self.btn_oneway, f"""
#         QPushButton {{ {common_btn} background:#dbeafe; border:1px solid #bfdbfe; }}
#         QPushButton:hover  {{ background:#cfe3fd; }}
#         QPushButton:pressed{{ background:#c3dcfc; }}
#         QPushButton:disabled{{ background:#eef6ff; color:#777; border-color:#e3efff; }}
#         """)
#         self._apply_btn_style(self.btn_twoway, self.btn_oneway.styleSheet())
#         self._apply_btn_style(self.btn_reset_lane, f"""
#         QPushButton {{ {common_btn} background:#fff3bf; border:1px solid #ffe066; }}
#         QPushButton:hover  {{ background:#ffeda3; }}
#         QPushButton:pressed{{ background:#ffe788; }}
#         QPushButton:disabled{{ background:#fff9dc; color:#777; border-color:#ffefb3; }}
#         """)

#         row_lane.addWidget(self.btn_oneway)
#         row_lane.addWidget(self.btn_twoway)
#         vl_lane.addLayout(row_lane)
#         vl_lane.addWidget(self.btn_reset_lane)

#         self.btn_oneway.clicked.connect(self.on_one_way_clicked)
#         self.btn_twoway.clicked.connect(self.on_two_way_clicked)
#         self.btn_reset_lane.clicked.connect(self.on_reset_lanes)
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
#             self.rb_gem.setToolTip("Chưa có GEMINI_API_KEY (.env hoặc biến môi trường) → dùng YOLO")
#         vside.addWidget(gb_ocr)

#         # THÔNG TIN XE VÀO
#         gb_in = QGroupBox("THÔNG TIN XE VÀO")
#         gl_in = QGridLayout(gb_in)
#         self.ed_date_in = QLineEdit(); 
#         self.ed_time_in = QLineEdit(); 
#         self.ed_plate_in = QLineEdit(); 
#         self.ed_plate_in.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
#         gl_in.addWidget(QLabel("Ngày vào:"),0,0); gl_in.addWidget(self.ed_date_in,0,1)
#         gl_in.addWidget(QLabel("Giờ vào:"), 1,0); gl_in.addWidget(self.ed_time_in, 1,1)
#         gl_in.addWidget(QLabel("Biển số vào:"),2,0); gl_in.addWidget(self.ed_plate_in,2,1)
#         vside.addWidget(gb_in)

#         # THÔNG TIN XE RA
#         gb_out = QGroupBox("THÔNG TIN XE RA")
#         gl_out = QGridLayout(gb_out)
#         self.ed_date_out = QLineEdit(); 
#         self.ed_time_out = QLineEdit(); 
#         self.ed_plate_out = QLineEdit(); 
#         self.ed_plate_out.setStyleSheet("color: #ff0000; font-size: 15px; font-weight: 700; height: 32px;")
#         gl_out.addWidget(QLabel("Ngày ra:"),0,0); gl_out.addWidget(self.ed_date_out,0,1)
#         gl_out.addWidget(QLabel("Giờ ra:"), 1,0); gl_out.addWidget(self.ed_time_out, 1,1)
#         gl_out.addWidget(QLabel("Biển số ra:"),2,0); gl_out.addWidget(self.ed_plate_out,2,1)
#         vside.addWidget(gb_out)

#         # BẢNG LỊCH SỬ (nút)
#         gb_hist_btns = QGroupBox("BẢNG LỊCH SỬ")
#         v_hist_btns = QVBoxLayout(gb_hist_btns)
#         self.btn_show_history = QPushButton("Xem bảng lịch sử")
#         self.btn_export_hist  = QPushButton("Export Excel")
#         self.btn_delete_hist  = QPushButton("Xóa bảng")
#         self.btn_search_hist  = QPushButton("Tìm kiếm")
#         self.btn_hide_history = QPushButton("Tắt bảng lịch sử"); self.btn_hide_history.hide()
#         self._normalize_button(self.btn_show_history, self.btn_export_hist, self.btn_delete_hist, self.btn_search_hist, self.btn_hide_history)

#         self._apply_btn_style(self.btn_show_history, f"""
#         QPushButton {{ {common_btn} background:#E6F4EA; border:1px solid #cde9d6; }}
#         QPushButton:hover  {{ background:#d9efe0; }}
#         QPushButton:pressed{{ background:#ccead6; }}
#         QPushButton:disabled{{ background:#f1faf4; color:#777; border-color:#e3f5e9; }}
#         """)
#         self._apply_btn_style(self.btn_hide_history, f"""
#         QPushButton {{ {common_btn} background:#fff3bf; border:1px solid #f5c6c2; }}
#         QPushButton:hover  {{ background:#ffeda3; }}
#         QPushButton:pressed{{ background:#ffe788; }}
#         QPushButton:disabled{{ background:#fff9dc; color:#777; border-color:#ffefb3; }}
#         """)
#         self._apply_btn_style(self.btn_export_hist, f"""
#         QPushButton {{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         QPushButton:pressed{{ background:#c8deff; }}
#         QPushButton:disabled{{ background:#eef5ff; color:#777; border-color:#ddeaff; }}
#         """)
#         self._apply_btn_style(self.btn_delete_hist, f"""
#         QPushButton {{ {common_btn} background:#ffe0e0; border:1px solid #ffb3b3; }}
#         QPushButton:hover  {{ background:#ffd1d1; }}
#         QPushButton:pressed{{ background:#ffc2c2; }}
#         QPushButton:disabled{{ background:#fff2f2; color:#777; border-color:#ffdede; }}
#         """)
#         self._apply_btn_style(self.btn_search_hist, f"""
#         QPushButton {{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         QPushButton:pressed{{ background:#c8deff; }}
#         QPushButton:disabled{{ background:#eef5ff; color:#777; border-color:#ddeaff; }}
#         """)
#         row_cmd = QHBoxLayout()

#         # FIX: không dùng stretch để nút không kéo dài
#         row_cmd.addWidget(self.btn_search_hist)
#         row_cmd.addWidget(self.btn_export_hist)
#         row_cmd.addWidget(self.btn_delete_hist)
#         v_hist_btns.addWidget(self.btn_show_history)
#         v_hist_btns.addLayout(row_cmd)
#         v_hist_btns.addWidget(self.btn_hide_history)
#         vside.addWidget(gb_hist_btns)

#         vside.addStretch(1)
#         root.addWidget(side)

#         # RIGHT
#         right_container = QVBoxLayout()
#         self.main_view = QWidget()
#         main_layout = QVBoxLayout(self.main_view)

#         top = QHBoxLayout()
#         self.lbl_cam1 = QLabel(); self.lbl_cam1.setScaledContents(False)
#         self.lbl_cam2 = QLabel(); self.lbl_cam2.setScaledContents(False)
#         for lbl in (self.lbl_cam1, self.lbl_cam2):
#             lbl.setAlignment(Qt.AlignCenter)
#             lbl.setStyleSheet("background:#ffffff; border-radius:12px;")
#             # FIX: không đặt minimumSize theo PANEL_W/H; chỉ đặt chiều cao gợi ý
#             lbl.setMinimumHeight(220)
#             lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         cam1_card, self.cam1_title = self._make_card("Cam 1 (Vào)", self.lbl_cam1)
#         cam2_card, self.cam2_title = self._make_card("Cam 2 (Vào)", self.lbl_cam2)
#         top.addWidget(cam1_card, 1); top.addWidget(cam2_card, 1)
#         main_layout.addLayout(top)

#         bottom = QHBoxLayout()
#         self.lbl_scene = QLabel(); self.lbl_scene.setScaledContents(False); self.lbl_scene.setAlignment(Qt.AlignCenter); self.lbl_scene.setStyleSheet("background:#ffffff; border-radius:12px;"); self.lbl_scene.setMinimumHeight(220); self.lbl_scene.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.lbl_roi   = QLabel(); self.lbl_roi.setScaledContents(False);   self.lbl_roi.setAlignment(Qt.AlignCenter);   self.lbl_roi.setStyleSheet("background:#ffffff; border-radius:12px;");   self.lbl_roi.setMinimumHeight(220);   self.lbl_roi.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
#         # self.txt_match    = QLabel("")
#         # THAY THẾ: Dùng QLineEdit thay cho QLabel để có giao diện khung
#         self.txt_match = QLineEdit()
#         self.txt_match.setReadOnly(True) 
#         self.txt_match.setStyleSheet("color: #0000ff; font-weight: 700;")
#         r=0
#         info_layout.addWidget(QLabel("Ngày vào:"), r,0); info_layout.addWidget(self.txt_date_in, r,1)
#         info_layout.addWidget(QLabel("Giờ vào:"),  r,2); info_layout.addWidget(self.txt_time_in, r,3)
#         info_layout.addWidget(QLabel("Biển số vào:"), r,4); info_layout.addWidget(self.txt_plate_in, r,5); r+=1
#         info_layout.addWidget(QLabel("Ngày ra:"),  r,0); info_layout.addWidget(self.txt_date_out, r,1)
#         info_layout.addWidget(QLabel("Giờ ra:"),   r,2); info_layout.addWidget(self.txt_time_out, r,3)
#         info_layout.addWidget(QLabel("Biển số ra:"), r,4); info_layout.addWidget(self.txt_plate_out, r,5); r+=1
#         info_layout.addWidget(QLabel("So khớp biển số:"), r,0); info_layout.addWidget(self.txt_match, r,1,1,2)
#         main_layout.addWidget(self.info_group)

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
#         self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected)

#         # Sửa lỗi bảng xen kẽ: Tắt
#         self.tbl_hist.setAlternatingRowColors(False)

#         header.setSectionResizeMode(QHeaderView.Stretch)
#         hist_v.addWidget(self.tbl_hist)
#         hist_layout.addWidget(hist_group)

#         # ==================== MỚI: TẠO TRANG CHI TIẾT (DETAIL_VIEW) (INDEX 2) ====================
#         self.detail_view = QWidget()
#         detail_layout = QVBoxLayout(self.detail_view)
        
#         # 1. Hàng nút "Quay lại"
#         row_btn_back = QHBoxLayout()
#         self.btn_back_to_history = QPushButton("⬅ Quay lại bảng lịch sử")
#         self._normalize_button(self.btn_back_to_history)
#         self._apply_btn_style(self.btn_back_to_history, f"""
#         QPushButton {{ {common_btn} background:#f3f4f6; border:1px solid #e5e7eb; }}
#         QPushButton:hover  {{ background:#eef0f3; }}
#         """)
#         row_btn_back.addWidget(self.btn_back_to_history)
#         row_btn_back.addStretch(1)
#         detail_layout.addLayout(row_btn_back)

#         # 2. Hàng hiển thị 2 ảnh
#         row_images = QHBoxLayout()
#         self.lbl_detail_scene = QLabel(); self.lbl_detail_scene.setScaledContents(False); self.lbl_detail_scene.setAlignment(Qt.AlignCenter); self.lbl_detail_scene.setStyleSheet("background:#ffffff; border-radius:12px;"); self.lbl_detail_scene.setMinimumHeight(320); self.lbl_detail_scene.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.lbl_detail_roi = QLabel(); self.lbl_detail_roi.setScaledContents(False); self.lbl_detail_roi.setAlignment(Qt.AlignCenter); self.lbl_detail_roi.setStyleSheet("background:#ffffff; border-radius:12px;"); self.lbl_detail_roi.setMinimumHeight(320); self.lbl_detail_roi.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
#         detail_scene_card, _ = self._make_card("Ảnh Chụp Vào (Image_IN)", self.lbl_detail_scene)
#         detail_roi_card, _ = self._make_card("Ảnh Chụp Ra (Image_OUT)", self.lbl_detail_roi)
        
#         row_images.addWidget(detail_scene_card, 1)
#         row_images.addWidget(detail_roi_card, 1)
#         detail_layout.addLayout(row_images, 1) # Cho phép ảnh co giãn

#         # 3. Hộp thông tin chi tiết
#         gb_detail_info = QGroupBox("Thông tin Lượt Gửi")
#         gl_detail = QGridLayout(gb_detail_info)
        
#         self.lbl_detail_plate_in = QLineEdit(); self.lbl_detail_plate_in.setReadOnly(True)
#         self.lbl_detail_date_in = QLineEdit(); self.lbl_detail_date_in.setReadOnly(True)
#         self.lbl_detail_time_in = QLineEdit(); self.lbl_detail_time_in.setReadOnly(True)
#         self.lbl_detail_plate_out = QLineEdit(); self.lbl_detail_plate_out.setReadOnly(True)
#         self.lbl_detail_date_out = QLineEdit(); self.lbl_detail_date_out.setReadOnly(True)
#         self.lbl_detail_time_out = QLineEdit(); self.lbl_detail_time_out.setReadOnly(True)
#         self.lbl_detail_match = QLineEdit(); self.lbl_detail_match.setReadOnly(True)
        
#         self.lbl_detail_plate_in.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: 700;")
#         self.lbl_detail_plate_out.setStyleSheet("color: #ff0000; font-size: 14px; font-weight: 700;")
#         self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")
        
#         gl_detail.addWidget(QLabel("Biển số vào:"), 0, 0); gl_detail.addWidget(self.lbl_detail_plate_in, 0, 1)
#         gl_detail.addWidget(QLabel("Ngày vào:"), 1, 0); gl_detail.addWidget(self.lbl_detail_date_in, 1, 1)
#         gl_detail.addWidget(QLabel("Giờ vào:"), 2, 0); gl_detail.addWidget(self.lbl_detail_time_in, 2, 1)
        
#         gl_detail.addWidget(QLabel("Biển số ra:"), 0, 2); gl_detail.addWidget(self.lbl_detail_plate_out, 0, 3)
#         gl_detail.addWidget(QLabel("Ngày ra:"), 1, 2); gl_detail.addWidget(self.lbl_detail_date_out, 1, 3)
#         gl_detail.addWidget(QLabel("Giờ ra:"), 2, 2); gl_detail.addWidget(self.lbl_detail_time_out, 2, 3)
        
#         gl_detail.addWidget(QLabel("Trạng thái:"), 3, 0); gl_detail.addWidget(self.lbl_detail_match, 3, 1, 1, 3) # Kéo dài 3 cột

#         detail_layout.addWidget(gb_detail_info)
#         # ==================== HẾT PHẦN DETAIL VIEW ====================

#         # ==================== MỚI: TẠO TRANG TÌM KIẾM (SEARCH_FILTER_VIEW) (INDEX 3) ====================
#         self.search_filter_view = QWidget()
#         sfv_layout = QVBoxLayout(self.search_filter_view)
#         sfv_layout.setContentsMargins(20, 20, 20, 20)
#         sfv_layout.setSpacing(15)

#         # 1. Tiêu đề
#         sfv_title = QLabel("Bộ lọc tìm kiếm lịch sử")
#         sfv_title.setStyleSheet("font-size: 20px; font-weight: 700; color: #333;")
#         sfv_title.setAlignment(Qt.AlignCenter)
#         sfv_layout.addWidget(sfv_title)

#         # 2. Form chứa các bộ lọc
#         sfv_form = QFrame()
#         sfv_form.setStyleSheet("QFrame { background: #f9f9f9; border: 1px solid #eee; border-radius: 10px; }")
#         sfv_form_layout = QGridLayout(sfv_form)
#         sfv_form_layout.setContentsMargins(25, 25, 25, 25)
#         sfv_form_layout.setSpacing(15)

#         sfv_form_layout.addWidget(QLabel("Từ ngày:"), 0, 0)
#         self.sfv_dt_start = QDateTimeEdit(self)
#         self.sfv_dt_start.setDateTime(datetime.now().replace(hour=0, minute=0, second=0))
#         self.sfv_dt_start.setCalendarPopup(True)
#         self.sfv_dt_start.setDisplayFormat("dd/MM/yyyy HH:mm:ss")
#         self.sfv_dt_start.setFixedHeight(34)
#         sfv_form_layout.addWidget(self.sfv_dt_start, 0, 1)

#         sfv_form_layout.addWidget(QLabel("Đến ngày:"), 1, 0)
#         self.sfv_dt_end = QDateTimeEdit(self)
#         self.sfv_dt_end.setDateTime(datetime.now())
#         self.sfv_dt_end.setCalendarPopup(True)
#         self.sfv_dt_end.setDisplayFormat("dd/MM/yyyy HH:mm:ss")
#         self.sfv_dt_end.setFixedHeight(34)
#         sfv_form_layout.addWidget(self.sfv_dt_end, 1, 1)

#         sfv_layout.addWidget(sfv_form) # Thêm form vào layout chính

#         # 3. Hàng nút (Quay lại, Tìm kiếm)
#         sfv_row_btn = QHBoxLayout()
#         self.sfv_btn_back = QPushButton("Quay lại")
#         self.sfv_btn_search = QPushButton("Tìm kiếm")
#         self._normalize_button(self.sfv_btn_back, self.sfv_btn_search)
        
#         # Style cho nút
#         self._apply_btn_style(self.sfv_btn_back, f"""
#         QPushButton {{ {common_btn} background:#f3f4f6; border:1px solid #e5e7eb; }}
#         QPushButton:hover  {{ background:#eef0f3; }}
#         """)
#         self._apply_btn_style(self.sfv_btn_search, f"""
#         QPushButton {{ {common_btn} background:#e0ecff; border:1px solid #c7dcff; }}
#         QPushButton:hover  {{ background:#d4e5ff; }}
#         """)
        
#         sfv_row_btn.addWidget(self.sfv_btn_back)
#         sfv_row_btn.addStretch(1)
#         sfv_row_btn.addWidget(self.sfv_btn_search)
        
#         sfv_layout.addLayout(sfv_row_btn)
#         sfv_layout.addStretch(1) # Đẩy mọi thứ lên trên
#         # ==================== HẾT PHẦN SEARCH FILTER VIEW ====================

#         self.stacked = QStackedWidget()
#         self.stacked.addWidget(self.main_view)      # index 0
#         self.stacked.addWidget(self.history_view)   # index 1
#         self.stacked.addWidget(self.detail_view)    # index 2
#         self.stacked.addWidget(self.search_filter_view) # index 3
#         self.stacked.setCurrentIndex(0)

#         right_container.addWidget(self.stacked, 1)
#         root.addLayout(right_container, 1)

#         self.update_titles_and_modes()

#         # Kết nối các nút lịch sử sau khi tạo UI
#         self.btn_show_history.clicked.connect(self.show_history_view)
#         self.btn_hide_history.clicked.connect(self.show_main_view)
#         self.btn_export_hist.clicked.connect(self.on_export_excel)
#         self.btn_delete_hist.clicked.connect(self.on_delete_history)
#         self.btn_search_hist.clicked.connect(self.on_search_history_clicked)
        
#         # Kết nối các nút của trang Chi Tiết (Detail) và Tìm Kiếm (Search)
#         self.btn_back_to_history.clicked.connect(self.show_history_view_only)
#         self.sfv_btn_back.clicked.connect(self.show_main_view)
#         self.sfv_btn_search.clicked.connect(self.on_run_search_from_page)

#     # ---- 8.8 Cập nhật hướng làn và thông báo cho worker ----
#     def update_titles_and_modes(self):
#         self.cam1_title.setText(f"Cam 1 ({'Vào' if self.lane1_dir=='VÀO' else 'Ra'})")
#         self.cam2_title.setText(f"Cam 2 ({'Vào' if self.lane2_dir=='VÀO' else 'Ra'})")
#         if self.cam1_worker: self.cam1_worker.set_mode("in" if self.lane1_dir=="VÀO" else "out")
#         if self.cam2_worker: self.cam2_worker.set_mode("in" if self.lane2_dir=="VÀO" else "out")

#     # ---- 8.9 Đặt lại hướng làn mặc định ----
#     @Slot()
#     def on_reset_lanes(self):
#         self.lane1_dir = "VÀO"; self.lane2_dir = "VÀO"
#         self.one_way_toggle_vao = True; self.two_way_toggle = True
#         self.update_titles_and_modes()
#         self.show_logo(1); self.show_logo(2)

#     # ---- 8.10 Chuyển đổi chế độ một chiều ----
#     @Slot()
#     def on_one_way_clicked(self):
#         if self.one_way_toggle_vao: self.lane1_dir="VÀO"; self.lane2_dir="VÀO"
#         else:                       self.lane1_dir="RA";  self.lane2_dir="RA"
#         self.one_way_toggle_vao = not self.one_way_toggle_vao
#         self.update_titles_and_modes()

#     # ---- 8.11 Chuyển đổi chế độ hai chiều ----
#     @Slot()
#     def on_two_way_clicked(self):
#         if self.two_way_toggle: self.lane1_dir="VÀO"; self.lane2_dir="RA"
#         else:                   self.lane1_dir="RA";  self.lane2_dir="VÀO"
#         self.two_way_toggle = not self.two_way_toggle
#         self.update_titles_and_modes()

#     # ---- 8.xx Hàm này cần được viết trong class MainWindow và kết nối với worker ----
#     @Slot(str)
#     def update_match_status(self, status: str):
#         display_status = status.replace('-', ' ').title()
#         self.txt_match.setText(display_status) # <-- Cập nhật QLineEdit

#         if "Khop Bien So" in display_status:
#             # Xanh lá cây
#             self.txt_match.setStyleSheet("color: #007700; font-weight: 700;") 
#         elif "Khong Khop Bien So" in display_status:
#             # Đỏ
#             self.txt_match.setStyleSheet("color: #ff0000; font-weight: 700;")
#         else:
#             # Xanh dương (Mặc định/Chờ)
#             self.txt_match.setStyleSheet("color: #007700; font-weight: 700;")

#     # ---- 8.xx MỚI: Nhận tín hiệu và phát âm thanh ----
#     @Slot(str)
#     def on_play_sound(self, mode):
#         """Phát âm thanh dựa trên chế độ (in/out)"""
#         if mode == "in":
#             self.sound_in.play()
#         elif mode == "out":
#             self.sound_out.play()
#         else:
#             print(f"Lỗi: Không tìm thấy file âm thanh!")

#     # ---- 8.12 Xử lý thay đổi chế độ OCR ----
#     @Slot()
#     def on_ocr_mode_changed(self):
#         self.current_ocr_mode = "gemini" if (self.rb_gem.isChecked() and GEMINI_READY) else "yolo"
#         if self.rb_gem.isChecked() and not GEMINI_READY:
#             QMessageBox.information(self, "Gemini", "Chưa cấu hình GEMINI_API_KEY (.env hoặc biến môi trường). Sẽ dùng YOLO OCR.")
#             self.rb_yolo.setChecked(True); self.current_ocr_mode = "yolo"
#         if self.cam1_worker: self.cam1_worker.set_ocr_mode(self.current_ocr_mode)
#         if self.cam2_worker: self.cam2_worker.set_ocr_mode(self.current_ocr_mode)

#     # ---- 8.13 Hiển thị chế độ xem Lịch sử ----
#     def show_history_view(self):
#         self.btn_show_how_history_view_only() 
#         self.refresh_history_data()   

#     # ---- 8.14 Hiển thị chế độ xem Camera chính ----
#     def show_main_view(self):
#         self.stacked.setCurrentIndex(0) # Về trang chính (index 0)
#         self.btn_hide_history.hide()
#         self.btn_show_history.show()

#     # ---- 8.15 Xuất dữ liệu lịch sử ra Excel ----
#     @Slot()
#     def on_export_excel(self):
#         df = self.db.fetch_history_df(limit=10000) if (self.db and self.db.ok) else pd.DataFrame()
#         if not df.empty and "STT" in df.columns: df = df.drop(columns=["STT"])
#         if df.empty: QMessageBox.information(self, "Export", "Không có dữ liệu để export."); return
#         path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "history.xlsx", "Excel Files (*.xlsx)")
#         if not path: return
#         try: df.to_excel(path, index=False); QMessageBox.information(self, "Export", f"Đã xuất Excel:\n{path}")
#         except Exception as e: QMessageBox.warning(self, "Export", f"Lỗi khi xuất Excel:\n{e}")

#     # ---- 8.16 Xóa dữ liệu lịch sử ----
#     @Slot()
#     def on_delete_history(self):
#         if not (self.db and self.db.ok):
#             QMessageBox.warning(self, "Xóa", "Chưa kết nối DB, không thể xóa."); return
#         dlg = DeleteDialog(self)
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
#             self.db.delete_by_ids(ids); self.refresh_history_data()
#         elif res == 2:
#             self.db.delete_all(); self.refresh_history_data()
#         else:
#             return

#     # ---- 8.17 image helpers ----
#     def qpix_logo(self):
#         if os.path.exists(LOGO_PATH):
#             return QPixmap(LOGO_PATH)
#         return QPixmap.fromImage(bgr_to_qimage(letterbox(None)))

#     # ---- 8.18 Hiển thị logo/ảnh mặc định trên camera ----
#     def show_logo(self, which: int):
#         pm = self._logo_pm
#         if which == 1: self._set_centered_pixmap(self.lbl_cam1, pm)
#         else:          self._set_centered_pixmap(self.lbl_cam2, pm)

#     # ---- 8.19 Nhận và hiển thị khung hình video ----
#     @Slot(np.ndarray, str)
#     def on_frame(self, frame_bgr, title):
#         sender = self.sender()
#         if sender is self.cam1_worker:
#             self._set_centered_pixmap(self.lbl_cam1, frame_bgr)
#         elif sender is self.cam2_worker:
#             self._set_centered_pixmap(self.lbl_cam2, frame_bgr)

#     # ---- 8.20 Nhận và hiển thị khung hình scene ----
#     @Slot(str)
#     def on_scene(self, path):
#         if os.path.exists(path):
#             bgr = cv2.imread(path)
#             self._set_centered_pixmap(self.lbl_scene, bgr)

#     # ---- 8.21 Nhận và hiển thị khung hình ROI ----
#     @Slot(str, str)
#     def on_roi(self, roi_path, mode):
#         if os.path.exists(roi_path):
#             bgr = cv2.imread(roi_path)
#             self._set_centered_pixmap(self.lbl_roi, bgr)

#     # ---- 8.22 Nhận và hiển thị thông tin xe ----
#     @Slot(dict)
#     def on_info(self, info):
#         if "date_in" in info:  self.txt_date_in.setText(info["date_in"]);   self.ed_date_in.setText(info["date_in"])
#         if "time_in" in info:  self.txt_time_in.setText(info["time_in"]);   self.ed_time_in.setText(info["time_in"])
#         if "plate_text_in" in info: self.txt_plate_in.setText(info["plate_text_in"]); self.ed_plate_in.setText(info["plate_text_in"])
#         if "date_out" in info: self.txt_date_out.setText(info["date_out"]); self.ed_date_out.setText(info["date_out"])
#         if "time_out" in info: self.txt_time_out.setText(info["time_out"]); self.ed_time_out.setText(info["time_out"])
#         if "plate_text_out" in info: self.txt_plate_out.setText(info["plate_text_out"]); self.ed_plate_out.setText(info["plate_text_out"])

#     # ---- 8.23 Nhận và hiển thị thông tin so khớp ----
#     @Slot(str)
#     def on_match(self, txt): 
#         self.txt_match.setText(txt.upper())

#     # ---- 8.24 Tải và cập nhật bảng lịch sử ----
#     @Slot()
#     def refresh_history_data(self, start_time=None, end_time=None):
#         """Hàm chính tải dữ liệu từ DB và cập nhật bảng"""
#         # 1. Tải dữ liệu (dùng hàm fetch_history_df đã sửa ở lần trước)
#         df = self.db.fetch_history_df(limit=10000, start_time=start_time, end_time=end_time) if (self.db and self.db.ok) else pd.DataFrame()
        
#         # 2. Lưu df để dùng cho việc nhấp chuột (rất quan trọng)
#         # Hàm này phải trả về cột 'STT'
#         self.history_df = df.copy() 
        
#         # 3. Chuẩn bị dữ liệu để hiển thị (bỏ cột STT khỏi view)
#         df_display = df.copy()
#         if not df_display.empty and "STT" in df_display.columns: 
#             df_display = df_display.drop(columns=["STT"])
        
#         # 4. Cập nhật QTableWidget
#         if df_display.empty:
#             self.tbl_hist.setRowCount(0)
#             cols = ["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
#             self.tbl_hist.setColumnCount(len(cols)); self.tbl_hist.setHorizontalHeaderLabels(cols)
#             hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#             self.tbl_hist.horizontalHeader().setFont(hfont); return

#         cols = list(df_display.columns)
#         self.tbl_hist.setRowCount(len(df_display)); self.tbl_hist.setColumnCount(len(cols))
#         self.tbl_hist.setHorizontalHeaderLabels(cols)
#         hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True)
#         self.tbl_hist.horizontalHeader().setFont(hfont)
#         self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
#         for i in range(len(df_display)):
#             for j, col in enumerate(cols):
#                 val = str(df_display.iloc[i, j]); item = QTableWidgetItem(val)
#                 item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
#                 self.tbl_hist.setItem(i, j, item)

#     # ---- 8.25 camera controls ----
#     def _connect_worker(self, w: VideoWorker):
#         w.frameSignal.connect(self.on_frame)
#         w.sceneSignal.connect(self.on_scene)
#         w.roiSignal.connect(self.on_roi)
#         w.infoSignal.connect(self.on_info)
#         w.matchSignal.connect(self.on_match)
#         w.histSignal.connect(self.on_history_signal_refresh)
#         w.playSoundSignal.connect(self.on_play_sound)

#     # ---- 8.xx MỚI: Slot trung gian cho Timer/Worker ----
#     @Slot()
#     def on_history_signal_refresh(self):
#         """Slot này nhận tín hiệu từ worker/timer và gọi hàm tải dữ liệu chính"""
#         # Chỉ làm mới nếu tab lịch sử đang được xem
#         if self.stacked.currentIndex() == 1:
#             # Tải mà không có bộ lọc thời gian
#             self.refresh_history_data() 

#     # ---- 8.xx MỚI: Chỉ chuyển tab lịch sử (không tải lại data) ----
#     @Slot()
#     def show_history_view_only(self):
#         """Chỉ chuyển tab, không tải lại dữ liệu"""
#         self.stacked.setCurrentIndex(1) # Chuyển về tab bảng (index 1)
#         self.btn_show_history.hide()
#         self.btn_hide_history.show()

#     # ---- 8.xx MỚI: Xử lý sự kiện nhấn 'Tìm kiếm' TỪ TRANG LỌC (INDEX 3) ----
#     @Slot()
#     def on_run_search_from_page(self):
#         """Lấy ngày giờ từ trang lọc, tìm kiếm và chuyển sang bảng kết quả"""
#         # 1. Lấy giá trị từ các QDateTimeEdit trên trang (index 3)
#         start_dt = self.sfv_dt_start.dateTime().toPython()
#         end_dt = self.sfv_dt_end.dateTime().toPython()

#         # 2. Gọi hàm tải dữ liệu (hàm đã sửa)
#         self.refresh_history_data(start_time=start_dt, end_time=end_dt)

#         # 3. Chuyển sang trang kết quả (bảng lịch sử, index 1)
#         self.stacked.setCurrentIndex(1)

#     # ---- 8.xx MỚI: Xử lý sự kiện nhấp vào nút Tìm kiếm (THANH BÊN) ----
#     @Slot()
#     def on_search_history_clicked(self):
#         """Mở trang bộ lọc tìm kiếm (index 3)"""
#         self.stacked.setCurrentIndex(3)
#         # Cập nhật nút ở thanh bên
#         self.btn_show_history.hide()
#         self.btn_hide_history.show()

#     # ---- 8.xx MỚI: Xử lý sự kiện nhấp vào hàng trong bảng ----
#     @Slot()
#     def on_history_row_selected(self):
#         selected_items = self.tbl_hist.selectedItems()
#         # Đảm bảo history_df đã được tải và có cột STT
#         if not selected_items or self.history_df.empty or "STT" not in self.history_df.columns:
#             return

#         try:
#             # Lấy chỉ số hàng đang xem
#             row_index_view = selected_items[0].row() 
            
#             # Lấy STT từ QTableWidget (cột đầu tiên của df là STT)
#             stt_col_index = 0 # Vì df_display đã bỏ cột STT, nên ID là cột 0
            
#             # SỬA LẠI: Chúng ta phải tìm cột STT trong self.history_df
#             if "STT" not in self.history_df.columns:
#                  print("Lỗi: self.history_df không có cột STT")
#                  return
                 
#             stt_col_view_index = list(self.history_df.columns).index("STT")
            
#             # Lấy STT từ QTableWidget
#             # Chú ý: self.tbl_hist KHÔNG có cột STT, nó chỉ có trong self.history_df
#             # Chúng ta phải lấy ID (cột 0 trong table) và tìm trong df
            
#             id_item = self.tbl_hist.item(row_index_view, 0) # Cột 0 là ID
#             if not id_item: return
            
#             row_id = int(id_item.text())

#             # Tìm hàng trong DataFrame gốc dựa trên ID
#             row_data_series = self.history_df[self.history_df['ID'] == row_id]
#             if row_data_series.empty: return
            
#             row_data = row_data_series.iloc[0] # Lấy dữ liệu của hàng đó

#             # 1. Cập nhật thông tin trên TRANG CHI TIẾT MỚI (index 2)
#             self.lbl_detail_plate_in.setText(str(row_data.get("Biển số vào", "")))
#             self.lbl_detail_date_in.setText(str(row_data.get("Ngày vào", "")))
#             self.lbl_detail_time_in.setText(str(row_data.get("Giờ vào", "")))
            
#             self.lbl_detail_plate_out.setText(str(row_data.get("Biển số ra", "")))
#             self.lbl_detail_date_out.setText(str(row_data.get("Ngày ra", "")))
#             self.lbl_detail_time_out.setText(str(row_data.get("Giờ ra", "")))
            
#             match_status = str(row_data.get("Trạng thái", "")).replace('-', ' ').title()
#             self.lbl_detail_match.setText(match_status)
            
#             # Đổi màu chữ trạng thái
#             if "Khop Bien So" in match_status:
#                 self.lbl_detail_match.setStyleSheet("color: #007700; font-weight: 700;")
#             elif "Khong Khop Bien So" in match_status:
#                 self.lbl_detail_match.setStyleSheet("color: #ff0000; font-weight: 700;")
#             else:
#                 self.lbl_detail_match.setStyleSheet("color: #0000ff; font-weight: 700;")

#             # 2. Cập nhật hình ảnh trên TRANG CHI TIẾT MỚI
#             img_in_path = str(row_data.get("Ảnh vào", ""))
#             img_out_path = str(row_data.get("Ảnh ra", "")) # Ảnh chụp lúc ra

#             if img_in_path and os.path.exists(img_in_path):
#                 bgr_in = cv2.imread(img_in_path)
#                 self._set_centered_pixmap(self.lbl_detail_scene, bgr_in)
#             else:
#                 self._set_centered_pixmap(self.lbl_detail_scene, self.qpix_logo())

#             if img_out_path and os.path.exists(img_out_path):
#                 bgr_out = cv2.imread(img_out_path)
#                 self._set_centered_pixmap(self.lbl_detail_roi, bgr_out)
#             else:
#                 self._set_centered_pixmap(self.lbl_detail_roi, self.qpix_logo())
                
#             # 3. Chuyển sang TRANG CHI TIẾT (index 2)
#             self.stacked.setCurrentIndex(2)
            
#         except Exception as e:
#             print(f"Lỗi khi chọn hàng: {e}")
#             import traceback
#             traceback.print_exc()

#     # ---- 8.26 Hàm chung để khởi động camera (1 hoặc 2) ----
#     def start_cam_generic(self, which: int):
#         if not self.models.ok:
#             QMessageBox.warning(self, "Model error", f"Không load được model:\n{self.models.err}")
#             return
#         if which == 1 and self.cam1_worker and self.cam1_worker.isRunning(): return
#         if which == 2 and self.cam2_worker and self.cam2_worker.isRunning(): return

#         ocr_mode = self.current_ocr_mode
#         default_api = API_MAP["DSHOW(Windows)"] 

#         if which == 1:
#             idx = int(self.spin_cam1.value())
#             mode = "in" if self.lane1_dir=="VÀO" else "out"
#             title = f"1) Cam 1 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam1_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam1_worker); self.cam1_worker.start()
#         else:
#             idx = int(self.spin_cam2.value())
#             mode = "in" if self.lane2_dir=="VÀO" else "out"
#             title = f"2) Cam 2 ({'Vào' if mode=='in' else 'Ra'})"
#             self.cam2_worker = VideoWorker(idx, default_api, mode, self.models, self.db, 1.2, ocr_mode=ocr_mode, title=title)
#             self._connect_worker(self.cam2_worker); self.cam2_worker.start()

#     # ---- 8.27 Hàm chung để dừng camera (1 hoặc 2) ----
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

#     # ---- 8.28 Xử lý sự kiện đóng cửa sổ ----
#     def closeEvent(self, event):
#         try: self.stop_cam_generic(1); self.stop_cam_generic(2)
#         except: pass
#         super().closeEvent(event)

# # ==================== 9. MAIN ====================
# def main():
#     QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
#     app = QApplication(sys.argv)
#     app.setStyle("Fusion")
#     w = MainWindow(); 
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

import os, sys, time, cv2, numpy as np, pandas as pd
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
    ts = datetime.now().strftime("%Ym%d_%H%M%S_%f")
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
                         status_filter=None, plate_filter=None) -> pd.DataFrame: # Thêm tham số mới
        columns = [
            "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
            "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
        ]
        if not self.ok:
            return pd.DataFrame(columns=["STT"] + columns)

        try:
            sql = f"""
                SELECT TOP ({limit})
                    id, image_in, plate_in, date_in, time_in,
                    image_out, plate_out, date_out, time_out, match_status
                FROM dbo.ParkingSessions
            """
            where_clauses = []
            sql_params = []

            # Lọc thời gian
            if start_time:
                where_clauses.append("created_at >= ?")
                sql_params.append(start_time)
            if end_time:
                where_clauses.append("created_at <= ?")
                sql_params.append(end_time)

            # Lọc trạng thái (MỚI)
            if status_filter: # Nếu danh sách status không rỗng
                # Tạo placeholder dạng (?, ?, ?)
                status_placeholders = ','.join('?' for _ in status_filter)
                where_clauses.append(f"match_status IN ({status_placeholders})")
                sql_params.extend(status_filter) # Thêm các status vào danh sách tham số

            # Lọc biển số (MỚI)
            if plate_filter: # Nếu có nhập biển số
                # Tìm kiếm gần đúng ở cả biển vào và biển ra
                where_clauses.append("(plate_in LIKE ? OR plate_out LIKE ?)")
                # Thêm dấu % vào đầu và cuối để tìm kiếm LIKE
                like_term = f"%{plate_filter}%"
                sql_params.append(like_term) # Cho plate_in LIKE ?
                sql_params.append(like_term) # Cho plate_out LIKE ?

            # Ghép mệnh đề WHERE
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY id DESC"

            # Thực thi truy vấn
            rows = self.cur.execute(sql, tuple(sql_params)).fetchall()

            # Tạo DataFrame
            df = pd.DataFrame.from_records(rows, columns=columns).astype(object).where(pd.notnull, "")
            df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"}) # Xử lý trạng thái NULL/rỗng
            df.insert(0, "STT", range(1, len(df)+1)) # Thêm cột STT
            return df

        except Exception as e:
            print(f"fetch_history error: {e}")
            import traceback
            traceback.print_exc() # In chi tiết lỗi
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
            model = genai.GenerativeModel('gemini-1.5-flash')
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
        self.history_view = QWidget(); hist_layout = QVBoxLayout(self.history_view)
        hist_group = QGroupBox("Bảng lịch sử (ParkingSessions)"); hist_v = QVBoxLayout(hist_group)
        self.tbl_hist = QTableWidget(0, 10); self.tbl_hist.setHorizontalHeaderLabels(["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"])
        header = self.tbl_hist.horizontalHeader(); hfont = QFont(header.font()); hfont.setBold(True); header.setFont(hfont)
        self.tbl_hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.tbl_hist.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.tbl_hist.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.tbl_hist.itemSelectionChanged.connect(self.on_history_row_selected); self.tbl_hist.setAlternatingRowColors(False); header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hist_v.addWidget(self.tbl_hist); hist_layout.addWidget(hist_group)

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

    # ---- 8.13 Hiển thị chế độ xem Lịch sử (THÊM PRINT ĐỂ DEBUG) ----
    def show_history_view(self):
        """Hàm này được gọi bởi nút 'Xem bảng lịch sử'"""

        # ***** THÊM DÒNG PRINT NÀY *****
        print("\n!!!!!!!!!! WARNING: show_history_view CALLED UNEXPECTEDLY? !!!!!!!!!\n")

        # ***** XÓA BỘ LỌC HIỆN TẠI *****
        self.current_filter_start = None
        self.current_filter_end = None
        self.current_filter_status = None
        self.current_filter_plate = None

        # Chuyển tab (gọi đúng tên hàm)
        self.show_history_view_only()
        # Tải lại toàn bộ dữ liệu (không có bộ lọc)
        self.refresh_history_data()

    # ---- 8.xx MỚI: Slot cho nút "Xem bảng lịch sử" (THÊM PRINT) ----
    @Slot()
    def on_show_all_history_clicked(self):
        """Slot này được kết nối với btn_show_history. Nó xóa bộ lọc và tải lại."""
        # ***** THÊM DÒNG PRINT NÀY *****
        print("\n!!!!!!!!!! DEBUG: on_show_all_history_clicked CALLED !!!!!!!!!\n")

        # 1. Xóa bộ lọc hiện tại
        self.current_filter_start = None
        self.current_filter_end = None
        self.current_filter_status = None
        self.current_filter_plate = None

        # 2. Chuyển sang tab lịch sử (nếu chưa ở đó)
        if self.stacked.currentIndex() != 1:
             self.show_history_view_only() # Chỉ chuyển tab và đổi nút

        # 3. Tải lại toàn bộ dữ liệu (không có bộ lọc)
        self.refresh_history_data()

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
        if not (self.db and self.db.ok): QMessageBox.warning(self, "Xóa", "Chưa kết nối DB."); return
        dlg = DeleteDialog(self); g = self.geometry(); dlg.adjustSize(); dlg.move(self.mapToGlobal(QPoint(g.width()-dlg.width()-40, 140)))
        res = dlg.exec()
        ids_to_delete = []
        if res == 1: # Xóa dòng chọn
            rows_view = sorted(set([idx.row() for idx in self.tbl_hist.selectedIndexes()]))
            if not rows_view: QMessageBox.information(self, "Xóa", "Bạn chưa chọn dòng nào."); return
            # Lấy ID từ bảng hiển thị
            for r_view in rows_view:
                 id_item = self.tbl_hist.item(r_view, 0) # Cột 0 là ID
                 if id_item: ids_to_delete.append(id_item.text())
            if not ids_to_delete: QMessageBox.warning(self, "Xóa", "Không lấy được ID các dòng chọn."); return
            self.db.delete_by_ids(ids_to_delete)
        elif res == 2: # Xóa tất cả
            confirm = QMessageBox.question(self, "Xác nhận", "Bạn chắc chắn muốn xóa TOÀN BỘ lịch sử?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.Yes:
                 self.db.delete_all()
            else: return # Hủy nếu không chọn Yes
        else: return # Hủy dialog
        # Tải lại dữ liệu sau khi xóa
        self.refresh_history_data()


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


    # ---- 8.24 Tải và cập nhật bảng lịch sử (ĐÃ CẬP NHẬT ĐỂ LỌC) ----
    @Slot()
    # Thêm tham số status_filter và plate_filter
    def refresh_history_data(self, start_time=None, end_time=None, status_filter=None, plate_filter=None):
        """Hàm chính tải dữ liệu từ DB và cập nhật bảng"""

        # ***** IN RA BỘ LỌC MÀ HÀM NÀY NHẬN ĐƯỢC *****
        print(f"+++ Refreshing with: Start={start_time}, End={end_time}, Status={status_filter}, Plate={plate_filter} +++")
        # 1. Tải dữ liệu (truyền các bộ lọc xuống DB)
        df = self.db.fetch_history_df(limit=10000, start_time=start_time, end_time=end_time,
                                     status_filter=status_filter, plate_filter=plate_filter) if (self.db and self.db.ok) else pd.DataFrame()

        # 2. Lưu df gốc (có STT)
        self.history_df = df.copy()

        # 3. Chuẩn bị df hiển thị (bỏ STT)
        df_display = df.copy()
        if not df_display.empty and "STT" in df_display.columns:
            df_display = df_display.drop(columns=["STT"])

        # 4. Cập nhật QTableWidget (code giữ nguyên)
        if df_display.empty:
            self.tbl_hist.setRowCount(0); cols = ["ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào","Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"]
            self.tbl_hist.setColumnCount(len(cols)); self.tbl_hist.setHorizontalHeaderLabels(cols)
            hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True); self.tbl_hist.horizontalHeader().setFont(hfont); return
        cols = list(df_display.columns)
        self.tbl_hist.setRowCount(len(df_display)); self.tbl_hist.setColumnCount(len(cols)); self.tbl_hist.setHorizontalHeaderLabels(cols)
        hfont = QFont(self.tbl_hist.horizontalHeader().font()); hfont.setBold(True); self.tbl_hist.horizontalHeader().setFont(hfont)
        # self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # === THAY THẾ DÒNG TRÊN BẰNG KHỐI CODE NÀY ===
        header = self.tbl_hist.horizontalHeader()
        # Cho phép cột ID, Ngày, Giờ, Trạng thái tự điều chỉnh theo nội dung
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # ID
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Ngày vào
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents) # Giờ vào
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents) # Ngày ra
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents) # Giờ ra
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.ResizeToContents) # Trạng thái
        # header.setSectionResizeMode(10, QHeaderView.ResizeMode.ResizeToContents) # Thời gian Lưu DB (nếu có)

        # Các cột còn lại (Ảnh vào, Biển số vào, Ảnh ra, Biển số ra) sẽ tự động kéo giãn để lấp đầy
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch) # Ảnh vào
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch) # Biển số vào
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch) # Ảnh ra
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch) # Biển số ra
        # === KẾT THÚC THAY THẾ ===
        for i in range(len(df_display)):
            for j, col in enumerate(cols):
                val = str(df_display.iloc[i, j]); item = QTableWidgetItem(val)
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable); self.tbl_hist.setItem(i, j, item)


    # ---- 8.xx MỚI: Slot trung gian cho Timer/Worker (THÊM PRINT ĐỂ DEBUG) ----
    @Slot()
    def on_history_signal_refresh(self):
        """Slot này nhận tín hiệu từ worker/timer và gọi hàm tải dữ liệu chính, SỬ DỤNG bộ lọc đã lưu (nếu có)"""
        # Chỉ làm mới nếu tab lịch sử đang được xem
        if self.stacked.currentIndex() == 1:
            # ***** IN RA BỘ LỌC HIỆN TẠI TRƯỚC KHI LÀM MỚI *****
            print("--- Timer Refresh ---")
            print(f"Current Start: {self.current_filter_start}")
            print(f"Current End: {self.current_filter_end}")
            print(f"Current Status: {self.current_filter_status}")
            print(f"Current Plate: {self.current_filter_plate}")
            print("---------------------")

            # Gọi refresh_history_data với các bộ lọc đã lưu
            self.refresh_history_data(start_time=self.current_filter_start,
                                     end_time=self.current_filter_end,
                                     status_filter=self.current_filter_status,
                                     plate_filter=self.current_filter_plate)

    # ---- 8.xx MỚI: Xử lý sự kiện nhấn 'Tìm kiếm' TỪ TRANG LỌC (INDEX 3) - (THÊM PRINT CHI TIẾT) ----
    @Slot()
    def on_run_search_from_page(self):
        """Lấy ngày giờ, trạng thái, biển số từ trang lọc, kiểm tra, LƯU BỘ LỌC, tìm kiếm và chuyển sang bảng kết quả"""
        print(">>> Entering on_run_search_from_page") # Print 1

        # 1. Lấy giá trị Ngày/Giờ và ghép thành datetime
        qdate_start = self.sfv_date_start.date(); qtime_start = self.sfv_time_start.time()
        qdate_end = self.sfv_date_end.date(); qtime_end = self.sfv_time_end.time()
        start_dt = QDateTime(qdate_start, qtime_start).toPython()
        end_dt = QDateTime(qdate_end, qtime_end).toPython()

        # 2. Kiểm tra ràng buộc ngày giờ
        if start_dt > end_dt:
            QMessageBox.warning(self, "Lỗi nhập liệu", "'Từ ngày/giờ' không được lớn hơn 'Đến ngày/giờ'.\nVui lòng kiểm tra lại.")
            print("<<< Exiting on_run_search_from_page (Date Error)") # Print Exit Point
            return

        # 3. Lấy giá trị bộ lọc Trạng thái
        selected_statuses = []
        if self.sfv_chk_pending.isChecked(): selected_statuses.append("PENDING")
        if self.sfv_chk_match.isChecked(): selected_statuses.append("KHOP-BIEN-SO")
        if self.sfv_chk_mismatch.isChecked(): selected_statuses.append("KHONG-KHOP-BIEN-SO")

        # 4. Lấy giá trị bộ lọc Biển số
        plate_text = self.sfv_txt_plate.text().strip()

        # 5. LƯU LẠI BỘ LỌC HIỆN TẠI
        self.current_filter_start = start_dt
        self.current_filter_end = end_dt
        self.current_filter_status = selected_statuses if selected_statuses else None
        self.current_filter_plate = plate_text if plate_text else None

        # ***** PRINT NGAY SAU KHI LƯU *****
        print(">>> Filters JUST SET in on_run_search:") # Print 2
        print(f"    Start: {self.current_filter_start}")
        print(f"    End: {self.current_filter_end}")
        print(f"    Status: {self.current_filter_status}")
        print(f"    Plate: {self.current_filter_plate}")

        # 6. Gọi hàm tải dữ liệu với ĐẦY ĐỦ bộ lọc
        print(">>> Calling refresh_history_data...") # Print 3
        self.refresh_history_data(start_time=self.current_filter_start,
                                 end_time=self.current_filter_end,
                                 status_filter=self.current_filter_status,
                                 plate_filter=self.current_filter_plate)
        print(">>> Returned from refresh_history_data.") # Print 4

        # ***** PRINT NGAY TRƯỚC KHI CHUYỂN TAB *****
        print(">>> Filters BEFORE setCurrentIndex(1):") # Print 5
        print(f"    Start: {self.current_filter_start}")
        print(f"    End: {self.current_filter_end}")
        print(f"    Status: {self.current_filter_status}")
        print(f"    Plate: {self.current_filter_plate}")

        # 7. Chuyển sang trang kết quả (bảng lịch sử, index 1)
        self.stacked.setCurrentIndex(1)
        print("<<< Exiting on_run_search_from_page (Success)") # Print 6

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