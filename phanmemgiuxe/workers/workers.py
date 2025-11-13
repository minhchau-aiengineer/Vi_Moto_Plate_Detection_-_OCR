import time, cv2, traceback
import numpy as np, pandas as pd
from datetime import datetime
from PySide6.QtCore import QThread, Signal
from ..models.models import Models, GEMINI_READY
from ..database.database import DB
from ..utils.utils import save_image





class VideoWorker(QThread):
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






    def set_title(self, title: str): 
        self.title = title
    def set_ocr_mode(self, mode: str): 
        self.ocr_mode = mode
    def set_mode(self, mode: str): 
        self.mode = mode





    def run(self):
        self._running = True
        try: 
             self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
             if not (self.cap and self.cap.isOpened()):
                  print(f"Lỗi: Không thể mở camera index {self.cam_idx} với API {self.api}")
                  self._running = False; 
                  return
        except Exception as e:
             print(f"Lỗi khi khởi tạo VideoCapture: {e}")
             self._running = False; 
             return

        try: 
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: 
            pass

        try: 
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except: 
            pass

        while self._running:
            try: 
                ok, frame = self.cap.read()
                if not ok or frame is None: 
                    self.stable_start = 0.0; self.captured = False
                    time.sleep(0.05); 
                    continue 

                # Gửi frame gốc lên UI
                self.frameSignal.emit(frame, self.title)

                # Phát hiện biển số
                plates, boxed_frame = self.models.detect_plates(frame)

                if not plates:
                    self.stable_start = 0.0; self.captured = False
                    time.sleep(0.01); 
                    continue

                # Chọn biển số tốt nhất (ví dụ: lớn nhất)
                best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
                roi_current = best[1]
                if roi_current is None or roi_current.size == 0: 
                     self.stable_start = 0.0; self.captured = False
                     time.sleep(0.01); 
                     continue

                # Logic ổn định
                if self.stable_start == 0.0: 
                     self.stable_start = time.time()
                elif self.captured: 
                     self.stable_start = time.time(); 
                     self.captured = False

                # Đủ thời gian ổn định và chưa chụp
                if (not self.captured) and (time.time() - self.stable_start >= self.stable_seconds):
                    
                    # Lưu ảnh (nên dùng ảnh đã vẽ hộp)
                    scene_img_to_save = boxed_frame if boxed_frame is not None else frame
                    scene_path = save_image(scene_img_to_save, "scene_in_boxed" if self.mode=="in" else "scene_out_boxed")
                   
                    # Lưu ROI
                    roi_path   = save_image(roi_current, "plate_in_roi" if self.mode=="in" else "plate_out_roi")

                    # Kiểm tra lưu ảnh thành công
                    if not scene_path or not roi_path:
                        print("Lỗi: Không thể lưu ảnh scene hoặc roi.")
                        self.captured = True
                        self.stable_start = 0.0 
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
                        self.sceneSignal.emit(scene_path) 
                        self.roiSignal.emit(roi_path, self.mode)

                        # Xử lý logic vào/ra và DB
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

                        self.captured = True # Đánh dấu đã chụp thành công
                        self.stable_start = 0.0 # Reset timer sau khi chụp thành công

            except Exception as e:
                 print(f"Lỗi trong vòng lặp VideoWorker: {e}")
                 import traceback
                 traceback.print_exc()
                 self.stable_start = 0.0 
                 self.captured = False
                 time.sleep(0.1) 

            time.sleep(0.01) 

        # Dọn dẹp khi dừng luồng
        try:
            if self.cap: self.cap.release()
        except Exception as e:
             print(f"Lỗi khi release camera: {e}")



    def stop(self): self._running = False













class HistoryLoaderWorker(QThread):
    resultReady = Signal(pd.DataFrame) 

    def __init__(self, db: DB, start_time=None, end_time=None, status_filter=None, plate_filter=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.start_time = start_time
        self.end_time = end_time
        self.status_filter = status_filter
        self.plate_filter = plate_filter





    def run(self):
        df = pd.DataFrame() 
        print("==> HistoryLoaderWorker bắt đầu chạy...")
        try:
             if self.db and self.db.ok:
                  df = self.db.fetch_history_df(limit=800,
                                             start_time=self.start_time,
                                             end_time=self.end_time,
                                             status_filter=self.status_filter,
                                             plate_filter=self.plate_filter)
        except Exception as e:
             print(f"==> Lỗi trong HistoryLoaderWorker.run: {e}")
             traceback.print_exc() 
        finally:
             self.resultReady.emit(df if df is not None else pd.DataFrame())
             print("==> HistoryLoaderWorker đã chạy xong.")











