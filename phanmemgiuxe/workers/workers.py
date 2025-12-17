import time, cv2, traceback
import numpy as np, pandas as pd
from typing import Optional
from datetime import datetime
from PySide6.QtCore import QThread, Signal
from ..models.models import Models, GEMINI_READY
from ..database.database import DB
from ..utils.utils import save_image





# ===== Video Processing Worker ======
class VideoWorker(QThread):
    frameSignal = Signal(np.ndarray, str)
    sceneSignal = Signal(str)
    roiSignal   = Signal(str, str)
    infoSignal  = Signal(dict)
    matchSignal = Signal(str)
    histSignal  = Signal()
    playSoundSignal = Signal(str)




    # === Init worker with params ===
    def __init__(
        self,
        cam_idx: int,
        api: int,
        mode: str,
        models: Models,
        db: Optional[DB],
        stable_seconds: float = 1.2,
        ocr_mode: str = "yolo",
        title: str = "",
        parent=None,
        *,
        camera_type: str = "WEBCAM",
        full_url: str = "",
    ):
        super().__init__(parent)
        self.cam_idx = cam_idx
        self.api = api
        self.mode = mode
        self.models = models
        self.db: Optional[DB] = db
        self.stable_seconds = stable_seconds
        self.ocr_mode = ocr_mode
        self.title = title or ("Cam 1" if self.mode=="in" else "Cam 2")

        # ===== mới =====
        self.camera_type = (camera_type or "WEBCAM").upper()
        self.full_url = (full_url or "").strip()

        self._running = False
        self.cap = None
        self.stable_start = 0.0
        self.captured = False


    
    
    
    
    # === Setters ===
    def set_title(self, title: str):
        self.title = title

    def set_ocr_mode(self, mode: str):
        self.ocr_mode = mode

    def set_mode(self, mode: str):
        self.mode = mode

    
    
    
    
    
    # === Mở nguồn camera ===
    def _open_capture(self) -> bool:
        """Mở camera theo camera_type, tối ưu timeout và log thời gian từng bước."""
        import time
        try:
            t0 = time.time()
            if self.camera_type == "WEBCAM":
                print(f"[VideoWorker] [{datetime.now()}] Bắt đầu mở WEBCAM index={self.cam_idx}, api={self.api}")
                self.cap = cv2.VideoCapture(int(self.cam_idx), self.api)
            else:
                print(f"[VideoWorker] [{datetime.now()}] Bắt đầu mở IP camera: {self.full_url}")
                if not self.full_url:
                    print(f"[VideoWorker] [{datetime.now()}] full_url rỗng")
                    return False
                self.cap = cv2.VideoCapture(self.full_url)

            t1 = time.time()
            print(f"[VideoWorker] [{datetime.now()}] Khởi tạo VideoCapture mất {t1-t0:.2f}s")

            # Tối ưu: chỉ thử mở tối đa 2 giây
            opened = False
            start = time.time()
            for _ in range(20):
                if self.cap and self.cap.isOpened():
                    opened = True
                    break
                time.sleep(0.1)
                if time.time() - start > 2:
                    break
            t2 = time.time()
            print(f"[VideoWorker] [{datetime.now()}] Kiểm tra mở camera mất {t2-t1:.2f}s")

            if not opened:
                print(f"[VideoWorker] [{datetime.now()}] Không mở được nguồn camera (timeout)")
                if self.cap:
                    self.cap.release()
                self.cap = None
                return False

            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            try:
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

            t3 = time.time()
            print(f"[VideoWorker] [{datetime.now()}] Set thuộc tính camera mất {t3-t2:.2f}s")

            print(f"[VideoWorker] [{datetime.now()}] Tổng thời gian mở camera: {t3-t0:.2f}s")
            return True
        except Exception as e:
            print(f"[VideoWorker] [{datetime.now()}] Lỗi _open_capture:", e)
            traceback.print_exc()
            return False






    # === Main loop ===
    def run(self):
        self._running = True

        if not self._open_capture():
            self._running = False
            return

        while self._running:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.stable_start = 0.0
                    self.captured = False
                    time.sleep(0.05)
                    continue

                # Gửi frame gốc lên UI
                self.frameSignal.emit(frame, self.title)

                # Phát hiện biển số
                plates, boxed_frame = self.models.detect_plates(frame)
                if not plates:
                    self.stable_start = 0.0
                    self.captured = False
                    time.sleep(0.01)
                    continue

                # Chọn biển số tốt nhất (lớn nhất)
                best = max(
                    plates,
                    key=lambda it: (it[0][2] - it[0][0]) * (it[0][3] - it[0][1]),
                )
                roi_current = best[1]
                if roi_current is None or roi_current.size == 0:
                    self.stable_start = 0.0
                    self.captured = False
                    time.sleep(0.01)
                    continue

                # Logic ổn định
                if self.stable_start == 0.0:
                    self.stable_start = time.time()
                elif self.captured:
                    self.stable_start = time.time()
                    self.captured = False

                # Đủ thời gian ổn định & chưa chụp
                if (not self.captured) and (
                    time.time() - self.stable_start >= self.stable_seconds
                ):
                    scene_img_to_save = boxed_frame if boxed_frame is not None else frame
                    scene_prefix = "scene_in_boxed" if self.mode == "in" else "scene_out_boxed"
                    roi_prefix = "plate_in_roi" if self.mode == "in" else "plate_out_roi"

                    scene_path = save_image(scene_img_to_save, scene_prefix)
                    roi_path = save_image(roi_current, roi_prefix)

                    if not scene_path or not roi_path:
                        print("[VideoWorker] Lỗi: Không thể lưu ảnh scene hoặc roi.")
                        self.captured = True
                        self.stable_start = 0.0
                        continue

                    # OCR
                    text_fmt, text_raw = "", ""
                    if self.ocr_mode == "gemini" and GEMINI_READY:
                        text_fmt, text_raw = self.models.ocr_plate_gemini_from_path(roi_path)
                    else:
                        text_fmt, text_raw = self.models.ocr_plate_yolo(roi_current)

                    if text_fmt or text_raw:
                        now = datetime.now()
                        d = now.strftime("%d/%m/%Y")
                        t = now.strftime("%H:%M:%S")
                        plate = text_fmt or text_raw

                        # Gửi tín hiệu lên UI
                        self.sceneSignal.emit(scene_path)
                        self.roiSignal.emit(roi_path, self.mode)

                        # Vào / Ra & DB
                        if self.mode == "in":
                            self.infoSignal.emit(
                                {"date_in": d, "time_in": t, "plate_text_in": plate}
                            )
                            if self.db and self.db.ok:
                                self.db.insert_in(plate, d, t, scene_path)
                                self.histSignal.emit()
                            self.playSoundSignal.emit("in")
                        else:
                            self.infoSignal.emit(
                                {"date_out": d, "time_out": t, "plate_text_out": plate}
                            )
                            if self.db and self.db.ok:
                                match = self.db.attach_out(plate, d, t, scene_path)
                                self.matchSignal.emit(match)
                                self.histSignal.emit()
                            self.playSoundSignal.emit("out")

                        self.captured = True
                        self.stable_start = 0.0

            except Exception as e:
                print(f"[VideoWorker] Lỗi trong vòng lặp VideoWorker: {e}")
                traceback.print_exc()
                self.stable_start = 0.0
                self.captured = False
                time.sleep(0.1)

            time.sleep(0.01)

        # Dọn dẹp
        try:
            if self.cap:
                self.cap.release()
        except Exception as e:
            print(f"[VideoWorker] Lỗi khi release camera: {e}")






    # === Stop worker ===
    def stop(self):
        self._running = False









class HistoryLoaderWorker(QThread):
    resultReady = Signal(pd.DataFrame) 

    def __init__(self, db: Optional[DB], start_time=None, end_time=None, status_filter=None, plate_filter=None, parent=None):
        super().__init__(parent)
        self.db: Optional[DB] = db
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











