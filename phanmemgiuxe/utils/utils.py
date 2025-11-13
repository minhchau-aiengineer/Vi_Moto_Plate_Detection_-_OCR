import os, cv2, numpy as np
from datetime import datetime
from PySide6.QtGui import QImage
from ..config.config import SAVE_DIR, PANEL_W, PANEL_H, PANEL_BG, OCR_MAP





def save_image(img, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    relative_path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    absolute_path = os.path.abspath(relative_path)

    try:
        cv2.imwrite(absolute_path, img)
        return absolute_path
    except Exception as e:
        print(f"Lỗi khi lưu ảnh {absolute_path}: {e}")
        return None 





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
    if bgr is None: 
        bgr = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape

    return QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)






def norm_char(x): 
    return OCR_MAP.get(str(x), str(x))



def plate_norm(s: str) -> str: 
    return (s or "").replace("-", "").replace(" ", "").upper()



def has_boxes(r):
    try: 
        return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
    except: 
        return False
    




def preprocess_for_ocr(roi):
    if roi is None: 
        return None
    if roi.shape[-1]==4: 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe,(3,3),0)
    
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)



