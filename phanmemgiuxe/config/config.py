import os, cv2



USE_SQL = True
try:
    import pyodbc
except Exception:
    USE_SQL = False



REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_ROOT = os.path.join(REPO_ROOT, "model")



# ===== CONFIG ĐƯỜNG DẪN MODEL =====
DETECT_MODEL_PATH = os.path.join(MODEL_ROOT, "detection_plates", "license_plate_detector.pt")
OCR_MODEL_PATH    = os.path.join(MODEL_ROOT, "ocr_plates", "License_Plate_OCR.pt")



# ===== CONFIG ĐƯỜNG DẪN ẢNH =====
SAVE_DIR = os.path.join(REPO_ROOT, "images"); os.makedirs(SAVE_DIR, exist_ok=True)
LOGO_PATH = os.path.join(REPO_ROOT, "logo", "logo_cholimex.jpg")



# ===== CONFIG ĐƯỜNG DẪN ÂM THANH =====
SOUND_IN_PATH = os.path.join(REPO_ROOT, "audio", "moi_vao_xin_cam_on.wav")
SOUND_OUT_PATH = os.path.join(REPO_ROOT, "audio", "moi_ra_xin_cam_on.wav")



# ===== CONFIG KẾT NỐI SQL SERVER =====
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"
)



# ===== CONFIG UI =====
PANEL_W, PANEL_H = 640, 360
PANEL_BG = (255, 255, 255)



# ===== CONFIG API CAMERA =====
API_MAP = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}


# ===== CONFIG MAP KÝ TỰ CHO OCR =====
OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
           "six":"6","seven":"7","eight":"8","nine":"9"}






