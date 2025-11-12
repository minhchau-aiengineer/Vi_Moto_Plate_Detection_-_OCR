import os, time, cv2, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from ultralytics import YOLO

# ============== CONFIG ==============
st.set_page_config(page_title="Detecting & OCR", layout="wide")

DETECT_MODEL_PATH = r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt"
OCR_MODEL_PATH    = r"D:/Documents/IUH_Student/OCR/model/ocr_plates/epoch199.pt"

SAVE_DIR = "images"; os.makedirs(SAVE_DIR, exist_ok=True)

USE_SQL = True
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"
)

PANEL_W, PANEL_H = 640, 360
PANEL_BG = (232, 239, 248)  
STABLE_SECONDS_IN  = 1.2
STABLE_SECONDS_OUT = 1.2

# ============== SIDEBAR ==============
with st.sidebar:
    st.subheader("Config camera")

    # Chọn cam
    cidx1, cidx2 = st.columns(2)
    with cidx1:
        cam1_idx = st.number_input("Index Cam 1", min_value=0, max_value=9, value=0, step=1)
    with cidx2:
        cam2_idx = st.number_input("Index Cam 2", min_value=0, max_value=9, value=0, step=1)

    # Chọn backend
    api_map = {"DSHOW(Windows)": cv2.CAP_DSHOW, "MSMF(Windows)": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
    api1 = st.selectbox("Backend Cam 1", list(api_map.keys()), index=0)
    api2 = st.selectbox("Backend Cam 2", list(api_map.keys()), index=0)
    st.caption("Chỉnh index/backend nếu cam không mở được.")

    # Bật/tắt cam
    st.markdown("---")
    st.subheader("Funtion camera")
    c1, c2 = st.columns(2)
    with c1:
        start_cam1 = st.button("Bật Cam 1", use_container_width=True)
    with c2:
        stop_cam1  = st.button("Tắt Cam 1", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        start_cam2 = st.button("Bật Cam 2", use_container_width=True)
    with c4:
        stop_cam2  = st.button("Tắt Cam 2", use_container_width=True)

# ============== STATE ==============
if "cam1_on" not in st.session_state: st.session_state.cam1_on = False
if "cam2_on" not in st.session_state: st.session_state.cam2_on = False
if "cap1" not in st.session_state:    st.session_state.cap1 = None
if "cap2" not in st.session_state:    st.session_state.cap2 = None

# Đóng camera
def close_cap(name):
    try:
        if st.session_state.get(name):
            st.session_state[name].release()
    except:
        pass
    st.session_state[name] = None

# ============== Xử lý bật/tắt cam ==============
# Cam 1
if start_cam1 and not st.session_state.cam1_on:
    st.session_state.cam1_on = True
if stop_cam1:
    st.session_state.cam1_on = False
    close_cap("cap1")
# Cam 2
if start_cam2 and not st.session_state.cam2_on:
    st.session_state.cam2_on = True
if stop_cam2:
    st.session_state.cam2_on = False
    close_cap("cap2")

# ============== Thông tin biển số ==============
# VÀO (Cam 1)
st.session_state.setdefault("date_in", "--/--/----")
st.session_state.setdefault("time_in", "--:--:--")
st.session_state.setdefault("plate_text_in", "---")
# RA (Cam 2)
st.session_state.setdefault("date_out", "--/--/----")
st.session_state.setdefault("time_out", "--:--:--")
st.session_state.setdefault("plate_text_out", "---")

# ROI riêng cho từng camera để hiển thị trong "Thông tin chi tiết"
st.session_state.setdefault("roi_in_path", "")   
st.session_state.setdefault("roi_out_path", "")  

# Ảnh chụp gần nhất (cho Ô3/Ô4)
st.session_state.setdefault("scene_path", "")
st.session_state.setdefault("roi_path", "")

# ổn định riêng cho 2 cam
st.session_state.setdefault("stable_start_in", 0.0)
st.session_state.setdefault("stable_start_out", 0.0)
st.session_state.setdefault("captured_in", False)
st.session_state.setdefault("captured_out", False)

# Lịch sử
st.session_state.setdefault("hist_tick", 0)
st.session_state.setdefault("match_text", "")

# ============== HÀM TIỆN ÍCH ==============
def letterbox(img, w=PANEL_W, h=PANEL_H, color=PANEL_BG):
    if img is None: return np.full((h, w, 3), color, dtype=np.uint8)
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0: return np.full((h, w, 3), color, dtype=np.uint8)
    s = min(w/iw, h/ih); nw, nh = int(iw*s), int(ih*s)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((h, w, 3), color, dtype=np.uint8)
    top, left = (h-nh)//2, (w-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

# ============== Vẽ title cho từng ô ==============
def draw_title(img, text):
    img = img.copy()
    x1,y1,x2,y2 = 12,12, min(PANEL_W-12, 12+340), 52
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(205,214,230),2)
    cv2.putText(img, text,(x1+6,y2-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(80,80,80),2, cv2.LINE_AA)
    return img

# ============== Hiển thị ảnh ==============
def show(ph, bgr, title):
    frame = letterbox(bgr); frame = draw_title(frame, title)
    ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=PANEL_W)

# ============== Hiển thị ảnh trống ==============
def show_blank(ph, title):
    show(ph, np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8), title)

# ============== MỞ CAM ==============
def open_cam(idx=0, api=cv2.CAP_DSHOW):
    cap = cv2.VideoCapture(int(idx), api)
    if not (cap and cap.isOpened()):
        try: cap.release()
        except: pass
        return None
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except: pass
    try: cap.set(cv2.CAP_PROP_FPS, 30)
    except: pass
    return cap

# ============== KẾT NỐI SQL ==============
cur = None; conn = None; sql_err = ""
if USE_SQL:
    try:
        import pyodbc
        conn = pyodbc.connect(CONN_STR, autocommit=True)
        cur  = conn.cursor()

        # Mỗi phiên có thể chỉ có IN hoặc chỉ có OUT
        # Nếu có IN thì match_status='PENDING' -> sau khi có OUT sẽ thành
        cur.execute("""
            IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
            CREATE TABLE dbo.ParkingSessions(
                id INT IDENTITY(1,1) PRIMARY KEY,
                -- IN
                plate_in NVARCHAR(64)  NULL,
                date_in  NVARCHAR(16)  NULL,
                time_in  NVARCHAR(16)  NULL,
                image_in NVARCHAR(255) NULL,
                -- OUT
                plate_out NVARCHAR(64)  NULL,
                date_out  NVARCHAR(16)  NULL,
                time_out  NVARCHAR(16)  NULL,
                image_out NVARCHAR(255) NULL,
                -- MATCH
                match_status NVARCHAR(16) NULL,  -- 'KHỚP' | 'KHÔNG KHỚP' | 'PENDING'
                created_at DATETIME DEFAULT GETDATE()
            );
        """)
    except Exception as e:
        sql_err = f"Không kết nối SQL: {e}"
        USE_SQL = False

# ============== LOAD MODEL ==============
@st.cache_resource(show_spinner=True)
def load_yolo(path): return YOLO(path)

# Load model
load_ok, load_err = True, ""

# nếu model lỗi thì vẫn chạy app, nhưng không detect/ocr được
try:
    detect_model = load_yolo(DETECT_MODEL_PATH)
    ocr_model    = load_yolo(OCR_MODEL_PATH)
except Exception as e:
    load_ok, load_err = False, str(e)

# ============== XỬ LÝ ẢNH & OCR ==============
def has_boxes(r):
    try: return hasattr(r,"boxes") and r.boxes is not None and len(r.boxes)>0
    except: return False

# ============== Chuẩn hoá ký tự OCR ==============
OCR_MAP = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
def norm_char(x): return OCR_MAP.get(str(x), str(x))

# ============== Xử lý ảnh trước khi OCR ==============
def preprocess_for_ocr(roi):
    if roi is None: return None
    if roi.shape[-1]==4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(clahe,(3,3),0)
    return cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# ============== Xử lý biển số ==============
def detect_plates(frame):
    plates, boxed = [], frame.copy()
    for r in detect_model(frame):
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

# ============== OCR biển số ==============
def ocr_plate(roi):
    roi_pre = preprocess_for_ocr(roi)
    res = ocr_model(roi_pre if roi_pre is not None else roi)
    text_raw=""

    # Nối ký tự biển số
    for r in res:
        if not has_boxes(r): continue
        names = getattr(r,'names',None) or getattr(ocr_model,'names',{}) or {}
        clses = r.boxes.cls.cpu().numpy().astype(int)
        xyxys= r.boxes.xyxy.cpu().numpy()
        boxes=[]

        # Lấy ký tự hợp lệ
        for i,cls in enumerate(clses):
            x1,y1,x2,y2 = xyxys[i]
            cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
            ch = norm_char(names.get(cls, str(cls)) if isinstance(names,dict) else str(cls))
            if ch.isdigit() or (ch.isalpha() and ch.isupper()):
                boxes.append((cy,cx,ch))
        if not boxes: continue
        ys=[b[0] for b in boxes]
        
        # Nối chuỗi biển số
        if len(boxes)<=7 or (max(ys)-min(ys) < 0.2*max(ys, default=1)):
            text_raw=''.join([b[2] for b in sorted(boxes,key=lambda b:b[1])])
        else:
            thr=(max(ys)+min(ys))/2.0
            l1=[b for b in boxes if b[0]<thr]; l2=[b for b in boxes if b[0]>=thr]
            t1=''.join([b[2] for b in sorted(l1,key=lambda b:b[1])])
            t2=''.join([b[2] for b in sorted(l2,key=lambda b:b[1])])
            text_raw=f"{t1}-{t2}" if t2 else t1
    raw=(text_raw or '').replace('-',' ').replace(' ','')
    text_fmt = f"{raw[:2]}-{raw[2:4]} {raw[4:]}" if len(raw)>=7 else (text_raw or "")
    return text_fmt, (text_raw or "")

# ============== LƯU ẢNH ==============
def save_image(img, prefix):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SAVE_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, img); return path

# ============== XỬ LÝ DỮ LIỆU VÀO/RA ==============
def plate_norm(s: str) -> str:
    """Chuẩn hoá biển để so khớp: bỏ khoảng trắng/gạch, upper-case."""
    return (s or "").replace("-", "").replace(" ", "").upper()

# ============== INSERT phiên mới khi có 'Vào' ==============
def session_insert_in(plate, d, t, img_path):
    """Tạo 1 phiên mới khi có 'Vào'."""
    if not USE_SQL: return
    try:
        cur.execute("""
            INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
            VALUES (?,?,?,?,?)
        """, (plate, d, t, img_path, 'PENDING'))
    except Exception:
        pass

# ============== GHÉP phiên khi có 'Ra' ==============
def session_attach_out(plate_out, d, t, img_path):
    """Gắn 'Ra' vào phiên 'Vào' gần nhất cùng biển (sau chuẩn hoá)."""
    if not USE_SQL: return "Khong khop bien so"
    try:
        rows = cur.execute("""
            SELECT TOP 50 id, plate_in FROM dbo.ParkingSessions
            WHERE plate_out IS NULL
            ORDER BY id DESC
        """).fetchall()
        match_sid = None

        # Tìm phiên 'Vào' gần nhất có biển khớp
        for r in rows:
            sid, plate_in = r
            if plate_norm(plate_in) == plate_norm(plate_out):
                match_sid = sid
                break

        # Cập nhật phiên
        if match_sid:
            cur.execute("""
                UPDATE dbo.ParkingSessions
                SET plate_out=?, date_out=?, time_out=?, image_out=?, match_status='KHOP-BIEN-SO'
                WHERE id=?
            """, (plate_out, d, t, img_path, match_sid))
            return "Khop bien so"
        else:
            cur.execute("""
                INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
                VALUES (?,?,?,?, 'KHONG-KHOP-BIEN-SO')
            """, (plate_out, d, t, img_path))
            return "Khong khop bien so"
    except Exception:
        return "Khong khop bien so"
    

# ============== HIỂN THỊ HTML AN TOÀN ==============
def safe_markdown(ph, html):
    if html:
        ph.markdown(html, unsafe_allow_html=True)
    else:
        ph.empty()

# ============== GIAO DIỆN CHÍNH ==============
st.title("------------------------ Phát hiện & OCR biển số xe ---------------------------------")
t1, t2 = st.columns(2, gap="small")
b1, b2 = st.columns(2, gap="small")
with t1: ph_cam1 = st.empty()   # 1) Cam 1 (Vào)
with t2: ph_cam2 = st.empty()   # 2) Cam 2 (Ra)
with b1: ph_scene = st.empty()  # 3) Ảnh toàn cảnh đã chụp + BOX (lần chụp gần nhất)
with b2: ph_roi   = st.empty()  # 4) ROI biển số (lần chụp gần nhất)


# Nếu model lỗi thì không chạy tiếp
if not load_ok:
    show_blank(ph_cam1, "1) Cam 1")
    show_blank(ph_cam2, f"2) Error_model: {load_err}")
    show_blank(ph_scene,"3) Image_BOX")
    show_blank(ph_roi,"4) Image_ROI")
    if sql_err: st.warning(sql_err)
    st.stop()

# ============== Thông tin chi tiết ==============
st.markdown("---")
st.markdown("## ----------------------------------------------- Thông tin chi tiết -----------------------------------------------")
st.markdown("""
<style>
.info-label{font-size:14px;color:#475569;margin:2px 0 4px 2px}
.info-card{border:1px solid #c7d7ff;background:#fff;border-radius:8px;
           padding:6px 10px;min-height:36px;font-size:18px;font-weight:700;
           display:inline-block;width:260px;text-align:left}
.info-card.plate{color:#d41111}
.info-section{display:flex;flex-direction:column;gap:8px;align-items:flex-start}
.match-box{border:1px dashed #9aaeff;padding:8px 10px;border-radius:8px;background:#f7faff}
</style>
""", unsafe_allow_html=True)

# Chia 2 cột: Vào (Cam 1) | Ra (Cam 2)
col_in, col_out = st.columns(2, gap="large")

with col_in:
    st.write("#### Vao (Cam 1)")
    date_in_box  = st.empty()
    time_in_box  = st.empty()
    plate_in_box = st.empty()
    roi_in_box   = st.empty()   

with col_out:
    st.write("#### Ra (Cam 2)")
    date_out_box  = st.empty()
    time_out_box  = st.empty()
    plate_out_box = st.empty()
    roi_out_box   = st.empty() 

# Cột dưới cùng để hiển thị kết quả so khớp biển số 
match_box = st.empty()


# ============== Hàm hiển thị thông tin chi tiết ==============
def render_info():
    date_in_box.markdown(
        f"<div class='info-section'><div class='info-label'>Ngày vào</div>"
        f"<div class='info-card'>{st.session_state.date_in}</div></div>", unsafe_allow_html=True)
    time_in_box.markdown(
        f"<div class='info-section'><div class='info-label'>Giờ vào</div>"
        f"<div class='info-card'>{st.session_state.time_in}</div></div>", unsafe_allow_html=True)
    plate_in_box.markdown(
        f"<div class='info-section'><div class='info-label'>Biển số xe</div>"
        f"<div class='info-card plate'>{st.session_state.plate_text_in}</div></div>", unsafe_allow_html=True)

    date_out_box.markdown(
        f"<div class='info-section'><div class='info-label'>Ngày ra</div>"
        f"<div class='info-card'>{st.session_state.date_out}</div></div>", unsafe_allow_html=True)
    time_out_box.markdown(
        f"<div class='info-section'><div class='info-label'>Giờ ra</div>"
        f"<div class='info-card'>{st.session_state.time_out}</div></div>", unsafe_allow_html=True)
    plate_out_box.markdown(
        f"<div class='info-section'><div class='info-label'>Biển số xe</div>"
        f"<div class='info-card plate'>{st.session_state.plate_text_out}</div></div>", unsafe_allow_html=True)

    # Ảnh ROI riêng cho từng camera
    if st.session_state.roi_in_path and os.path.exists(st.session_state.roi_in_path):
        roi_in_box.image(st.session_state.roi_in_path, caption="ROI Cam 1 (Vào)", width=280, use_container_width=False)
    else:
        roi_in_box.empty()

    if st.session_state.roi_out_path and os.path.exists(st.session_state.roi_out_path):
        roi_out_box.image(st.session_state.roi_out_path, caption="ROI Cam 2 (Ra)", width=280, use_container_width=False)
    else:
        roi_out_box.empty()

    if st.session_state.match_text:
        match_box.markdown(f"<div class='match-box'>So khớp: <b>{st.session_state.match_text.upper()}</b></div>", unsafe_allow_html=True)
    else:
        match_box.markdown("", unsafe_allow_html=True)

render_info()

# ============== Lịch sử bảng Plates ==============
st.markdown("## -------------------------------------------- Thông tin bảng Plates -------------------------------------------")
history_box = st.empty()

# Làm mới bảng lịch sử
def refresh_history():
    if USE_SQL:
        try:
            rows = cur.execute("""
                SELECT TOP 100
                    id, image_in, plate_in, date_in, time_in,
                    image_out, plate_out, date_out, time_out, match_status
                FROM dbo.ParkingSessions
                ORDER BY id DESC
            """).fetchall()

            df = pd.DataFrame.from_records(
                rows,
                columns=[
                    "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
                    "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
                ]
            )

            # Ẩn None/NaN -> chuỗi rỗng
            df = df.astype(object).where(pd.notnull(df), "")
            # Nếu trạng thái trống thì hiển thị 'PENDING'
            df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})

            # Thêm STT
            df.insert(0, "STT", range(1, len(df) + 1))

            history_box.dataframe(df, use_container_width=True, height=420)
        except Exception as e:
            st.warning(f"Lỗi tải lịch sử: {e}")


refresh_history()

# ============== VÒNG LẶP CHÍNH ==============
# Đọc frame từ camera
def read_frame(cap):
    if cap is None or (not cap.isOpened()):
        return None
    ok, f = cap.read()
    if not ok:
        return None
    return f

# Đảm bảo camera đã mở
def ensure_cap(which):
    """Mở camera theo lựa chọn sidebar."""
    if which == 1:
        if st.session_state.cam1_on and (st.session_state.cap1 is None or not st.session_state.cap1.isOpened()):
            st.session_state.cap1 = open_cam(cam1_idx, api_map[api1])
            if st.session_state.cap1 is None: st.session_state.cam1_on = False
    else:
        if st.session_state.cam2_on and (st.session_state.cap2 is None or not st.session_state.cap2.isOpened()):
            st.session_state.cap2 = open_cam(cam2_idx, api_map[api2])
            if st.session_state.cap2 is None: st.session_state.cam2_on = False

# Xử lý luồng video từ cam 1 (Vào) hoặc cam 2 (Ra)
def process_stream(frame, mode="in"):
    """
    mode = 'in'  -> cập nhật cột Vào + ParkingSessions (insert hàng mới, PENDING)
    mode = 'out' -> cập nhật cột Ra  + ghép với 'Vào' (KHỚP/ KHÔNG KHỚP)
    """

    # Nếu không có frame thì reset ổn định
    if frame is None:
        if mode == "in":
            st.session_state.stable_start_in = 0.0
            st.session_state.captured_in = False
        else:
            st.session_state.stable_start_out = 0.0
            st.session_state.captured_out = False
        return

    # Phát hiện biển số
    plates, boxed = detect_plates(frame)
    if not plates:
        if mode == "in":
            st.session_state.stable_start_in = 0.0
            st.session_state.captured_in = False
        else:
            st.session_state.stable_start_out = 0.0
            st.session_state.captured_out = False
        return

    best = max(plates, key=lambda it:(it[0][2]-it[0][0])*(it[0][3]-it[0][1]))
    roi_current = best[1]

    # Xử lý ổn định & OCR
    if mode == "in":
        if st.session_state.stable_start_in == 0.0 or st.session_state.captured_in:
            st.session_state.stable_start_in = time.time()
            st.session_state.captured_in = False
        if (not st.session_state.captured_in) and (time.time() - st.session_state.stable_start_in >= STABLE_SECONDS_IN):
            text_fmt, text_raw = ocr_plate(roi_current)
            if text_fmt or text_raw:
                now = datetime.now()
                st.session_state.date_in = now.strftime("%d/%m/%Y")
                st.session_state.time_in = now.strftime("%H:%M:%S")
                st.session_state.plate_text_in = text_fmt or text_raw

                scene_path = save_image(boxed if boxed is not None else frame, "scene_in_boxed")
                roi_path   = save_image(roi_current, "plate_in_roi")
                st.session_state.scene_path = scene_path
                st.session_state.roi_path   = roi_path
                st.session_state.roi_in_path = roi_path  # hiển thị ROI Cam 1 ở phần thông tin

                # Hiển thị
                scene_img = cv2.imread(scene_path); show(ph_scene, scene_img, "3) Image_BOX")
                roi_img   = cv2.imread(roi_path);   show(ph_roi, roi_img, "4) ROI_Plates")
                render_info()

                # Ghi phiên IN
                session_insert_in(st.session_state.plate_text_in, st.session_state.date_in, st.session_state.time_in, scene_path)
                refresh_history()
                st.session_state.captured_in = True

    # Xử lý cam 2 (Ra)
    else:  
        if st.session_state.stable_start_out == 0.0 or st.session_state.captured_out:
            st.session_state.stable_start_out = time.time()
            st.session_state.captured_out = False
        if (not st.session_state.captured_out) and (time.time() - st.session_state.stable_start_out >= STABLE_SECONDS_OUT):
            text_fmt, text_raw = ocr_plate(roi_current)
            if text_fmt or text_raw:
                now = datetime.now()
                st.session_state.date_out = now.strftime("%d/%m/%Y")
                st.session_state.time_out = now.strftime("%H:%M:%S")
                st.session_state.plate_text_out = text_fmt or text_raw

                scene_path = save_image(boxed if boxed is not None else frame, "scene_out_boxed")
                roi_path   = save_image(roi_current, "plate_out_roi")
                st.session_state.scene_path = scene_path
                st.session_state.roi_path   = roi_path
                st.session_state.roi_out_path = roi_path  # hiển thị ROI Cam 2 ở phần thông tin

                # Hiển thị
                scene_img = cv2.imread(scene_path); show(ph_scene, scene_img, "3) Image_BOX")
                roi_img   = cv2.imread(roi_path);   show(ph_roi, roi_img, "4) ROI_Plates")
                render_info()

                # Ghép phiên OUT
                match_result = session_attach_out(st.session_state.plate_text_out, st.session_state.date_out, st.session_state.time_out, scene_path)
                st.session_state.match_text = match_result
                render_info()
                refresh_history()

                st.session_state.captured_out = True

# ============== VÒNG LẶP CHÍNH ==============
# Nếu có cam nào đang bật thì đọc frame từ cam đó
if st.session_state.cam1_on or st.session_state.cam2_on:
    try:
        while st.session_state.cam1_on or st.session_state.cam2_on:
            ensure_cap(1); ensure_cap(2)

            f1 = read_frame(st.session_state.cap1) if st.session_state.cam1_on else None
            f2 = read_frame(st.session_state.cap2) if st.session_state.cam2_on else None

            show(ph_cam1, f1, "1) Cam 1 (Vao)") if f1 is not None else show_blank(ph_cam1, "1) Cam 1 (Vao)")
            show(ph_cam2, f2, "2) Cam 2 (Ra)")  if f2 is not None else show_blank(ph_cam2, "2) Cam 2 (Ra)")

            if f1 is not None: process_stream(f1, mode="in")
            if f2 is not None: process_stream(f2, mode="out")

            st.session_state.hist_tick += 1
            if st.session_state.hist_tick % 30 == 0:
                refresh_history()

            time.sleep(0.02)
    finally:
        close_cap("cap1"); close_cap("cap2")
        st.session_state.cam1_on = False
        st.session_state.cam2_on = False

# Nếu không có cam nào bật thì hiển thị trống
else:
    show_blank(ph_cam1,"1) Cam 1")
    show_blank(ph_cam2,"2) Cam 2")
    if st.session_state.scene_path and os.path.exists(st.session_state.scene_path):
        scene_img = cv2.imread(st.session_state.scene_path); show(ph_scene, scene_img, "3) Image_BOX")
    else:
        show_blank(ph_scene,"3) Image_BOX")
    if st.session_state.roi_path and os.path.exists(st.session_state.roi_path):
        roi_img = cv2.imread(st.session_state.roi_path); show(ph_roi, roi_img, "4) ROI_Plates")
    else:
        show_blank(ph_roi,"4) ROI_Plates")

# Thông báo lỗi SQL
if sql_err:
    st.warning(sql_err)
