# ----- Import thư viện -----
import os
import cv2
import time
import pyodbc
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from datetime import datetime

# ----- Tiêu đề trang -----
st.set_page_config(page_title="Test Biển Số", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #fff; background: linear-gradient(90deg, #1e90ff, #00c6ff); width: 800px; padding: 16px; border-radius: 12px; font-weight: bold; box-shadow: 0 2px 8px rgba(30,144,255,0.2);'>
        TEST PHÁT HIỆN & TRÍCH XUẤT BIỂN SỐ  LƯU SQL SERVER
    </h1>
    """, 
    unsafe_allow_html=True
)

# ----- Kết nối SQL Server -----
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"
)
conn = pyodbc.connect(conn_str)
cur = conn.cursor()

# ----- Tạo bảng Plates nếu chưa  -----
cur.execute("""
IF OBJECT_ID('Plates', 'U') IS NULL
CREATE TABLE Plates (
    id INT IDENTITY(1,1) PRIMARY KEY,
    plate_idx NVARCHAR(10),
    image_path NVARCHAR(255),
    ocr_text NVARCHAR(64),
    timestamp DATETIME
)
""")
conn.commit()

# ----- Load model an toàn + báo lỗi rõ -----
def load_yolo_or_fail(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    try:
        return YOLO(path)
    except Exception as e:
        raise RuntimeError(f"Lỗi load YOLO ({path}): {e}")

detect_model = load_yolo_or_fail(r"D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt")
ocr_model    = load_yolo_or_fail(r"D:/Documents/IUH_Student/OCR/model/ocr_plates/OCR_HOAN_CHINH.pt")


def _has_boxes(r):
    try:
        return hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
    except Exception:
        return False
    
# ----- Hàm phát hiện và cắt vùng biển số -----
def detect_plate(frame):
    results = detect_model(frame)
    plates = []
    for r in results:
        if not _has_boxes(r):
            continue
        xyxy_all = r.boxes.xyxy.cpu().numpy().astype(int)
        for i, (x1, y1, x2, y2) in enumerate(xyxy_all):
            pad = 10
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1]-1, x2 + pad); y2 = min(frame.shape[0]-1, y2 + pad)
            roi = frame[y1:y2, x1:x2].copy()
            plates.append((roi, (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "License Plate", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return plates, frame

# def detect_plate(frame):
#     results = detect_model(frame)
#     plates = []
#     for r in results:
#         if r.boxes is None:
#             continue
#         for b in r.boxes:
#             xyxy = b.xyxy[0].cpu().numpy()
#             x1, y1, x2, y2 = map(int, xyxy)
#             pad = 10
#             x1 = max(0, x1 - pad)
#             y1 = max(0, y1 - pad)
#             x2 = min(frame.shape[1]-1, x2 + pad)
#             y2 = min(frame.shape[0]-1, y2 + pad)
#             roi = frame[y1:y2, x1:x2].copy()
#             plates.append((roi, (x1, y1, x2, y2)))
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, "License Plate", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
#     return plates, frame

#  ----- Hàm trích xuất và định dạng biển số -----
def ocr_plate_image_best(roi):
    results = ocr_model(roi)
    chars = []
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            names = r.names if hasattr(r, 'names') else {}
            clses = r.boxes.cls.cpu().numpy().astype(int)
            xyxys = r.boxes.xyxy.cpu().numpy()
            char_boxes = []
            for i, cls in enumerate(clses):
                x1, y1, x2, y2 = xyxys[i]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                char = names.get(cls, str(cls))
                if char.isdigit() or (char.isalpha() and char.isupper()):
                    char_boxes.append((cy, cx, char))
            if len(char_boxes) > 0:
                ys = [b[0] for b in char_boxes]
                if len(char_boxes) <= 7 or (max(ys) - min(ys) < 0.2 * (max(ys) if max(ys) else 1)):
                    char_boxes_sorted = sorted(char_boxes, key=lambda b: b[1])
                    text_out = ''.join([b[2] for b in char_boxes_sorted])
                else:
                    y_thresh = (max(ys) + min(ys)) / 2
                    line1 = [b for b in char_boxes if b[0] < y_thresh]
                    line2 = [b for b in char_boxes if b[0] >= y_thresh]
                    line1_sorted = sorted(line1, key=lambda b: b[1])
                    line2_sorted = sorted(line2, key=lambda b: b[1])
                    line1_text = ''.join([b[2] for b in line1_sorted])
                    line2_text = ''.join([b[2] for b in line2_sorted])
                    text_out = f"{line1_text}-{line2_text}" if line2_text else line1_text
            else:
                text_out = ''
        else:
            text_out = ''

    # Chuẩn hóa output của OCR trích xuất
    raw_text = text_out.replace('-', ' ').replace(' ', '')
    if len(raw_text) >= 7:
        part1 = raw_text[:2]
        part2 = raw_text[2:4]
        part3 = raw_text[4:]
        text_out = f"{part1}-{part2} {part3}"
    return text_out

# ----- Hàm hiển thị bảng dữ liệu -----
def show_table():
    st.markdown(
    "<h2 style='color:#1e90ff; font-weight:700; margin-top:24px;'>Thông tin dữ liệu bảng Plates </h2>",
    unsafe_allow_html=True)
    query = "SELECT * FROM Plates ORDER BY id"
    try:
        df = pd.read_sql(query, conn)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {e}")

# ----- Tạo sidebar -----
with st.sidebar:
    # Upload ảnh
    st.markdown(
    """
    <div style='
        text-align:center;
        background: linear-gradient(90deg, #1e90ff, #00c6ff);
        color: #fff;
        padding: 12px 0 12px 0;
        border-radius: 10px;
        font-size: 23px;
        font-weight: bold;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(30,144,255,0.15);'>
        Upload ảnh để test nhận diện biển số
    </div>
    """, unsafe_allow_html=True)

    # Tạo thanh chọn ảnh
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"])
    btn_style = """
        <style>
            .custom-btn {
                width: 100%;
                padding: 10px 0;
                margin-bottom: 8px;
                border-radius: 8px;
                border: none;
                background: linear-gradient(90deg, #1e90ff, #00c6ff);
                color: #fff;
                font-size: 16px;
                font-weight: bold;
                transition: background 0.3s, box-shadow 0.3s;
                box-shadow: 0 2px 8px rgba(30,144,255,0.10);
                cursor: pointer;}
            .custom-btn:hover {
                background: linear-gradient(90deg, #00c6ff, #1e90ff);
                box-shadow: 0 4px 16px rgba(30,144,255,0.18);}
        </style>
    """
    
    # Camera 
    st.markdown(
    """
    <div style='
        text-align:center;
        background: linear-gradient(90deg, #1e90ff, #00c6ff);
        color: #fff;
        padding: 12px 0 12px 0;
        border-radius: 10px;
        font-size: 23px;
        font-weight: bold;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(30,144,255,0.15);'>
        Chọn camera để test nhận diện biển số
    </div>
    """, unsafe_allow_html=True)
    
    # Nút bật/tắt camera (gán vào session_state)
    cam_placeholder = st.empty()
    col_cam = st.columns([1, 1])
    with col_cam[0]:
        if st.button("Bật camera", key="start_cam_btn"):
            st.session_state.camera_running = True
            st.session_state.stop_cam = False
    with col_cam[1]:
        if st.button("Tắt camera", key="stop_cam_btn"):
            st.session_state.stop_cam = True
            st.session_state.camera_running = False

    # Tiêu đề truy vấn SQL Server
    st.markdown(
    """
    <div style='
        text-align:center;
        background: linear-gradient(90deg, #00c6ff, #1e90ff);
        color: #fff;
        padding: 12px 0 12px 0;
        border-radius: 10px;
        font-size: 23px;
        font-weight: bold;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(30,144,255,0.15);'>
        Truy vấn SQL Server
    </div>
    """, unsafe_allow_html=True)

    # Xữ lý 2 nút sql
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        btn_show = st.button("Xem dữ liệu bảng Plates", key="show_table_btn")
    with col_btn2:
        btn_del = st.button("Xóa dữ liệu bảng Plates", key="del_table_btn")
    if "sidebar_msg" in st.session_state and st.session_state.sidebar_msg:
        st.info(st.session_state.sidebar_msg)
        st.session_state.sidebar_msg = ""

# ----- Khởi tạo các biến trong st.session_state của Streamlit -----
# Lưu ảnh chụp từ camera
if 'captured_image' not in st.session_state:      
    st.session_state.captured_image = None

# Đếm lần upload / lưu biển số.
if 'upload_count' not in st.session_state:
    st.session_state.upload_count = 1

# Bật/tắt hiển thị bảng SQL.
if 'show_table' not in st.session_state:
    st.session_state.show_table = False

# Thông báo ở sidebar.
if 'sidebar_msg' not in st.session_state:
    st.session_state.sidebar_msg = ""

# List kết quả OCR từ camera.
if 'camera_results' not in st.session_state:
    st.session_state.camera_results = []

# Frame hiện tại của camera.
if 'camera_frame' not in st.session_state:
    st.session_state.camera_frame = None

# Thời gian để điều khiển debounce/phát hiện biển số.
if 'last_plate_time' not in st.session_state:
    st.session_state.last_plate_time = 0

# Thời gian để điều khiển debounce/phát hiện biển số.
if 'plate_detected_time' not in st.session_state:
    st.session_state.plate_detected_time = 0

# Cờ báo đã phát hiện biển số.
if 'plate_detected' not in st.session_state:
    st.session_state.plate_detected = False

# Cờ báo camera đang chạy.
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# --- thêm khởi tạo stop_cam và camera_running nếu chưa có ---
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'stop_cam' not in st.session_state:
    st.session_state.stop_cam = False

if btn_show:
    st.session_state.show_table = not st.session_state.show_table
if btn_del:
    cur.execute("DELETE FROM Plates")
    cur.execute("DBCC CHECKIDENT ('Plates', RESEED, -1);")
    conn.commit()
    st.session_state.upload_count = 1
    st.session_state.sidebar_msg = "Đã xóa toàn bộ dữ liệu trong bảng Plates!"
    st.session_state.camera_results = []




# ----- Xử lý upload ảnh -----
if uploaded_file and not st.session_state.camera_running:
    st.markdown("<h2 style='color:#1e90ff; font-weight:700; text-align:left; margin-top:24px;'>Phát hiện biển số</h2>", unsafe_allow_html=True)
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Lỗi đọc ảnh: {e}")
        frame = None

    # Chia 2 cột: ảnh gốc (trái) và ảnh có bounding-box (phải)
    if frame is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)
        plates, frame_boxed = detect_plate(frame)
        with col2:
            st.image(cv2.cvtColor(frame_boxed, cv2.COLOR_BGR2RGB), caption="Ảnh phát hiện biển số", use_container_width=True)
        
        # Nếu có ít nhất 1 plate -> xử lý từng ROI
        if plates:
            st.success(f"Phát hiện {len(plates)} biển số!")
            cols = st.columns(len(plates))
            for idx, (roi, (x1, y1, x2, y2)) in enumerate(plates):
                plate_idx = f"{st.session_state.upload_count}.{idx+1}"
                with cols[idx]:
                    # Tiêu đề cho kết quả OCR từng ROI
                    st.markdown("<h2 style='color:#00c6ff; font-weight:700; margin-top:24px;'>Kết quả thông tin trích xuất</h2>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption=f"Biển số {plate_idx}", width=200)
                    
                    # Chạy OCR trên ROI -> trả về chuỗi biển số
                    text = ocr_plate_image_best(roi)
                    st.write(f"Kết quả trích xuất: {text or '---'}")
                    
                    # Chuẩn bị lưu ảnh ROI ra ổ đĩa
                    roi_path = f"images/plate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{plate_idx}.jpg"
                    try:
                        # Lưu ảnh ROI
                        cv2.imwrite(roi_path, roi)
                    except Exception as e:
                        st.error(f"Lỗi lưu file ảnh: {e}")
                        roi_path = ""
                    
                    # Kiểm tra trùng trong DB theo ocr_text -> chỉ INSERT nếu chưa tồn tại
                    try:
                        cur.execute("SELECT COUNT(*) FROM Plates WHERE ocr_text = ?", (text,))
                        row = cur.fetchone()
                        exists = row[0] if row is not None else 0
                        if exists == 0:
                            cur.execute(
                                "INSERT INTO Plates (plate_idx, image_path, ocr_text, timestamp) VALUES (?, ?, ?, ?)",
                                (plate_idx, roi_path, text, datetime.now())
                            )
                            conn.commit()
                            st.info(f"Đã lưu vào SQL Server: {roi_path}, {text}, plate_idx={plate_idx}")
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Lỗi khi lưu vào DB: {e}")
            st.session_state.upload_count += 1
        else:
            st.warning("Không phát hiện được biển số xe!")


# -----Xữ lí camera -----
result_placeholder = st.empty()  

# Start camera loop nếu bật
if st.session_state.camera_running and not st.session_state.stop_cam:
    cap = cv2.VideoCapture(0)
    st.session_state.camera_results = []
    st.session_state.camera_frame = None
    frame_count = 0

    if not cap.isOpened():
        st.error("Không mở được camera.")
        st.session_state.camera_running = False
        st.session_state.stop_cam = True
    else:
        try:
            while cap.isOpened() and not st.session_state.stop_cam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Không lấy được frame từ camera!")
                    break
                frame = cv2.resize(frame, (350, 300))
                plates, frame_boxed = detect_plate(frame)
                st.session_state.camera_frame = frame_boxed.copy()
                # Hiển thị feed
                result_container = cam_placeholder.container()
                result_container.image(cv2.cvtColor(frame_boxed, cv2.COLOR_BGR2RGB), caption="Màn hình cam", use_container_width=True)

                now_ts = time.time()
                if plates:
                    if not st.session_state.plate_detected:
                        st.session_state.plate_detected_time = now_ts
                        st.session_state.plate_detected = True
                    elif now_ts - st.session_state.plate_detected_time >= 2.0: 
                        st.session_state.captured_image = frame_boxed.copy()
                        result_list = []
                        for idx, (roi, (x1, y1, x2, y2)) in enumerate(plates):
                            plate_idx = f"{st.session_state.upload_count}.{idx+1}"
                            text = ocr_plate_image_best(roi)
                            roi_path = f"images/plate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{plate_idx}.jpg"
                            try:
                                cv2.imwrite(roi_path, roi)
                            except Exception as e:
                                roi_path = ""
                            try:
                                cur.execute("SELECT COUNT(*) FROM Plates WHERE ocr_text = ?", (text,))
                                row = cur.fetchone()
                                exists = row[0] if row is not None else 0
                                if exists == 0:
                                    cur.execute(
                                        "INSERT INTO Plates (plate_idx, image_path, ocr_text, timestamp) VALUES (?, ?, ?, ?)",
                                        (plate_idx, roi_path, text, datetime.now())
                                    )
                                    conn.commit()
                            except Exception as e:
                                conn.rollback()
                            result_list.append({
                                "roi": roi,
                                "plate_idx": plate_idx,
                                "text": text,
                                "roi_path": roi_path
                            })
                        st.session_state.camera_results = result_list
                        st.session_state.upload_count += 1
                        st.session_state.plate_detected = False

                        # Hiển thị kết quả
                        with result_placeholder.container():
                            st.markdown("<h2 style='color:#1e90ff; font-weight:700; text-align:left; margin-top:24px;'>Ảnh chụp phát hiện biển số</h2>", unsafe_allow_html=True)
                            st.image(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB), caption="Ảnh chụp phát hiện biển số", use_container_width=True)
                            st.success(f"Phát hiện {len(st.session_state.camera_results)} biển số! (Camera)")
                            cols = st.columns(len(st.session_state.camera_results))
                            for idx, result in enumerate(st.session_state.camera_results):
                                with cols[idx]:
                                    st.markdown("<h2 style='color:#00c6ff; font-weight:700; margin-top:24px;'>Kết quả thông tin trích xuất</h2>", unsafe_allow_html=True)
                                    st.image(cv2.cvtColor(result["roi"], cv2.COLOR_BGR2RGB), caption=f"Biển số {result['plate_idx']}", width=200)
                                    st.write(f"Kết quả trích xuất: {result['text'] or '---'}")
                                    st.info(f"Đã lưu vào SQL Server: {result['roi_path']}, {result['text']}, plate_idx={result['plate_idx']}")
                else:
                    st.session_state.plate_detected = False
                    st.session_state.plate_detected_time = 0

                # nhỏ delay để giảm tải
                time.sleep(0.03)

                if st.session_state.stop_cam:
                    break
        finally:
            cap.release()
            st.session_state.camera_running = False
            st.session_state.stop_cam = False

# ------ Nếu camera không đang chạy nhưng có kết quả cũ, hiển thị -----
if not st.session_state.camera_running:
    if st.session_state.captured_image is not None:
        st.markdown("<h2 style='color:#1e90ff; font-weight:700; text-align:left; margin-top:24px;'>Ảnh chụp phát hiện biển số</h2>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(st.session_state.captured_image, cv2.COLOR_BGR2RGB), caption="Ảnh chụp phát hiện biển số", use_container_width=True)
    
    if st.session_state.camera_results:
        st.success(f"Phát hiện {len(st.session_state.camera_results)} biển số! (Camera)")
        cols = st.columns(len(st.session_state.camera_results))
        for idx, result in enumerate(st.session_state.camera_results):
            with cols[idx]:
                st.markdown("<h2 style='color:#00c6ff; font-weight:700; margin-top:24px;'>Kết quả thông tin trích xuất</h2>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(result["roi"], cv2.COLOR_BGR2RGB), caption=f"Biển số {result['plate_idx']}", width=200)
                st.write(f"Kết quả trích xuất: {result['text'] or '---'}")
                st.info(f"Đã lưu vào SQL Server: {result['roi_path']}, {result['text']}, plate_idx={result['plate_idx']}")


# ----- Hiển thị bảng nếu yêu cầu -----
if st.session_state.show_table:
    show_table()