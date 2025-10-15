# Import các thư viện cần thiết
import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import pyodbc
from datetime import datetime


#---------- Tiêu đề trang ----------
st.set_page_config(page_title="Test Biển Số", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #fff; background: linear-gradient(90deg, #1e90ff, #00c6ff); width: 800px; padding: 16px; border-radius: 12px; font-weight: bold; box-shadow: 0 2px 8px rgba(30,144,255,0.2);'>
        TEST PHÁT HIỆN & TRÍCH XUẤT BIỂN SỐ  LƯU SQL SERVER
    </h1>
    """, 
    unsafe_allow_html=True)


#---------- Kết nối SQL Server ----------
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=123456"  
)
conn = None
cur = None
db_available = False
try:
    conn = pyodbc.connect(conn_str, timeout=5)
    cur = conn.cursor()
    db_available = True
except Exception as e:
    conn = None
    cur = None
    db_available = False
    st.error("Không thể kết nối tới SQL Server. Kiểm tra service/instance/driver/xác thực.")
    st.error(f"Chi tiết: {e}")



#---------- Tạo bảng Plates nếu chưa có ----------
if db_available:
    try:
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
    except Exception as e:
        conn.rollback()
        st.error(f"Lỗi khi tạo bảng Plates: {e}")
else:
    st.warning("DB không khả dụng — chức năng lưu/đọc sẽ bị vô hiệu hoá.")


#---------- Model phát hiện và trích xuất biển số ----------
detect_model = YOLO("D:/Documents/IUH_Student/OCR/model/detection_plates/license_plate_detector.pt")
ocr_model = YOLO("D:/Documents/IUH_Student/OCR/model/ocr_plates/plates_xe_may_best.pt")


#---------- Hàm phát hiện và cắt vùng biển số ----------
def detect_plate(frame):
    results = detect_model(frame)
    plates = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1]-1, x2 + pad)
            y2 = min(frame.shape[0]-1, y2 + pad)
            roi = frame[y1:y2, x1:x2].copy()
            plates.append((roi, (x1, y1, x2, y2)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "License Plate", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return plates, frame


#---------- Hàm trích xuất và định dạng biển số ----------
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
    
    # Cấu hình đầu ra của OCR trích xuất
    raw_text = text_out.replace('-', ' ').replace(' ', '')
    if len(raw_text) >= 7:
        part1 = raw_text[:2]
        part2 = raw_text[2:4]
        part3 = raw_text[4:]
        text_out = f"{part1}-{part2} {part3}"
    return text_out


# ---------- Hàm hiển thị bảng dữ liệu ----------
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


# ---------- Tạo thanh upload ảnh và các nút nằm ở sidebar ----------
with st.sidebar:
    # Tiêu đề Upload ảnh
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
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"])
    
    # Tiêu đề Truy vấn SQL
    st.markdown(
    """
    <div style='
        text-align:center;
        background: linear-gradient(90deg, #00c6ff, #1e90ff);
        color: #fff;
        padding: 10px 0 10px 0;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        margin-top: 18px;
        box-shadow: 0 2px 8px rgba(30,144,255,0.10);'>
        Truy vấn SQL Server
    </div>
    """, unsafe_allow_html=True)
    
    # Định dạng 2 nút
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
    st.markdown(btn_style, unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        btn_show = st.button("Xem dữ liệu bảng Plates", key="show_table_btn")
    with col_btn2:
        btn_del = st.button("Xóa dữ liệu bảng Plates", key="del_table_btn")
   
    # Hiển thị thông báo bên dưới 2 nút
    if "sidebar_msg" in st.session_state and st.session_state.sidebar_msg:
        st.info(st.session_state.sidebar_msg)
        st.session_state.sidebar_msg = ""


# ---------- Khởi tạo biến đếm số lần upload ảnh ----------
if 'upload_count' not in st.session_state:
    st.session_state.upload_count = 1


# ---------- Xử lý nút ở sidebar ----------
if btn_del:
    if not db_available:
        st.warning("DB không khả dụng — không thể xóa.")
    else:
        try:
            cur.execute("DELETE FROM Plates")
            cur.execute("DBCC CHECKIDENT ('Plates', RESEED, -1);")
            conn.commit()
            st.session_state.upload_count = 1
            st.session_state.sidebar_msg = "Đã xóa toàn bộ dữ liệu trong bảng Plates!"
            st.success("Đã xóa dữ liệu và reset identity.")
        except Exception as e:
            conn.rollback()
            st.error(f"Lỗi khi xóa dữ liệu: {e}")


#---------- Thêm biến trạng thái để điều khiển hiển thị bảng ----------
if 'show_table' not in st.session_state:
    st.session_state.show_table = False


#---------- Khởi tạo biến trạng thái sidebar_msg ----------
if 'sidebar_msg' not in st.session_state:
    st.session_state.sidebar_msg = ""


# ---------- Xử lý luồng dữ liệu ----------
if uploaded_file:
    st.markdown("<h2 style='color:#1e90ff; font-weight:700; text-align:left; margin-top:24px;'>Phát hiện biển số</h2>", unsafe_allow_html=True)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Tạo 2 cột để hiển thị ảnh gốc và ảnh phát hiện biển số trên cùng 1 hàng
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_container_width=True)
    plates, frame_boxed = detect_plate(frame)
    with col2:
        st.image(cv2.cvtColor(frame_boxed, cv2.COLOR_BGR2RGB), caption="Ảnh phát hiện biển số", use_container_width=True)

    # Phát hiện và trích xuất biển số
    if plates:
        st.success(f"Phát hiện {len(plates)} biển số!")
        cols = st.columns(len(plates))
        for idx, (roi, (x1, y1, x2, y2)) in enumerate(plates):
            plate_idx = f"{st.session_state.upload_count}.{idx+1}"
            with cols[idx]:
                st.markdown("<h2 style='color:#00c6ff; font-weight:700; margin-top:24px;'>Kết quả thông tin trích xuất</h2>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption=f"Biển số {plate_idx}", width=200)
                text = ocr_plate_image_best(roi)
                st.write(f"Kết quả trích xuất: {text}")
                
                # Lưu ảnh ROI tạm ra file
                roi_path = f"images/plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{plate_idx}.jpg"
                cv2.imwrite(roi_path, roi)
                
                # --- CHANGED: remove unconditional cur.execute above and only use DB when db_available ---
                if db_available:
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
                            st.info("Đã lưu vào SQL Server")
                        else:
                            st.info("Bỏ qua lưu do trùng")
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Lỗi khi lưu vào DB: {e}")
                else:
                    st.warning("DB không khả dụng — bỏ qua lưu vào SQL Server.")
        st.session_state.upload_count += 1
    else:
        st.warning("Không phát hiện được biển số xe!")


#---------- Nếu bảng đang hiện thì tự động cập nhật ----------
if st.session_state.show_table:
    show_table()

# Run in Python REPL or a .py file
import pyodbc
print("ODBC drivers:", pyodbc.drivers())
# (no filepath)
import pyodbc, traceback

tests = [
    r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\SQLEXPRESS;UID=sa;PWD=123456",
    r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=127.0.0.1,1433;UID=sa;PWD=123456",
    r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=.\SQLEXPRESS;UID=sa;PWD=123456",
    r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=(localdb)\MSSQLLocalDB;Trusted_Connection=yes"
]

for s in tests:
    try:
        print("Trying:", s)
        conn = pyodbc.connect(s, timeout=5)
        print("OK")
        conn.close()
    except Exception as e:
        print("ERR:", e)
        traceback.print_exc()
    print("-" * 60)







