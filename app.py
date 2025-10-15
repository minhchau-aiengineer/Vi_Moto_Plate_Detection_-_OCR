import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import easyocr
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Phát Hiện Biển Số Xe", layout="centered")
st.title("Phát Hiện Biển Số Xe")

# Khởi tạo database SQLite
DB_PATH = "plates.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame INTEGER,
    plate_idx INTEGER,
    image_path TEXT,
    ocr_text TEXT,
    timestamp TEXT
)''')
conn.commit()

WEIGHTS_PATH = "license_plate_model_sauravdb.pt"
model = YOLO(WEIGHTS_PATH)

# Hàm vẽ khung biển số lên ảnh
def draw_boxes(frame, results):
    found_plate = False
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
            pad = 0.12
            w = x2 - x1; h = y2 - y1
            x1p = max(0, int(x1 - pad*w)); y1p = max(0, int(y1 - pad*h))
            x2p = min(frame.shape[1]-1, int(x2 + pad*w)); y2p = min(frame.shape[0]-1, int(y2 + pad*h))
            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0,255,0), 2)
            cv2.putText(frame, "License Plate", (x1p, max(0, y1p-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            found_plate = True
    return frame, found_plate

# Hàm lưu vùng biển số (ROI) ra file
def save_plate_roi(frame, results, out_path_prefix):
    for r in results:
        if r.boxes is None:
            continue
        for idx, b in enumerate(r.boxes):
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
            roi = frame[y1:y2, x1:x2].copy()
            roi_path = f"{out_path_prefix}_plate_{idx}.jpg"
            cv2.imwrite(roi_path, roi)

#---------------------------------------------------------------------------- 

# Hàm nhận diện biển số bằng EasyOCR
def ocr_plate_image(image_path, txt_path=None):
    img = cv2.imread(image_path)
    if img is None or len(img.shape) not in [2, 3]:
        text_out = 'Không nhận diện được (ảnh lỗi)'
        if txt_path:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text_out)
        return text_out
    reader = easyocr.Reader(['en', 'vi'])
    result = reader.readtext(img)
    texts = [r[1] for r in result]
    text_out = ' '.join(texts) if texts else 'Không nhận diện được'
    if txt_path:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_out)
    return text_out

# Hàm đo độ nét của ảnh biển số
def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()
#----------------------------------------------------------------------------
# Giao diện chọn chức năng
option = st.sidebar.radio(
    "Chọn chức năng:",
    ["📷 Upload ảnh", "🎬 Upload video", "🎥 Dùng webcam"]
)

if option == "📷 Upload ảnh":
    st.header("Upload ảnh để phát hiện biển số xe")
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model(frame)
        frame, found_plate = draw_boxes(frame, results)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Kết quả phát hiện", use_container_width=True)
        if found_plate:
            out_dir = "output_images"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, "image_result.jpg")
            cv2.imwrite(out_path, frame)
            save_plate_roi(frame, results, os.path.join(out_dir, "image_result"))
            st.success("Đã phát hiện biển số xe! Ảnh đã được xử lý và cắt vùng biển số.")
            st.subheader("Kết quả nhận diện biển số:")
            idx = 0
            while True:
                roi_path = os.path.join(out_dir, f"image_result_plate_{idx}.jpg")
                txt_path = os.path.join(out_dir, f"image_result_plate_{idx}.txt")
                if not os.path.exists(roi_path):
                    break
                st.image(roi_path, caption=f"Biển số {idx+1}", width=200)
                text = ocr_plate_image(roi_path, txt_path)
                st.write(f"Biển số: {text}")
                idx += 1
        else:
            st.warning("Không phát hiện được biển số xe!")

elif option == "🎬 Upload video":
    st.header("Upload video để phát hiện biển số xe")
    uploaded_video = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_count = 0
        out_dir = "output_images"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plate_images = []  # Lưu thông tin các ảnh biển số: [{'text':..., 'sharp':..., 'path':..., 'frame':..., 'idx':...}]
        prev_found_plate = False
        last_plate_frame = None
        last_plate_results = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            frame, found_plate = draw_boxes(frame, results)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", use_container_width=True)
            if found_plate:
                out_path = os.path.join(out_dir, f"video_frame_{frame_count}.jpg")
                cv2.imwrite(out_path, frame)
                # Chỉ cắt và lưu vùng biển số, không nhận diện ngay
                for r in results:
                    if r.boxes is None:
                        continue
                    for idx, b in enumerate(r.boxes):
                        xyxy = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                        roi = frame[y1:y2, x1:x2].copy()
                        roi_path = os.path.join(out_dir, f"video_frame_{frame_count}_plate_{idx}.jpg")
                        cv2.imwrite(roi_path, roi)
                        # Lưu thông tin để xử lý OCR sau
                        plate_images.append({'frame': frame_count, 'idx': idx, 'roi_path': roi_path})
            frame_count += 1
        cap.release()
        time.sleep(0.5)
        try:
            os.unlink(tfile.name)
        except PermissionError:
            st.warning(f"Không thể xóa file tạm: {tfile.name}. Bạn có thể xóa thủ công sau.")
        # Sau khi video xử lý xong, thực hiện OCR cho các vùng biển số đã cắt và lưu vào database
        st.subheader("Kết quả nhận diện biển số từ video:")
        for info in plate_images:
            roi_path = info['roi_path']
            frame_num = info['frame']
            plate_idx = info['idx']
            txt_path = os.path.splitext(roi_path)[0] + '.txt'
            ocr_text = ocr_plate_image(roi_path, txt_path)
            cur.execute("INSERT INTO plates (frame, plate_idx, image_path, ocr_text, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (frame_num, plate_idx, roi_path, ocr_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            if os.path.exists(roi_path):
                st.image(roi_path, caption=f"Frame {frame_num} - Biển số {plate_idx+1}", width=200)
            st.write(f"Biển số: {ocr_text} (Frame: {frame_num})")

        # Sau khi video xử lý xong, thực hiện OCR cho tất cả các vùng biển số đã cắt
        st.subheader("Kết quả nhận diện biển số từ video:")
        idx = 0
        while True:
            roi_path = os.path.join(out_dir, f"video_frame_{idx}_plate_0.jpg")
            if not os.path.exists(roi_path):
                break
            txt_path = os.path.join(out_dir, f"video_frame_{idx}_plate_0.txt")
            text = ocr_plate_image(roi_path, txt_path)
            st.image(roi_path, caption=f"Frame {idx} - Biển số", width=200)
            st.write(f"Biển số: {text}")
            idx += 1

        # So sánh, chỉ giữ lại ảnh rõ nhất cho mỗi biển số
        plate_dict = {}
        for info in plate_images:
            plate = info['text'].strip()
            if plate == '' or plate == 'Không nhận diện được':
                continue
            if plate not in plate_dict:
                plate_dict[plate] = []
            plate_dict[plate].append(info)
        best_images = []
        for plate, infos in plate_dict.items():
            # Chọn ảnh có sharpness lớn nhất
            best = max(infos, key=lambda x: x['sharp'])
            best_images.append(best)
            # Xóa các ảnh còn lại
            for img in infos:
                if img != best:
                    try:
                        os.remove(img['path'])
                    except Exception:
                        pass
        st.success(f"Video đã xử lý xong! Đã lưu {len(best_images)} biển số rõ nhất vào output_images.")
        st.subheader("Kết quả nhận diện biển số từ video:")
        for info in best_images:
            st.image(info['path'], caption=f"Frame {info['frame']} - Biển số: {info['text']}", width=200)
            st.write(f"Biển số: {info['text']} (Độ nét: {info['sharp']:.2f})")

elif option == "🎥 Dùng webcam":
    st.header("Dùng webcam để phát hiện biển số xe")
    run_webcam = st.button("Bắt đầu webcam", key="start_webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        stframe = st.empty()
        frame_count = 0
        saved_count = 0
        saved_files = []
        out_dir = "output_images"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        stop = False
        stop_btn = st.button("Dừng webcam", key="stop_webcam")
        prev_found_plate = False
        last_plate_frame = None
        last_plate_results = None
        last_captured = None
        while not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Không đọc được frame từ webcam!")
                break
            results = model(frame)
            frame, found_plate = draw_boxes(frame, results)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Webcam Frame {frame_count}", use_container_width=True)
            if found_plate:
                last_plate_frame = frame.copy()
                last_plate_results = results
            if prev_found_plate and not found_plate and last_plate_frame is not None:
                out_path = os.path.join(out_dir, f"webcam_{frame_count-1}.jpg")
                cv2.imwrite(out_path, last_plate_frame)
                save_plate_roi(last_plate_frame, last_plate_results, os.path.join(out_dir, f"webcam_{frame_count-1}"))
                saved_files.append(out_path)
                saved_count += 1
                last_captured = out_path
                last_plate_frame = None
                last_plate_results = None
            prev_found_plate = found_plate
            frame_count += 1
            if stop_btn:
                stop = True
        cap.release()
        st.success(f"Đã dừng webcam! Đã lưu {saved_count} ảnh có biển số và vùng biển số vào output_images.")
        if last_captured:
            st.subheader("Ảnh vừa chụp:")
            st.image(last_captured, caption=os.path.basename(last_captured), width=300)
            st.write(last_captured)
            st.subheader("Kết quả nhận diện biển số vừa chụp:")
            idx = 0
            while True:
                roi_path = os.path.join(out_dir, f"webcam_{frame_count-1}_plate_{idx}.jpg")
                txt_path = os.path.join(out_dir, f"webcam_{frame_count-1}_plate_{idx}.txt")
                if not os.path.exists(roi_path):
                    break
                st.image(roi_path, caption=f"Biển số {idx+1}", width=200)
                text = ocr_plate_image(roi_path, txt_path)
                st.write(f"Biển số: {text}")
                idx += 1
        if saved_files:
            st.subheader("Các ảnh đã chụp có biển số:")
            for f in saved_files:
                st.image(f, caption=os.path.basename(f), width=200)
                st.write(f)
