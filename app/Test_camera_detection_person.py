# import cv2
# import matplotlib.pyplot as plt

# cap = cv2.VideoCapture("rtsp://admin:CMF@2025@172.16.120.10:554/Streaming/Channels/101?rtsp_transport=tcp")
# if not cap.isOpened():
#     raise Exception("❌ Cannot open camera")
    
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Không nhận được khung hình")
#         break
#     frame_resized = cv2.resize(frame, (640, 480))
#     cv2.imshow("Camera RTSP", frame_resized)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------- 
# Ung dung camera trong cty de phat hien doi tuong (nguoi)


import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Đường dẫn RTSP camera
rtsp_url = "rtsp://admin:CMF@2025@172.16.120.10:554/Streaming/Channels/101?rtsp_transport=tcp"

# Kết nối camera
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
	raise Exception("❌ Cannot open camera")

# Load YOLOv8 model
model = YOLO("C:/Users/Admin/Documents/IUH_Student/OCR/model/detection_person/yolov8n.pt")
tracker = DeepSort(max_age=30)
frame_count = 0

while True:
	ret, frame = cap.read()
	if not ret:
		print("❌ Không nhận được khung hình")
		break
	frame_resized = cv2.resize(frame, (640, 480))
	detections = []
	frame_count += 1
	if frame_count % 3 == 0:
		results = model(frame_resized)
		boxes = results[0].boxes
		for box in boxes:
			cls_id = int(box.cls[0])
			conf = float(box.conf[0])
			if cls_id == 0 and conf > 0.3:
				x1, y1, x2, y2 = map(int, box.xyxy[0])
				detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
		tracks = tracker.update_tracks(detections, frame=frame_resized)
		for track in tracks:
			if not track.is_confirmed():
				continue
			track_id = track.track_id
			ltrb = track.to_ltrb()
			x1, y1, x2, y2 = map(int, ltrb)
			cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.putText(frame_resized, f"ID {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

	cv2.imshow("YOLOv8 + DeepSORT Tracking", frame_resized)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()