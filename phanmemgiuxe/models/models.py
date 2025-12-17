import os, cv2, re
from ultralytics import YOLO
from ..utils.utils import has_boxes, norm_char, preprocess_for_ocr
from ..config.config import DETECT_MODEL_PATH, OCR_MODEL_PATH
from dotenv import load_dotenv
from phanmemgiuxe.utils.log_helper import log_info



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
    log_info("Gemini init failed: {_e}")
    GEMINI_READY = False





class Models:

    def __init__(self, det_path: str, ocr_path: str):
        self.ok = True; self.err = ""
        try:
            self.det = YOLO(det_path)
            self.ocr = YOLO(ocr_path)
        except Exception as e:
            self.ok = False; self.err = str(e)
            log_info(f"[Models] Lỗi khởi tạo model: {e}")




    def detect_plates(self, frame):
        plates, boxed = [], None 
        try:
            boxed = frame.copy() 
            results = self.det(frame, verbose=False)
            for r in results:
                if not has_boxes(r): 
                    continue
                xyxy_np = r.boxes.xyxy.cpu().numpy().astype(int)
                for (x1,y1,x2,y2) in xyxy_np:
                    pad=8
                    fh, fw = frame.shape[:2] 
                    x1=max(0,x1-pad); y1=max(0,y1-pad)
                    x2=min(fw-1,x2+pad); y2=min(fh-1,y2+pad)

                    # Cắt ROI từ frame gốc
                    roi = frame[y1:y2, x1:x2].copy()
                    if roi.size == 0: 
                        continue 
                    plates.append(((x1,y1,x2,y2), roi))

                    # Vẽ lên ảnh copy
                    cv2.rectangle(boxed,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(boxed,"License Plate",(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        except Exception as e:
            print(f"Lỗi detect_plates: {e}")
            log_info(f"Lỗi detect_plates: {e}")
            return [], frame

        return plates, boxed if boxed is not None else frame





    def ocr_plate_yolo(self, roi):
        if roi is None or roi.size == 0: 
            return "", "" 
        try:
            roi_pre = preprocess_for_ocr(roi)
            input_roi = roi_pre if roi_pre is not None and roi_pre.size > 0 else roi
            res = self.ocr(input_roi, verbose=False) 
            text_raw=""

            for r in res:
                if not has_boxes(r): 
                    continue
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
                if len(boxes)<=7 or (max(ys)-min(ys) < 0.2 * h_roi): 
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
            log_info(f"Lỗi ocr_plate_yolo: {e}")
            return "", "" 





    def ocr_plate_gemini_from_path(self, image_path: str):
        if not GEMINI_READY: 
            return "", ""
        try:
            img = Image.open(image_path)
        except Exception as e:
            print("Gemini open image error:", e); 
            log_info("Gemini open image error: {e}")
            return "", ""
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = ("Đây là ảnh biển số xe Việt Nam. "
                      "Hãy trích xuất CHÍNH XÁC chuỗi biển số và chỉ trả về chuỗi đó. "
                      "VD: '29-P1 123.45' hoặc '50-Z8 888.88'.")
            resp = model.generate_content([prompt, img])
            raw = (resp.text or "").strip().replace("\n", " ")
            return self._format_text(raw)
        except gexceptions.GoogleAPICallError as e:
            print("Gemini API error:", e); 
            log_info("Gemini API error: {e}")
            return "", ""
        except Exception as e:
            print("Gemini unknown error:", e); 
            log_info("Gemini unknown error: {e}")
            return "", ""







    @staticmethod
    def _format_text(text_raw: str):
        """
        Chuẩn hóa chuỗi biển số: giữ lại A–Z, 0–9; viết hoa.
        Trả về (formatted, raw_clean) với formatted theo quy tắc VN:
        - NN-XX 12345  (2 chữ cái)
        - NN-X1 12345  (1 chữ + 1 số, kiểu C1)
        - Nếu không khớp, trả về nguyên gốc.
        LƯU Ý: KHÔNG chèn dấu chấm giữa 5 chữ số theo yêu cầu người dùng.
        """
        raw = re.sub(r'[^A-Za-z0-9]', '', (text_raw or '')).upper()
        if len(raw) < 6:
            return (text_raw or "", text_raw or "")

        # Phải bắt đầu bằng 2 chữ số tỉnh/thành
        if not raw[:2].isdigit():
            return (text_raw or "", text_raw or "")

        # Tách phần series sau 2 số đầu: 1–2 chữ cái, có thể kèm 1 số (ví dụ C1)
        i = 2
        # gom tối đa 2 chữ cái
        letters = ""
        while i < len(raw) and raw[i].isalpha() and len(letters) < 2:
            letters += raw[i]
            i += 1

        # tùy biến: nếu sau letters là 1 chữ số (kiểu C1) thì đưa vào series
        digit_in_series = ""
        if i < len(raw) and raw[i].isdigit() and len(letters) in (1,):  # chỉ gắn thêm khi dạng chữ+1 số (C1)
            digit_in_series = raw[i]
            i += 1

        series = letters + digit_in_series
        number = raw[i:]

        # Chỉ định dạng khi phần còn lại toàn số
        if not number.isdigit() or len(series) == 0:
            return (text_raw or "", text_raw or "")

        # KHÔNG chấm: giữ 5 số liền nhau theo yêu cầu ('04953' -> '04953')
        formatted = f"{raw[:2]}-{series} {number}"
        return formatted, text_raw or ""
