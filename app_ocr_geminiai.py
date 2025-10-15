
import os
from dotenv import load_dotenv  # <-- THÊM DÒNG NÀY
from google import generativeai as genai  
from google.api_core import exceptions
from PIL import Image


load_dotenv()  

# Đọc biến môi trường đã được tải
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Chưa đặt biến môi trường GEMINI_API_KEY hoặc không tìm thấy file .env")

# SỬA 3: Cấu hình API key theo cách mới
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini: {e}")
    exit()

def extract_license_plate_number(image_path: str) -> str:
    """
    Sử dụng Gemini API để trích xuất biển số xe từ ảnh đã cắt.
    """
    print(f"Đang xử lý ảnh: {image_path}...")
    
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return f"Lỗi: Không tìm thấy tệp ảnh tại {image_path}"
    except Exception as e:
        return f"Lỗi khi mở ảnh: {e}"

    # SỬA 4: Khởi tạo model theo cách mới và dùng đúng tên model
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = (
        "Đây là một ảnh chụp biển số xe máy của Việt Nam. Biển số có thể bị mờ, "
        "chói, hoặc khó đọc. Nhiệm vụ của bạn là trích xuất chính xác chuỗi "
        "ký tự và số trên biển số. "
        "**Chỉ trả lời bằng chuỗi biển số xe đã trích xuất, không thêm bất kỳ văn bản giải thích nào.** "
        "Ví dụ: '29-P1 123.45' hoặc '50-Z8 888.88'."
    )
    
    contents = [prompt, img]
    
    try:
        # SỬA 5: Gọi hàm generate_content từ đối tượng model
        response = model.generate_content(contents)
        
        # Làm sạch kết quả
        result = response.text.strip().replace('\n', ' ')
        return result

    # Xử lý các lỗi phổ biến của API
    except exceptions.GoogleAPICallError as e:
        return f"Lỗi gọi API Google: {e}"
    except Exception as e:
        return f"Lỗi không xác định khi gọi Gemini: {e}"

# --- Ví dụ Sử dụng ---
if __name__ == "__main__":
    MOCK_IMAGE_PATH = "D:/Documents/IUH_Student/OCR/images/anh3.jpg"

    if not os.path.exists(MOCK_IMAGE_PATH):
        print(f"Không tìm thấy ảnh tại: {MOCK_IMAGE_PATH}. Tạo ảnh placeholder...")
        try:
            mock_img = Image.new('RGB', (300, 100), color = 'white')
            # Thêm text giả để model có gì đó để đọc
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mock_img)
            draw.text((10, 10), "51-F8 123.45", fill="black")
            mock_img.save(MOCK_IMAGE_PATH)
            print("Đã tạo ảnh placeholder. Hãy thay thế bằng ảnh biển số thực của bạn.")
        except Exception as e:
            print(f"Không thể tạo ảnh placeholder: {e}")

    plate_number = extract_license_plate_number(MOCK_IMAGE_PATH)
    
    print("\n-------------------------------------------")
    print(f"Kết quả trích xuất từ Gemini: **{plate_number}**")
    print("-------------------------------------------")