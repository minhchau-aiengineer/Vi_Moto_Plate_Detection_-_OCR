#                        🚗 HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE THÔNG MINH

**                  🎓 ĐỒ ÁN TỐT NGHIỆP - QUẢN LÝ BÃI ĐỖ XE THÔNG MINH**

*                   Ứng dụng desktop sử dụng AI nhận dạng biển số xe tự động*

---

## ----------👥 THÀNH VIÊN NHÓM----------
**SINH VIÊN 1**
Họ tên: Trần Minh Châu
MSSV: 21070211
Lớp: DHKHDL19A

**SINH VIÊN 2**
Họ tên: Phạm Thanh Thảo
MSSV: 22695701
Lớp: DHKHDL19A



## ----------📋 MỤC LỤC----------

|   Chương   |               Nội dung      |          Trang              |
|:----------:|:-----=================------|:---------------------------:|
|   **1**    | [👥 Thành viên nhóm]       | [↑](#-thành-viên-nhóm)       |
|   **2**    | [📖 Giới thiệu dự án]      | [↓](#-giới-thiệu-dự-án)      |
|   **3**    | [🎯 Tính năng chính]       | [↓](#-tính-năng-chính)       |
|   **4**    | [🏗️ Kiến trúc dự án]       | [↓](#️-kiến-trúc-dự-án)       |
|   **5**    | [⚙️ Yêu cầu hệ thống]      | [↓](#️-yêu-cầu-hệ-thống)      |
|   **6**    | [🔧 Cài đặt và triển khai] | [↓](#-cài-đặt-và-triển-khai) |
|   **7**    | [� Giao diện]              | [↓](#-giao-diện)             |
|   **8**    | [📁 Cấu trúc dự án]        | [↓](#-cấu-trúc-dự-án)        |
|   **9**    | [🤖 AI Models]             | [↓](#-ai-models)             |
|   **10**   | [🗄️ Database]              | [↓](#️-database)              |
|   **11**   | [� Xử lý sự cố]            | [↓](#-xử-lý-sự-cố)           |
|   **12**   | [� Cấu hình bảo mật]       | [↓](#-cấu-hình-bảo-mật)      |
|   **13**   | [� Hướng dẫn sử dụng]      | [↓](#-hướng-dẫn-sử-dụng)     |

---

## ----------📖 GIỚI THIỆU DỰ ÁN----------

### � **TẦM NHÌN DỰ ÁN**
*"Xây dựng hệ thống quản lý bãi đỗ xe thông minh, tự động hóa hoàn toàn quy trình kiểm soát ra vào bằng công nghệ AI, nhằm nâng cao hiệu quả quản lý và trải nghiệm người dùng."*

### 📋 **TỔNG QUAN**
Hệ thống Quản lý Bãi Đỗ Xe Thông minh là một ứng dụng desktop được phát triển bằng Python, tích hợp các công nghệ AI tiên tiến để tự động nhận diện biển số xe, quản lý thông tin ra vào và tạo báo cáo thống kê chi tiết.

### 🎯 **MỤC TIÊU DỰ ÁN**
|    **Mục tiêu**    |                   **Mô tả chi tiết**                            |
|:------------------:|:----------------------------------------------------------------|
| Tự động hóa        | Giảm thiểu can thiệp thủ công trong quá trình quản lý xe ra vào |
| Chính xác cao      | Đạt độ chính xác >95% trong việc nhận diện biển số xe Việt Nam  |
| Hiệu suất tối ưu   | Xử lý real-time với độ trễ <2 giây cho mỗi giao dịch            |
| Báo cáo chi tiết   | Cung cấp thống kê đa dạng và báo cáo Excel tự động              |
| Bảo mật dữ liệu    | Đảm bảo an toàn thông tin với mã hóa và backup tự động          |


### 🔄 **LUỒNG XỬ LÝ CHÍNH**


*            🎥 [Camera Input] 

*                    ↓

🎯 [YOLOv8 Detection] → 📦 [Bounding Box + Confidence]

*                    ↓

*    📝 [OCR Processing] → 🔤 [Text Recognition]

*                    ↓

*   ❓ [Quality Check] → 🤖 [Gemini AI Fallback]

*                    ↓

* 🔄 [Auto-matching Algorithm] → 🚗 [Vehicle Pairing]

*                    ↓

*   💾 [Database Storage] → 📊 [Statistics Update]

*                    ↓

*     🎵 [Audio Notification] + 🖥️ [UI Update]




## ----------⚙️ YÊU CẦU HỆ THỐNG----------

|    **COMPONENT**  |  **VERSION** |        **MÔ TẢ**         |
|:-----------------:|:-------------|:-------------------------|
| **Python**        | `3.9 - 3.11` | Ngôn ngữ lập trình chính |
| **pip**           | `21.0+`      | Package manager          |
| **SQL Server**    | `2017+`      | Database (tùy chọn)      |
| **Audio Driver**  | `DirectSound`| Cho thông báo âm thanh   |
| **Camera Driver** | `DirectShow` | Cho camera trên Windows  |

---

## 🔧 CÀI ĐẶT VÀ TRIỂN KHAI

### **Bước 1:  Tải về source code**
```bash
# Clone repository từ GitHub
git clone https://github.com/aiengineer/Vi_Moto_Plate_Detection_OCR.git

# Di chuyển vào thư mục dự án
cd PHAN-MEM-GIU-XE
```

### **Bước 2:  Tạo môi trường Python**
```bash
# Tạo môi trường Conda (KHUYẾN NGHỊ)
conda create -n giuxe_new python=3.11 -y
conda activate giuxe_new

# HOẶC tạo môi trường Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### **Bước 3:  Cài đặt dependencies**
```bash
# Cài đặt PyTorch (CPU version)
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Cài đặt các package khác
pip install -r requirements.txt

# Cài đặt package bổ sung 
pip install openpyxl psutil
```

#### **Bước 4:  Cấu hình Database**
```sql
-- Tạo database SQL Server (tùy chọn)
CREATE DATABASE plates_db;
USE plates_db;

-- Bảng sẽ được tự động tạo khi chạy ứng dụng lần đầu
```

#### **Bước 5:  Cấu hình môi trường**
```bash
# Copy file cấu hình mẫu
cp .env .env

# Chỉnh sửa file .env với thông tin của bạn
nano .env       # Linux/Mac
notepad .env    # Windows
```

#### **Bước 6: Chạy ứng dụng**
```bash
# Di chuyển vào thư mục app
cd GIU_XE/app

# Chạy ứng dụng
python main.py
```


### 🛠️ **CẤU HÌNH CHI TIẾT**

#### **📁 Cấu hình đường dẫn trong `config.py`**
```python
# ===== CONFIG ĐƯỜNG DẪN MODEL =====
DETECT_MODEL_PATH = r"./model/detection_plates/license_plate_detector.pt"
OCR_MODEL_PATH = r"./model/ocr_plates/License_Plate_OCR.pt"

# ===== CONFIG KẾT NỐI SQL SERVER =====
CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=plates_db;"
    "UID=sa;"
    "PWD=your_password"
)
```

#### ** Cấu hình âm thanh**
```python
# Đường dẫn file âm thanh
SOUND_IN_PATH = "audio/entry_sound.wav"
SOUND_OUT_PATH = "audio/exit_sound.wav"
```


---


## ---------📁 CẤU TRÚC DỰ ÁN----------

### 🗂️ **CẤU TRÚC THỦ MỤC TỔNG QUAN**

```
📁 PHAN-MEM-GIU-XE/
├── 📁 GIU_XE/                          # 🎯 Ứng dụng chính
│   ├── 📁 app/                         # 💻 Source code
│   │   ├── 🐍 main.py                  # 🚀 Entry point
│   │   ├── 🎨 ui.py                    # 🖥️ Giao diện người dùng
│   │   ├── 🤖 models.py                # 🧠 AI Models Manager
│   │   ├── 🗄️ database.py              # 💾 Database Handler
│   │   ├── ⚙️ config.py                # 🔧 Cấu hình hệ thống
│   │   ├── 🛠️ utils.py                 # 🔨 Utility Functions
│   │   ├── 👷 workers.py               # ⚡ Multi-threading
│   │   ├── 📊 statistics.py            # 📈 Analytics Engine
│   │   └── 💬 dialogs.py               # 🪟 UI Dialogs
│   ├── 📁 images/                      # 🖼️ Captured Images  
│   └── 📁 file_out/                    # 📤 Exported Files
├── 📁 model/                           # 🧠 AI Models Repository
│   ├── 📁 detection_plates/            # 🎯 License Plate Detection
│   │   ├── 📄 license_plate_detector.pt
│   │   ├── 📄 detection_plates.pt
│   │   └── 📄 plates_1.pt
│   └── 📁 ocr_plates/                  # 📝 OCR Recognition
│       ├── 📄 License_Plate_OCR.pt
│       ├── 📄 OCR_PLATES_BEST.pt
│       └── 📄 OCR_HOAN_CHINH.pt
├── 📁 app/                             # 🧪 Test Applications
│   ├── 🐍 app_giu_xe.py                # 🏠 Main parking app
│   ├── 🐍 app_camera_detection_ocr.py  # 📹 Camera test
│   └── 🐍 app_ocr_geminiai.py          # 🤖 Gemini AI test
├── 📁 audio/                           # 🔊 Sound Effects
│   ├── 🎵 moi_vao_xin_cam_on.wav       # 📥 Entry sound
│   └── 🎵 moi_ra_xin_cam_on.wav        # 📤 Exit sound
├── 📁 logo/                            # 🎨 Application Assets
│   └── 🖼️ logo_iuh.webp                # 🏫 University logo
├── 📄 requirements.txt                 # 📋 Python Dependencies
├── 📄 .env.example                     # 🔐 Environment Template
└── 📄 README.md                        # 📖 Documentation
```