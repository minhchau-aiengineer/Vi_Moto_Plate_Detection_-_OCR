import logging
import os
from logging.handlers import RotatingFileHandler



# === Đường dẫn thư mục log và file log ===
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'log')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'log')



# === Thiết lập logger ===
logger = logging.getLogger('phanmemgiuxe')
logger.setLevel(logging.INFO)



# === Handler ghi log ra file, xoay vòng khi file lớn ===
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
file_handler.setFormatter(file_formatter)



# === Handler log ra console ===
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)



# === Thêm handler vào logger ===
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)



# === Hàm tiện ích log ===
def log_info(msg):
    logger.info(msg)

def log_warning(msg):
    logger.warning(msg)

def log_error(msg):
    logger.error(msg)



# === Log test khi import để xác nhận hoạt động ===
if __name__ == "__main__":
    log_info("[TEST] Logger hoạt động - ghi vào app.log")
