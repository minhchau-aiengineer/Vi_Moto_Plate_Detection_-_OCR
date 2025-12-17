import sys
import time

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog

from ..ui.main_window import MainWindow
from ..auth import AuthService
from ..dialogs import LoginDialog
from phanmemgiuxe.utils.log_helper import log_info




# ======= Hàm main khởi động ứng dụng ======
def main() -> None:
    log_info("Khởi động hệ thống giữ xe!")
    print("\U0001F680 Starting Parking Management App...")
    t0 = time.time()


    # Giữ tỉ lệ scale chính xác (tránh bị làm tròn gây mờ)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


    # --- Bước 0: Khởi tạo QApplication ---
    t1 = time.time()
    print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo QApplication...")
    log_info(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo QApplication...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    
    # --- Bước 1: Khởi tạo AuthService và hiển thị dialog đăng nhập ---
    t2 = time.time()
    print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo QApplication mất {t2-t1:.2f}s")
    log_info(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo QApplication mất {t2-t1:.2f}s")
    auth_service = AuthService()


    # --- Bước 2: Hiển thị dialog đăng nhập ---
    print(f"[Startup] [{time.strftime('%H:%M:%S')}] Hiển thị dialog đăng nhập...")
    log_info(f"[Startup] [{time.strftime('%H:%M:%S')}] Hiển thị dialog đăng nhập...")
    login_dlg = LoginDialog(auth_service)
    if login_dlg.exec() != QDialog.DialogCode.Accepted or login_dlg.logged_in_user is None:
        sys.exit(0)


    # --- Bước 3: Khởi tạo MainWindow ---
    user_obj = login_dlg.logged_in_user
    if hasattr(user_obj, '__dict__'):
        user_dict = user_obj.__dict__
    else:
        user_dict = {
            'username': getattr(user_obj, 'username', ''),
            'full_name': getattr(user_obj, 'full_name', ''),
            'role': getattr(user_obj, 'role', 'GUARD'),
        }

    print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo MainWindow...")
    log_info(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo MainWindow...")
    
    
    # --- Khởi tạo MainWindow với thông tin user ---
    t3 = time.time()
    w = MainWindow(current_user=user_dict)
    t4 = time.time()
    print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo MainWindow mất {t4-t3:.2f}s")
    log_info(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo MainWindow mất {t4-t3:.2f}s")
    w.show()


    # --- Kết thúc khởi động ---
    print(f"[Startup] Tổng thời gian khởi động app: {time.time()-t0:.2f}s")
    log_info(f"[Startup] Tổng thời gian khởi động app: {time.time()-t0:.2f}s")
    
    
    # --- Chạy vòng lặp chính của ứng dụng ---
    sys.exit(app.exec())





# ===== Chạy hàm main nếu chạy file này trực tiếp =====
if __name__ == "__main__":
    main()




