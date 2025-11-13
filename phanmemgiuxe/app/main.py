import sys

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from ..ui.main_window import MainWindow


def main() -> None:
    # Bật High DPI cho màn hình 4K / scale Windows
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # Giữ tỉ lệ scale chính xác (tránh bị làm tròn gây mờ)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    print("🚀 Starting Parking Management App...")
    
    try:
        main()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")