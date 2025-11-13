# ui/theme.py
"""
Định nghĩa theme (giao diện) chung cho toàn bộ ứng dụng.

- apply_global_theme(window): áp dụng stylesheet chung cho QMainWindow.
- normalize_button(*buttons): chuẩn hoá hành vi nút (không auto default, không focus viền xanh).
- apply_button_style(button, css): gán stylesheet cho nút (thường dùng với common_btn_style).

File này chỉ chứa phần "look & feel", không chứa logic nghiệp vụ.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QPushButton

# ======================================================================
#  STYLESHEET CHUNG
# ======================================================================

# Đây là stylesheet bạn dùng trong _init_theme cũ, mình gom lại thành hằng số.
BASE_STYLESHEET = """
* { 
    color: #000000; 
}

QMainWindow, QWidget { 
    background: #ffffff; 
}

/* Sidebar bên trái */
QWidget#SideBar { 
    background: #ffffff; 
    border-right: 2px solid #e6e6e6;
}

/* ScrollArea chung */
QScrollArea {
    border: none;
    background: #ffffff;
}

/* GroupBox chung (khung chức năng) */
QGroupBox { 
    background: #ffffff; 
    font-weight: 600; 
    border: 2px solid #e6e6e6; 
    border-radius: 12px; 
    margin-top: 8px; 
    padding-top: 10px; 
}

QGroupBox::title { 
    subcontrol-origin: margin; 
    left: 10px; 
    padding: 0 6px; 
    background: #ffffff; 
}

/* Card layout cho các khung ảnh/camera/... */
QFrame[class="card-wrap"] { 
    background: #e6e6e6; 
    border-radius: 14px; 
}

QFrame[class="card"] { 
    background: #ffffff; 
    border-radius: 12px; 
}

QFrame[class="title-wrap"]{ 
    background: #e6e6e6; 
    border-radius: 12px; 
}

QLabel[class="title"] { 
    font: 700 18px "Segoe UI"; 
    padding: 6px 10px; 
    background: #ffffff; 
    border-radius: 10px; 
}

/* Input text chung */
QLineEdit { 
    height: 28px; 
    background: #ffffff; 
    border: 1px solid #e0e0e0; 
    border-radius: 8px; 
    padding: 2px 6px; 
}

/* Bảng dữ liệu chung */
QTableWidget { 
    background: #ffffff; 
    gridline-color: #e6e6e6; 
}
"""

# Bạn có thể xài chuỗi này để tái sử dụng khi style nút:
COMMON_BUTTON_STYLE = "height:34px; font-weight:600; border-radius:10px; padding:4px 10px;"


# ======================================================================
#  HÀM ÁP DỤNG THEME
# ======================================================================

def apply_global_theme(window: QMainWindow) -> None:
    """
    Áp dụng theme (stylesheet) chung cho toàn bộ ứng dụng.

    Gọi trong MainWindow.__init__:
        apply_global_theme(self)
    """
    # Font mặc định (tuỳ bạn, có thể bỏ nếu không cần)
    default_font = QFont("Segoe UI", 9)
    window.setFont(default_font)

    # Áp dụng stylesheet
    window.setStyleSheet(BASE_STYLESHEET)


# ======================================================================
#  HELPER STYLE NÚT
# ======================================================================

def normalize_button(*buttons: QPushButton) -> None:
    """
    Chuẩn hoá các QPushButton:
    - Không auto-default, không default.
    - Không flat.
    - Không focus (đỡ viền xanh / dotted khi tab).
    - SizePolicy: Minimum x Fixed.
    """
    for b in buttons:
        if b is None:
            continue
        b.setAutoDefault(False)
        b.setDefault(False)
        b.setFlat(False)
        b.setFocusPolicy(Qt.NoFocus)
        b.setSizePolicy(
            b.sizePolicy().horizontalPolicy(),
            b.sizePolicy().verticalPolicy() if b.sizePolicy().verticalPolicy() != 0 else Qt.SizeHintRole,
        )
        # Thực tế chỉ cần:
        # b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # nhưng tránh import thêm QSizePolicy ở đây nếu không cần.


def apply_button_style(button: QPushButton, css: str) -> None:
    """
    Gán stylesheet cho button.

    Thường kết hợp với COMMON_BUTTON_STYLE, ví dụ:

        from .theme import COMMON_BUTTON_STYLE, apply_button_style

        apply_button_style(
            self.btn_start1,
            f"QPushButton{{ {COMMON_BUTTON_STYLE} background:#d1fadf; border:1px solid #a6f4c5; }}"
            "QPushButton:hover{{ background:#c3f7d6; }}"
            "QPushButton:disabled{{ background:#ecfdf3; color:#777; border-color:#d7fbe7; }}"
        )
    """
    if button is not None:
        button.setStyleSheet(css)
