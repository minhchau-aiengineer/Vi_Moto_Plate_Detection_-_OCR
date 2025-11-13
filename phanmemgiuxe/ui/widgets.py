# ui/widgets.py
"""
Các UI component nhỏ dùng chung trong toàn ứng dụng.

Bao gồm:
- add_shadow(frame): thêm hiệu ứng đổ bóng cho QFrame.
- StatsCard: card khung dùng trong trang thống kê.
- KPIChip: ô hiển thị KPI nổi bật.
- make_card(title, content): tạo khung "card" có tiêu đề dùng cho các vùng hiển thị ảnh/camera.

Các component này chỉ lo phần trình bày, không chứa logic nghiệp vụ.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
    QGraphicsDropShadowEffect,
)


# ======================================================================
#  HIỆU ỨNG BÓNG (SHADOW)
# ======================================================================

def add_shadow(frame: QFrame, blur: int = 14, alpha: int = 30, dy: int = 2) -> None:
    """
    Thêm hiệu ứng đổ bóng (shadow) cho một QFrame.

    :param frame: QFrame cần đổ bóng.
    :param blur: Độ mờ của shadow.
    :param alpha: Độ đậm (alpha) của màu shadow (0-255).
    :param dy: Độ lệch theo trục dọc (offset Y).
    """
    if frame is None:
        return

    eff = QGraphicsDropShadowEffect(frame)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(QColor(0, 0, 0, alpha))
    frame.setGraphicsEffect(eff)


# ======================================================================
#  CARD THỐNG KÊ (StatsCard)
# ======================================================================

class StatsCard(QFrame):
    """
    Card component dùng trong trang thống kê.

    - Có viền và nền trắng.
    - Có một tiêu đề (QLabel) trên cùng.
    - Bên trong là một QVBoxLayout để bạn add thêm widget/bảng/etc.

    Ví dụ sử dụng:

        card = StatsCard("Xe đang trong bãi")
        inner_layout = card.layout()  # QVBoxLayout
        inner_layout.addWidget(table_widget)

    Stylesheet có thể target bằng:
        QFrame#StatsCard { ... }
        QLabel#StatsCardTitle { ... }
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatsCard")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title_label = QLabel(title, self)
        title_label.setObjectName("StatsCardTitle")
        layout.addWidget(title_label)

        # Thêm shadow nhẹ cho card
        add_shadow(self)


# ======================================================================
#  KPI CHIP (ô KPI lớn nổi bật)
# ======================================================================

class KPIChip(QFrame):
    """
    Component KPI dạng "chip" (ô lớn) dùng để hiển thị số liệu nổi bật.

    Các phần tử:
    - title_label (QLabel) : tiêu đề, nhỏ, bold.
    - value_label (QLabel) : giá trị, to, đậm.
    - background màu mềm (bg) tuỳ truyền vào.

    Stylesheet có thể target bằng:
        QFrame#KPIChip { ... }
        QLabel#KpiTitle { ... }
        QLabel#KpiValue { ... }
    """

    def __init__(self, title: str, value: str = "--", bg: str = "#FFFFFF",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("KPIChip")
        self.bg = bg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(6)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("KpiTitle")
        layout.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.value_label = QLabel(value, self)
        self.value_label.setObjectName("KpiValue")
        layout.addWidget(self.value_label, 0, Qt.AlignmentFlag.AlignHCenter)

        # Đổ bóng nhẹ
        add_shadow(self, blur=10, alpha=25, dy=1)

        # Đặt background color
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(bg))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    # --------------------------------------------------------------
    #  CẬP NHẬT NỘI DUNG
    # --------------------------------------------------------------

    def update_value(self, value: str) -> None:
        """Cập nhật giá trị hiển thị (value_label)."""
        self.value_label.setText(value)

    def update_title(self, title: str) -> None:
        """Cập nhật tiêu đề hiển thị (title_label)."""
        self.title_label.setText(title)


# ======================================================================
#  CARD BỌC TIÊU ĐỀ + NỘI DUNG (cho camera / ảnh)
# ======================================================================

def make_card(title: str, content: QWidget) -> tuple[QFrame, QLabel]:
    """
    Tạo một "card" để bọc nội dung với tiêu đề.

    Cấu trúc:

        wrap (QFrame, class="card-wrap")
          └─ card (QFrame, class="card")
             ├─ title_wrap (QFrame, class="title-wrap")
             │    └─ title_label (QLabel, class="title")
             └─ content (QWidget do bạn truyền vào)

    Trả về:
        (wrap, title_label)

    Bạn có thể dùng như sau:

        scene_card, scene_title_lbl = make_card("Image_BOX", self.lbl_scene)
        layout.addWidget(scene_card)

    Stylesheet:
        QFrame[class="card-wrap"]  { ... }
        QFrame[class="card"]       { ... }
        QFrame[class="title-wrap"] { ... }
        QLabel[class="title"]      { ... }
    """
    # Frame ngoài cùng
    wrap = QFrame()
    wrap.setProperty("class", "card-wrap")
    wrap_layout = QVBoxLayout(wrap)
    wrap_layout.setContentsMargins(2, 2, 2, 2)
    wrap_layout.setSpacing(0)

    # Card bên trong
    card = QFrame(wrap)
    card.setProperty("class", "card")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(8, 8, 8, 8)
    card_layout.setSpacing(8)

    # Thanh tiêu đề
    title_wrap = QFrame(card)
    title_wrap.setProperty("class", "title-wrap")
    title_layout = QHBoxLayout(title_wrap)
    title_layout.setContentsMargins(2, 2, 2, 2)
    title_layout.setSpacing(4)

    title_label = QLabel(title, title_wrap)
    title_label.setProperty("class", "title")
    title_layout.addWidget(title_label)

    # Thêm tiêu đề và nội dung vào card
    card_layout.addWidget(title_wrap)
    if content is not None:
        card_layout.addWidget(content, 1)

    wrap_layout.addWidget(card)

    return wrap, title_label
