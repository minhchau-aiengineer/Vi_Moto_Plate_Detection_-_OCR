# ui/__init__.py
"""
Package giao diện (UI) của ứng dụng.

Cấu trúc:

    ui/
        __init__.py          # file này
        main_window.py       # class MainWindow (kết hợp các page)
        theme.py             # style / stylesheet chung
        widgets.py           # các UI component nhỏ dùng chung
        pages/
            __init__.py
            statistics.py    # trang / logic THỐNG KÊ
            history.py       # trang / logic BẢNG LỊCH SỬ + DETAIL
            search.py        # trang / logic TRANG TÌM KIẾM
            camera.py        # trang / logic CAMERA + hiển thị ảnh

Khi dùng từ bên ngoài (vd: trong main.py), bạn chỉ cần:

    from ui import MainWindow

hoặc nếu cần truy cập trực tiếp từng page:

    from .pages import StatisticsPageBuilder
"""

from .main_window import MainWindow  # export MainWindow ra ngoài

# (tuỳ chọn) nếu sau này bạn có các class/factory chính trong ui/pages,
# bạn có thể re-export ở đây cho tiện. Ví dụ:
#
# from .pages.statistics import StatisticsPageBuilder
# from .pages.history import HistoryPageBuilder
# from .pages.search import SearchPageBuilder
# from .pages.camera import CameraPageBuilder
#
# __all__ = [
#     "MainWindow",
#     "StatisticsPageBuilder",
#     "HistoryPageBuilder",
#     "SearchPageBuilder",
#     "CameraPageBuilder",
# ]

# Hiện tại, chỉ cần export MainWindow là đủ:
__all__ = ["MainWindow"]
