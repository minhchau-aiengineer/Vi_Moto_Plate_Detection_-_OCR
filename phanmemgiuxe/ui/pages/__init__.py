"""
Các trang (page) chính của ứng dụng:
- CameraPage    : màn hình camera / giám sát
- HistoryPage   : lịch sử xe ra vào
- SearchPage    : trang lọc / tìm kiếm nâng cao
- StatisticsPage: thống kê
"""

from .cameras.camera import CameraPage
from .historis.history import HistoryPage
from .historis.search import SearchPage
from .statistics.statistics import StatisticsPages
from .config import ConfigPage

__all__ = [
    "CameraPage",
    "HistoryPage",
    "SearchPage",
    "StatisticsPages",
    "ConfigPage",
]
