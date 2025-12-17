from __future__ import annotations

from typing import Optional, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QSizePolicy,
    QFrame,
)

from .pages import CameraPage, HistoryPage, ConfigPage
from .pages.statistics import StatisticsPages
from ..statistics.statistics import ParkingStatistics
from .pages.cameras.camera_4view import Camera4ViewPage

try:
    from .theme import APP_STYLESHEET  # type: ignore
except (ImportError, AttributeError):
    APP_STYLESHEET = ""

try:
    from ..auth import User  # type: ignore
except Exception:
    class User:  # type: ignore
        def __init__(self) -> None:
            self.username = "unknown"
            self.full_name = "Unknown"
            self.role = "GUARD"




# ===== PHÂN QUYỀN THEO ROLE =====
ROLE_PERMISSIONS = {
    "ADMIN": {"camera","camera2", "history", "statistics", "config"},
    "MANAGER": {"camera", "camera2", "history", "statistics", "config"},
    "GUARD": {"camera", "camera2", "history"},  
}



# ===== NHÃN ROLE BẰNG TIẾNG VIỆT =====

ROLE_LABELS_VN = {
    "ADMIN": "Admin",
    "MANAGER": "Quản lý",
    "GUARD": "Bảo vệ",
}






# ===== Main Window =====
class MainWindow(QMainWindow, StatisticsPages):
    """
    Cửa sổ chính của ứng dụng:

    - Thanh tab trên cùng: Camera / Lịch sử / Tìm kiếm / Thống kê / Cấu hình
    - Khu vực thân: QStackedWidget chứa các page:
        + CameraPage
        + HistoryPage
        + StatisticsPage
        + ConfigPage
    - Phân quyền theo current_user.role:
        + GUARD : chỉ Camera + Lịch sử
        + MANAGER/ADMIN: full
    """





    # === Khởi tạo cửa sổ chính ===
    def __init__(self, current_user=None, parent=None) -> None:
        super().__init__(parent)

        # Cho phép truyền vào dict hoặc object User
        if isinstance(current_user, dict):
            class UserObj:
                def __init__(self, d):
                    self.username = d.get('username', '')
                    self.full_name = d.get('full_name', '')
                    self.role = d.get('role', 'GUARD')
            self.current_user = UserObj(current_user)
        else:
            self.current_user = current_user

        # Service thống kê đọc dữ liệu từ DB
        self.stats_service = ParkingStatistics()

        # Dict lưu button và index page trong QStackedWidget
        self._nav_buttons: Dict[str, QPushButton] = {}
        self._page_indexes: Dict[str, int] = {}

        self.setWindowTitle("HỆ THỐNG GIỮ XE")
        self.resize(1200, 720)

        self._init_theme()
        self._init_ui()
        self._apply_role_permissions()






    # === Áp dụng theme / stylesheet chung cho toàn app ===
    def _init_theme(self) -> None:
        base = APP_STYLESHEET or ""

        extra = """
        /* NỀN CHUNG CẢ APP */
        QMainWindow {
            background-color: #f5f5f7;
        }
        QWidget#CentralRoot {
            background-color: #f5f5f7;
        }
        QFrame#MainBody {
            background-color: #f5f5f7;
        }
        QFrame#NavBar {
            background-color: #f5f5f7;
            border-bottom: 1px solid #e5e7eb;
        }

        /* Sidebar trong các page (nếu có đặt objectName) */
        QFrame#SideBarFrame, QWidget#SideBarFrame {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        /* Các page chính dùng nền xám rất nhạt */
        QWidget#HistoryPageRoot,
        QWidget#SearchPageRoot,
        QWidget#CameraPageRoot {
            background-color: #f5f5f7;
        }

        /* ===== BẢNG LỊCH SỬ: nền trắng, CHỮ MÀU ĐEN ===== */
        QTableView, QTableWidget {
            background-color: #ffffff;
            color: #111827;                 /* chữ đen */
            gridline-color: #9ca3af;
        }
        QHeaderView::section {
            background-color: #e5e7eb;
            color: #111827;
            padding: 4px;
            border: 1px solid #9ca3af;
        }

        /* ===== KHUNG (GROUPBOX) VIỀN ĐEN ĐẬM ===== */
        QGroupBox {
            border: 1px solid #111827;
            border-radius: 4px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #111827;
            background-color: transparent;
        }

        /* ===== NÚT TAB TRÊN CÙNG: CAMERA / LỊCH SỬ... ===== */
        QPushButton#TopNavButton {
            background-color: #d1d5db;          /* nền xám rõ hơn */
            border: 1px solid #9ca3af;
            border-radius: 4px;
            padding: 6px 18px;
            font-weight: 600;
            color: #111827;
        }
        QPushButton#TopNavButton:hover {
            background-color: #9ca3af;          /* hover xám đậm hơn */
        }
        QPushButton#TopNavButton:checked {
            background-color: #6b7280;          /* khi đang được chọn: xám đậm */
            border: 1px solid #374151;
            color: #ffffff;                     /* chữ trắng cho dễ nhìn */
        }
        """

        self.setStyleSheet(base + extra)

    
    
    
    
    
    # === Khởi tạo UI chính: thanh tab + stacked pages + status bar ===
    def _init_ui(self) -> None:
        import time
        # -------- Central widget + layout root --------
        central = QWidget(self)
        central.setObjectName("CentralRoot")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ===== Thanh TAB TRÊN CÙNG =====
        nav_bar = QFrame()
        nav_bar.setObjectName("NavBar")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 8, 8, 8)
        nav_layout.setSpacing(4)

        self.btn_tab_camera = self._create_nav_button("Camera", key="camera")
        self.btn_tab_history = self._create_nav_button("Lịch sử", key="history")
        self.btn_tab_stats = self._create_nav_button("Thống kê", key="statistics")
        self.btn_tab_config = self._create_nav_button("Cấu hình", key="config")
        self.btn_tab_camera2 = self._create_nav_button("Camera 2", key="camera2")

        nav_layout.addWidget(self.btn_tab_camera)
        nav_layout.addWidget(self.btn_tab_history)
        nav_layout.addWidget(self.btn_tab_stats)
        nav_layout.addWidget(self.btn_tab_config)
        nav_layout.addWidget(self.btn_tab_camera2)
        nav_layout.addStretch(1)

        root.addWidget(nav_bar, 0)

        # ===== Khu vực thân chính (Stacked pages) =====
        body = QFrame()
        body.setObjectName("MainBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.setObjectName("MainStack")

        # --- Page Camera ---
        t_cam = time.time()
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo CameraPage...")
        self.page_camera = CameraPage(parent=self)
        self.page_camera.setObjectName("CameraPageRoot")
        self._page_indexes["camera"] = self.stack.addWidget(self.page_camera)
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo CameraPage mất {time.time()-t_cam:.2f}s")

        # --- Page History ---
        t_hist = time.time()
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo HistoryPage...")
        self.page_history = HistoryPage(parent=self)
        self.page_history.setObjectName("HistoryPageRoot")
        self._page_indexes["history"] = self.stack.addWidget(self.page_history)
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo HistoryPage mất {time.time()-t_hist:.2f}s")

        # --- Page Statistics ---
        t_stats = time.time()
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo StatisticsPage...")
        self.page_statistics = self.build_statistics_page("")
        self._page_indexes["statistics"] = self.stack.addWidget(self.page_statistics)
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo StatisticsPage mất {time.time()-t_stats:.2f}s")

        # --- Page Config (Cấu hình) ---
        t_cfg = time.time()
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo ConfigPage...")
        self.page_config = ConfigPage(parent=self)
        self.page_config.setObjectName("ConfigPageRoot")
        self._page_indexes["config"] = self.stack.addWidget(self.page_config)
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo ConfigPage mất {time.time()-t_cfg:.2f}s")

        # --- Page Camera 2 ---
        t_cam2 = time.time()
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Bắt đầu khởi tạo Camera4ViewPage...")
        self.page_camera2 = Camera4ViewPage(parent=self)
        self._page_indexes["camera2"] = self.stack.addWidget(self.page_camera2)
        print(f"[Startup] [{time.strftime('%H:%M:%S')}] Khởi tạo Camera4ViewPage mất {time.time()-t_cam2:.2f}s")

        body_layout.addWidget(self.stack)
        root.addWidget(body, 1)

        # ===== Status bar: thông tin user đăng nhập =====
        if self.current_user:
            role = (self.current_user.role or "").upper()
            role_label = ROLE_LABELS_VN.get(role, role)
            self.statusBar().showMessage(
                f"Đăng nhập: {self.current_user.username} ({role_label})"
            )

        # Chọn page mặc định (sau đó _apply_role_permissions có thể đổi lại)
        self._switch_page("camera")

    
    
    
    
    
    # === Tạo nút tab trên cùng ===
    def _create_nav_button(self, text: str, key: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setMinimumHeight(36)
        btn.setObjectName("TopNavButton")

        self._nav_buttons[key] = btn
        btn.clicked.connect(self._on_nav_button_clicked)

        return btn

    
    
    
    
    
    # === Handler chung cho nút tab trên cùng ===
    def _on_nav_button_clicked(self) -> None:
        """
        Handler chung cho tất cả nút tab.
        Tự tìm xem nút nào phát tín hiệu rồi gọi _switch_page(key).
        """
        sender = self.sender()
        if not isinstance(sender, QPushButton):
            return

        for key, btn in self._nav_buttons.items():
            if btn is sender:
                self._switch_page(key)
                break

    
    
    
    
    
    # === Chuyển page trong QStackedWidget ===
    def _switch_page(self, key: str) -> None:
        """Chuyển sang page theo key ('camera', 'history', 'statistics', 'config')."""
        if key not in self._page_indexes:
            return

        index = self._page_indexes[key]
        self.stack.setCurrentIndex(index)

        # Cập nhật trạng thái checked của các tab
        for k, btn in self._nav_buttons.items():
            btn.setChecked(k == key)

    
    
    
    
    
    # === Áp dụng phân quyền theo role hiện tại ===
    def _apply_role_permissions(self) -> None:
        """
        Ẩn / khoá các tab theo role hiện tại.
        Nếu role không hợp lệ => coi như GUARD.
        """
        role = "GUARD"
        if self.current_user and self.current_user.role:
            role = self.current_user.role.upper()

        # Đổi title cho dễ theo dõi ai đăng nhập
        if self.current_user is not None:
            self.setWindowTitle(
                f"HỆ THỐNG GIỮ XE - {self.current_user.full_name} [{role}]"
            )

        allowed = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["GUARD"])

        # Ẩn / khoá nút theo quyền
        for key, btn in self._nav_buttons.items():
            can_use = key in allowed
            btn.setVisible(can_use)
            btn.setEnabled(can_use)

        # Chọn page mặc định là page đầu tiên mà role được phép dùng
        for prefer in ("camera", "history", "statistics", "config"):
            if prefer in allowed and prefer in self._page_indexes:
                self._switch_page(prefer)
                break
