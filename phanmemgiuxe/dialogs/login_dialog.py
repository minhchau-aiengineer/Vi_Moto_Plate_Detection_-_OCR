from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpacerItem, QSizePolicy
)

from ..auth import AuthService, User



# ======= Dialog Đăng nhập =======
class LoginDialog(QDialog):
    
    
    
    # === Khởi tạo dialog =====
    def __init__(self, auth_service: AuthService, parent=None):
        super().__init__(parent)
        
        self.auth_service: AuthService = auth_service
        self.current_user: Optional[User] = None
        self.logged_in_user: Optional[User] = None

        self.setWindowTitle("Đăng nhập hệ thống")
        self.setModal(True)
        self.setFixedSize(480, 350)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self._build_ui()
        self._apply_styles()





    # === Xây dựng giao diện UI =====
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --------- TITLE (center) ----------
        title_label = QLabel("ĐĂNG NHẬP HỆ THỐNG GIỮ XE")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # --------- Role hint (center) ---------
        role_hint = QLabel("Tài khoản được phân quyền: Bảo vệ / Quản lý.")
        role_hint.setObjectName("RoleHint")
        role_hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(role_hint)

        # --------- Username ----------
        lbl_user = QLabel("Tài khoản:")
        lbl_user.setObjectName("FieldLabel")
        layout.addWidget(lbl_user)

        # --------- Username Edit ----------
        self.ed_username = QLineEdit()
        self.ed_username.setPlaceholderText("Nhập tài khoản...")
        self.ed_username.setObjectName("UsernameEdit")
        layout.addWidget(self.ed_username)

        # --------- Password ----------
        lbl_pass = QLabel("Mật khẩu:")
        lbl_pass.setObjectName("FieldLabel")
        layout.addWidget(lbl_pass)

        # --------- Password Edit ----------
        self.ed_password = QLineEdit()
        self.ed_password.setPlaceholderText("Nhập mật khẩu...")
        self.ed_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.ed_password.setObjectName("PasswordEdit")
        layout.addWidget(self.ed_password)

        # --------- Error ----------
        self.lbl_error = QLabel("")
        self.lbl_error.setObjectName("ErrorLabel")
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        # --------- Buttons (center, rounded) ----------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        # --------- Buttons (center, rounded) ----------
        btn_row.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding))
        self.btn_login = QPushButton("Đăng nhập")
        self.btn_login.setObjectName("LoginButton")
        self.btn_login.setMinimumWidth(130)

        self.btn_cancel = QPushButton("Thoát")
        self.btn_cancel.setObjectName("CancelButton")
        self.btn_cancel.setMinimumWidth(130)

        btn_row.addWidget(self.btn_login)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding))

        layout.addLayout(btn_row)

        # --------- Signals ----------
        self.btn_login.clicked.connect(self.on_login_clicked)
        self.btn_cancel.clicked.connect(self.reject)
        self.ed_password.returnPressed.connect(self.on_login_clicked)


    # === Áp dụng style cho dialog =====
    def _apply_styles(self):
        self.setStyleSheet("""
        QDialog {
            background: #F3F4F6;
        }

        QLabel#TitleLabel {
            font-size: 18px;
            font-weight: 800;
            color: #2563EB;
        }

        QLabel#RoleHint {
            font-size: 12px;
            color: #6B7280;
            margin-bottom: 10px;
        }

        QLabel#FieldLabel {
            font-size: 12px;
            font-weight: 700;
            color: #111827;
            margin-top: 6px;
        }

        QLineEdit {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            padding: 6px 8px;
            color: #000000;
            border-radius: 6px;
        }
        QLineEdit:focus {
            border: 1px solid #2563EB;
        }

        QLabel#ErrorLabel {
            font-size: 11px;
            color: #DC2626;
        }

        QPushButton {
            font-size: 13px;
            font-weight: 600;
            padding: 8px 16px;
            min-height: 36px;
            border-radius: 8px;
        }

        QPushButton#LoginButton {
            background-color: #2563EB;
            color: white;
            border: none;
        }
        QPushButton#LoginButton:hover {
            background-color: #1E40AF;
        }

        QPushButton#CancelButton {
            background-color: #E5E7EB;
            color: #111827;
            border: none;
        }
        QPushButton#CancelButton:hover {
            background-color: #D4D4D8;
        }
        """)

    # === Hiển thị lỗi =====
    def show_error(self, msg: str):
        if not msg:
            self.lbl_error.hide()
            return
        self.lbl_error.setText(msg)
        self.lbl_error.show()




    # === Xử lý khi nhấn nút Đăng nhập =====
    def on_login_clicked(self):
        username = self.ed_username.text().strip()
        password = self.ed_password.text()

        if not username or not password:
            self.show_error("Vui lòng nhập đầy đủ thông tin.")
            return

        ok, user, msg = self.auth_service.login(username, password)
        if not ok:
            self.show_error(msg)
            return

        # đồng bộ cả 2 thuộc tính
        self.current_user = user
        self.logged_in_user = user

        self.accept()




