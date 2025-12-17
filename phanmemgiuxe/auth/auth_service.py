# phanmemgiuxe/auth/auth_service.py

from __future__ import annotations

import os
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional, Tuple, List

from ..config.config import USE_SQL, CONN_STR

try:
    import pyodbc
except Exception:  
    pyodbc = None



# ======= Model User =======
@dataclass
class User:
    """Model đơn giản đại diện cho 1 user đăng nhập."""
    user_id: int
    username: str
    full_name: Optional[str]
    role: str           # 'GUARD', 'MANAGER'
    is_active: bool





# ======= AuthService =======
class AuthService:
    """
    Service làm việc với bảng dbo.Users:
    - Hash / verify mật khẩu
    - Đăng nhập (login)
    - Tạo / đổi mật khẩu user
    """

    # số vòng lặp cho PBKDF2 (có thể tăng thêm nếu máy mạnh)
    DEFAULT_ITERATIONS = 120_000





    # ------- Khởi tạo -------
    def __init__(self, conn_str: Optional[str] = None, use_sql: Optional[bool] = None):
        self.use_sql = USE_SQL if use_sql is None else use_sql
        self.conn_str = conn_str or CONN_STR
        self.conn = None
        self.cur = None
        if self.use_sql and pyodbc is not None:
            try:
                self.conn = pyodbc.connect(self.conn_str, autocommit=True)
                self.cur = self.conn.cursor()
                self._ensure_users_table()
            except Exception as e:
                print("[AuthService] Lỗi kết nối SQL:", e)
                self.conn = None
                self.cur = None
        else:
            print("[AuthService] USE_SQL = False hoặc thiếu pyodbc → chạy chế độ offline (no-DB).")





    # ------- Private methods -------
    def _ensure_users_table(self) -> None:
        """Đảm bảo bảng dbo.Users tồn tại (chỉ gọi khi có SQL)."""
        if not self.cur:
            return
        try:
            self.cur.execute("""
                IF OBJECT_ID('dbo.Users', 'U') IS NULL
                BEGIN
                    CREATE TABLE dbo.Users (
                        user_id        INT IDENTITY(1,1) PRIMARY KEY,
                        username       NVARCHAR(50) NOT NULL UNIQUE,
                        password_hash  NVARCHAR(256) NOT NULL,
                        full_name      NVARCHAR(100) NULL,
                        role           NVARCHAR(20) NOT NULL,
                        is_active      BIT NOT NULL DEFAULT (1),
                        last_login_at  DATETIME NULL,
                        created_at     DATETIME NOT NULL DEFAULT (GETDATE())
                    );
                END
            """)
        except Exception as e:
            print("[AuthService] Lỗi kiểm tra / tạo bảng Users:", e)





    # ------- Password hashing / verification -------
    @classmethod
    def hash_password(cls, plain_password: str) -> str:
        """
        Hash mật khẩu với PBKDF2-HMAC-SHA256.
        Trả về chuỗi: iterations$salt_hex$hash_hex
        """
        if not plain_password:
            raise ValueError("Mật khẩu không được rỗng")
        iterations = cls.DEFAULT_ITERATIONS
        salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )
        return f"{iterations}${salt.hex()}${dk.hex()}"






    # ------- Password hashing / verification -------
    @staticmethod
    def verify_password(plain_password: str, stored: str) -> bool:
        """
        Kiểm tra mật khẩu thường với hash đã lưu.
        stored dạng: iterations$salt_hex$hash_hex
        """
        try:
            parts = stored.split("$")
            if len(parts) != 3:
                return hmac.compare_digest(plain_password, stored)
            iterations = int(parts[0])
            salt = bytes.fromhex(parts[1])
            stored_hash = bytes.fromhex(parts[2])
            new_hash = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                iterations,
            )
            return hmac.compare_digest(stored_hash, new_hash)
        except Exception:
            return False





    # ------- Login -------
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Lấy thông tin user (không trả về password)."""
        if not self.cur:
            return None
        row = self.cur.execute(
            """
            SELECT user_id, username, full_name, role, is_active
            FROM dbo.Users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
        if not row:
            return None
        return User(
            user_id=row[0],
            username=row[1],
            full_name=row[2],
            role=row[3],
            is_active=bool(row[4]),
        )





    # ------- Login -------
    def _get_user_with_password(self, username: str):
        """Lấy full record (bao gồm password_hash) – chỉ dùng nội bộ."""
        if not self.cur:
            return None
        return self.cur.execute(
            """
            SELECT user_id, username, password_hash, full_name, role, is_active
            FROM dbo.Users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()





    # ------- Login -------
    def login(self, username: str, password: str) -> Tuple[bool, Optional[User], str]:
        """
        Đăng nhập:
        - ok: bool
        - user: User hoặc None
        - message: thông báo lỗi (nếu có)
        """
        username = (username or "").strip()
        if not username or not password:
            return False, None, "Vui lòng nhập đầy đủ tài khoản và mật khẩu."
        if not self.cur:
            offline_user = User(
                user_id=-1,
                username="offline",
                full_name="Offline Admin",
                role="MANAGER",
                is_active=True,
            )
            if username == "admin" and password == "admin":
                return True, offline_user, ""
            return False, None, "Không kết nối được CSDL và tài khoản không hợp lệ (offline)."
        row = self._get_user_with_password(username)
        if not row:
            return False, None, "Tên đăng nhập hoặc mật khẩu không đúng."
        user_id, u_name, pwd_hash, full_name, role, is_active = row
        if not is_active:
            return False, None, "Tài khoản đã bị khoá. Liên hệ quản lý để kích hoạt."
        if not self.verify_password(password, pwd_hash):
            return False, None, "Tên đăng nhập hoặc mật khẩu không đúng."
        try:
            self.cur.execute(
                "UPDATE dbo.Users SET last_login_at = GETDATE() WHERE user_id = ?",
                (user_id,),
            )
        except Exception as e:
            print("[AuthService] Không thể cập nhật last_login_at:", e)
        user = User(
            user_id=user_id,
            username=u_name,
            full_name=full_name,
            role=role,
            is_active=bool(is_active),
        )
        return True, user, ""





    # ------- Create user -------
    def create_user(
        self,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        role: str = "GUARD",
        is_active: bool = True,
    ) -> Tuple[bool, str]:
        """
        Tạo user mới.
        Trả về (ok, message)
        """
        if not self.cur:
            return False, "Không có kết nối CSDL."

        username = (username or "").strip()
        if not username or not password:
            return False, "Tài khoản và mật khẩu không được rỗng."

        # kiểm tra trùng
        exists = self.cur.execute(
            "SELECT 1 FROM dbo.Users WHERE username = ?",
            (username,),
        ).fetchone()
        if exists:
            return False, "Tên đăng nhập đã tồn tại."

        pwd_hash = self.hash_password(password)

        try:
            self.cur.execute(
                """
                INSERT INTO dbo.Users (username, password_hash, full_name, role, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, pwd_hash, full_name, role, 1 if is_active else 0),
            )
            return True, "Tạo tài khoản thành công."
        except Exception as e:
            print("[AuthService] Lỗi tạo user:", e)
            return False, "Lỗi khi tạo tài khoản."





    # ------- Update password -------
    def update_password(self, username: str, new_password: str) -> Tuple[bool, str]:
        """
        Đổi mật khẩu cho user.
        """
        if not self.cur:
            return False, "Không có kết nối CSDL."

        username = (username or "").strip()
        if not username or not new_password:
            return False, "Tài khoản và mật khẩu mới không được rỗng."

        row = self._get_user_with_password(username)
        if not row:
            return False, "Không tìm thấy tài khoản."

        pwd_hash = self.hash_password(new_password)
        try:
            self.cur.execute(
                "UPDATE dbo.Users SET password_hash = ? WHERE username = ?",
                (pwd_hash, username),
            )
            return True, "Đổi mật khẩu thành công."
        except Exception as e:
            print("[AuthService] Lỗi đổi mật khẩu:", e)
            return False, "Lỗi khi đổi mật khẩu."





    # ------- List users -------
    def list_users(self) -> List[User]:
        """
        Lấy danh sách users (không trả về password_hash).
        Dùng cho UI quản lý tài khoản sau này.
        """
        if not self.cur:
            return []

        rows = self.cur.execute(
            """
            SELECT user_id, username, full_name, role, is_active
            FROM dbo.Users
            ORDER BY user_id ASC
            """
        ).fetchall()

        return [
            User(
                user_id=row[0],
                username=row[1],
                full_name=row[2],
                role=row[3],
                is_active=bool(row[4]),
            )
            for row in rows
        ]





    # ------- Close connection -------
    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.cur = None

    def __del__(self) -> None:  
        self.close()
