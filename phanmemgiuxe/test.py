# phanmemgiuxe/test.py
"""
Script tạo các tài khoản mặc định cho hệ thống giữ xe.

Chạy:
    python -m phanmemgiuxe.test
"""

from __future__ import annotations
from typing import Any, List, Tuple
from .auth import AuthService




# === Main function ===
def main() -> None:
    svc = AuthService()

    # username, password, full_name, role
    default_users: List[Tuple[str, str, str, str]] = [
        ("admin",   "admin123", "Quản trị hệ thống", "ADMIN"),
        ("manager", "manager123", "Quản lý 1",        "MANAGER"),
        ("baove",   "baove123", "Bảo vệ 1",          "GUARD"),
    ]

    print("=== TẠO TÀI KHOẢN MẶC ĐỊNH ===")
    for username, password, full_name, role in default_users:
        ok, msg = svc.create_user(
            username=username,
            password=password,
            full_name=full_name,
            role=role,
        )
        status = "OK  " if ok else "SKIP"
        print(f"[{status}] {username:8s} ({role}) -> {msg}")

    print("\n=== TEST ĐĂNG NHẬP THỬ ===")
    for username, password, _, _ in default_users:  # type: ignore[reportGeneralTypeIssues]
        res: Any = svc.login(username, password)
        ok: bool = False
        user: Any = None
        msg: str = ""

        if isinstance(res, tuple):
            if len(res) == 2:
                ok, payload = res
                if ok:
                    user = payload
                    msg = "OK"
                else:
                    msg = str(payload)
            elif len(res) == 3:
                ok, user, msg = res
            else:
                ok = False
                msg = f"Unexpected login return: {res!r}"
        else:
            ok = bool(res)
            user = None
            msg = str(res)

        if ok and user is not None:
            print(f"[LOGIN OK]   {username:8s} -> role={getattr(user, 'role', '?')}")
        elif ok:
            print(f"[LOGIN OK]   {username:8s} -> (không có object User, res={res!r})")
        else:
            print(f"[LOGIN FAIL] {username:8s} -> {msg}")





# === Entry point ===
if __name__ == "__main__":
    main()
