# phanmemgiuxe/database/database.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple

from matplotlib import category
import pandas as pd

from ..config.config import USE_SQL as CONFIG_USE_SQL, CONN_STR
from ..utils.utils import plate_norm
from .fee_rules import SessionForFee, compute_fee_for_session


# ------------------------------------------------------------------
# Kiểm tra pyodbc, tránh lỗi nếu môi trường chưa cài
# ------------------------------------------------------------------
try:
    import pyodbc  # type: ignore

    HAVE_PYODBC = True
except Exception:
    pyodbc = None  # type: ignore
    HAVE_PYODBC = False

# Chỉ thực sự dùng SQL nếu cả config bật và có pyodbc
USE_SQL = bool(CONFIG_USE_SQL and HAVE_PYODBC)

# ------------------------------------------------------------------
# Phân loại phiên gửi xe (dùng cho UI)
# ------------------------------------------------------------------
SESSION_CAT_TRANSIENT = "VISITOR"   # vãng lai
SESSION_CAT_INTERNAL = "INTERNAL"   # nội bộ

VEHICLE_GROUP_LABELS = {
    SESSION_CAT_TRANSIENT: "Vãng lai",
    SESSION_CAT_INTERNAL: "Nội bộ",
}


class DB:
    """
    Lớp làm việc với SQL Server.

    - Kết nối theo CONN_STR trong config.
    - Nếu USE_SQL = False hoặc không có pyodbc thì ok=False và các hàm sẽ
      không làm gì, chỉ in log.
    """

    def __init__(self, conn_str: str = CONN_STR):
        self.ok: bool = False
        self.conn: Optional["pyodbc.Connection"] = None  # type: ignore
        self.cur: Optional["pyodbc.Cursor"] = None       # type: ignore

        if not USE_SQL:
            print("[DB] USE_SQL = False hoặc thiếu pyodbc, không kết nối SQL.")
            return

        try:
            self.conn = pyodbc.connect(conn_str, autocommit=True)  # type: ignore
            self.cur = self.conn.cursor()  # type: ignore

            # Nếu bảng chưa có thì tạo bảng tối thiểu.
            # Nếu bạn đã tạo bảng với nhiều cột hơn thì đoạn này KHÔNG chạy.
            self.cur.execute(
                """
                IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
                BEGIN
                    CREATE TABLE dbo.ParkingSessions(
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        plate_in  NVARCHAR(64)  NULL,
                        date_in   NVARCHAR(16)  NULL,
                        time_in   NVARCHAR(16)  NULL,
                        image_in  NVARCHAR(MAX) NULL,
                        plate_out NVARCHAR(64)  NULL,
                        date_out  NVARCHAR(16)  NULL,
                        time_out  NVARCHAR(16)  NULL,
                        image_out NVARCHAR(MAX) NULL,
                        match_status NVARCHAR(32) NULL,
                        created_at   DATETIME     DEFAULT GETDATE(),

                        -- Các cột mở rộng (tùy DB thật của bạn)
                        session_category NVARCHAR(32) NULL,
                        vehicle_id       INT NULL,
                        vehicle_type_id  INT NULL,
                        -- Cột phục vụ tính phí
                        fee_rule_id      INT NULL,
                        fee_amount       INT NULL
                    );
                END
                """
            )

            self.ok = True
        except Exception as e:
            print("[DB] connect error:", e)
            self.ok = False

    # ==================================================================
    # TIỆN ÍCH CHUNG
    # ==================================================================
    def _execute_one(self, sql: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        """
        Thực thi 1 câu SELECT ... WHERE ... và trả về 1 dict duy nhất (hoặc None).

        Được dùng bởi:
        - logic tính phí (fee_rules.DBProtocol)
        - các hàm helper khác trong DB nếu cần.
        """
        if not self.ok or self.conn is None:
            return None

        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(sql, params)
            row = cur.fetchone()
            if not row:
                return None

            cols = [c[0] for c in cur.description]
            return {cols[i]: row[i] for i in range(len(cols))}
        except Exception as e:
            print("[DB._execute_one] error:", e)
            return None

    # ==================================================================
    # TIỆN ÍCH NỘI BỘ
    # ==================================================================
    def _classify_plate_for_session(
        self,
        plate: str,
    ) -> tuple[str, Optional[int], Optional[int]]:
        """
        Phân loại biển số cho 1 phiên gửi xe:

        Trả về (session_category, vehicle_id, vehicle_type_id)
        - Nếu tìm thấy trong Vehicles => Nội bộ (INTERNAL)
        - Không thấy => Vãng lai (VISITOR)
        """
        if not plate:
            return SESSION_CAT_TRANSIENT, None, None

        v = self.get_vehicle_by_plate(plate)
        if not v:
            return SESSION_CAT_TRANSIENT, None, None

        return SESSION_CAT_INTERNAL, v["id"], v.get("vehicle_type_id")

    def _resolve_session_category_from_record(
        self,
        record: dict,
        plate_in: str,
        plate_out: str,
    ) -> tuple[str, Optional[int], Optional[int]]:
        """
        Dùng trong insert_history_record / update_history_record.

        Ưu tiên theo thứ tự:
        1) Nếu record có key "Loại xe" = 'Nội bộ' / 'Vãng lai'
             -> dùng trực tiếp (INTERNAL / VISITOR).
        2) Nếu record có "session_category" (INTERNAL/VISITOR/0/1)
             -> normalize về INTERNAL/VISITOR.
        3) Nếu không có gì -> tự phân loại bằng biển số:
             - ưu tiên biển vào, nếu không có thì dùng biển ra.
        """
        loai_xe_raw = (record.get("Loại xe") or "").strip().lower()

        # 1) Dùng "Loại xe" (text) nếu có
        sc: Optional[str] = None
        if loai_xe_raw:
            if "nội bộ" in loai_xe_raw or "noi bo" in loai_xe_raw:
                sc = SESSION_CAT_INTERNAL
            elif "vãng lai" in loai_xe_raw or "vang lai" in loai_xe_raw:
                sc = SESSION_CAT_TRANSIENT

        # 2) Nếu không xác định được từ text, thử lấy session_category trong record (nếu có)
        if sc is None:
            sc_val = record.get("session_category", None)
            if sc_val is not None:
                s = str(sc_val).strip().upper()
                if s in ("1", "INTERNAL", "NOI BO", "NỘI BỘ"):
                    sc = SESSION_CAT_INTERNAL
                elif s in ("0", "VISITOR", "VANG LAI", "VÃNG LAI"):
                    sc = SESSION_CAT_TRANSIENT

        # 3) Nếu vẫn chưa có -> tự phân loại bằng biển số
        if sc is None:
            ref_plate = plate_in or plate_out
            sc, vid, vtid = self._classify_plate_for_session(ref_plate)
            return sc, vid, vtid

        # Nếu sc đã xác định, cần tra thêm vehicle_id / vehicle_type_id nếu là INTERNAL
        if sc == SESSION_CAT_INTERNAL:
            ref_plate = plate_in or plate_out
            v = self.get_vehicle_by_plate(ref_plate)
            if v:
                return sc, v["id"], v.get("vehicle_type_id")
            return sc, None, None

        # sc == VISITOR -> không có vehicle_id
        return sc, None, None

    # ==================================================================
    # HÀM HỖ TRỢ UI: LẤY NHÃN "LOẠI XE" TỪ BIỂN SỐ
    # ==================================================================
    def get_vehicle_group_label_by_plate(self, plate: str) -> str:
        """
        Trả về chuỗi 'Nội bộ' / 'Vãng lai' / '' cho 1 biển số.

        - Nếu plate nằm trong Vehicles -> 'Nội bộ'
        - Nếu không -> 'Vãng lai'
        - Nếu DB không ok hoặc plate rỗng -> ''
        """
        if not self.ok or not plate:
            return ""

        cat, _, _ = self._classify_plate_for_session(plate)
        return VEHICLE_GROUP_LABELS.get(cat, "")

    # ==================================================================
    # XE NỘI BỘ – tra bảng Vehicles (lookup theo biển số)
    # ==================================================================
    def get_vehicle_by_plate(self, plate: str) -> Optional[dict]:
        """
        Tìm xe nội bộ theo biển số trong bảng dbo.Vehicles.

        Schema hiện tại (theo plates_db):
            vehicle_id, plate_number, vehicle_type_id, ...
        """
        if not self.ok or not plate:
            return None

        plate = plate.strip()
        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                SELECT TOP (1) vehicle_id, vehicle_type_id
                FROM dbo.Vehicles
                WHERE plate_number = ?
                """,
                plate,
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": int(getattr(row, "vehicle_id", row[0])),
                "vehicle_type_id": getattr(row, "vehicle_type_id", row[1]),
            }
        except Exception:
            # Nếu có lỗi (không có bảng/column) thì coi như không có xe nội bộ
            return None

    # ==================================================================
    # GHI LƯỢT VÀO / RA TỰ ĐỘNG TỪ CAMERA
    # ==================================================================
    def insert_in(self, plate: str, d: str, t: str, img_path: str) -> None:
        """
        Thêm lượt gửi XE VÀO từ camera.

        Quy trình:
        - Chuẩn hoá biển số.
        - Tra Vehicles:
            + Nếu có -> session_category = Nội bộ, lưu kèm vehicle_id, vehicle_type_id
            + Nếu không có -> session_category = Vãng lai
        - Ghi vào ParkingSessions.
        """
        if not self.ok or not img_path:
            return

        plate = (plate or "").strip()
        if not plate:
            print("[DB.insert_in] plate is empty, skip insert.")
            return

        cat, v_id, vt_id = self._classify_plate_for_session(plate)

        # Thử INSERT có cả session_category / vehicle_id / vehicle_type_id
        try:
            self.cur.execute(  # type: ignore
                """
                INSERT INTO dbo.ParkingSessions(
                    plate_in, date_in, time_in, image_in,
                    match_status, created_at,
                    session_category, vehicle_id, vehicle_type_id
                )
                VALUES (?,?,?,?, 'PENDING', GETDATE(), ?, ?, ?)
                """,
                (plate, d, t, img_path, cat, v_id, vt_id),
            )
            return
        except Exception as e:
            print("[DB.insert_in] full insert error, fallback simple insert:", e)

        # Fallback: schema cũ không có 3 cột trên
        try:
            self.cur.execute(  # type: ignore
                """
                INSERT INTO dbo.ParkingSessions(
                    plate_in, date_in, time_in, image_in, match_status, created_at
                )
                VALUES (?,?,?,?, 'PENDING', GETDATE())
                """,
                (plate, d, t, img_path),
            )
        except Exception as e2:
            print("[DB.insert_in] fallback insert error:", e2)

    def attach_out(self, plate_out: str, d: str, t: str, img_path: str) -> str:
        """
        Ghép XE RA với các lượt vào chưa có plate_out.
        Nếu khớp biển -> KHOP-BIEN-SO, nếu không -> KHONG-KHOP-BIEN-SO

        Đồng thời phân loại:
        - Nếu xe nội bộ -> session_category = INTERNAL
        - Ngược lại -> VISITOR
        """
        if not self.ok or not img_path:
            return "KHONG-KHOP-BIEN-SO"

        plate_out = (plate_out or "").strip()
        cat, v_id, vt_id = self._classify_plate_for_session(plate_out)

        try:
            rows = self.cur.execute(  # type: ignore
                """
                SELECT TOP 50 id, plate_in, session_category, vehicle_id, vehicle_type_id
                FROM dbo.ParkingSessions
                WHERE plate_out IS NULL
                ORDER BY id DESC
                """
            ).fetchall()

            match_sid: Optional[int] = None
            existing_cat = None
            existing_vid = None
            existing_vtid = None

            for r in rows:
                sid = int(getattr(r, "id", r[0]))
                plate_in = getattr(r, "plate_in", r[1]) or ""
                sc = getattr(r, "session_category", None)
                vid = getattr(r, "vehicle_id", None)
                vtid = getattr(r, "vehicle_type_id", None)

                if plate_norm(plate_in) == plate_norm(plate_out):
                    match_sid = sid
                    existing_cat = sc
                    existing_vid = vid
                    existing_vtid = vtid
                    break

            if match_sid:
                # Nếu bản ghi IN đã có session_category thì giữ nguyên,
                # nếu chưa có thì dùng cat/v_id/vt_id vừa phân loại.
                use_cat = existing_cat if existing_cat is not None else cat
                use_vid = existing_vid if existing_vid is not None else v_id
                use_vtid = existing_vtid if existing_vtid is not None else vt_id

                # Thử UPDATE với 3 cột session_category, vehicle_id, vehicle_type_id
                try:
                    self.cur.execute(  # type: ignore
                        """
                        UPDATE dbo.ParkingSessions
                        SET plate_out = ?, date_out = ?, time_out = ?,
                            image_out = ?, match_status = 'KHOP-BIEN-SO',
                            session_category = ?,
                            vehicle_id       = ?,
                            vehicle_type_id  = ?
                        WHERE id = ?
                        """,
                        (
                            plate_out,
                            d,
                            t,
                            img_path,
                            use_cat,
                            use_vid,
                            use_vtid,
                            match_sid,
                        ),
                    )
                except Exception as e:
                    print("[DB.attach_out] update with category error, fallback:", e)
                    try:
                        self.cur.execute(  # type: ignore
                            """
                            UPDATE dbo.ParkingSessions
                            SET plate_out = ?, date_out = ?, time_out = ?,
                                image_out = ?, match_status = 'KHOP-BIEN-SO'
                            WHERE id = ?
                            """,
                            (plate_out, d, t, img_path, match_sid),
                        )
                    except Exception as e2:
                        print("[DB.attach_out] fallback update error:", e2)

                # Sau khi đã có date_out/time_out -> tự động tính phí
                self._auto_compute_and_update_fee(match_sid)

                return "KHOP-BIEN-SO"

            # Không ghép được -> tạo bản ghi RA lẻ
            try:
                self.cur.execute(  # type: ignore
                    """
                    INSERT INTO dbo.ParkingSessions(
                        plate_out, date_out, time_out, image_out,
                        match_status, created_at,
                        session_category, vehicle_id, vehicle_type_id
                    )
                    VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO', GETDATE(), ?, ?, ?)
                    """,
                    (plate_out, d, t, img_path, cat, v_id, vt_id),
                )
            except Exception as e:
                print("[DB.attach_out] insert out with category error, fallback:", e)
                try:
                    self.cur.execute(  # type: ignore
                        """
                        INSERT INTO dbo.ParkingSessions(
                            plate_out, date_out, time_out, image_out,
                            match_status, created_at
                        )
                        VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO', GETDATE())
                        """,
                        (plate_out, d, t, img_path),
                    )
                except Exception as e2:
                    print("[DB.attach_out] fallback insert out error:", e2)

            return "KHONG-KHOP-BIEN-SO"

        except Exception as e:
            print("[DB.attach_out] error:", e)
            return "KHONG-KHOP-BIEN-SO"

    # ==================================================================
    # LẤY LỊCH SỬ (BẢNG) – dùng cho HistoryPage (DataFrame)
    # ==================================================================
    def fetch_history_df(
        self,
        limit: int = 10000,
        start_time=None,
        end_time=None,
        status_filter=None,
        plate_filter=None,
        vehicle_group_filter=None,
    ) -> pd.DataFrame:
        """
        Lọc theo:
        - Khoảng thời gian VÀO/RA (dựa trên date_in+time_in và date_out+time_out, đều là NVARCHAR)
        - Trạng thái (match_status)
        - Biển số (plate_in/plate_out LIKE)
        - Nhóm xe (Nội bộ / Vãng lai) – dựa trên session_category + fallback Vehicles
        Không dùng created_at.

        Trả về DataFrame với cột hiển thị:
            STT, ID,
            Ảnh vào, Biển số vào, Ngày vào, Giờ vào,
            Ảnh ra,  Biển số ra,  Ngày ra,  Giờ ra,
            Nhóm xe (Nội bộ / Vãng lai / ''), Tiền phí, Trạng thái
        """

        display_columns = [
            "ID",
            "Ảnh vào",
            "Biển số vào",
            "Ngày vào",
            "Giờ vào",
            "Ảnh ra",
            "Biển số ra",
            "Ngày ra",
            "Giờ ra",
            "Nhóm xe",
            "Tiền phí",
            "Trạng thái",
        ]

        # Các cột thực lấy từ SQL (có session_category + fee_amount để xử lý nội bộ)
        sql_columns = [
            "ID",
            "Ảnh vào",
            "Biển số vào",
            "Ngày vào",
            "Giờ vào",
            "Ảnh ra",
            "Biển số ra",
            "Ngày ra",
            "Giờ ra",
            "Trạng thái",
            "session_category",
            "fee_amount",
        ]

        if not self.ok:
            return pd.DataFrame(columns=["STT"] + display_columns)

        try:
            dt_in = "TRY_CONVERT(datetime, date_in  + ' ' + time_in , 103)"
            dt_out = "TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)"

            # LẤY THÊM session_category + fee_amount ĐỂ XỬ LÝ BÊN PYTHON
            sql = f"""
                SELECT TOP ({limit})
                    id,
                    image_in,
                    plate_in,
                    date_in,
                    time_in,
                    image_out,
                    plate_out,
                    date_out,
                    time_out,
                    match_status,
                    session_category,
                    fee_amount
                FROM dbo.ParkingSessions
            """

            where_clauses = []
            sql_params: List = []

            # --- Lọc thời gian ---
            if start_time and end_time:
                where_clauses.append(
                    f"( ({dt_in}  BETWEEN ? AND ?) OR ({dt_out} BETWEEN ? AND ?) )"
                )
                sql_params += [start_time, end_time, start_time, end_time]
            elif start_time:
                where_clauses.append(f"( {dt_in} >= ? OR {dt_out} >= ? )")
                sql_params += [start_time, start_time]
            elif end_time:
                where_clauses.append(f"( {dt_in} <= ? OR {dt_out} <= ? )")
                sql_params += [end_time, end_time]

            # --- Lọc trạng thái ---
            if status_filter and len(status_filter) > 0:
                placeholders = ",".join("?" for _ in status_filter)
                where_clauses.append(f"match_status IN ({placeholders})")
                sql_params += list(status_filter)

            # --- Lọc biển số gần đúng ---
            if plate_filter and len(str(plate_filter).strip()) > 0:
                where_clauses.append("(plate_in LIKE ? OR plate_out LIKE ?)")
                like_term = f"%{str(plate_filter).strip()}%"
                sql_params += [like_term, like_term]

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            # Ưu tiên giờ RA, nếu chưa RA thì dùng giờ VÀO
            sql += f" ORDER BY COALESCE({dt_out}, {dt_in}) DESC, id DESC"

            rows = self.cur.execute(sql, tuple(sql_params)).fetchall()  # type: ignore

            # DataFrame thô (có cột session_category + fee_amount)
            df = (
                pd.DataFrame.from_records(rows, columns=sql_columns)
                .astype(object)
                .where(pd.notnull, "")
            )

            # Chuẩn hoá Trạng thái rỗng -> PENDING
            df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})

            # ----------------------------------------------------------
            # CHUẨN BỊ DANH SÁCH BIỂN SỐ NỘI BỘ ĐỂ FALLBACK
            # ----------------------------------------------------------
            internal_plate_set = set()
            try:
                cur = self.conn.cursor()  # type: ignore
                cur.execute("SELECT plate_number FROM dbo.Vehicles")
                v_rows = cur.fetchall()
                for r in v_rows:
                    plate_raw = getattr(r, "plate_number", r[0])
                    if plate_raw:
                        internal_plate_set.add(plate_norm(str(plate_raw)))
            except Exception as e:
                # Nếu lỗi (không có bảng/column) thì coi như danh sách rỗng
                print("[DB.fetch_history_df] load Vehicles error:", e)
                internal_plate_set = set()

            # ----------------------------------------------------------
            # HÀM TÍNH "Nhóm xe" CHO TỪNG DÒNG (DÙNG CHUỖI, KHÔNG DÙNG INT)
            # ----------------------------------------------------------
            def compute_vehicle_group(row) -> str:
                sc = (row.get("session_category") or "").strip().upper()

                # 1) ƯU TIÊN session_category NẾU HỢP LỆ
                if sc == str(SESSION_CAT_INTERNAL).upper():
                    return VEHICLE_GROUP_LABELS.get(SESSION_CAT_INTERNAL, "Nội bộ")
                if sc == str(SESSION_CAT_TRANSIENT).upper():
                    return VEHICLE_GROUP_LABELS.get(SESSION_CAT_TRANSIENT, "Vãng lai")

                # 2) FALLBACK: TRA BIỂN SỐ VÀO TRONG Vehicles
                plate_in = row.get("Biển số vào") or ""
                if not plate_in:
                    return ""
                norm = plate_norm(str(plate_in))
                if norm in internal_plate_set:
                    return VEHICLE_GROUP_LABELS[SESSION_CAT_INTERNAL]
                else:
                    return VEHICLE_GROUP_LABELS[SESSION_CAT_TRANSIENT]

            df["Nhóm xe"] = df.apply(compute_vehicle_group, axis=1)

            # ----------------------------------------------------------
            # LỌC THEO NHÓM XE (NẾU CÓ)
            # vehicle_group_filter: list[str] với giá trị "Nội bộ", "Vãng lai", ...
            # ----------------------------------------------------------
            if vehicle_group_filter:
                wanted = {g.strip() for g in vehicle_group_filter if g}
                if wanted:
                    df = df[df["Nhóm xe"].isin(wanted)]

            # ----------------------------------------------------------
            # XỬ LÝ CỘT "Tiền phí"
            # ----------------------------------------------------------
            def format_fee(v):
                if v in ("", None):
                    return ""
                try:
                    return f"{int(v):,}"
                except Exception:
                    return str(v)

            df["Tiền phí"] = df["fee_amount"].apply(format_fee)

            # Bỏ các cột nội bộ, sắp xếp lại thứ tự cột hiển thị
            df = df[
                [
                    "ID",
                    "Ảnh vào",
                    "Biển số vào",
                    "Ngày vào",
                    "Giờ vào",
                    "Ảnh ra",
                    "Biển số ra",
                    "Ngày ra",
                    "Giờ ra",
                    "Nhóm xe",
                    "Tiền phí",
                    "Trạng thái",
                ]
            ]

            # Đảm bảo tên cột đúng như mô tả (display_columns)
            df.columns = display_columns

            # Thêm STT
            df.insert(0, "STT", range(1, len(df) + 1))

            return df

        except Exception as e:
            print("[DB.fetch_history_df] error:", e)
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=["STT"] + display_columns)


    # ==================================================================
    # XÓA LỊCH SỬ
    # ==================================================================
    def delete_by_ids(self, ids: List[int] | List[str]) -> None:
        if not self.ok or not ids:
            return
        try:
            placeholders = ",".join("?" for _ in ids)
            sql = f"DELETE FROM dbo.ParkingSessions WHERE id IN ({placeholders})"
            self.cur.execute(sql, tuple(int(sid) for sid in ids))  # type: ignore
        except Exception as e:
            print("delete_by_ids error:", e)

    def delete_all(self) -> None:
        if not self.ok:
            return
        try:
            self.cur.execute("DELETE FROM dbo.ParkingSessions")  # type: ignore
        except Exception as e:
            print("delete_all error:", e)

    # ==================================================================
    # THÊM / SỬA BẢN GHI LỊCH SỬ (dialog Thêm / Sửa)
    # ==================================================================
    def insert_history_record(self, record: dict) -> Optional[int]:
        """
        Thêm 1 bản ghi lịch sử đầy đủ (dùng cho dialog Thêm).
        """
        if not self.ok:
            return None

        new_id: Optional[int] = None

        try:
            img_in = record.get("Ảnh vào") or ""
            plate_in = (record.get("Biển số vào") or "").strip()
            date_in = record.get("Ngày vào") or ""
            time_in = record.get("Giờ vào") or ""

            img_out = record.get("Ảnh ra") or ""
            plate_out = (record.get("Biển số ra") or "").strip()
            date_out = record.get("Ngày ra") or ""
            time_out = record.get("Giờ ra") or ""

            match_status = (record.get("Trạng thái") or "").strip() or "PENDING"

            # Lấy session_category / vehicle_id / vehicle_type_id từ record hoặc tự phân loại
            sc, vid, vtid = self._resolve_session_category_from_record(
                record, plate_in, plate_out
            )

            try:
                # Thử insert đầy đủ
                self.cur.execute(  # type: ignore
                    """
                    INSERT INTO dbo.ParkingSessions (
                        image_in, plate_in, date_in, time_in,
                        image_out, plate_out, date_out, time_out,
                        match_status, created_at,
                        session_category, vehicle_id, vehicle_type_id
                    )
                    OUTPUT INSERTED.id
                    VALUES (?,?,?,?,?,?,?,?,?, GETDATE(), ?, ?, ?)
                    """,
                    (
                        img_in,
                        plate_in,
                        date_in,
                        time_in,
                        img_out,
                        plate_out,
                        date_out,
                        time_out,
                        match_status,
                        sc,
                        vid,
                        vtid,
                    ),
                )
            except Exception as e:
                print("insert_history_record full-insert error, fallback:", e)
                self.cur.execute(  # type: ignore
                    """
                    INSERT INTO dbo.ParkingSessions (
                        image_in, plate_in, date_in, time_in,
                        image_out, plate_out, date_out, time_out,
                        match_status, created_at
                    )
                    OUTPUT INSERTED.id
                    VALUES (?,?,?,?,?,?,?,?,?, GETDATE())
                    """,
                    (
                        img_in,
                        plate_in,
                        date_in,
                        time_in,
                        img_out,
                        plate_out,
                        date_out,
                        time_out,
                        match_status,
                    ),
                )

            # Lấy ID vừa insert (dù là full hay fallback)
            row = self.cur.fetchone()  # type: ignore
            new_id = int(row[0]) if row else None

            if new_id is not None:
                try:
                    self._auto_compute_and_update_fee(new_id)
                except Exception as fee_err:
                    print("[DB.insert_history_record] auto-compute fee error:", fee_err)
            # ====================================================

            return new_id

        except Exception as e:
            print("insert_history_record error:", e)
            return None



    def update_history_record(self, record_id: int, record: dict) -> None:
        """
        Cập nhật 1 bản ghi lịch sử theo ID.

        record có thể có thêm:
            'Loại xe'          -> 'Nội bộ' / 'Vãng lai'
            'session_category' -> INTERNAL / VISITOR / 1 / 0
        """
        if not self.ok:
            return

        try:
            img_in = record.get("Ảnh vào") or ""
            plate_in = (record.get("Biển số vào") or "").strip()
            date_in = record.get("Ngày vào") or ""
            time_in = record.get("Giờ vào") or ""

            img_out = record.get("Ảnh ra") or ""
            plate_out = (record.get("Biển số ra") or "").strip()
            date_out = record.get("Ngày ra") or ""
            time_out = record.get("Giờ ra") or ""

            match_status = (record.get("Trạng thái") or "").strip() or "PENDING"

            sc, vid, vtid = self._resolve_session_category_from_record(
                record, plate_in, plate_out
            )

            try:
                self.cur.execute(  # type: ignore
                    """
                    UPDATE dbo.ParkingSessions
                    SET image_in      = ?,
                        plate_in      = ?,
                        date_in       = ?,
                        time_in       = ?,
                        image_out     = ?,
                        plate_out     = ?,
                        date_out      = ?,
                        time_out      = ?,
                        match_status  = ?,
                        session_category = ?,
                        vehicle_id       = ?,
                        vehicle_type_id  = ?
                    WHERE id = ?
                    """,
                    (
                        img_in,
                        plate_in,
                        date_in,
                        time_in,
                        img_out,
                        plate_out,
                        date_out,
                        time_out,
                        match_status,
                        sc,
                        vid,
                        vtid,
                        int(record_id),
                    ),
                )
            except Exception as e:
                print("update_history_record full-update error, fallback:", e)
                self.cur.execute(  # type: ignore
                    """
                    UPDATE dbo.ParkingSessions
                    SET image_in      = ?,
                        plate_in      = ?,
                        date_in       = ?,
                        time_in       = ?,
                        image_out     = ?,
                        plate_out     = ?,
                        date_out      = ?,
                        time_out      = ?,
                        match_status  = ?
                    WHERE id = ?
                    """,
                    (
                        img_in,
                        plate_in,
                        date_in,
                        time_in,
                        img_out,
                        plate_out,
                        date_out,
                        time_out,
                        match_status,
                        int(record_id),
                    ),
                )

            # Sau khi cập nhật lại IN/OUT bằng tay -> tự động tính phí
            self._auto_compute_and_update_fee(int(record_id))

        except Exception as e:
            print("update_history_record error:", e)

    # ==================================================================
    # HỖ TRỢ TÍNH PHÍ GIỮ XE
    # ==================================================================
    def get_parking_session_for_fee(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu thô của 1 phiên gửi xe để phục vụ tính phí.

        Trả về dict với các key:
            id, vehicle_type_id, session_category,
            date_in, time_in, date_out, time_out

        !!! Lưu ý:
        - Nếu bảng ParkingSessions của bạn có tên cột khác thì chỉnh lại SELECT.
        """
        if not self.ok:
            return None

        sql = """
            SELECT
                id,
                vehicle_type_id,
                session_category,
                date_in,
                time_in,
                date_out,
                time_out
            FROM dbo.ParkingSessions
            WHERE id = ?
        """
        return self._execute_one(sql, (int(session_id),))

    def update_parking_fee(
        self,
        session_id: int,
        fee_rule_id: Optional[int],
        fee_amount: Optional[int],
    ) -> None:
        """
        Cập nhật thông tin tính phí cho 1 phiên gửi xe.

        YÊU CẦU:
        - Bảng dbo.ParkingSessions có 2 cột:
              fee_rule_id INT NULL,
              fee_amount  INT NULL
        Nếu chưa có 2 cột này, câu UPDATE sẽ lỗi nhẹ và được log ra.
        """
        if not self.ok:
            return

        try:
            self.cur.execute(  # type: ignore
                """
                UPDATE dbo.ParkingSessions
                SET fee_rule_id = ?,
                    fee_amount  = ?
                WHERE id = ?
                """,
                (fee_rule_id, fee_amount, int(session_id)),
            )
        except Exception as e:
            print("[DB.update_parking_fee] error:", e)


    
    def get_latest_fee_for_plate(self, plate: str) -> Optional[int]:
        """
        Lấy fee_amount mới nhất của 1 lượt đã RA theo biển số.
        Dùng để hiển thị ô 'Tiền phí' ở màn Camera.
        """
        if not self.ok or not plate:
            return None

        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                SELECT TOP (1) fee_amount
                FROM dbo.ParkingSessions
                WHERE plate_out = ? AND fee_amount IS NOT NULL
                ORDER BY id DESC
                """,
                (plate.strip(),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return int(row[0])
        except Exception as e:
            print("[DB.get_latest_fee_for_plate] error:", e)
            return None



    # ==================================================================
    # CẤU HÌNH LOẠI XE (VehicleTypes) - THEO SCHEMA plates_db
    # ==================================================================
    def get_vehicle_types(self, include_inactive: bool = False) -> List[Dict]:
        """
        Lấy danh sách loại xe.
        Nếu include_inactive = False -> chỉ lấy is_active = 1

        Trả về mỗi item:
            {
                "vehicle_type_id": int,
                "id": int,               # alias cho UI cũ
                "code": str,
                "name": str,
                "description": str,
                "is_active": bool,
            }
        """
        if not self.ok:
            return []

        try:
            cur = self.conn.cursor()  # type: ignore
            sql = """
                SELECT vehicle_type_id, code, name, description, is_active
                FROM dbo.VehicleTypes
            """
            if not include_inactive:
                sql += " WHERE is_active = 1"

            sql += " ORDER BY vehicle_type_id"
            cur.execute(sql)
            rows = cur.fetchall()

            result: List[Dict] = []
            for r in rows:
                vt_id = int(getattr(r, "vehicle_type_id", r[0]))
                result.append(
                    {
                        "vehicle_type_id": vt_id,
                        "id": vt_id,  # alias cho các chỗ UI cũ đang dùng vt["id"]
                        "code": getattr(r, "code", r[1]),
                        "name": getattr(r, "name", r[2]),
                        "description": getattr(r, "description", r[3]),
                        "is_active": bool(getattr(r, "is_active", r[4])),
                    }
                )
            return result

        except Exception as e:
            print("[DB.get_vehicle_types] error:", e)
            return []

    def insert_vehicle_type(
        self,
        name: str,
        description: str = "",
        is_active: bool = True,
    ) -> Optional[int]:
        """
        Thêm 1 loại xe mới.

        - code: tạm dùng luôn name cho đơn giản.
        - Nếu trùng UNIQUE (tên hoặc code đã tồn tại) thì:
            + Không xem là lỗi "toang app"
            + Lấy lại vehicle_type_id của dòng cũ và trả về
            + Nếu dòng cũ đang inactive thì bật is_active = 1
        """
        if not self.ok or not name.strip():
            return None

        name_clean = name.strip()

        try:
            cur = self.conn.cursor()  # type: ignore

            # Thử INSERT trước
            cur.execute(
                """
                INSERT INTO dbo.VehicleTypes(
                    code,
                    name,
                    description,
                    is_active,
                    created_at
                )
                OUTPUT INSERTED.vehicle_type_id
                VALUES (?,?,?,?, GETDATE())
                """,
                (
                    name_clean,  # code
                    name_clean,  # name
                    description or "",
                    1 if is_active else 0,
                ),
            )
            row = cur.fetchone()
            vt_id = int(row[0]) if row else None
            print(f"[DB.insert_vehicle_type] Đã thêm loại xe mới '{name_clean}' id={vt_id}")
            return vt_id

        except Exception as e:
            msg = str(e)
            # Nếu là lỗi UNIQUE KEY (trùng), ta không coi là lỗi nặng
            if "2627" in msg or "UNIQUE" in msg or "duplicate" in msg:
                try:
                    cur = self.conn.cursor()  # type: ignore
                    # Tìm lại dòng cũ theo name hoặc code
                    cur.execute(
                        """
                        SELECT TOP (1) vehicle_type_id, is_active
                        FROM dbo.VehicleTypes
                        WHERE name = ? OR code = ?
                        """,
                        (name_clean, name_clean),
                    )
                    row = cur.fetchone()
                    if row:
                        vt_id = int(getattr(row, "vehicle_type_id", row[0]))
                        is_act = bool(getattr(row, "is_active", row[1]))

                        # Nếu đang inactive thì bật lại
                        if not is_act and is_active:
                            cur.execute(
                                """
                                UPDATE dbo.VehicleTypes
                                SET is_active = 1,
                                    description = ?
                                WHERE vehicle_type_id = ?
                                """,
                                (description or "", vt_id),
                            )

                        print(
                            f"[DB.insert_vehicle_type] Loại xe '{name_clean}' đã tồn tại, dùng lại id={vt_id}."
                        )
                        return vt_id
                except Exception as e2:
                    print("[DB.insert_vehicle_type] follow-up select error:", e2)
                    return None

            # Các lỗi khác (không phải duplicate) vẫn in ra
            print("[DB.insert_vehicle_type] error:", e)
            return None

    def update_vehicle_type(
        self,
        vt_id: int,
        name: str,
        description: str = "",
        is_active: bool = True,
    ) -> None:
        """
        Cập nhật 1 loại xe theo vehicle_type_id.
        """
        if not self.ok:
            return
        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                UPDATE dbo.VehicleTypes
                SET name = ?, description = ?, is_active = ?
                WHERE vehicle_type_id = ?
                """,
                (name.strip(), description or "", 1 if is_active else 0, int(vt_id)),
            )
        except Exception as e:
            print("[DB.update_vehicle_type] error:", e)

    def delete_vehicle_type(self, vt_id: int) -> None:
        """
        Xóa 1 loại xe (theo vehicle_type_id).

        - Nếu xóa được (không bị ràng buộc FK) thì DELETE bình thường.
        - Nếu bị lỗi FK (đang có xe dùng loại này) thì chuyển sang "xóa mềm":
          UPDATE is_active = 0 để loại xe biến mất khỏi UI nhưng dữ liệu cũ vẫn an toàn.
        """
        if not self.ok:
            return

        try:
            cur = self.conn.cursor()  # type: ignore
            try:
                # Thử xóa cứng trước
                cur.execute(
                    "DELETE FROM dbo.VehicleTypes WHERE vehicle_type_id = ?",
                    (int(vt_id),),
                )
                print(f"[DB.delete_vehicle_type] Đã DELETE vehicle_type_id={vt_id}")
            except Exception as e:
                msg = str(e)
                # Nếu vướng FK (mã lỗi 547, có chữ REFERENCE constraint)
                if "547" in msg or "REFERENCE constraint" in msg:
                    # Xóa mềm: chỉ tắt is_active
                    cur.execute(
                        """
                        UPDATE dbo.VehicleTypes
                        SET is_active = 0
                        WHERE vehicle_type_id = ?
                        """,
                        (int(vt_id),),
                    )
                    print(
                        f"[DB.delete_vehicle_type] vehicle_type_id={vt_id} đang được dùng, "
                        "chuyển sang is_active = 0 (xóa mềm)."
                    )
                else:
                    # Lỗi khác thì in ra
                    print("[DB.delete_vehicle_type] error:", e)
        except Exception as e:
            print("[DB.delete_vehicle_type] outer error:", e)

    # ==================================================================
    # CẤU HÌNH XE NỘI BỘ (Vehicles) - THEO SCHEMA plates_db
    # ==================================================================
    def get_vehicles_with_type(self) -> List[Dict]:
        """
        Lấy danh sách xe nội bộ kèm tên loại xe.

        Bảng Vehicles:
            vehicle_id, plate_number, vehicle_type_id, category, owner_name,
            owner_phone, department, note, is_active, created_at

        Trả về list[dict]:
            id, plate, owner_name, vehicle_type_id, vehicle_type_name, note, is_active
        """
        if not self.ok:
            return []

        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                SELECT v.vehicle_id,
                       v.plate_number,
                       v.owner_name,
                       v.vehicle_type_id,
                       vt.name AS vehicle_type_name,
                       v.note,
                       v.is_active
                FROM dbo.Vehicles AS v
                LEFT JOIN dbo.VehicleTypes AS vt
                    ON v.vehicle_type_id = vt.vehicle_type_id
                ORDER BY v.vehicle_id ASC
                """
            )
            rows = cur.fetchall()
            result: List[Dict] = []
            for r in rows:
                result.append(
                    {
                        "id": int(getattr(r, "vehicle_id", r[0])),
                        "plate": getattr(r, "plate_number", ""),
                        "owner_name": getattr(r, "owner_name", ""),
                        "vehicle_type_id": getattr(r, "vehicle_type_id", None),
                        "vehicle_type_name": getattr(r, "vehicle_type_name", ""),
                        "note": getattr(r, "note", ""),
                        "is_active": bool(getattr(r, "is_active", True)),
                    }
                )
            return result
        except Exception as e:
            print("[DB.get_vehicles_with_type] error:", e)
            return []

    def insert_vehicle(
        self,
        plate: str,
        owner_name: str,
        vehicle_type_id: Optional[int],
        note: str = "",
        is_active: bool = True,
        category: int = 1,  # 1 = xe nội bộ (mặc định)
    ) -> Optional[int]:
        """
        Thêm 1 xe nội bộ mới.

        Ghi vào: plate_number, vehicle_type_id, category, owner_name, owner_phone,
        department, note, is_active, created_at
        """
        if not self.ok or not plate.strip():
            return None
        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                INSERT INTO dbo.Vehicles(
                    plate_number,
                    vehicle_type_id,
                    category,
                    owner_name,
                    owner_phone,
                    department,
                    note,
                    is_active,
                    created_at
                )
                OUTPUT INSERTED.vehicle_id
                VALUES (?,?,?,?,?,?,?,?, GETDATE())
                """,
                (
                    plate.strip(),
                    vehicle_type_id,
                    category,  # KHÔNG để NULL nữa
                    owner_name.strip() if owner_name else "",
                    "",  # owner_phone (rỗng)
                    "",  # department (rỗng)
                    note or "",
                    1 if is_active else 0,
                ),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None
        except Exception as e:
            print("[DB.insert_vehicle] error:", e)
            return None

    def update_vehicle(
        self,
        v_id: int,
        plate: str,
        owner_name: str,
        vehicle_type_id: Optional[int],
        note: str = "",
        is_active: bool = True,
    ) -> None:
        """
        Cập nhật xe nội bộ theo vehicle_id.
        """
        if not self.ok:
            return
        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                UPDATE dbo.Vehicles
                SET plate_number    = ?,
                    owner_name      = ?,
                    vehicle_type_id = ?,
                    note            = ?,
                    is_active       = ?
                WHERE vehicle_id = ?
                """,
                (
                    plate.strip(),
                    owner_name.strip() if owner_name else "",
                    vehicle_type_id,
                    note or "",
                    1 if is_active else 0,
                    int(v_id),
                ),
            )
        except Exception as e:
            print("[DB.update_vehicle] error:", e)

    def delete_vehicle(self, v_id: int) -> None:
        """
        Xóa 1 xe nội bộ theo vehicle_id.
        (hoặc sau này có thể đổi thành UPDATE is_active=0 để xóa mềm)
        """
        if not self.ok:
            return
        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                "DELETE FROM dbo.Vehicles WHERE vehicle_id = ?",
                (int(v_id),),
            )
        except Exception as e:
            print("[DB.delete_vehicle] error:", e)








    
    # ==================================================================
    # CẤU HÌNH PHÍ GIỮ XE (FeeRules)
    # ==================================================================
    def get_fee_rules(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        Đọc danh sách rule phí gửi xe từ bảng dbo.FeeRules.

        Trả về list dict với các key mà UI đang dùng:
            fee_rule_id, id, fee_id,
            vehicle_type_id,
            session_category, rule_type,
            price, fee_amount,
            unit, description,
            effective_from, effective_to,
            is_active
        """
        if not self.ok:
            return []

        try:
            cur = self.conn.cursor()  # type: ignore
            sql = """
                SELECT
                    fee_rule_id,
                    vehicle_type_id,
                    category,
                    rule_type,
                    price,
                    effective_from,
                    effective_to,
                    is_active,
                    description,
                    unit
                FROM dbo.FeeRules
            """
            if not include_inactive:
                sql += " WHERE is_active = 1"

            sql += " ORDER BY fee_rule_id"

            cur.execute(sql)
            rows = cur.fetchall()

            rules: List[Dict[str, Any]] = []

            for r in rows:
                fee_rule_id = int(getattr(r, "fee_rule_id", r[0]))
                vehicle_type_id = getattr(r, "vehicle_type_id", None)

                raw_category = (getattr(r, "category", "") or "").strip().upper()
                raw_rule_type = (getattr(r, "rule_type", "") or "").strip().upper()
                price_val = int(getattr(r, "price", 0) or 0)
                eff_from = getattr(r, "effective_from", None)
                eff_to = getattr(r, "effective_to", None)
                is_active = bool(getattr(r, "is_active", False))
                desc_db = getattr(r, "description", "") or ""
                unit_db = getattr(r, "unit", None)

                # chuẩn hoá session_category = INTERNAL / VISITOR / ALL
                if raw_category in ("INTERNAL", "VISITOR", "ALL"):
                    session_category = raw_category
                else:
                    session_category = "ALL"

                # Nếu DB chưa có unit thì suy ra từ rule_type
                rt = raw_rule_type
                if unit_db:
                    unit = str(unit_db)
                else:
                    if rt in ("PER_ENTRY", "PER_TURN", "PER_VISIT", "DAYTIME", "OVERNIGHT_24H"):
                        unit = "Lượt"
                    elif rt in ("PER_HOUR", "HOURLY"):
                        unit = "Giờ"
                    elif rt in ("PER_DAY", "DAILY"):
                        unit = "Ngày"
                    else:
                        unit = "Lượt"

                rules.append(
                    {
                        # cho DB / logic
                        "fee_rule_id": fee_rule_id,
                        "id": fee_rule_id,

                        # cho UI cũ đang đọc fee_id
                        "fee_id": fee_rule_id,

                        "vehicle_type_id": vehicle_type_id,
                        "session_category": session_category,
                        "rule_type": raw_rule_type,

                        # GIỮ CẢ HAI KEY: price và fee_amount
                        "price": price_val,
                        "fee_amount": price_val,

                        "unit": unit,
                        "description": desc_db,
                        "effective_from": eff_from,
                        "effective_to": eff_to,
                        "is_active": is_active,
                    }
                )

            return rules

        except Exception as e:
            print("[DB.get_fee_rules] error:", e)
            return []

    def insert_fee_rule(
        self,
        *,
        vehicle_type_id=None,
        session_category="ALL",
        rule_type="PER_ENTRY",
        price=None,
        fee_amount=None,
        unit=None,
        effective_from=None,
        effective_to=None,
        is_active=True,
        **extra,
    ) -> Optional[int]:
        """
        Thêm rule phí mới.

        UI có thể truyền price hoặc fee_amount hoặc cả hai.
        Có thể truyền thêm:
            - description: mô tả rule (ghi vào cột description).
        """

        if not self.ok:
            return None

        # DB yêu cầu vehicle_type_id NOT NULL
        if vehicle_type_id is None:
            print("[DB.insert_fee_rule] vehicle_type_id is None, skip insert.")
            return None


        # Ưu tiên price, nếu không có thì dùng fee_amount
        if price is None:
            price = fee_amount
        if price is None:
            price = 0
        else:
            price = int(price)

        # category
        cat = (session_category or "ALL").strip().upper()
        if cat not in ("INTERNAL", "VISITOR", "ALL"):
            cat = "ALL"

        # rule_type
        rt = (rule_type or "").strip().upper()
        if not rt:
            rt = "PER_ENTRY"

        # unit
        if unit is not None:
            unit = str(unit).strip() or None

        # description (tùy chọn)
        description = extra.get("description")
        if description is not None:
            description = str(description).strip() or None

        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                INSERT INTO dbo.FeeRules(
                    vehicle_type_id,
                    category,
                    rule_type,
                    price,
                    unit,
                    effective_from,
                    effective_to,
                    is_active,
                    description
                )
                OUTPUT INSERTED.fee_rule_id
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    vehicle_type_id,
                    cat,
                    rt,
                    price,
                    unit,
                    effective_from,
                    effective_to,
                    1 if is_active else 0,
                    description,
                ),
            )

            row = cur.fetchone()
            return int(row[0]) if row else None

        except Exception as e:
            print("[DB.insert_fee_rule] error:", e)
            return None

    def update_fee_rule(
        self,
        fee_rule_id: Optional[int] = None,
        fee_id: Optional[int] = None,
        *,
        vehicle_type_id: Optional[int],
        session_category: str,
        rule_type: str,
        price: int,
        unit: Optional[str] = None,
        effective_from: Optional[str] = None,
        effective_to: Optional[str] = None,
        is_active: bool = True,
        **extra: Any,
    ) -> None:
        """
        Cập nhật 1 rule phí.

        Hỗ trợ cả hai kiểu gọi:
            update_fee_rule(fee_rule_id=..., ...)
            update_fee_rule(fee_id=..., ...)
        """
        if not self.ok:
            return

        rule_id = fee_rule_id if fee_rule_id is not None else fee_id
        if rule_id is None:
            print("[DB.update_fee_rule] missing fee_rule_id/fee_id")
            return

        cat = (session_category or "ALL").strip().upper()
        if cat not in ("INTERNAL", "VISITOR", "ALL"):
            cat = "ALL"

        rt = (rule_type or "").strip().upper()
        if not rt:
            rt = "PER_ENTRY"

        if unit is not None:
            unit = str(unit).strip() or None

        # description có thể đi qua extra
        description = extra.get("description")
        if description is not None:
            description = str(description).strip() or None

        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                """
                UPDATE dbo.FeeRules
                SET vehicle_type_id = ?,
                    category       = ?,
                    rule_type      = ?,
                    price          = ?,
                    unit           = ?,
                    effective_from = ?,
                    effective_to   = ?,
                    is_active      = ?,
                    description    = ?
                WHERE fee_rule_id  = ?
                """,
                (
                    vehicle_type_id,
                    cat,
                    rt,
                    int(price),
                    unit,
                    effective_from,
                    effective_to,
                    1 if is_active else 0,
                    description,
                    int(rule_id),
                ),
            )
        except Exception as e:
            print("[DB.update_fee_rule] error:", e)

    def delete_fee_rule(self, fee_rule_id: int | str) -> None:
        """
        Xóa 1 rule phí theo fee_rule_id.
        (nếu sau này dính FK với ParkingSessions, có thể sửa thành is_active=0)
        """
        if not self.ok:
            return
        try:
            cur = self.conn.cursor()  # type: ignore
            cur.execute(
                "DELETE FROM dbo.FeeRules WHERE fee_rule_id = ?",
                (int(fee_rule_id),),
            )
        except Exception as e:
            print("[DB.delete_fee_rule] error:", e)







    def _auto_compute_and_update_fee(self, session_id: int) -> None:
        """
        Sau khi đã UPDATE date_out/time_out cho một phiên gửi,
        hàm này sẽ:
          - lấy dữ liệu phiên từ DB
          - tính phí bằng compute_fee_for_session
          - UPDATE lại FeeRulesId + FeeAmount trong ParkingSessions
        Nếu thiếu dữ liệu hoặc không có rule -> im lặng bỏ qua (không crash app).
        """
        if not self.ok:
            return

        try:
            raw = self.get_parking_session_for_fee(session_id)
            if not raw:
                return

            session = SessionForFee(
                id=int(raw["id"]),
                vehicle_type_id=raw.get("vehicle_type_id"),
                session_category=str(raw.get("session_category") or ""),
                date_in=str(raw.get("date_in") or ""),
                time_in=str(raw.get("time_in") or ""),
                date_out=str(raw.get("date_out") or ""),
                time_out=str(raw.get("time_out") or ""),
            )

            fee_rule_id, fee_amount = compute_fee_for_session(self, session)

            # Không tính được (chưa đủ giờ ra / thiếu rule / INTERNAL chưa config FREE / ...)
            if fee_amount is None:
                return

            # Cập nhật vào DB
            self.update_parking_fee(session_id, fee_rule_id, fee_amount)

        except Exception as e:
            print("[DB._auto_compute_and_update_fee] error:", e)