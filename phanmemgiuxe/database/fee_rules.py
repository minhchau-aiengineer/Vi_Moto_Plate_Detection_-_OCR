from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Optional, Dict, Any, Tuple, Protocol






# ======= Giao diện DB tối thiểu để dùng cho logic tính phí =======
class DBProtocol(Protocol):
    """
    Giao diện tối thiểu DB cần có để dùng cho logic tính phí.
    Lớp DB thật trong database.py đã đáp ứng, vì có .conn và .ok
    và có hàm _execute_one().
    """

    ok: bool
    conn: Any  # pyodbc.Connection

    def _execute_one(self, sql: str, params: tuple) -> Optional[Dict[str, Any]]:
        ...







# ======= Lớp thông tin phiên gửi xe để tính phí =======
@dataclass
class SessionForFee:
    """
    Thông tin 1 phiên gửi xe dùng để tính phí.

    - vehicle_type_id: loại xe (nếu có), dùng để chọn rule theo loại xe
    - session_category: VISITOR / INTERNAL
    - date/time: dạng text giống trong DB (dd/MM/yyyy, HH:mm[:ss])
    """

    id: int
    vehicle_type_id: Optional[int]
    session_category: str  # VISITOR / INTERNAL / ...
    date_in: str
    time_in: str
    date_out: Optional[str]
    time_out: Optional[str]



# === Hàm phụ trợ cho logic tính phí ===
def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """
    Chuyển date + time (string) thành datetime.

    Mặc định dùng format:
        - date:  dd/MM/yyyy  (vd: 19/11/2025)
        - time:  HH:mm[:ss]  (vd: 08:30 hoặc 08:30:15)

    Nếu parse không được -> trả về None.
    """
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()
    if not date_str or not time_str:
        return None

    # Chuẩn hoá time: nếu không có giây thì thêm :00
    if len(time_str.split(":")) == 2:
        time_str = time_str + ":00"

    dt_raw = f"{date_str} {time_str}"

    # Thử vài format phổ biến
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_raw, fmt)
        except Exception:
            continue
    return None





# === Chuẩn hoá category ===
def _normalize_category(cat: Any) -> str:
    """
    Chuẩn hoá session_category về:
        - 'VISITOR'  (vãng lai)
        - 'INTERNAL' (nội bộ)
    """
    if cat is None:
        return ""

    s = str(cat).strip().upper()
    if s in ("VISITOR", "VANG LAI", "VÃNG LAI", "0"):
        return "VISITOR"
    if s in ("INTERNAL", "NOI BO", "NỘI BỘ", "1"):
        return "INTERNAL"
    return s




# === Lấy rule tính phí từ DB theo vehicle_type_id, category, rule_type ===
def _pick_rule(
    db: DBProtocol, vehicle_type_id: Optional[int], category: str, rule_type: str
) -> Optional[Dict[str, Any]]:
    """
    Lấy 1 dòng từ bảng FeeRules theo:

        - vehicle_type_id (có thể NULL, khi đó lấy rule chung cho category)
        - category        (VISITOR / INTERNAL / ...)
        - rule_type       (DAYTIME / OVERNIGHT_24H / PER_DAY / FREE / ...)

    Ưu tiên:
        1) rule trùng vehicle_type_id
        2) nếu không có thì rule vehicle_type_id IS NULL (rule chung)
    """
    if not db.ok:
        return None

    # Tìm rule với vehicle_type_id cụ thể
    base_sql = """
        SELECT TOP (1)
            fee_rule_id,
            vehicle_type_id,
            category,
            rule_type,
            price,
            unit,
            description,
            is_active
        FROM dbo.FeeRules
        WHERE category = ?
          AND rule_type = ?
          AND is_active = 1
    """

    if vehicle_type_id is not None:
        sql = base_sql + " AND vehicle_type_id = ? ORDER BY fee_rule_id"
        row = db._execute_one(sql, (category, rule_type, vehicle_type_id))
        if row:
            return row

    # Nếu không có rule theo loại xe -> lấy rule chung (vehicle_type_id IS NULL hoặc ignore)
    sql2 = base_sql + " ORDER BY fee_rule_id"
    return db._execute_one(sql2, (category, rule_type))




# === Hàm chính: tính phí cho 1 phiên gửi xe ===
def compute_fee_for_session(
    db: DBProtocol, session: SessionForFee
) -> Tuple[Optional[int], Optional[int]]:
    """
    Tính (fee_rule_id, fee_amount) cho 1 phiên gửi xe.

    QUY ĐỊNH (theo mô tả của bạn):

      - INTERNAL (nội bộ): dùng rule FREE, giá = 0.
      - VISITOR (vãng lai):

            + DAYTIME (ngày):     từ 6h đến 18h cùng 1 ngày, giá 5.000
              (lấy từ FeeRules: VISITOR / DAYTIME / price = 5000)

            + OVERNIGHT_24H:      gửi qua đêm, trong vòng 24h,
                                   từ khoảng 18h -> 6h, giá 10.000
              (FeeRules: VISITOR / OVERNIGHT_24H / price = 10000)

            + PER_DAY (theo ngày): nếu thời gian gửi > 24h
               => mỗi block 24h tính 1 ngày:
                    tiền = ceil( Số giờ / 24 ) * 10.000
               (FeeRules: VISITOR / PER_DAY / price = 10000)

      Giá cụ thể (5.000 / 10.000 / 0) lấy từ bảng FeeRules,
      KHÔNG hard-code trong code, nên admin vẫn chỉnh được trong DB.

    Trả về:
        (None, None)  -> không tính được (thiếu dữ liệu, DB off, không có rule, ...)
        (rule_id, tiền)
    """
    if not db.ok:
        return None, None

    cat = _normalize_category(session.session_category)
    if not cat:
        return None, None

    # Chưa có giờ ra -> chưa tính
    if not session.date_out or not session.time_out:
        return None, None

    dt_in = _parse_datetime(session.date_in, session.time_in)
    dt_out = _parse_datetime(session.date_out, session.time_out)
    if not dt_in or not dt_out or dt_out <= dt_in:
        # Dữ liệu lỗi -> bỏ qua, không cho app crash
        return None, None


    # ----- 1) NỘI BỘ (INTERNAL) -----
    if cat == "INTERNAL":
        rule = _pick_rule(db, session.vehicle_type_id, "INTERNAL", "FREE")
        if not rule:
            # Chưa cấu hình rule FREE cho INTERNAL -> không tính được
            return None, None
        return int(rule["fee_rule_id"]), 0


    # ----- 2) VÃNG LAI (VISITOR) -----
    if cat != "VISITOR":
        return None, None
    delta_hours = (dt_out - dt_in).total_seconds() / 3600.0
    if delta_hours <= 0.0:
        return None, 0


    # ----- 2a) Tính phí theo ngày nếu > 24h -----
    if delta_hours > 24.0:
        rule = _pick_rule(db, session.vehicle_type_id, "VISITOR", "PER_DAY")
        if not rule:
            return None, None

        days = ceil(delta_hours / 24.0)  # block 24h
        price = int(rule["price"] or 0)  # thường = 10000
        amount = days * price
        return int(rule["fee_rule_id"]), amount


    # ----- 2b) Tính phí DAYTIME hoặc OVERNIGHT_24H nếu <= 24h -----
    same_day = (dt_in.date() == dt_out.date())

    # DAYTIME: trong cùng 1 ngày, nằm trong khung 6h–18h
    in_day_range = (dt_in.hour >= 6)
    out_day_range = (dt_out.hour <= 18)

    if same_day and in_day_range and out_day_range:
        rule_type = "DAYTIME"
    else:
        # Còn lại nhưng vẫn <= 24h => coi là gửi qua đêm (OVERNIGHT_24H)
        rule_type = "OVERNIGHT_24H"

    rule2 = _pick_rule(db, session.vehicle_type_id, "VISITOR", rule_type)
    if not rule2:
        return None, None

    price2 = int(rule2["price"] or 0)
    return int(rule2["fee_rule_id"]), price2
