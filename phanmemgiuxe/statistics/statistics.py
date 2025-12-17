# phanmemgiuxe/statistics/parking_statistics.py
"""
ParkingStatistics

Lớp thống kê cho hệ thống giữ xe.
Đọc dữ liệu từ SQL Server thông qua lớp DB (database/database.py)
và cung cấp API đơn giản cho UI (StatisticsPageMixin).

Các nhóm API chính:
- Tổng quan hệ thống
- Thống kê theo ngày / khoảng thời gian / 7 ngày gần nhất
- Danh sách xe đang trong bãi
- Phân tích thời gian đỗ
- Top biển số thường xuyên
- Export báo cáo ra file .txt
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, date

from ..database.database import DB
from ..config.config import CONN_STR, USE_SQL




# ===== Lớp thống kê giữ xe =====
class ParkingStatistics:
    def __init__(self, db: Optional[DB] = None) -> None:
        """
        Nếu truyền db từ ngoài vào thì dùng luôn db đó.
        Nếu không, tự tạo DB mới (nếu USE_SQL bật).
        """
        if db is not None:
            self.db = db
        else:
            self.db: DB | None = DB(CONN_STR) if USE_SQL else None

  
  
  
  
    # ===== THỐNG KÊ XE TRONG BÃI =====
    def get_cars_currently_inside(self) -> Dict[str, Any]:
        """
        Đếm số lượng xe hiện đang trong bãi (PENDING status)
        Trả về:
        {
          "total": int,
          "list": [
             {
               "id": ...,
               "plate": ...,
               "date_in": ...,
               "time_in": ...,
               "image_in": ...,
               "duration_minutes": ...,
               "session_category": "INTERNAL"/"VISITOR"/"",
               "vehicle_type_name": "Xe máy", ...
             }, ...
          ],
          "error": None | msg
        }
        """
        if not self.db or not self.db.ok:
            return {"total": 0, "list": [], "error": "Database not available"}

        try:
            query = """
                SELECT s.id,
                       s.plate_in,
                       s.date_in,
                       s.time_in,
                       s.image_in,
                       s.session_category,
                       s.vehicle_type_id,
                       vt.name AS vehicle_type_name
                FROM dbo.ParkingSessions AS s
                LEFT JOIN dbo.VehicleTypes AS vt
                    ON s.vehicle_type_id = vt.vehicle_type_id
                WHERE s.match_status = 'PENDING'
                  AND s.plate_out IS NULL
                ORDER BY s.date_in DESC, s.time_in DESC, s.id DESC
            """
            rows = self.db.cur.execute(query).fetchall()

            result_list: List[Dict[str, Any]] = []
            for row in rows:
                date_in = row[2]
                time_in = row[3]
                duration_minutes = self._calculate_duration_from_entry(date_in, time_in)

                session_category = (row[5] or "").upper()
                if session_category == "INTERNAL":
                    session_cat_label = "Nội bộ"
                elif session_category == "VISITOR":
                    session_cat_label = "Vãng lai"
                else:
                    session_cat_label = ""

                result_list.append(
                    {
                        "id": row[0],
                        "plate": row[1],
                        "date_in": date_in,
                        "time_in": time_in,
                        "image_in": row[4],
                        "duration_minutes": duration_minutes,
                        "session_category": session_category,
                        "session_category_label": session_cat_label,
                        "vehicle_type_name": row[7] or "",
                    }
                )

            return {
                "total": len(result_list),
                "list": result_list,
                "error": None,
            }
        except Exception as e:
            return {"total": 0, "list": [], "error": str(e)}



    # ===THỐNG KÊ THEO NGÀY ===
    def get_daily_statistics(self, target_date: str | None = None) -> Dict[str, Any]:
        """
        Thống kê trong 1 ngày (theo cột date_in, date_out – định dạng NVARCHAR dd/MM/yyyy)

        target_date: 'dd/MM/yyyy' hoặc None (mặc định hôm nay).
        Trả về:
        {
            "date": str,
            "entries_today": int,
            "exits_today": int,
            "matched_exits": int,
            "unmatched_exits": int,
            "pending_cars": int,
            "success_rate": float,
            "net_change": int,
            "error": None | str
        }
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}

        if target_date is None:
            target_date = datetime.now().strftime("%d/%m/%Y")

        try:
            query = """
                SELECT 
                    -- xe vào trong ngày
                    SUM(CASE WHEN date_in = ? THEN 1 ELSE 0 END) AS entries_today,
                    -- xe ra trong ngày
                    SUM(CASE WHEN date_out = ? THEN 1 ELSE 0 END) AS exits_today,
                    -- xe ra khớp biển số
                    SUM(CASE WHEN date_out = ? AND match_status = 'KHOP-BIEN-SO' THEN 1 ELSE 0 END) AS matched_exits,
                    -- xe ra không khớp
                    SUM(CASE WHEN date_out = ? AND match_status = 'KHONG-KHOP-BIEN-SO' THEN 1 ELSE 0 END) AS unmatched_exits,
                    -- xe vào nhưng chưa ra (tính đến ngày đó)
                    SUM(CASE WHEN date_in <= ? AND match_status = 'PENDING' THEN 1 ELSE 0 END) AS pending_cars
                FROM dbo.ParkingSessions
            """
            row = self.db.cur.execute(  # type: ignore
                query,
                (target_date, target_date, target_date, target_date, target_date),
            ).fetchone()

            entries_today = int(row[0] or 0)
            exits_today = int(row[1] or 0)
            matched_exits = int(row[2] or 0)
            unmatched_exits = int(row[3] or 0)
            pending_cars = int(row[4] or 0)

            success_rate = (matched_exits / exits_today * 100.0) if exits_today > 0 else 0.0

            return {
                "date": target_date,
                "entries_today": entries_today,
                "exits_today": exits_today,
                "matched_exits": matched_exits,
                "unmatched_exits": unmatched_exits,
                "pending_cars": pending_cars,
                "success_rate": round(success_rate, 2),
                "net_change": entries_today - exits_today,
                "error": None,
            }
        except Exception as e:
            print("[ParkingStatistics.get_daily_statistics] error:", e)
            return {"error": str(e)}


    # === THỐNG KÊ 7 NGÀY (DÙNG CHO BÁO CÁO / BIỂU ĐỒ) ===
    def get_weekly_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Thống kê từng ngày trong N ngày gần nhất (mặc định 7).

        Trả về:
        {
            "start_date": ...,
            "end_date": ...,
            "daily_data": [ {thống kê theo ngày}, ... ],
            "weekly_summary": {...},
            "error": None | str
        }
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days - 1)

            daily_data: List[Dict[str, Any]] = []
            for i in range(days):
                d = start_date + timedelta(days=i)
                d_str = d.strftime("%d/%m/%Y")
                day_stats = self.get_daily_statistics(d_str)
                day_stats["day_name"] = d.strftime("%A")
                daily_data.append(day_stats)

            total_entries = sum(int(d["entries_today"] or 0) for d in daily_data)
            total_exits = sum(int(d["exits_today"] or 0) for d in daily_data)
            total_matched = sum(int(d["matched_exits"] or 0) for d in daily_data)
            total_unmatched = sum(int(d["unmatched_exits"] or 0) for d in daily_data)

            avg_success_rate = (total_matched / total_exits * 100.0) if total_exits > 0 else 0.0

            return {
                "start_date": start_date.strftime("%d/%m/%Y"),
                "end_date": end_date.strftime("%d/%m/%Y"),
                "daily_data": daily_data,
                "weekly_summary": {
                    "total_entries": total_entries,
                    "total_exits": total_exits,
                    "total_matched": total_matched,
                    "total_unmatched": total_unmatched,
                    "avg_success_rate": round(avg_success_rate, 2),
                    "net_change": total_entries - total_exits,
                },
                "error": None,
            }
        except Exception as e:
            print("[ParkingStatistics.get_weekly_statistics] error:", e)
            return {"error": str(e)}





    # ===== THỐNG KÊ THEO KHOẢNG THỜI GIAN TÙY CHỈNH =====
    def get_statistics_by_range(self, range_type: str = "today") -> Dict[str, Any]:
        """
        Thống kê theo khoảng thời gian được chọn cho UI Thống kê.

        range_type: "today", "7days", "month"

        Trả về:
        {
          "range": "...",
          "daily": [...],
          "revenue_total": int,
          "revenue_internal": int,
          "revenue_visitor": int,
          "error": None | msg
        }
        """
        if not self.db or not self.db.ok:
            return {
                "range": "",
                "daily": [],
                "revenue_total": 0,
                "revenue_internal": 0,
                "revenue_visitor": 0,
                "error": "Database not available",
            }

        try:
            now = datetime.now()

            if range_type in ("today", "Hôm nay"):
                start_date = now
                end_date = now
            elif range_type in ("7days", "7 ngày"):
                start_date = now - timedelta(days=6)
                end_date = now
            elif range_type in ("month", "Tháng này"):
                start_date = now.replace(day=1)
                end_date = now
            else:
                # mặc định: hôm nay
                start_date = now
                end_date = now

            return self._build_range_aggregates(start_date, end_date)

        except Exception as e:
            return {
                "range": "",
                "daily": [],
                "revenue_total": 0,
                "revenue_internal": 0,
                "revenue_visitor": 0,
                "error": str(e),
            }





    # === THỐNG KÊ THEO KHOẢNG TÙY CHỈNH ===
    def _get_range_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Nội bộ: tính thống kê trong khoảng [start_date, end_date] theo date_in/date_out (dd/MM/yyyy)
        Trả về format giống get_daily_statistics.
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}

        try:
            start_str = start_date.strftime("%d/%m/%Y")
            end_str = end_date.strftime("%d/%m/%Y")

            sql = """
                SELECT 
                    SUM(CASE WHEN date_in  >= ? AND date_in  <= ? THEN 1 ELSE 0 END) AS entries_range,
                    SUM(CASE WHEN date_out >= ? AND date_out <= ? THEN 1 ELSE 0 END) AS exits_range,
                    SUM(CASE WHEN date_out >= ? AND date_out <= ? AND match_status = 'KHOP-BIEN-SO' THEN 1 ELSE 0 END) AS matched_exits,
                    SUM(CASE WHEN date_out >= ? AND date_out <= ? AND match_status = 'KHONG-KHOP-BIEN-SO' THEN 1 ELSE 0 END) AS unmatched_exits,
                    SUM(CASE WHEN date_in  >= ? AND date_in  <= ? AND match_status = 'PENDING' THEN 1 ELSE 0 END) AS pending_cars
                FROM dbo.ParkingSessions
            """
            row = self.db.cur.execute(  # type: ignore
                sql,
                (
                    start_str,
                    end_str,  # entries
                    start_str,
                    end_str,  # exits
                    start_str,
                    end_str,  # matched
                    start_str,
                    end_str,  # unmatched
                    start_str,
                    end_str,  # pending
                ),
            ).fetchone()

            entries = int(row[0] or 0)
            exits = int(row[1] or 0)
            matched = int(row[2] or 0)
            unmatched = int(row[3] or 0)
            pending = int(row[4] or 0)

            success_rate = (matched / exits * 100.0) if exits > 0 else 0.0

            return {
                "range_type": f"{start_str} - {end_str}",
                "entries_today": entries,
                "exits_today": exits,
                "matched_exits": matched,
                "unmatched_exits": unmatched,
                "pending_cars": pending,
                "success_rate": round(success_rate, 2),
                "net_change": entries - exits,
                "error": None,
            }
        except Exception as e:
            print("[ParkingStatistics._get_range_statistics] error:", e)
            return {"error": str(e)}




    # === XÂY DỰNG THỐNG KÊ THEO KHOẢNG TÙY CHỈNH ===
    def _build_range_aggregates(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Gom thống kê theo ngày trong khoảng [start_date, end_date] (theo created_at).

        Trả về:
        {
          "range": "dd/MM/yyyy - dd/MM/yyyy",
          "daily": [ { ... per day ... }, ... ],
          "revenue_total": int,
          "revenue_internal": int,
          "revenue_visitor": int,
          "error": None | msg
        }
        """
        if not self.db or not self.db.ok:
            return {
                "range": "",
                "daily": [],
                "revenue_total": 0,
                "revenue_internal": 0,
                "revenue_visitor": 0,
                "error": "Database not available",
            }

        try:
            start_dt = datetime(
                start_date.year, start_date.month, start_date.day, 0, 0, 0
            )
            end_dt = datetime(
                end_date.year, end_date.month, end_date.day, 23, 59, 59
            )

            query = """
                SELECT
                    CONVERT(date, created_at) AS day,
                    session_category,
                    ISNULL(fee_amount, 0) AS fee_amount
                FROM dbo.ParkingSessions
                WHERE created_at >= ? AND created_at <= ?
            """
            rows = self.db.cur.execute(query, (start_dt, end_dt)).fetchall()

            # gom theo từng ngày
            agg: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                day: date = row[0]
                day_key = day.strftime("%d/%m/%Y")
                if day_key not in agg:
                    dt_tmp = datetime(day.year, day.month, day.day)
                    agg[day_key] = {
                        "date": day_key,
                        "day_name": dt_tmp.strftime("%A"),
                        "total_sessions": 0,
                        "internal_sessions": 0,
                        "visitor_sessions": 0,
                        "revenue_total": 0,
                        "revenue_internal": 0,
                        "revenue_visitor": 0,
                    }

                rec = agg[day_key]
                rec["total_sessions"] += 1
                fee = int(row[2] or 0)
                rec["revenue_total"] += fee

                cat = (row[1] or "").upper()
                if cat == "INTERNAL":
                    rec["internal_sessions"] += 1
                    rec["revenue_internal"] += fee
                elif cat == "VISITOR":
                    rec["visitor_sessions"] += 1
                    rec["revenue_visitor"] += fee

            # bảo đảm ngày nào trong range cũng có dòng (kể cả 0)
            daily: List[Dict[str, Any]] = []
            cur = start_dt
            while cur.date() <= end_dt.date():
                key = cur.strftime("%d/%m/%Y")
                if key in agg:
                    daily.append(agg[key])
                else:
                    daily.append(
                        {
                            "date": key,
                            "day_name": cur.strftime("%A"),
                            "total_sessions": 0,
                            "internal_sessions": 0,
                            "visitor_sessions": 0,
                            "revenue_total": 0,
                            "revenue_internal": 0,
                            "revenue_visitor": 0,
                        }
                    )
                cur += timedelta(days=1)

            # sort theo ngày tăng dần cho đẹp
            daily.sort(
                key=lambda d: datetime.strptime(d["date"], "%d/%m/%Y")
            )

            rev_total = sum(d["revenue_total"] for d in daily)
            rev_internal = sum(d["revenue_internal"] for d in daily)
            rev_visitor = sum(d["revenue_visitor"] for d in daily)

            return {
                "range": f"{start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}",
                "daily": daily,
                "revenue_total": rev_total,
                "revenue_internal": rev_internal,
                "revenue_visitor": rev_visitor,
                "error": None,
            }

        except Exception as e:
            return {
                "range": "",
                "daily": [],
                "revenue_total": 0,
                "revenue_internal": 0,
                "revenue_visitor": 0,
                "error": str(e),
            }





    # === THỐNG KÊ THEO KHOẢNG TÙY CHỈNH ===
    def get_weekly_real_data(self) -> Dict[str, Any]:
        """
        Dữ liệu 7 ngày gần nhất (hôm nay lùi về 6 ngày trước).
        Format đúng như StatisticsPageMixin đang dùng:
        {
            "weekly_data": [
                {"date": ..., "day_name": ..., "entries": ..., "exits": ..., "matched": ..., "unmatched": ..., "success_rate": ...},
                ...
            ],
            "summary": {
                "total_in": ...,
                "total_out": ...,
                "total_matched": ...,
                "overall_success": ...
            },
            "error": None | str
        }
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}

        try:
            weekly_data: List[Dict[str, Any]] = []
            total_in = 0
            total_out = 0
            total_matched = 0

            for i in range(7):
                d = datetime.now() - timedelta(days=i)
                d_str = d.strftime("%d/%m/%Y")
                stats = self.get_daily_statistics(d_str)
                if stats.get("error"):
                    continue

                day_in = int(stats.get("entries_today", 0) or 0)
                day_out = int(stats.get("exits_today", 0) or 0)
                day_matched = int(stats.get("matched_exits", 0) or 0)
                day_unmatched = int(stats.get("unmatched_exits", 0) or 0)
                success_rate = float(stats.get("success_rate", 0) or 0.0)

                total_in += day_in
                total_out += day_out
                total_matched += day_matched

                weekly_data.append(
                    {
                        "date": d_str,
                        "day_name": d.strftime("%A"),
                        "entries": day_in,
                        "exits": day_out,
                        "matched": day_matched,
                        "unmatched": day_unmatched,
                        "success_rate": success_rate,
                    }
                )

            overall_success = (total_matched / total_out * 100.0) if total_out > 0 else 0.0

            return {
                "weekly_data": weekly_data,
                "summary": {
                    "total_in": total_in,
                    "total_out": total_out,
                    "total_matched": total_matched,
                    "overall_success": round(overall_success, 2),
                },
                "error": None,
            }
        except Exception as e:
            print("[ParkingStatistics.get_weekly_real_data] error:", e)
            return {"error": str(e)}





    # === THỐNG KÊ TỔNG QUAN TOÀN HỆ THỐNG ===
    def get_overview_statistics(self) -> Dict[str, Any]:
        """
        Thống kê tổng quan toàn hệ thống.

        Trả về cấu trúc mới cho UI Thống kê:

        {
          "totals": {
             "total_sessions": ...,
             "internal_sessions": ...,
             "visitor_sessions": ...,
             "current_inpark": ...,
          },
          "revenue": {
             "total": ...,
             "internal": ...,
             "visitor": ...,
             "unpaid_amount": ...,
             "unpaid_count": ...,
          },
          "today": {...},
          "longest_parking": [...],
          "error": None,
          # các key cũ vẫn giữ lại cho export_comprehensive_report
          "total_sessions": ...,
          "current_cars_inside": ...,
          "completed_sessions": ...,
          "unmatched_sessions": ...,
          "overall_success_rate": ...,
        }
        """
        if not self.db or not self.db.ok:
            return {
                "totals": {
                    "total_sessions": 0,
                    "internal_sessions": 0,
                    "visitor_sessions": 0,
                    "current_inpark": 0,
                },
                "revenue": {
                    "total": 0,
                    "internal": 0,
                    "visitor": 0,
                    "unpaid_amount": 0,
                    "unpaid_count": 0,
                },
                "today": {},
                "longest_parking": [],
                "error": "Database not available",
                "total_sessions": 0,
                "current_cars_inside": 0,
                "completed_sessions": 0,
                "unmatched_sessions": 0,
                "overall_success_rate": 0.0,
            }

        try:
            # --- Tổng phiên, xe đang trong bãi, phiên khớp / không khớp ---
            overview_query = """
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN match_status = 'PENDING' THEN 1 ELSE 0 END) as current_cars,
                    SUM(CASE WHEN match_status = 'KHOP-BIEN-SO' THEN 1 ELSE 0 END) as completed_sessions,
                    SUM(CASE WHEN match_status = 'KHONG-KHOP-BIEN-SO' THEN 1 ELSE 0 END) as unmatched_sessions,
                    MIN(created_at) as first_record,
                    MAX(created_at) as latest_record
                FROM dbo.ParkingSessions
            """
            overview_row = self.db.cur.execute(overview_query).fetchone()

            total_sessions = int(overview_row[0] or 0)
            current_cars = int(overview_row[1] or 0)
            completed_sessions = int(overview_row[2] or 0)
            unmatched_sessions = int(overview_row[3] or 0)
            first_record = overview_row[4]
            latest_record = overview_row[5]

            total_exits = completed_sessions + unmatched_sessions
            overall_success_rate = (
                completed_sessions / total_exits * 100 if total_exits > 0 else 0.0
            )

            # --- Đếm lượt nội bộ / vãng lai ---
            cat_query = """
                SELECT
                    SUM(CASE WHEN session_category = 'INTERNAL' THEN 1 ELSE 0 END) AS internal_sessions,
                    SUM(CASE WHEN session_category = 'VISITOR'  THEN 1 ELSE 0 END) AS visitor_sessions
                FROM dbo.ParkingSessions
            """
            cat_row = self.db.cur.execute(cat_query).fetchone()
            internal_sessions = int(cat_row[0] or 0)
            visitor_sessions = int(cat_row[1] or 0)

            # --- Doanh thu theo loại phiên ---
            revenue_query = """
                SELECT
                    SUM(CASE WHEN fee_amount IS NOT NULL THEN fee_amount ELSE 0 END) AS total_revenue,
                    SUM(CASE WHEN session_category = 'INTERNAL' THEN ISNULL(fee_amount,0) ELSE 0 END) AS internal_revenue,
                    SUM(CASE WHEN session_category = 'VISITOR'  THEN ISNULL(fee_amount,0) ELSE 0 END) AS visitor_revenue,
                    SUM(CASE WHEN fee_amount IS NULL AND date_out IS NOT NULL THEN 1 ELSE 0 END) AS unpaid_count
                FROM dbo.ParkingSessions
            """
            rev_row = self.db.cur.execute(revenue_query).fetchone()
            rev_total = int(rev_row[0] or 0)
            rev_internal = int(rev_row[1] or 0)
            rev_visitor = int(rev_row[2] or 0)
            unpaid_count = int(rev_row[3] or 0)
            unpaid_amount = 0  # nếu cần có thể tính thêm sau

            # --- Thống kê hôm nay (dùng hàm cũ cho export) ---
            today_stats = self.get_daily_statistics()

            # --- 5 xe đỗ lâu nhất (đang trong bãi) ---
            longest_parking_query = """
                SELECT TOP 5 
                    plate_in, date_in, time_in,
                    DATEDIFF(MINUTE, 
                        TRY_CONVERT(datetime, date_in + ' ' + time_in, 103), 
                        GETDATE()) as duration_minutes
                FROM dbo.ParkingSessions
                WHERE match_status = 'PENDING'
                ORDER BY duration_minutes DESC
            """
            longest_rows = self.db.cur.execute(longest_parking_query).fetchall()

            longest_parking_list: List[Dict[str, Any]] = []
            for row in longest_rows:
                minutes = int(row[3] or 0)
                longest_parking_list.append(
                    {
                        "plate": row[0],
                        "date_in": row[1],
                        "time_in": row[2],
                        "duration_hours": round(minutes / 60, 1) if minutes else 0.0,
                        "duration_minutes": minutes,
                    }
                )

            totals = {
                "total_sessions": total_sessions,
                "internal_sessions": internal_sessions,
                "visitor_sessions": visitor_sessions,
                "current_inpark": current_cars,
            }
            revenue = {
                "total": rev_total,
                "internal": rev_internal,
                "visitor": rev_visitor,
                "unpaid_amount": unpaid_amount,
                "unpaid_count": unpaid_count,
            }

            return {
                "totals": totals,
                "revenue": revenue,
                "today": today_stats,
                "longest_parking": longest_parking_list,
                "first_record_date": first_record,
                "latest_record_date": latest_record,
                "error": None,
                # các key cũ cho export_comprehensive_report
                "total_sessions": total_sessions,
                "current_cars_inside": current_cars,
                "completed_sessions": completed_sessions,
                "unmatched_sessions": unmatched_sessions,
                "overall_success_rate": round(overall_success_rate, 2),
            }

        except Exception as e:
            return {
                "totals": {
                    "total_sessions": 0,
                    "internal_sessions": 0,
                    "visitor_sessions": 0,
                    "current_inpark": 0,
                },
                "revenue": {
                    "total": 0,
                    "internal": 0,
                    "visitor": 0,
                    "unpaid_amount": 0,
                    "unpaid_count": 0,
                },
                "today": {},
                "longest_parking": [],
                "error": str(e),
                "total_sessions": 0,
                "current_cars_inside": 0,
                "completed_sessions": 0,
                "unmatched_sessions": 0,
                "overall_success_rate": 0.0,
            }





    # === TÍNH TỔNG DOANH THU THEO KHOẢNG NGÀY ===
    def get_total_revenue(self, start_date: str, end_date: str) -> int:
        """
        Tính tổng fee_amount trong khoảng date_out [start_date, end_date]
        start_date, end_date: chuỗi 'dd/MM/yyyy'
        """
        if not self.db or not self.db.ok:
            return 0

        try:
            sql = """
                SELECT SUM(fee_amount)
                FROM dbo.ParkingSessions
                WHERE fee_amount IS NOT NULL
                  AND date_out >= ? AND date_out <= ?
            """
            row = self.db.cur.execute(sql, (start_date, end_date)).fetchone()  # type: ignore
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            print("[ParkingStatistics.get_total_revenue] error:", e)
            return 0





    # === TÍNH TỔNG DOANH THU THEO LOẠI RANGE CHO UI ===
    def get_total_revenue_by_range_type(self, range_type: str = "today") -> int:
        """
        Helper cho UI: lấy tổng doanh thu theo loại range giống combo.
        """
        now = datetime.now()

        if range_type in ("today", "Hôm nay"):
            d = now.strftime("%d/%m/%Y")
            return self.get_total_revenue(d, d)

        if range_type in ("7days", "7 ngày"):
            start = (now - timedelta(days=6)).strftime("%d/%m/%Y")
            end = now.strftime("%d/%m/%Y")
            return self.get_total_revenue(start, end)

        if range_type in ("month", "Tháng này"):
            start = now.replace(day=1).strftime("%d/%m/%Y")
            end = now.strftime("%d/%m/%Y")
            return self.get_total_revenue(start, end)

        # fallback
        d = now.strftime("%d/%m/%Y")
        return self.get_total_revenue(d, d)


    
    
    
    
    # === EXPORT BÁO CÁO TỔNG HỢP RA FILE TEXT ===
    def export_comprehensive_report(self, file_path: str) -> bool:
        """
        Export báo cáo tổng hợp ra file text.
        UI gọi khi bấm nút "Export báo cáo".
        """
        try:
            overview = self.get_overview_statistics()
            cars_inside = self.get_cars_currently_inside()
            daily_stats = self.get_daily_statistics()
            weekly_stats = self.get_weekly_statistics()
            duration_stats = self.get_parking_duration_statistics()
            frequent_stats = self.get_frequent_plates_statistics()

            plates: List[Dict[str, Any]] = list(frequent_stats.get("frequent_plates", []))

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("     BÁO CÁO THỐNG KÊ HỆ THỐNG QUẢN LÝ BÃI GIỮ XE\n")
                f.write("=" * 60 + "\n")
                f.write(
                    "Thời gian tạo báo cáo: "
                    + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    + "\n\n"
                )

                # Tổng quan
                f.write("📊 TỔNG QUAN HỆ THỐNG\n")
                f.write("-" * 30 + "\n")
                f.write(f"• Tổng số phiên gửi xe: {overview.get('total_sessions', 0)}\n")
                f.write(
                    f"• Xe hiện đang trong bãi: {overview.get('current_cars_inside', 0)}\n"
                )
                f.write(
                    f"• Phiên hoàn thành: {overview.get('completed_sessions', 0)}\n"
                )
                f.write(
                    f"• Phiên không khớp: {overview.get('unmatched_sessions', 0)}\n"
                )
                f.write(
                    f"• Tỷ lệ nhận diện đúng tổng: "
                    f"{overview.get('overall_success_rate', 0)}%\n\n"
                )

                # Hôm nay
                f.write("📅 THỐNG KÊ HÔM NAY\n")
                f.write("-" * 30 + "\n")
                f.write(f"• Xe vào: {daily_stats.get('entries_today', 0)}\n")
                f.write(f"• Xe ra: {daily_stats.get('exits_today', 0)}\n")
                f.write(f"• Khớp biển số: {daily_stats.get('matched_exits', 0)}\n")
                f.write(f"• Không khớp: {daily_stats.get('unmatched_exits', 0)}\n")
                f.write(f"• Tỷ lệ thành công: {daily_stats.get('success_rate', 0)}%\n\n")

                # Xe đang trong bãi
                f.write("🚗 XE ĐANG TRONG BÃI\n")
                f.write("-" * 30 + "\n")
                f.write(f"Tổng số: {cars_inside.get('total', 0)} xe\n")
                for car in list(cars_inside.get("list", []))[:10]:
                    f.write(
                        f"• {car.get('plate')} - vào lúc "
                        f"{car.get('date_in')} {car.get('time_in')} - "
                        f"đã đỗ {car.get('duration_minutes')} phút\n"
                    )
                f.write("\n")

                # Top biển số
                f.write("🔢 TOP BIỂN SỐ THƯỜNG XUYÊN\n")
                f.write("-" * 30 + "\n")
                for p in plates[:10]:
                    f.write(
                        f"• {p.get('plate_number')}: {p.get('frequency')} lượt, "
                        f"tỷ lệ khớp ~ {p.get('success_rate')}%\n"
                    )
                f.write("\n")

                # Thời gian đỗ
                if not duration_stats.get("error"):
                    f.write("⏱️ PHÂN TÍCH THỜI GIAN ĐỖ XE\n")
                    f.write("-" * 30 + "\n")
                    f.write(
                        f"• Thời gian đỗ trung bình: "
                        f"{duration_stats.get('average_duration_hours', 0)} giờ\n"
                    )
                    f.write(
                        f"• Thời gian đỗ ngắn nhất: "
                        f"{duration_stats.get('min_duration_minutes', 0)} phút\n"
                    )
                    f.write(
                        f"• Thời gian đỗ dài nhất: "
                        f"{duration_stats.get('max_duration_minutes', 0)} phút\n"
                    )
                    dist = duration_stats.get("distribution", {})
                    st = dist.get("short_term", {})
                    mt = dist.get("medium_term", {})
                    lt = dist.get("long_term", {})
                    f.write(
                        f"• Đỗ ngắn hạn (≤1h): {st.get('count', 0)} "
                        f"({st.get('percentage', 0)}%)\n"
                    )
                    f.write(
                        f"• Đỗ trung hạn (1–8h): {mt.get('count', 0)} "
                        f"({mt.get('percentage', 0)}%)\n"
                    )
                    f.write(
                        f"• Đỗ dài hạn (>8h): {lt.get('count', 0)} "
                        f"({lt.get('percentage', 0)}%)\n"
                    )

                f.write("\n" + "=" * 60 + "\n")
                f.write("Báo cáo được tạo bởi Hệ thống Quản lý Bãi giữ xe\n")

            return True
        except Exception as e:
            print("[ParkingStatistics.export_comprehensive_report] error:", e)
            return False

 
    
    
    # === THỐNG KÊ BIỂN SỐ THƯỜNG XUYÊN =====
    def get_frequent_plates_statistics(self, limit: int = 10) -> Dict[str, Any]:
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}

        try:
            sql = f"""
                SELECT TOP {limit}
                    COALESCE(plate_in, plate_out) AS plate_number,
                    COUNT(*) AS frequency,
                    SUM(CASE WHEN match_status = 'KHOP-BIEN-SO' THEN 1 ELSE 0 END) AS successful_matches,
                    SUM(CASE WHEN match_status = 'PENDING' THEN 1 ELSE 0 END) AS currently_inside,
                    MAX(COALESCE(created_at, GETDATE())) AS last_seen
                FROM dbo.ParkingSessions
                WHERE COALESCE(plate_in, plate_out) IS NOT NULL
                  AND COALESCE(plate_in, plate_out) <> ''
                GROUP BY COALESCE(plate_in, plate_out)
                ORDER BY frequency DESC
            """
            rows = self.db.cur.execute(sql).fetchall()  # type: ignore

            plates: List[Dict[str, Any]] = []
            for r in rows:
                freq = int(r[1] or 0)
                success = int(r[2] or 0)
                success_rate = success / freq * 100.0 if freq > 0 else 0.0
                plates.append(
                    {
                        "plate_number": r[0] or "",
                        "frequency": freq,
                        "successful_matches": success,
                        "currently_inside": bool(r[3] and r[3] > 0),
                        "last_seen": r[4],
                        "success_rate": round(success_rate, 2),
                    }
                )

            return {"frequent_plates": plates, "total_unique_plates": len(plates), "error": None}
        except Exception as e:
            print("[ParkingStatistics.get_frequent_plates_statistics] error:", e)
            return {"error": str(e)}





    # === PHÂN TÍCH THỜI GIAN ĐỖ ===
    def get_parking_duration_statistics(self) -> Dict[str, Any]:
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}

        try:
            sql = """
                SELECT 
                    DATEDIFF(
                        MINUTE,
                        TRY_CONVERT(datetime, date_in + ' ' + time_in, 103),
                        TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)
                    ) AS duration_minutes
                FROM dbo.ParkingSessions
                WHERE match_status = 'KHOP-BIEN-SO'
                  AND date_in IS NOT NULL AND time_in IS NOT NULL
                  AND date_out IS NOT NULL AND time_out IS NOT NULL
                  AND TRY_CONVERT(datetime, date_in + ' ' + time_in, 103) IS NOT NULL
                  AND TRY_CONVERT(datetime, date_out + ' ' + time_out, 103) IS NOT NULL
            """
            rows = self.db.cur.execute(sql).fetchall()  # type: ignore
            durations = [int(r[0]) for r in rows if r[0] is not None and r[0] > 0]

            if not durations:
                return {"error": "No completed sessions"}

            durations.sort()
            total = len(durations)
            avg = sum(durations) / total
            min_dur = durations[0]
            max_dur = durations[-1]
            median = durations[total // 2]

            short_term = len([d for d in durations if d <= 60])
            medium_term = len([d for d in durations if 60 < d <= 480])
            long_term = len([d for d in durations if d > 480])

            def pct(n: int) -> float:
                return round(n / total * 100.0, 2) if total > 0 else 0.0

            return {
                "total_completed_sessions": total,
                "average_duration_minutes": round(avg, 2),
                "average_duration_hours": round(avg / 60.0, 2),
                "min_duration_minutes": min_dur,
                "max_duration_minutes": max_dur,
                "median_duration_minutes": median,
                "distribution": {
                    "short_term": {"count": short_term, "percentage": pct(short_term)},
                    "medium_term": {"count": medium_term, "percentage": pct(medium_term)},
                    "long_term": {"count": long_term, "percentage": pct(long_term)},
                },
                "error": None,
            }
        except Exception as e:
            print("[ParkingStatistics.get_parking_duration_statistics] error:", e)
            return {"error": str(e)}

    
    
    
    
    
    # === TÍNH THỜI GIAN ĐỖ TỪ THỜI ĐIỂM VÀO ĐẾN HIỆN TẠI ===
    def _calculate_duration_from_entry(self, date_in: str, time_in: str) -> int:
        """
        Tính số phút từ thời điểm vào (date_in + time_in, định dạng dd/MM/yyyy HH:mm:ss) đến hiện tại.
        Nếu parse lỗi thì trả 0.
        """
        try:
            s = f"{date_in} {time_in}"
            t_in = datetime.strptime(s, "%d/%m/%Y %H:%M:%S")
            diff = datetime.now() - t_in
            return int(diff.total_seconds() / 60)
        except Exception:
            return 0

 
    
    
    # === ĐẾM SỐ PHIÊN THEO TRẠNG THÁI ===
    def count_sessions_by_status(self, status: str) -> int:
        if not self.db or not self.db.ok:
            return 0
        try:
            sql = "SELECT COUNT(*) FROM dbo.ParkingSessions WHERE match_status = ?"
            row = self.db.cur.execute(sql, (status,)).fetchone()  # type: ignore
            return int(row[0]) if row else 0
        except Exception as e:
            print("[ParkingStatistics.count_sessions_by_status] error:", e)
            return 0
