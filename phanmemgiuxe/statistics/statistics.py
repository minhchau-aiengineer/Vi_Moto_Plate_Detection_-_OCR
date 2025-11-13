import pandas as pd
from datetime import datetime, timedelta
from ..database.database import DB
from ..config.config import CONN_STR, USE_SQL






class ParkingStatistics:
    """
    Module thống kê cho hệ thống quản lý bãi đỗ xe
    Cung cấp các chức năng thống kê và báo cáo chi tiết
    """
    
    def __init__(self):
        self.db = DB(CONN_STR) if USE_SQL else None
    
    
    
    
    
    # ===== THỐNG KÊ XE TRONG BÃI =====
    def get_cars_currently_inside(self):
        """
        Đếm số lượng xe hiện đang trong bãi (PENDING status)
        Returns: dict với thông tin chi tiết
        """
        if not self.db or not self.db.ok:
            return {"total": 0, "list": [], "error": "Database not available"}
        
        try:
            # Lấy danh sách xe đang trong bãi
            query = """
                SELECT id, plate_in, date_in, time_in, image_in
                FROM dbo.ParkingSessions 
                WHERE match_status = 'PENDING' AND plate_out IS NULL
                ORDER BY date_in DESC, time_in DESC
            """
            cars_inside = self.db.cur.execute(query).fetchall()
            
            result = {
                "total": len(cars_inside),
                "list": [
                    {
                        "id": row[0],
                        "plate": row[1],
                        "date_in": row[2],
                        "time_in": row[3],
                        "image_in": row[4],
                        "duration_minutes": self._calculate_duration_from_entry(row[2], row[3])
                    }
                    for row in cars_inside
                ],
                "error": None
            }
            return result
            
        except Exception as e:
            return {"total": 0, "list": [], "error": str(e)}




    # ===== THỐNG KÊ THEO NGÀY =====
    def get_daily_statistics(self, target_date=None):
        """
        Thống kê chi tiết theo ngày
        Args: target_date (str): 'dd/MM/yyyy' hoặc None (hôm nay)
        Returns: dict với các thống kê
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
        
        if target_date is None:
            target_date = datetime.now().strftime("%d/%m/%Y")
        
        try:
            query = """
                SELECT 
                    -- Xe vào hôm nay
                    SUM(CASE WHEN date_in = ? THEN 1 ELSE 0 END) as entries_today,
                    
                    -- Xe ra hôm nay  
                    SUM(CASE WHEN date_out = ? THEN 1 ELSE 0 END) as exits_today,
                    
                    -- Xe ra khớp biển số
                    SUM(CASE WHEN date_out = ? AND match_status = 'KHOP-BIEN-SO' THEN 1 ELSE 0 END) as matched_exits,
                    
                    -- Xe ra không khớp biển số
                    SUM(CASE WHEN date_out = ? AND match_status = 'KHONG-KHOP-BIEN-SO' THEN 1 ELSE 0 END) as unmatched_exits,
                    
                    -- Xe vào nhưng chưa ra (tính đến hôm nay)
                    SUM(CASE WHEN date_in <= ? AND match_status = 'PENDING' THEN 1 ELSE 0 END) as pending_cars
                    
                FROM dbo.ParkingSessions
            """
            
            result = self.db.cur.execute(query, (target_date, target_date, target_date, target_date, target_date)).fetchone()
            
            entries_today = result[0] or 0
            exits_today = result[1] or 0
            matched_exits = result[2] or 0
            unmatched_exits = result[3] or 0
            pending_cars = result[4] or 0
            
            # Tính tỷ lệ nhận dạng thành công
            success_rate = (matched_exits / exits_today * 100) if exits_today > 0 else 0
            
            return {
                "date": target_date,
                "entries_today": entries_today,
                "exits_today": exits_today,
                "matched_exits": matched_exits,
                "unmatched_exits": unmatched_exits,
                "pending_cars": pending_cars,
                "success_rate": round(success_rate, 2),
                "net_change": entries_today - exits_today,
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}




    # ===== THỐNG KÊ THEO TUẦN =====
    def get_weekly_statistics(self, weeks_back=1):
        """
        Thống kê 7 ngày gần nhất
        Args: weeks_back (int): Số tuần về trước (1 = tuần này)
        Returns: dict với thống kê từng ngày
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7 * weeks_back)
            
            weekly_data = []
            for i in range(7):
                current_date = start_date + timedelta(days=i)
                date_str = current_date.strftime("%d/%m/%Y")
                daily_stats = self.get_daily_statistics(date_str)
                daily_stats["day_name"] = current_date.strftime("%A")
                weekly_data.append(daily_stats)
            
            # Tính tổng tuần
            total_entries = sum(day.get("entries_today", 0) for day in weekly_data)
            total_exits = sum(day.get("exits_today", 0) for day in weekly_data)
            total_matched = sum(day.get("matched_exits", 0) for day in weekly_data)
            total_unmatched = sum(day.get("unmatched_exits", 0) for day in weekly_data)
            
            avg_success_rate = (total_matched / total_exits * 100) if total_exits > 0 else 0
            
            return {
                "start_date": start_date.strftime("%d/%m/%Y"),
                "end_date": end_date.strftime("%d/%m/%Y"),
                "daily_data": weekly_data,
                "weekly_summary": {
                    "total_entries": total_entries,
                    "total_exits": total_exits,
                    "total_matched": total_matched,
                    "total_unmatched": total_unmatched,
                    "avg_success_rate": round(avg_success_rate, 2),
                    "net_change": total_entries - total_exits
                },
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}


    # ===== THỐNG KÊ THEO KHOẢNG THỜI GIAN TÙY CHỈNH =====
    def get_statistics_by_range(self, range_type="today"):
        """
        Thống kê theo khoảng thời gian được chọn
        Args: range_type (str): "today", "7days", "month"
        Returns: dict với thống kê tương ứng
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
            
        try:
            now = datetime.now()
            
            if range_type == "today" or range_type == "Hôm nay":
                # Thống kê hôm nay
                return self.get_daily_statistics()
                
            elif range_type == "7days" or range_type == "7 ngày":
                # Thống kê 7 ngày qua
                start_date = now - timedelta(days=7)
                return self._get_range_statistics(start_date, now)
                
            elif range_type == "month" or range_type == "Tháng này":
                # Thống kê tháng này
                start_date = now.replace(day=1)
                return self._get_range_statistics(start_date, now)
                
            else:
                return self.get_daily_statistics()  # Default hôm nay
                
        except Exception as e:
            return {"error": str(e)}
    
    
    def _get_range_statistics(self, start_date, end_date):
        """
        Lấy thống kê trong khoảng thời gian từ start_date đến end_date
        """
        try:
            start_str = start_date.strftime("%d/%m/%Y")
            end_str = end_date.strftime("%d/%m/%Y")
            
            range_query = """
                SELECT 
                    -- Xe vào trong khoảng
                    SUM(CASE WHEN date_in >= ? AND date_in <= ? THEN 1 ELSE 0 END) as entries_range,
                    -- Xe ra trong khoảng 
                    SUM(CASE WHEN date_out >= ? AND date_out <= ? THEN 1 ELSE 0 END) as exits_range,
                    -- Xe ra khớp trong khoảng
                    SUM(CASE WHEN date_out >= ? AND date_out <= ? AND match_status = 'KHOP-BIEN-SO' THEN 1 ELSE 0 END) as matched_exits,
                    -- Xe ra không khớp trong khoảng
                    SUM(CASE WHEN date_out >= ? AND date_out <= ? AND match_status = 'KHONG-KHOP-BIEN-SO' THEN 1 ELSE 0 END) as unmatched_exits,
                    -- Xe pending (vào trong khoảng nhưng chưa ra)
                    SUM(CASE WHEN date_in >= ? AND date_in <= ? AND match_status = 'PENDING' THEN 1 ELSE 0 END) as pending_cars
                FROM dbo.ParkingSessions
            """
            
            result = self.db.cur.execute(range_query, (
                start_str, end_str,  # entries_range
                start_str, end_str,  # exits_range  
                start_str, end_str,  # matched_exits
                start_str, end_str,  # unmatched_exits
                start_str, end_str   # pending_cars
            )).fetchone()
            
            entries_range = result[0] or 0
            exits_range = result[1] or 0
            matched_exits = result[2] or 0
            unmatched_exits = result[3] or 0
            pending_cars = result[4] or 0
            
            # Tính success rate
            success_rate = (matched_exits / exits_range * 100) if exits_range > 0 else 0
            
            return {
                "range_type": f"{start_str} - {end_str}",
                "entries_today": entries_range,  # Dùng tên giống daily để UI tương thích
                "exits_today": exits_range,
                "matched_exits": matched_exits,
                "unmatched_exits": unmatched_exits,
                "pending_cars": pending_cars,
                "success_rate": round(success_rate, 2),
                "net_change": entries_range - exits_range,
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}


    def get_weekly_real_data(self):
        """
        Lấy dữ liệu thống kê thực theo 7 ngày gần nhất
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
            
        try:
            weekly_data = []
            total_in = total_out = total_matched = 0
            
            for i in range(7):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%d/%m/%Y")
                day_stats = self.get_daily_statistics(date_str)
                
                if not day_stats.get("error"):
                    day_in = day_stats.get("entries_today", 0)
                    day_out = day_stats.get("exits_today", 0)
                    day_matched = day_stats.get("matched_exits", 0)
                    day_unmatched = day_stats.get("unmatched_exits", 0)
                    success_rate = day_stats.get("success_rate", 0)
                    
                    total_in += day_in
                    total_out += day_out
                    total_matched += day_matched
                    
                    weekly_data.append({
                        "date": date_str,
                        "day_name": date.strftime("%A"),
                        "entries": day_in,
                        "exits": day_out,
                        "matched": day_matched,
                        "unmatched": day_unmatched,
                        "success_rate": success_rate
                    })
            
            overall_success = (total_matched / total_out * 100) if total_out > 0 else 0
            
            return {
                "weekly_data": weekly_data,
                "summary": {
                    "total_in": total_in,
                    "total_out": total_out,
                    "total_matched": total_matched,
                    "overall_success": round(overall_success, 2)
                },
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}


    # ===== THỐNG KÊ TỔNG QUAN =====
    def get_overview_statistics(self):
        """
        Thống kê tổng quan toàn hệ thống
        Returns: dict với các chỉ số quan trọng
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
        
        try:
            # Thống kê tổng quan
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
            
            overview = self.db.cur.execute(overview_query).fetchone()
            
            # Thống kê hôm nay
            today_stats = self.get_daily_statistics()
            
            # Thống kê xe có thời gian đỗ lâu nhất
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
            
            longest_parking = self.db.cur.execute(longest_parking_query).fetchall()
            
            # Tính thời gian đỗ trung bình cho xe đã ra (KHOP-BIEN-SO sessions)
            avg_parking_query = """
                SELECT AVG(
                    DATEDIFF(MINUTE, 
                        TRY_CONVERT(datetime, date_in + ' ' + time_in, 103),
                        TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)
                    )
                ) as avg_parking_minutes
                FROM dbo.ParkingSessions
                WHERE match_status = 'KHOP-BIEN-SO' 
                  AND date_in IS NOT NULL AND time_in IS NOT NULL
                  AND date_out IS NOT NULL AND time_out IS NOT NULL
            """
            
            avg_result = self.db.cur.execute(avg_parking_query).fetchone()
            avg_parking_minutes = avg_result[0] if avg_result and avg_result[0] else 0
            
            # Tính success rate tổng
            total_exits = overview[2] + overview[3]  # matched + unmatched
            overall_success_rate = (overview[2] / total_exits * 100) if total_exits > 0 else 0
            
            return {
                "total_sessions": overview[0] or 0,
                "current_cars_inside": overview[1] or 0,
                "completed_sessions": overview[2] or 0,
                "unmatched_sessions": overview[3] or 0,
                "avg_parking_minutes": round(avg_parking_minutes, 1) if avg_parking_minutes else 0,
                "overall_success_rate": round(overall_success_rate, 2),
                "first_record_date": overview[4],
                "latest_record_date": overview[5],
                "today": today_stats,
                "longest_parking": [
                    {
                        "plate": row[0],
                        "date_in": row[1],
                        "time_in": row[2],
                        "duration_hours": round(row[3] / 60, 1),
                        "duration_minutes": row[3]
                    }
                    for row in longest_parking
                ],
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}




    # ===== THỐNG KÊ TOP BIỂN SỐ =====
    def get_frequent_plates_statistics(self, limit=10):
        """
        Thống kê các biển số xuất hiện thường xuyên nhất
        Args: limit (int): Số lượng kết quả trả về
        Returns: dict với danh sách biển số
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
        
        try:
            query = f"""
                SELECT TOP {limit}
                    COALESCE(plate_in, plate_out) as plate_number,
                    COUNT(*) as frequency,
                    SUM(CASE WHEN match_status = 'MATCHED' THEN 1 ELSE 0 END) as successful_matches,
                    SUM(CASE WHEN match_status = 'PENDING' THEN 1 ELSE 0 END) as currently_inside,
                    MAX(COALESCE(created_at, GETDATE())) as last_seen
                FROM dbo.ParkingSessions
                WHERE COALESCE(plate_in, plate_out) IS NOT NULL
                    AND COALESCE(plate_in, plate_out) != ''
                GROUP BY COALESCE(plate_in, plate_out)
                ORDER BY frequency DESC
            """
            
            results = self.db.cur.execute(query).fetchall()
            
            frequent_plates = [
                {
                    "plate_number": row[0],
                    "frequency": row[1],
                    "successful_matches": row[2],
                    "currently_inside": row[3] > 0,
                    "last_seen": row[4],
                    "success_rate": round((row[2] / row[1] * 100), 2) if row[1] > 0 else 0
                }
                for row in results
            ]
            
            return {
                "frequent_plates": frequent_plates,
                "total_unique_plates": len(frequent_plates),
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}




    # ===== THỐNG KÊ THỜI GIAN ĐỖ =====
    def get_parking_duration_statistics(self):
        """
        Thống kê phân tích thời gian đỗ xe
        Returns: dict với phân tích thời gian
        """
        if not self.db or not self.db.ok:
            return {"error": "Database not available"}
        
        try:
            # Thống kê các xe đã hoàn thành (MATCHED)
            duration_query = """
                SELECT 
                    DATEDIFF(MINUTE, 
                        TRY_CONVERT(datetime, date_in + ' ' + time_in, 103),
                        TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)
                    ) as duration_minutes
                FROM dbo.ParkingSessions
                WHERE match_status = 'MATCHED'
                    AND date_in IS NOT NULL AND time_in IS NOT NULL
                    AND date_out IS NOT NULL AND time_out IS NOT NULL
                    AND TRY_CONVERT(datetime, date_in + ' ' + time_in, 103) IS NOT NULL
                    AND TRY_CONVERT(datetime, date_out + ' ' + time_out, 103) IS NOT NULL
            """
            
            durations = [row[0] for row in self.db.cur.execute(duration_query).fetchall() if row[0] is not None and row[0] > 0]
            
            if not durations:
                return {"error": "No completed parking sessions found"}
            
            # Phân tích thống kê
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            median_duration = sorted(durations)[len(durations) // 2]
            
            # Phân loại thời gian đỗ
            short_term = len([d for d in durations if d <= 60])      # <= 1 giờ
            medium_term = len([d for d in durations if 60 < d <= 480])  # 1-8 giờ
            long_term = len([d for d in durations if d > 480])          # > 8 giờ
            
            return {
                "total_completed_sessions": len(durations),
                "average_duration_minutes": round(avg_duration, 2),
                "average_duration_hours": round(avg_duration / 60, 2),
                "min_duration_minutes": min_duration,
                "max_duration_minutes": max_duration,
                "median_duration_minutes": median_duration,
                "distribution": {
                    "short_term": {"count": short_term, "percentage": round(short_term/len(durations)*100, 2)},
                    "medium_term": {"count": medium_term, "percentage": round(medium_term/len(durations)*100, 2)},
                    "long_term": {"count": long_term, "percentage": round(long_term/len(durations)*100, 2)}
                },
                "error": None
            }
            
        except Exception as e:
            return {"error": str(e)}




    # ===== HELPER FUNCTIONS =====
    def _calculate_duration_from_entry(self, date_in, time_in):
        """
        Tính thời gian đỗ từ khi vào đến hiện tại
        Args: date_in (str), time_in (str)
        Returns: int (phút)
        """
        try:
            entry_time_str = f"{date_in} {time_in}"
            entry_time = datetime.strptime(entry_time_str, "%d/%m/%Y %H:%M:%S")
            now = datetime.now()
            duration = now - entry_time
            return int(duration.total_seconds() / 60)
        except:
            return 0




    # ===== EXPORT BÁO CÁO =====
    def export_comprehensive_report(self, file_path="parking_report.txt"):
        """
        Export báo cáo tổng hợp ra file text
        Args: file_path (str): Đường dẫn file
        Returns: bool (success/failure)
        """
        try:
            overview = self.get_overview_statistics()
            cars_inside = self.get_cars_currently_inside()
            daily_stats = self.get_daily_statistics()
            weekly_stats = self.get_weekly_statistics()
            duration_stats = self.get_parking_duration_statistics()
            frequent_plates = self.get_frequent_plates_statistics()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("     BÁO CÁO THỐNG KÊ HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE\n")
                f.write("="*60 + "\n")
                f.write(f"Thời gian tạo báo cáo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                
                # Tổng quan
                f.write("📊 TỔNG QUAN HỆ THỐNG\n")
                f.write("-" * 30 + "\n")
                f.write(f"• Tổng số phiên làm việc: {overview.get('total_sessions', 0)}\n")
                f.write(f"• Xe hiện đang trong bãi: {overview.get('current_cars_inside', 0)}\n")
                f.write(f"• Phiên hoàn thành: {overview.get('completed_sessions', 0)}\n")
                f.write(f"• Phiên không khớp: {overview.get('unmatched_sessions', 0)}\n")
                f.write(f"• Tỷ lệ thành công tổng: {overview.get('overall_success_rate', 0)}%\n\n")
                
                # Thống kê hôm nay
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
                for car in cars_inside.get('list', [])[:10]:  # Top 10
                    f.write(f"• {car['plate']} - Vào lúc: {car['date_in']} {car['time_in']} - Đỗ: {car['duration_minutes']} phút\n")
                f.write("\n")
                
                # Biển số thường xuyên
                f.write("🔢 TOP BIỂN SỐ THƯỜNG XUYÊN\n")
                f.write("-" * 30 + "\n")
                for plate in frequent_plates.get('frequent_plates', [])[:5]:
                    f.write(f"• {plate['plate_number']}: {plate['frequency']} lần - Thành công: {plate['success_rate']}%\n")
                f.write("\n")
                
                # Thời gian đỗ
                if not duration_stats.get('error'):
                    f.write("⏱️ PHÂN TÍCH THỜI GIAN ĐỖ XE\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"• Thời gian đỗ trung bình: {duration_stats.get('average_duration_hours', 0)} giờ\n")
                    f.write(f"• Thời gian đỗ ngắn nhất: {duration_stats.get('min_duration_minutes', 0)} phút\n")
                    f.write(f"• Thời gian đỗ dài nhất: {duration_stats.get('max_duration_minutes', 0)} phút\n")
                    dist = duration_stats.get('distribution', {})
                    f.write(f"• Đỗ ngắn hạn (≤1h): {dist.get('short_term', {}).get('count', 0)} ({dist.get('short_term', {}).get('percentage', 0)}%)\n")
                    f.write(f"• Đỗ trung hạn (1-8h): {dist.get('medium_term', {}).get('count', 0)} ({dist.get('medium_term', {}).get('percentage', 0)}%)\n")
                    f.write(f"• Đỗ dài hạn (>8h): {dist.get('long_term', {}).get('count', 0)} ({dist.get('long_term', {}).get('percentage', 0)}%)\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("Báo cáo được tạo bởi Hệ thống Quản lý Bãi đỗ xe Thông minh\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
            
    def count_sessions_by_status(self, status):
        """
        Đếm số sessions theo status cụ thể
        Args: status - string như "KHOP-BIEN-SO", "KHONG-KHOP-BIEN-SO", "PENDING"
        Returns: int - số lượng sessions
        """
        if not self.db or not self.db.ok:
            return 0
            
        try:
            query = """
                SELECT COUNT(*) 
                FROM dbo.ParkingSessions 
                WHERE match_status = ?
            """
            result = self.db.cur.execute(query, (status,)).fetchone()
            return result[0] if result else 0
            
        except Exception as e:
            print(f"Error counting sessions by status {status}: {e}")
            return 0




# ===== DEMO USAGE =====
if __name__ == "__main__":
    stats = ParkingStatistics()
    
    print("🚗 DEMO THỐNG KÊ HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE")
    print("="*50)
    
    # Test các function
    overview = stats.get_overview_statistics()
    cars_inside = stats.get_cars_currently_inside()
    daily = stats.get_daily_statistics()
    
    print(f"📊 Tổng quan: {overview.get('total_sessions', 0)} phiên, {overview.get('current_cars_inside', 0)} xe trong bãi")
    print(f"📅 Hôm nay: {daily.get('entries_today', 0)} vào, {daily.get('exits_today', 0)} ra")
    print(f"🚗 Chi tiết xe trong bãi: {cars_inside.get('total', 0)} xe")
    
    # Export báo cáo
    if stats.export_comprehensive_report("demo_report.txt"):
        print("✅ Đã export báo cáo ra file: demo_report.txt")
    else:
        print("❌ Lỗi khi export báo cáo")