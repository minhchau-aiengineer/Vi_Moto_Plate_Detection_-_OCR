import pandas as pd, pyodbc
from ..config.config import USE_SQL, CONN_STR
from ..utils.utils import plate_norm


USE_SQL = True
try:
    import pyodbc
except Exception:
    USE_SQL = False





class DB:
    def __init__(self, conn_str: str):
        self.ok = False; self.conn = None; self.cur  = None
        if not USE_SQL: return
        try:
            self.conn = pyodbc.connect(conn_str, autocommit=True)
            self.cur  = self.conn.cursor()
            self.cur.execute("""
                IF OBJECT_ID('dbo.ParkingSessions','U') IS NULL
                CREATE TABLE dbo.ParkingSessions(
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    plate_in NVARCHAR(64)  NULL,
                    date_in  NVARCHAR(16)  NULL,
                    time_in  NVARCHAR(16)  NULL,
                    image_in NVARCHAR(MAX) NULL,
                    plate_out NVARCHAR(64)  NULL,
                    date_out  NVARCHAR(16)  NULL,
                    time_out  NVARCHAR(16)  NULL,
                    image_out NVARCHAR(MAX) NULL,
                    match_status NVARCHAR(32) NULL,
                    created_at DATETIME DEFAULT GETDATE()
                );
            """)
            self.ok = True
        except Exception as e:
            print("DB connect error:", e); self.ok = False





    def insert_in(self, plate, d, t, img_path):
        if not self.ok or not img_path: 
            return 
        try:
            self.cur.execute("""
                INSERT INTO dbo.ParkingSessions(plate_in,date_in,time_in,image_in,match_status)
                VALUES (?,?,?,?,?)
            """, (plate, d, t, img_path, 'PENDING'))
        except Exception as e: 
            print("insert_in error:", e)





    def attach_out(self, plate_out, d, t, img_path) -> str:
        if not self.ok or not img_path: 
            return "KHONG-KHOP-BIEN-SO" 
        try:
            rows = self.cur.execute("""
                SELECT TOP 50 id, plate_in FROM dbo.ParkingSessions
                WHERE plate_out IS NULL
                ORDER BY id DESC
            """).fetchall()
            match_sid = None

            for sid, plate_in in rows:
                if plate_norm(plate_in) == plate_norm(plate_out):
                    match_sid = sid; break
            if match_sid:
                self.cur.execute("""
                    UPDATE dbo.ParkingSessions
                    SET plate_out=?, date_out=?, time_out=?, image_out=?, match_status='KHOP-BIEN-SO'
                    WHERE id=?
                """, (plate_out, d, t, img_path, match_sid))
                return "KHOP-BIEN-SO"
            else:
                self.cur.execute("""
                    INSERT INTO dbo.ParkingSessions(plate_out,date_out,time_out,image_out,match_status)
                    VALUES (?,?,?,?,'KHONG-KHOP-BIEN-SO')
                """, (plate_out, d, t, img_path))
                return "KHONG-KHOP-BIEN-SO"
        except Exception as e:
            print("attach_out error:", e); return "KHONG-KHOP-BIEN-SO"





    def fetch_history_df(self, limit=10000, start_time=None, end_time=None,
                     status_filter=None, plate_filter=None) -> pd.DataFrame:
        """
        Lọc theo:
        - Khoảng thời gian VÀO/RA (dựa trên date_in+time_in và date_out+time_out, đều là NVARCHAR)
        - Trạng thái (match_status)
        - Biển số (plate_in/plate_out LIKE)
        Không dùng created_at.
        """
        columns = [
            "ID","Ảnh vào","Biển số vào","Ngày vào","Giờ vào",
            "Ảnh ra","Biển số ra","Ngày ra","Giờ ra","Trạng thái"
        ]
        if not self.ok:
            return pd.DataFrame(columns=["STT"] + columns)

        try:
            dt_in  = "TRY_CONVERT(datetime, date_in  + ' ' + time_in , 103)"
            dt_out = "TRY_CONVERT(datetime, date_out + ' ' + time_out, 103)"

            sql = f"""
                SELECT TOP ({limit})
                    id, image_in, plate_in, date_in, time_in,
                    image_out, plate_out, date_out, time_out, match_status
                FROM dbo.ParkingSessions
            """

            where_clauses = []
            sql_params = []

            if start_time and end_time:
                where_clauses.append(f"( ({dt_in}  BETWEEN ? AND ?) OR ({dt_out} BETWEEN ? AND ?) )")
                sql_params += [start_time, end_time, start_time, end_time]
            elif start_time:
                where_clauses.append(f"( {dt_in}  >= ? OR {dt_out} >= ? )")
                sql_params += [start_time, start_time]
            elif end_time:
                where_clauses.append(f"( {dt_in}  <= ? OR {dt_out} <= ? )")
                sql_params += [end_time, end_time]

            # ------- Lọc Trạng thái -------
            if status_filter and len(status_filter) > 0:
                placeholders = ",".join("?" for _ in status_filter)
                where_clauses.append(f"match_status IN ({placeholders})")
                sql_params += status_filter

            # ------- Lọc Biển số gần đúng ở cả vào/ra -------
            if plate_filter and len(plate_filter.strip()) > 0:
                where_clauses.append("(plate_in LIKE ? OR plate_out LIKE ?)")
                like_term = f"%{plate_filter.strip()}%"
                sql_params += [like_term, like_term]

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += f" ORDER BY COALESCE({dt_out}, {dt_in}) DESC, id DESC"

            rows = self.cur.execute(sql, tuple(sql_params)).fetchall()

            df = pd.DataFrame.from_records(rows, columns=columns).astype(object).where(pd.notnull, "")
            df["Trạng thái"] = df["Trạng thái"].replace({"": "PENDING"})
            df.insert(0, "STT", range(1, len(df) + 1))
            return df

        except Exception as e:
            print(f"fetch_history_df error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=["STT"] + columns)





    def delete_by_ids(self, ids):
        if not self.ok or not ids: 
            return
        try:
            placeholders = ','.join('?' for _ in ids)
            sql = f"DELETE FROM dbo.ParkingSessions WHERE id IN ({placeholders})"
            self.cur.execute(sql, tuple(int(sid) for sid in ids))
        except Exception as e: 
            print("delete_by_ids error:", e)





    def delete_all(self):
        if not self.ok: 
            return
        try: 
            self.cur.execute("DELETE FROM dbo.ParkingSessions")
        except Exception as e: 
            print("delete_all error:", e)

