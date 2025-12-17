# Quản lý truy xuất cấu hình camera từ bảng dbo.Cameras (SQL Server)
import pyodbc
from ..config.config import CONN_STR




# ===== Lớp quản lý cấu hình camera trong database =====
class CameraConfigDB:
    
    
    
    
    # === Lấy cấu hình mapping camera theo chức năng ===
    def get_camera_mapping_configs(self) -> dict:
            """
            Trả về dict: function_type -> camera config dict (chỉ lấy camera đang active)
            Ví dụ: {
                'vao_truoc': {...},
                'vao_sau': {...},
                'ra_truoc': {...},
                'ra_sau': {...}
            }
            """
            if not self.conn:
                self._connect()
            if not self.conn:
                return {}
            try:
                rows = self._execute_all(
                    '''
                    SELECT m.function_type, c.*
                    FROM dbo.CameraMapping m
                    JOIN dbo.Cameras c ON m.camera_id = c.camera_id
                    WHERE c.is_active = 1
                    ''',
                )
                mapping = {}
                for row in rows:
                    func = row.get('function_type')
                    if func:
                        mapping[func] = row
                return mapping
            except Exception as e:
                print(f"[CameraConfigDB] get_camera_mapping_configs error: {e}")
                return {}
            
    
    
    
    # === Khởi tạo kết nối database ===
    def __init__(self, conn_str=None):
        self.conn_str = conn_str or CONN_STR
        self.conn = None
        self._connect()




    # === Kết nối database ===
    def _connect(self):
        try:
            self.conn = pyodbc.connect(self.conn_str)
        except Exception as e:
            print(f"[CameraConfigDB] DB connect error: {e}")
            self.conn = None




    # === Thực thi câu lệnh SQL ===
    def _execute(self, sql, params=()):
        if not self.conn:
            self._connect()
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, params)
                self.conn.commit()
            return True
        except Exception as e:
            print(f"[CameraConfigDB] _execute error: {e}")
            return False




    # === Thực thi câu lệnh SQL và trả về một bản ghi ===
    def _execute_one(self, sql, params=()):
        if not self.conn:
            self._connect()
        if not self.conn:
            return None
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, params)
                row = cursor.fetchone()
                if not row:
                    return None
                columns = [column[0] for column in cursor.description]
                return dict(zip(columns, row))
        except Exception as e:
            print(f"[CameraConfigDB] _execute_one error: {e}")
            return None




    # === Thực thi câu lệnh SQL và trả về tất cả bản ghi ===
    def _execute_all(self, sql, params=()):
        if not self.conn:
            self._connect()
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                columns = [column[0] for column in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"[CameraConfigDB] _execute_all error: {e}")
            return []




    # === Lấy cấu hình camera theo tên ===
    def get_camera_by_name(self, camera_name: str):
        """
        Trả về dict cấu hình camera theo tên (camera_name), chỉ lấy camera đang active.
        """
        if not self.conn:
            self._connect()
        if not self.conn:
            return None
        try:
            row = self._execute_one(
                """
                SELECT TOP (1)
                    camera_id,
                    camera_name,
                    camera_type,
                    source_index,
                    ip_address,
                    port,
                    url_path,
                    full_url,
                    username,
                    password,
                    direction,
                    view_role,
                    is_active,
                    note
                FROM dbo.Cameras
                WHERE camera_name = ? AND is_active = 1
                ORDER BY camera_id
                """,
                (camera_name,)
            )
            return row
        except Exception as e:
            print(f"[CameraConfigDB] get_camera_by_name error ({camera_name}):", e)
            return None




    # === Lấy tất cả cấu hình camera đang active ===
    def get_all_active_cameras(self):
        """
        Trả về list các dict cấu hình của tất cả camera đang active.
        """
        if not self.conn:
            self._connect()
        if not self.conn:
            return []
        try:
            rows = self._execute_all(
                """
                SELECT camera_id,
                       camera_name,
                       camera_type,
                       source_index,
                       ip_address,
                       port,
                       url_path,
                       full_url,
                       username,
                       password,
                       direction,
                        view_role,
                        is_active,
                        note
                FROM dbo.Cameras
                WHERE is_active = 1
                ORDER BY camera_id
                """
            )
            return rows
        except Exception as e:
            print("[CameraConfigDB] get_all_active_cameras error:", e)
            return []




    # === Thêm mới camera ===
    def add_camera(self, camera_info: dict):
        """
        Thêm mới một camera vào bảng Cameras.
        camera_info: dict chứa các trường cần thiết.
        """
        if not self.conn:
            self._connect()
        if not self.conn:
            return False
        try:
            self._execute(
                """
                INSERT INTO dbo.Cameras (
                    camera_name, camera_type, source_index, ip_address, port, url_path, full_url, username, password, direction, view_role, is_active, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_info.get('camera_name'),
                    camera_info.get('camera_type'),
                    camera_info.get('source_index'),
                    camera_info.get('ip_address'),
                    camera_info.get('port'),
                    camera_info.get('url_path'),
                    camera_info.get('full_url'),
                    camera_info.get('username'),
                    camera_info.get('password'),
                    camera_info.get('direction'),
                    camera_info.get('view_role', None),
                    camera_info.get('is_active', 1),
                    camera_info.get('note'),
                )
            )
            return True
        except Exception as e:
            print("[CameraConfigDB] add_camera error:", e)
            return False




    # === Cập nhật thông tin camera ===
    def update_camera(self, camera_id: int, update_fields: dict):
        """
        Cập nhật thông tin camera theo camera_id.
        update_fields: dict các trường cần cập nhật.
        """
        if not self.conn:
            self._connect()
        if not self.conn:
            return False
        try:
            set_clause = ', '.join([f"{k} = ?" for k in update_fields.keys()])
            params = list(update_fields.values())
            params.append(camera_id)
            sql = f"UPDATE dbo.Cameras SET {set_clause} WHERE camera_id = ?"
            self._execute(sql, tuple(params))
            return True
        except Exception as e:
            print(f"[CameraConfigDB] update_camera error: {e}")
            return False




    # === Xóa camera ===
    def delete_camera(self, camera_id: int):
        """
        Xóa hoàn toàn bản ghi camera khỏi bảng Cameras.
        """
        if not self.conn:
            self._connect()
        if not self.conn:
            return False
        try:
            sql = "DELETE FROM dbo.Cameras WHERE camera_id = ?"
            return self._execute(sql, (camera_id,))
        except Exception as e:
            print(f"[CameraConfigDB] delete_camera error: {e}")
            return False
