import pyodbc
from phanmemgiuxe.config.config import CONN_STR



# ===== Lớp quản lý cấu hình barrier trong database =====
class BarrierConfigDB:
    
    
    
    
    
    # === Khởi tạo kết nối database ===
    def __init__(self):
        self.conn_str = CONN_STR




    # === Lấy tất cả cấu hình barrier ===
    def get_all(self):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM BarrierConfigs")
            return cursor.fetchall()




    # === Thêm cấu hình barrier ===
    def add(self, data):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO BarrierConfigs (name, lane, ip_address, port_number, serial_number, account, password, port_name, relay, open_delay_ms, close_delay_ms, auto_open_on_match, is_active, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['name'], data['lane'], data['ip_address'], data['port_number'],
                data['serial_number'], data['account'], data['password'], data['port_name'],
                data['relay'], data['open_delay_ms'], data['close_delay_ms'],
                data['auto_open_on_match'], data['is_active'], data['note']
            ))
            conn.commit()




    # === Cập nhật cấu hình barrier ===
    def update(self, id, data):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE BarrierConfigs SET name=?, lane=?, ip_address=?, port_number=?, serial_number=?, account=?, password=?, port_name=?, relay=?, open_delay_ms=?, close_delay_ms=?, auto_open_on_match=?, is_active=?, note=?, created_at=created_at WHERE barrier_id=?
            """, (
                data['name'], data['lane'], data['ip_address'], data['port_number'],
                data['serial_number'], data['account'], data['password'], data['port_name'],
                data['relay'], data['open_delay_ms'], data['close_delay_ms'],
                data['auto_open_on_match'], data['is_active'], data['note'], id
            ))
            conn.commit()




    # === Xóa cấu hình barrier ===
    def delete(self, id):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM BarrierConfigs WHERE barrier_id=?", (id,))
            conn.commit()




    # === Lấy cấu hình barrier theo ID ===
    def get_by_id(self, id):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM BarrierConfigs WHERE barrier_id=?", (id,))
            return cursor.fetchone()
