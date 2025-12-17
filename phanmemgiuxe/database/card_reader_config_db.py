import pyodbc
from phanmemgiuxe.config.config import CONN_STR




# ===== Lớp quản lý cấu hình đầu đọc thẻ trong database =====
class CardReaderConfigDB:
    
    
    
    
    # === Khởi tạo kết nối database ===
    def __init__(self):
        self.conn_str = CONN_STR




    # === Lấy tất cả cấu hình đầu đọc thẻ ===
    def get_all(self):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM CardReaders")
            return cursor.fetchall()




    # === Thêm cấu hình đầu đọc thẻ ===
    def add(self, data):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO CardReaders (device_name, port_name, ip_address, port_number, serial_number, reader_id, status, device_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['device_name'], data['port_name'], data['ip_address'], data['port_number'],
                data['serial_number'], data['reader_id'], data['status'], data['device_type']
            ))
            conn.commit()




    # === Cập nhật cấu hình đầu đọc thẻ ===
    def update(self, id, data):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE CardReaders SET device_name=?, port_name=?, ip_address=?, port_number=?, serial_number=?, reader_id=?, status=?, device_type=?, updated_at=GETDATE() WHERE id=?
            """, (
                data['device_name'], data['port_name'], data['ip_address'], data['port_number'],
                data['serial_number'], data['reader_id'], data['status'], data['device_type'], id
            ))
            conn.commit()




    # === Xóa cấu hình đầu đọc thẻ ===
    def delete(self, id):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM CardReaders WHERE id=?", (id,))
            conn.commit()




    # === Lấy cấu hình đầu đọc thẻ theo ID ===
    def get_by_id(self, id):
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM CardReaders WHERE id=?", (id,))
            return cursor.fetchone()
