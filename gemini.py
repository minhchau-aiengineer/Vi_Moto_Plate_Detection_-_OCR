import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# ================== 1. TẢI CẤU HÌNH TỪ .ENV ==================
load_dotenv()

# --- Cấu hình Gemini AI ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Chưa cung cấp biến môi trường GEMINI_API_KEY")

# --- Cấu hình Database ---
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# ================== 2. KHỞI TẠO GEMINI MODEL ==================
try:
    genai.configure(api_key=GEMINI_API_KEY)

    SYSTEM_VI = (
        "Bạn là một trợ lý AI nói TIẾNG VIỆT, chuyên nghiệp và thân thiện. "
        "Mặc định trả lời chi tiết và trọng điểm câu hỏi đưa ra. "
        "Bạn luôn trả lời bằng TIẾNG VIỆT. "
        "Bạn luôn trả lời trong phạm vi kiến thức đã được huấn luyện, "
        "Bạn không bao giờ nói rằng bạn là một mô hình ngôn ngữ. "
        "Bạn KHÔNG BAO GIỜ từ chối trả lời câu hỏi trừ khi nó vi phạm chính sách. "
        "Bạn luôn tuân thủ nguyên tắc"
        "Bạn chỉ trả lời trong phạm vi kiến thức đã được huấn luyện, không đưa ra thông tin sai lệch. "
        "Khi người dùng xin CODE, hãy trả về trong khối ```lang ...``` và KHÔNG giải thích dài."
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", # Sử dụng model mới nhất và hợp lệ
        system_instruction=SYSTEM_VI,
        generation_config=genai.GenerationConfig(
            temperature=0.3, top_p=0.9, top_k=40, max_output_tokens=1024
        ),
    )
except Exception as e:
    print(f"Lỗi khởi tạo Gemini Model: {e}")
    exit()

# ================== 3. CÁC HÀM TƯƠNG TÁC DATABASE (PostgreSQL) ==================
def get_db_connection():
    """Tạo và trả về một kết nối tới database PostgreSQL."""
    conn = psycopg2.connect(
        dbname=DB_NAME, 
        user=DB_USER, 
        password=DB_PASSWORD, 
        host=DB_HOST, 
        port=DB_PORT
    )
    return conn

def init_db():
    """Khởi tạo bảng 'messages' trong database nếu chưa tồn tại."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                role VARCHAR(10) NOT NULL,
                parts TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Database 'messages' table initialized successfully.")
    except psycopg2.OperationalError as e:
        print(f"Lỗi kết nối hoặc khởi tạo DB: {e}")
        print("Vui lòng kiểm tra lại thông tin kết nối trong file .env và đảm bảo PostgreSQL đang chạy.")
        exit()


def load_history_from_db():
    """Tải lịch sử từ PostgreSQL và định dạng lại cho Gemini."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT role, parts FROM messages ORDER BY timestamp ASC;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [{"role": row["role"], "parts": [row["parts"]]} for row in rows]

def save_message_to_db(role: str, parts: str):
    """Lưu một tin nhắn vào PostgreSQL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (role, parts) VALUES (%s, %s);", (role, parts)
    )
    conn.commit()
    cursor.close()
    conn.close()

# ================== 4. CÁC HÀM HỖ TRỢ XỬ LÝ TEXT ==================
def extract_code_or_text(model_text: str):
    if not model_text:
        return {"text": "", "is_code": False}
    # Sửa lại regex cho đúng
    m = re.search(r"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```", model_text)
    if m:
        return {"text": m.group(2).strip(), "is_code": True}
    return {"text": model_text.strip(), "is_code": False}

# ================== 5. FLASK APPLICATION ==================
app = Flask(__name__)

# --- Giao diện Front-end ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8" />
    <title>Gemini Chatbot</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f0f2f5; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        #chat-container { width: 100%; max-width: 750px; height: 90vh; background: #fff; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        #chat-log { flex: 1; overflow-y: auto; border-bottom: 1px solid #ddd; padding: 20px; }
        .msg { margin-bottom: 18px; line-height: 1.6; max-width: 90%; display: flex; flex-direction: column; }
        .sender { font-weight: 600; margin-bottom: 4px; }
        .user { align-self: flex-end; }
        .user .sender { color: #007bff; text-align: right; }
        .bot { align-self: flex-start; }
        .bot .sender { color: #28a745; }
        .msg-content { padding: 10px 15px; border-radius: 18px; }
        .user .msg-content { background-color: #007bff; color: white; }
        .bot .msg-content { background-color: #e9ecef; color: #333; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; overflow-x: auto; font-family: 'Consolas', 'Courier New', monospace; font-size: 14px; }
        #input-area { display: flex; padding: 15px; border-top: 1px solid #ddd; }
        #user-input { flex: 1; padding: 12px; border: 1px solid #ccc; border-radius: 20px; font-size: 16px; resize: none; }
        #send-btn { margin-left: 10px; padding: 12px 20px; background-color: #007bff; color: white; border: none; border-radius: 20px; cursor: pointer; font-size: 16px; }
        #send-btn:disabled { background-color: #a0c7e8; cursor: not-allowed; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="chat-log"></div>
        <form id="input-area" onsubmit="sendMessage(event);">
            <input type="text" id="user-input" autocomplete="off" placeholder="Hỏi Gemini điều gì đó..." required />
            <button id="send-btn" type="submit">Gửi</button>
        </form>
    </div>
    <script>
        const chatLog = document.getElementById('chat-log');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        function renderMessage(sender, payload) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'msg ' + (sender === 'Bạn' ? 'user' : 'bot');
            
            const senderSpan = document.createElement('div');
            senderSpan.className = 'sender';
            senderSpan.textContent = sender;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'msg-content';

            if (payload.is_code) {
                const pre = document.createElement('pre');
                pre.textContent = payload.text;
                contentDiv.appendChild(pre);
            } else {
                contentDiv.textContent = payload.text;
            }
            
            msgDiv.appendChild(senderSpan);
            msgDiv.appendChild(contentDiv);
            chatLog.appendChild(msgDiv);
            chatLog.scrollTop = chatLog.scrollHeight;
        }

        async function sendMessage(event) {
            event.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;
            renderMessage('Bạn', { text, is_code: false });
            userInput.value = '';
            sendBtn.disabled = true;
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                renderMessage('Gemini', data);
            } catch (e) {
                renderMessage('Gemini', { text: 'Lỗi: Không thể kết nối tới máy chủ.', is_code: false });
            } finally {
                sendBtn.disabled = false;
                userInput.focus();
            }
        }
    </script>
</body>
</html>
"""

# --- Routes ---
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify({"text": "Vui lòng nhập câu hỏi.", "is_code": False})

    try:
        history = load_history_from_db()
        chat = model.start_chat(history=history)
        
        response = chat.send_message(user_input)
        raw_text = response.text or ""

        save_message_to_db("user", user_input)
        save_message_to_db("model", raw_text)

        payload = extract_code_or_text(raw_text)
        if not payload["text"]:
            payload = {"text": "Xin lỗi, mình chưa thể trả lời câu này.", "is_code": False}
        
        return jsonify(payload)

    except psycopg2.Error as db_error:
        print(f"DATABASE ERROR: {db_error}")
        return jsonify({"text": "Lỗi: Không thể kết nối tới database.", "is_code": False}), 500
    except Exception as e:
        print(f"MODEL/OTHER ERROR: {e}")
        return jsonify({"text": f"Lỗi: Có sự cố từ phía model AI. {e}", "is_code": False}), 500

# ================== 6. KHỞI CHẠY ỨNG DỤNG ==================
if __name__ == "__main__":
    init_db()  # Quan trọng: Đảm bảo bảng đã tồn tại trước khi chạy app
    app.run(debug=True, port=5000)