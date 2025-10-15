from flask import Flask, render_template_string, request, jsonify
import os
import re
import google.generativeai as genai

# ================== CONFIG MODEL ==================
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDfR0gZcb3xQTrPZgrYw_wWicQSzrt1FLA")
genai.configure(api_key=API_KEY)

SYSTEM_VI = (
    "Bạn là trợ lý AI nói TIẾNG VIỆT. "
    "Mặc định trả lời chi tiết và trọng điểm câu hỏi đưa ra. "
    "Khi người dùng xin CODE, hãy trả về trong khối ```lang ...``` và KHÔNG giải thích dài."
)

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_VI,
    generation_config=genai.GenerationConfig(
        temperature=0.2, top_p=0.9, top_k=40, max_output_tokens=512
    ),
)
chat = model.start_chat(history=[])

# ================== FLASK APP ==================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Gemini Chatbot</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f4f4f4; margin:0; }
    #chat-container { width:100%; max-width:700px; margin:32px auto; background:#fff; border-radius:10px;
                      box-shadow:0 2px 8px rgba(0,0,0,.08); padding:20px; }
    #chat-log { height:480px; overflow-y:auto; border:1px solid #e6e6e6; border-radius:8px; padding:12px; background:#fafafa; margin-bottom:12px; }
    .msg { margin:10px 0; line-height:1.5; }
    .sender { font-weight:700; margin-right:6px; }
    .user .sender { color:#1976d2; }
    .bot  .sender { color:#388e3c; }
    pre { background:#0f172a; color:#e2e8f0; padding:12px; border-radius:8px; overflow:auto; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    #input-area { display:flex; gap:8px; }
    #user-input { flex:1; padding:10px; border:1px solid #ccc; border-radius:6px; }
    #send-btn { padding:10px 16px; background:#1976d2; color:#fff; border:none; border-radius:6px; cursor:pointer; }
    #send-btn:disabled { opacity:.6; cursor:not-allowed; }
  </style>
</head>
<body>
  <div id="chat-container">
    <h2>Gemini Chatbot</h2>
    <div id="chat-log"></div>
    <form id="input-area" onsubmit="return sendMessage();">
      <input type="text" id="user-input" autocomplete="off" placeholder="Nhập tin nhắn..." required />
      <button id="send-btn" type="submit">Gửi</button>
    </form>
  </div>
  <script>
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    function renderMessage(sender, payload) {
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + (sender === 'Bạn' ? 'user' : 'bot');

      const s = document.createElement('span');
      s.className = 'sender';
      s.textContent = sender + ':';
      wrap.appendChild(s);

      if (payload.is_code) {
        // Hiển thị code block
        const pre = document.createElement('pre');
        const code = document.createElement('code');
        code.textContent = payload.text; // giữ nguyên xuống dòng/khoảng trắng
        pre.appendChild(code);
        wrap.appendChild(pre);
      } else {
        // Text thường: tôn trọng xuống dòng
        const p = document.createElement('div');
        p.style.whiteSpace = 'pre-wrap';
        p.textContent = payload.text;
        wrap.appendChild(p);
      }

      chatLog.appendChild(wrap);
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return false;
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
        renderMessage('Gemini', { text: 'Lỗi kết nối máy chủ.', is_code: false });
      } finally {
        sendBtn.disabled = false;
        userInput.focus();
      }
      return false;
    }

    userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

# -------- Helpers --------
CODE_HINTS = (
    "code", "mã", "đoạn mã", "snippet", "python", "java", "c++", "c#", "js", "javascript",
    "go", "rust", "ruby", "php", "typescript", "viết code", "hàm", "function", "class", "```"
)

def looks_like_code_request(user_text: str, model_text: str = "") -> bool:
    t = (user_text or "").lower()
    if any(k in t for k in CODE_HINTS): return True
    if "```" in (model_text or ""): return True
    return False

def extract_code_or_text(model_text: str):
    """
    Nếu có khối ```...``` thì trả về nội dung code & is_code=True.
    Nếu không, trả về text bình thường & is_code=False.
    """
    if not model_text:
        return {"text": "", "is_code": False}

    # Tìm code fence đầu tiên
    m = re.search(r"```([a-zA-Z0-9_+-]*)\\n([\\s\\S]*?)```", model_text)
    if m:
        code_body = m.group(2).strip()
        return {"text": code_body, "is_code": True}

    # Nếu không có code fence, giữ nguyên text (tôn trọng xuống dòng)
    return {"text": model_text.strip(), "is_code": False}

def summarize_vi(text: str, max_sent=3) -> str:
    """Rút gọn câu trả lời thường (không phải code)."""
    parts = re.split(r"(?<=[.!?…])\\s+", text.strip())
    clipped = " ".join(parts[:max_sent]).strip()
    return clipped

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify({"text": "Bạn chưa nhập gì.", "is_code": False})

    try:
        # Nhắc ngắn gọn + TV ở user turn
        user_turn = f"Trả lời bằng TIẾNG VIỆT. Nếu người dùng yêu cầu CODE, trả về trong ```lang ...```: {user_input}"
        resp = chat.send_message(user_turn)
        raw = getattr(resp, "text", "") or ""

        # Nếu là yêu cầu/phản hồi code -> trả code block; ngược lại thì rút gọn
        if looks_like_code_request(user_input, raw):
            payload = extract_code_or_text(raw)
        else:
            payload = {"text": summarize_vi(raw, max_sent=3), "is_code": False}

        if not payload["text"]:
            payload = {"text": "Mình chưa hiểu yêu cầu, bạn nói rõ hơn nhé.", "is_code": False}

        return jsonify(payload)
    except Exception as e:
        return jsonify({"text": f"Lỗi model: {e}", "is_code": False}), 500

if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template_string, request, jsonify
# import os
# import re
# import google.generativeai as genai

# # ================== CONFIG MODEL ==================
# # Ưu tiên lấy từ biến môi trường; nếu không có sẽ dùng chuỗi sẵn (KHÔNG khuyến nghị).
# API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDfR0gZcb3xQTrPZgrYw_wWicQSzrt1FLA")

# genai.configure(api_key=API_KEY)

# # System prompt: ép tiếng Việt + ngắn gọn
# SYSTEM_VI = (
#     "Bạn là trợ lý AI nói TIẾNG VIỆT. "
#     "Luôn trả lời TRỌNG TÂM CÂU HỎI CỦA TÔI. "
#     "Ưu tiên trực tiếp vào trọng tâm, không rào trước đón sau, không định dạng rườm rà."
# )

# # Model + cấu hình sinh: giảm lan man
# model = genai.GenerativeModel(
#     model_name="gemini-2.0-flash",     # giữ nguyên tên bạn đang dùng
#     system_instruction=SYSTEM_VI,
#     generation_config=genai.GenerationConfig(
#         temperature=0.2,               # ít 'bay'
#         top_p=0.9,
#         top_k=40,
#         max_output_tokens=256          # chặn trả lời quá dài
#     ),
# )

# # Tạo phiên chat (có thể thay bằng generate_content nếu muốn stateless)
# chat = model.start_chat(history=[])

# # ================== FLASK APP ==================
# app = Flask(__name__)

# HTML = """
# <!DOCTYPE html>
# <html lang="vi">
# <head>
#   <meta charset="UTF-8">
#   <title>Gemini Chatbot</title>
#   <style>
#     body { font-family: Arial, sans-serif; background:#f4f4f4; margin:0; }
#     #chat-container { width:100%; max-width:600px; margin:40px auto; background:#fff; border-radius:8px; box-shadow:0 2px 8px #ccc; padding:20px; }
#     #chat-log { height:400px; overflow-y:auto; border:1px solid #ddd; border-radius:4px; padding:10px; background:#fafafa; margin-bottom:10px; }
#     .msg { margin:8px 0; }
#     .user { color:#1976d2; }
#     .bot { color:#388e3c; }
#     #input-area { display:flex; }
#     #user-input { flex:1; padding:8px; border:1px solid #ccc; border-radius:4px; }
#     #send-btn { padding:8px 18px; margin-left:8px; background:#1976d2; color:#fff; border:none; border-radius:4px; cursor:pointer; }
#     #send-btn:disabled { background:#aaa; }
#   </style>
# </head>
# <body>
#   <div id="chat-container">
#     <h2>Gemini Chatbot</h2>
#     <div id="chat-log"></div>
#     <form id="input-area" onsubmit="return sendMessage();">
#       <input type="text" id="user-input" autocomplete="off" placeholder="Nhập tin nhắn..." required />
#       <button id="send-btn" type="submit">Gửi</button>
#     </form>
#   </div>
#   <script>
#     const chatLog = document.getElementById('chat-log');
#     const userInput = document.getElementById('user-input');
#     const sendBtn = document.getElementById('send-btn');

#     function appendMessage(sender, text) {
#       const div = document.createElement('div');
#       div.className = 'msg ' + (sender === 'Bạn' ? 'user' : 'bot');
#       div.innerHTML = '<b>' + sender + ':</b> ' + text;
#       chatLog.appendChild(div);
#       chatLog.scrollTop = chatLog.scrollHeight;
#     }

#     async function sendMessage() {
#       const text = userInput.value.trim();
#       if (!text) return false;
#       appendMessage('Bạn', text);
#       userInput.value = '';
#       sendBtn.disabled = true;
#       try {
#         const res = await fetch('/chat', {
#           method: 'POST',
#           headers: { 'Content-Type': 'application/json' },
#           body: JSON.stringify({ message: text })
#         });
#         const data = await res.json();
#         appendMessage('Gemini', data.reply || '(không có phản hồi)');
#       } catch (e) {
#         appendMessage('Gemini', 'Lỗi kết nối máy chủ.');
#       } finally {
#         sendBtn.disabled = false;
#         userInput.focus();
#       }
#       return false;
#     }

#     userInput.addEventListener('keydown', function(e) {
#       if (e.key === 'Enter' && !e.shiftKey) {
#         e.preventDefault();
#         sendMessage();
#       }
#     });
#   </script>
# </body>
# </html>
# """

# @app.route("/")
# def index():
#     return render_template_string(HTML)

# # --------- Helpers làm sạch & rút gọn đầu ra ---------
# def strip_markdown(text: str) -> str:
#     """Bỏ các dấu **, *, tiêu đề #… để tránh rối."""
#     if not text:
#         return ""
#     text = text.replace("**", "").replace("*", "")
#     # bỏ các đầu dòng bullet
#     text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
#     # gom nhiều xuống dòng về tối đa 2
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     return text.strip()

# def limit_sentences_vi(text: str, max_sent=3) -> str:
#     """Cắt tối đa N câu tiếng Việt (tách thô theo dấu chấm cảm hỏi)."""
#     if not text:
#         return ""
#     # Tách câu theo . ! ? (kể cả có khoảng trắng sau đó)
#     parts = re.split(r"(?<=[.!?])\s+", text.strip())
#     clipped = " ".join(parts[:max_sent]).strip()
#     return clipped

# @app.route("/chat", methods=["POST"])
# def chat_api():
#     data = request.get_json(silent=True) or {}
#     user_input = (data.get("message") or "").strip()

#     if not user_input:
#         return jsonify({"reply": "Bạn chưa nhập gì."})

#     if user_input.lower() in {"exit", "quit"}:
#         return jsonify({"reply": "Tạm biệt nhé!"})

#     # Nhấn mạnh thêm yêu cầu tiếng Việt + ngắn gọn ở user turn
#     user_turn = f"Trả lời bằng TIẾNG VIỆT, ngắn gọn (≤ 3 câu): {user_input}"

#     try:
#         resp = chat.send_message(user_turn)
#         text = getattr(resp, "text", "") or ""
#         text = strip_markdown(text)
#         text = limit_sentences_vi(text, max_sent=3)
#         if not text:
#             text = "Mình chưa hiểu yêu cầu, bạn nói rõ hơn được không?"
#         return jsonify({"reply": text})
#     except Exception as e:
#         return jsonify({"reply": f"Lỗi model: {e}"}), 500

# if __name__ == "__main__":
#     # Mặc định chạy localhost:5000; đổi host='0.0.0.0' nếu muốn truy cập từ máy khác
#     app.run(debug=True)
