from flask import Flask, render_template_string, request, jsonify
import google.generativeai as genai

API = "AIzaSyDfR0gZcb3xQTrPZgrYw_wWicQSzrt1FLA"
genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gemini Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; margin: 0; }
        #chat-container { width: 100%; max-width: 600px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 20px; }
        #chat-log { height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #fafafa; margin-bottom: 10px; }
        .msg { margin: 8px 0; }
        .user { color: #1976d2; }
        .bot { color: #388e3c; }
        #input-area { display: flex; }
        #user-input { flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        #send-btn { padding: 8px 18px; margin-left: 8px; background: #1976d2; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
        #send-btn:disabled { background: #aaa; }
    </style>
</head>
<body>
    <div id="chat-container">
        <h2>Gemini Chatbot</h2>
        <div id="chat-log"></div>
        <form id="input-area" onsubmit="return sendMessage();">
            <input type="text" id="user-input" autocomplete="off" placeholder="Type your message..." required />
            <button id="send-btn" type="submit">Send</button>
        </form>
    </div>
    <script>
        const chatLog = document.getElementById('chat-log');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        function appendMessage(sender, text) {
            const div = document.createElement('div');
            div.className = 'msg ' + (sender === 'You' ? 'user' : 'bot');
            div.innerHTML = '<b>' + sender + ':</b> ' + text;
            chatLog.appendChild(div);
            chatLog.scrollTop = chatLog.scrollHeight;
        }

        function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return false;
            appendMessage('You', text);
            userInput.value = '';
            sendBtn.disabled = true;
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            })
            .then(res => res.json())
            .then(data => {
                appendMessage('Gemini', data.reply);
                sendBtn.disabled = false;
                userInput.focus();
            });
            return false;
        }

        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_input = data.get("message", "")
    if not user_input.strip():
        return jsonify({"reply": ""})
    if user_input.lower() == "exit":
        return jsonify({"reply": "Goodbye!"})
    response = chat.send_message(user_input)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    app.run(debug=True)