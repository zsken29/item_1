import os
import sys
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

API_KEY = os.environ.get('MINIMAX_API_KEY', 'sk-cp-ALlW5lD3StgyNRUCc4QnaUNJs8WngxIiEswevo9TcclPrwapHd5QwSLL1VXMuBZoXqg6T4TpiwKrttEb27ko0Qn4NmaGflgvjvR6hX05GdqsLfQI2ZXlvh8')
API_URL = 'https://api.minimax.chat/v1/chat/completions'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if data is None:
        return jsonify({'error': '无效的JSON数据'}), 400

    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400

    conversation_history = data.get('history', []) or []

    conversation_history.append({
        'role': 'user',
        'content': user_message
    })

    print(f"[DEBUG] user_message: {user_message}", file=sys.stderr)
    print(f"[DEBUG] history length: {len(conversation_history)}", file=sys.stderr)

    try:
        resp = requests.post(
            API_URL,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {API_KEY}'
            },
            json={
                'model': 'MiniMax-M2.7',
                'messages': conversation_history,
                'stream': False
            },
            timeout=60
        )

        print(f"[DEBUG] API status: {resp.status_code}", file=sys.stderr)
        print(f"[DEBUG] API response: {resp.text[:200]}", file=sys.stderr)

        result = resp.json()
        if resp.ok:
            ai_response = result['choices'][0]['message']['content']
        else:
            return jsonify({'error': f'API错误: {resp.status_code}', 'detail': result}), resp.status_code

        conversation_history.append({
            'role': 'assistant',
            'content': ai_response
        })

        return jsonify({
            'response': ai_response,
            'history': conversation_history
        })

    except requests.exceptions.Timeout:
        return jsonify({'error': '请求超时，请稍后重试'}), 504
    except Exception as e:
        print(f"[DEBUG] Exception: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
