let conversationHistory = [];

function handleEnter(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function addMessage(content, isUser = false) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = `avatar ${isUser ? 'user' : 'ai'}`;
    avatarDiv.textContent = isUser ? '👤' : '🤖';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';
    bubbleDiv.textContent = content;

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(bubbleDiv);
    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function showTyping() {
    const chatArea = document.getElementById('chatArea');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing';
    typingDiv.className = 'typing';
    typingDiv.innerHTML = '<div class="dots"><span></span><span></span><span></span></div> 思考中...';
    chatArea.appendChild(typingDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function hideTyping() {
    const typing = document.getElementById('typing');
    if (typing) typing.remove();
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const message = input.value.trim();

    if (!message) return;

    addMessage(message, true);
    input.value = '';
    sendBtn.disabled = true;
    showTyping();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                history: conversationHistory
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `错误: ${response.status}`);
        }

        const data = await response.json();
        const aiResponse = data.response;

        hideTyping();
        addMessage(aiResponse);

        conversationHistory = data.history;
    } catch (error) {
        hideTyping();
        const chatArea = document.getElementById('chatArea');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = '抱歉，发生错误: ' + error.message;
        chatArea.appendChild(errorDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    sendBtn.disabled = false;
}
