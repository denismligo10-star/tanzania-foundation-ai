
const API_URL = "https://tanzania-foundation-ai.onrender.com";

const body = document.body;
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeText = document.getElementById("themeText");
const messageInput = document.getElementById("messageInput");
const chatForm = document.getElementById("chatForm");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const newChatButton = document.getElementById("newChat");
const chatHistory = document.getElementById("chatHistory");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("overlay");

let chats = JSON.parse(localStorage.getItem("tanzania_ai_chats") || "[]");
let currentChatId = null;
let isSending = false;

function saveChats() {
    localStorage.setItem("tanzania_ai_chats", JSON.stringify(chats));
}

function loadTheme() {
    const savedTheme = localStorage.getItem("tanzania_ai_theme") || "light";
    body.classList.toggle("dark", savedTheme === "dark");
    updateThemeButton();
}

function updateThemeButton() {
    const dark = body.classList.contains("dark");
    themeIcon.textContent = dark ? "☀" : "☾";
    themeText.textContent = dark ? "Light mode" : "Dark mode";

    document.querySelector('meta[name="theme-color"]')
        ?.setAttribute("content", dark ? "#111111" : "#ffffff");
}

themeToggle.addEventListener("click", () => {
    body.classList.toggle("dark");
    localStorage.setItem(
        "tanzania_ai_theme",
        body.classList.contains("dark") ? "dark" : "light"
    );
    updateThemeButton();
});

function createId() {
    return Date.now().toString();
}

function renderHistory() {
    chatHistory.innerHTML = "";

    chats.slice().reverse().forEach(chat => {
        const item = document.createElement("div");
        item.className = "history-item";
        item.textContent = chat.title || "New chat";
        item.addEventListener("click", () => openChat(chat.id));
        chatHistory.appendChild(item);
    });
}

function addMessage(role, content) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    if (role === "assistant") {
        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = "T";
        wrapper.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = "message-content";
    bubble.textContent = content;
    wrapper.appendChild(bubble);

    messages.appendChild(wrapper);

    const conversation = document.getElementById("conversation");
    conversation.scrollTop = conversation.scrollHeight;

    return bubble;
}

function startNewChat() {
    currentChatId = createId();
    messages.innerHTML = "";
    welcome.style.display = "block";
    messageInput.value = "";
    closeSidebar();
}

function openChat(id) {
    const chat = chats.find(item => item.id === id);
    if (!chat) return;

    currentChatId = id;
    messages.innerHTML = "";
    welcome.style.display = "none";

    chat.messages.forEach(message => {
        addMessage(message.role, message.content);
    });

    closeSidebar();
}

function saveCurrentMessage(role, content) {
    let chat = chats.find(item => item.id === currentChatId);

    if (!chat) {
        chat = {
            id: currentChatId || createId(),
            title: role === "user" ? content.slice(0, 35) : "New chat",
            messages: []
        };

        currentChatId = chat.id;
        chats.push(chat);
    }

    chat.messages.push({ role, content });
    saveChats();
    renderHistory();
}

function closeSidebar() {
    sidebar.classList.remove("open");
    overlay.classList.remove("active");
}

function openSidebar() {
    sidebar.classList.add("open");
    overlay.classList.add("active");
}

document.getElementById("menuButton")
    .addEventListener("click", openSidebar);

document.getElementById("closeSidebar")
    .addEventListener("click", closeSidebar);

overlay.addEventListener("click", closeSidebar);
newChatButton.addEventListener("click", startNewChat);

async function sendMessage(text) {
    text = text.trim();

    if (!text || isSending) return;

    isSending = true;
    welcome.style.display = "none";

    addMessage("user", text);
    saveCurrentMessage("user", text);

    messageInput.value = "";
    messageInput.style.height = "auto";

    const reply = addMessage("assistant", "Tanzania AI inafikiria...");

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: text
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || `Hitilafu ya server: ${response.status}`
            );
        }

        const answer = data.reply || data.response || "Hakuna jibu lililorudishwa.";

        reply.textContent = answer;
        saveCurrentMessage("assistant", answer);

    } catch (error) {
        const errorMessage =
            "Imeshindikana kupata jibu kutoka kwenye AI: " + error.message;

        reply.textContent = errorMessage;
        saveCurrentMessage("assistant", errorMessage);

    } finally {
        isSending = false;

        const conversation = document.getElementById("conversation");
        conversation.scrollTop = conversation.scrollHeight;
    }
}

chatForm.addEventListener("submit", event => {
    event.preventDefault();
    sendMessage(messageInput.value);
});

messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height =
        Math.min(messageInput.scrollHeight, 160) + "px";
});

messageInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

document.querySelectorAll(".suggestion").forEach(button => {
    button.addEventListener("click", () => {
        sendMessage(button.dataset.prompt);
    });
});

loadTheme();
renderHistory();
startNewChat();
