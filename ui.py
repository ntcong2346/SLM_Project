import streamlit as st
import requests
from qdrant_client import models
from database_manager import init_vector_store, get_all_sources, delete_source_from_db
import history_db

# --- KHỞI TẠO STATE & DATABASE ---
history_db.init_db()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "current_node_id" not in st.session_state:
    st.session_state.current_node_id = None
# State mới để ẩn/hiện sidebar bên phải
if "show_history" not in st.session_state:
    st.session_state.show_history = True

st.set_page_config(page_title="Bio-SLM AI Assistant", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1b2e 0%, #0f101a 100%);
        color: #ffffff;
    }

    /* Center the chat container */
    [data-testid="stVerticalBlock"] > div:has(div.stChatMessage) {
        max-width: 850px;
        margin: 0 auto;
    }

    /* Copilot Chat Bubbles */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
    }
    
    /* User message specific style */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: rgba(60, 65, 210, 0.2) !important;
        border: 1px solid rgba(100, 110, 255, 0.3);
    }

    /* Floating Input Bar Styling */
    .stChatInputContainer {
        padding: 20px !important;
        background: transparent !important;
        border: none !important;
    }
    
    .stChatInputContainer > div {
        background-color: #25273d !important;
        border: 1px solid #3e416d !important;
        border-radius: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8) !important;
    }

    /* Typography */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(90deg, #fff, #8a94ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 50px 0 !important;
    }
    
    /* Sidebar styling for 'Library' feel */
    section[data-testid="stSidebar"] {
        background-color: #0c0d14 !important;
        border-right: 1px solid #1f212e !important;
    }

    /* Làm mờ hoặc ẩn nền của avatar mặc định */
[data-testid="stChatMessageAvatarUser"], 
[data-testid="stChatMessageAvatarAssistant"] {
    background-color: transparent !important;
    border: none !important;
}


    /* Ẩn nút đóng sidebar mặc định để dùng giao diện tùy chỉnh nếu muốn */
    [data-testid="stSidebarNav"] {display: none;}

    /* Làm cho sidebar trông giống một bảng điều khiển nổi */
    section[data-testid="stSidebar"] {
        background-color: #0b0c12 !important;
        border-right: 1px solid #1f212e !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.5);
    }

    /* Style lại các nút trong lịch sử cho giống Copilot Library */
    .stButton button {
        border: none !important;
        background-color: transparent !important;
        text-align: left !important;
        padding: 5px 10px !important;
        transition: 0.3s;
    }

    
    .stButton button:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }
            
/* Tùy chỉnh kích thước icon cho cân đối */
.stChatMessage img {
    border-radius: 50%; /* Làm icon hình tròn cho giống Copilot */
    width: 35px;
    height: 35px;
}
            
    .stButton button {
        border-radius: 12px !important;
        background-color: #1f212e !important;
        border: 1px solid #3e416d !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Đang kết nối Vector DB...")
def get_vector_db():
    return init_vector_store()

vector_db = get_vector_db()

# ==========================================
# 1. LEFT SIDEBAR: CẤU HÌNH RAG
# ==========================================
with st.sidebar:
    st.header("Cấu hình SLM")
    if vector_db:
        st.success(" Vector DB: Ready")
    else:
        st.error(" Lỗi kết nối Vector DB")

    st.markdown("---")
    st.header(" Nguồn tài liệu")
    if st.button(" Làm mới danh sách", use_container_width=True):
        st.rerun()

    selected_sources = []
    if vector_db:
        all_docs = get_all_sources()
        if all_docs:
            for doc in all_docs:
                col_check, col_del = st.columns([4, 1])
                with col_check:
                    if st.checkbox(doc, value=True, key=f"check_{doc}"):
                        selected_sources.append(doc)
                with col_del:
                    if st.button("✖", key=f"del_{doc}"):
                        delete_source_from_db(doc)
                        st.rerun()

# ==========================================
# MAIN LAYOUT: CHAT (Trái) & HISTORY (Phải)
# ==========================================
# Nút Toggle History ở góc trên bên phải
col_title, col_toggle = st.columns([8, 1])
with col_title:
    st.title("Bio-SLM Assistant")
with col_toggle:
    toggle_text = "Ẩn Lịch sử" if st.session_state.show_history else " Hiện Lịch sử"
    if st.button(toggle_text, use_container_width=True):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()

# Thay đổi tỷ lệ cột dựa trên trạng thái toggle
if st.session_state.show_history:
    chat_col, history_col = st.columns([3, 1], gap="large")
else:
    chat_col = st.container() # Chat chiếm toàn màn hình nếu ẩn lịch sử

# ------------------------------------------
# 2. LỊCH SỬ CHAT (Chỉ render nếu show_history là True)
# ------------------------------------------
if st.session_state.show_history:
    with history_col:
        st.header(" Lịch sử Chat")
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.current_chat_id = None
            st.session_state.current_node_id = None
            st.rerun()
            
        st.divider()
        chats = history_db.get_all_chats()
        
        for chat in chats:
            title = chat['title'] if len(chat['title']) < 30 else chat['title'][:27] + "..."
            st.markdown('<div class="history-btn">', unsafe_allow_html=True)
            col_t, col_d = st.columns([4, 1])
            
            with col_t:
                if st.button(f" {title}", key=f"chat_{chat['chat_id']}", use_container_width=True):
                    # FIX LỖI LOAD CHAT CŨ: Cập nhật CẢ Chat ID và Node ID
                    st.session_state.current_chat_id = chat['chat_id']
                    st.session_state.current_node_id = history_db.get_latest_node_for_chat(chat['chat_id'])
                    st.rerun()
            
            with col_d:
                if st.button("✖", key=f"del_chat_{chat['chat_id']}"):
                    history_db.delete_chat(chat['chat_id'])
                    if st.session_state.current_chat_id == chat['chat_id']:
                        st.session_state.current_chat_id = None
                        st.session_state.current_node_id = None
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# 3. CHAT INTERFACE
# ------------------------------------------
with chat_col:
    branch = []
    message_container = st.container()
    if st.session_state.current_node_id:
        branch = history_db.get_message_branch(st.session_state.current_node_id)
    
    if not branch:
        st.info("Hãy đặt câu hỏi để bắt đầu cuộc trò chuyện mới!")
    else:
        for msg in branch:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                if msg["role"] == "assistant":
                    if st.button("Trích xuất nhánh (Fork)", key=f"fork_{msg['message_id']}"):
                        # Lấy tên chat cũ và tạo tên fork mới
                        old_chat = history_db.get_chat(st.session_state.current_chat_id)
                        old_title = old_chat['title'] if old_chat else "Unknown"
                        fork_title = f"Fork từ: {old_title}"
                        
                        new_chat, new_node = history_db.fork_chat(
                            st.session_state.current_chat_id, 
                            msg['message_id'],
                            new_title=fork_title
                        )
                        st.session_state.current_chat_id = new_chat
                        st.session_state.current_node_id = new_node
                        st.rerun()

    # Nhập tin nhắn mới
    if prompt := st.chat_input("Hỏi về tài liệu Sinh học đã chọn..."):
        if not st.session_state.current_chat_id:
            new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            st.session_state.current_chat_id = history_db.create_chat(title=new_title)
    
            
        user_node_id = history_db.add_message(
            chat_id=st.session_state.current_chat_id,
            parent_id=st.session_state.current_node_id,
            role="user",
            content=prompt
        )
        st.session_state.current_node_id = user_node_id
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm & suy luận..."):
                context = ""
                if vector_db and selected_sources:
                    search_filter = models.Filter(must=[models.FieldCondition(key="metadata.source", match=models.MatchAny(any=selected_sources))])
                    docs = vector_db.similarity_search(prompt, k=3, filter=search_filter)
                    if docs:
                        context = "\n\n---\n\n".join([d.page_content for d in docs])
                        with st.expander(" Xem nguồn trích dẫn"):
                            for i, d in enumerate(docs):
                                st.caption(f"Nguồn: {d.metadata.get('source', 'Unknown')}")
                                st.write(d.page_content[:200] + "...")

                groq_messages = [{"role": "system", "content": f"Bạn là trợ lý Sinh học 12. Ngữ cảnh:\n{context}"}]
                for m in branch:
                    groq_messages.append({"role": m["role"], "content": m["content"]})
                groq_messages.append({"role": "user", "content": prompt})
                
                try:
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}", "Content-Type": "application/json"}
                    data = {"model": "llama-3.1-8b-instant", "messages": groq_messages, "temperature": 0.3}
                    
                    response = requests.post(url, json=data, headers=headers)
                    if response.status_code == 200:
                        res_text = response.json()['choices'][0]['message']['content']
                        st.markdown(res_text)
                        
                        ast_node_id = history_db.add_message(
                            chat_id=st.session_state.current_chat_id,
                            parent_id=st.session_state.current_node_id,
                            role="assistant",
                            content=res_text
                        )
                        st.session_state.current_node_id = ast_node_id
                    else:
                        st.error(f"Lỗi API: {response.text}")
                except Exception as e:
                    st.error(f"Lỗi: {e}")