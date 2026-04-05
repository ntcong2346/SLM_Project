import streamlit as st
import requests
from qdrant_client import models # Cần import models để dùng bộ lọc (Filter) của Qdrant
from database_manager import init_vector_store, get_all_sources, delete_source_from_db 

# --- CẤU HÌNH TRANG CHUYÊN NGHIỆP ---
st.set_page_config(
    page_title="Bio-SLM AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: 1px solid #30363d; }
    .stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }
    h1 { color: #58a6ff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .status-box { padding: 10px; border-radius: 10px; border: 1px solid #30363d; background-color: #0d1117; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- TỐI ƯU HÓA: CACHE VECTOR DB ---
@st.cache_resource(show_spinner="Đang kết nối Vector DB...")
def get_vector_db():
    return init_vector_store()

# --- TIÊU ĐỀ ---
col1, col2 = st.columns([1, 4])
with col2:
    st.title("Bio-SLM AI Assistant")
    st.markdown("*Hệ thống RAG hỗ trợ học tập Sinh học 12 (NotebookLM Style)*")

st.divider()

# --- SIDEBAR: QUẢN LÝ ---
with st.sidebar:
    st.header("⚙️ Cấu hình SLM")
    
    try:
        vector_db = get_vector_db()
        st.success("🟢 Qdrant DB & Jina Embeddings: Ready")
    except Exception as e:
        vector_db = None
        st.error(f"🔴 Lỗi kết nối Vector DB: {e}")

    st.markdown("---")
    
    # --- TÍNH NĂNG MỚI: CHỌN NGUỒN TÀI LIỆU ---
    st.header("📚 Nguồn tài liệu")
    
    if st.button("🔄 Làm mới danh sách", use_container_width=True):
        st.rerun()

    all_docs = get_all_sources() if vector_db else []
    selected_sources = []
    if vector_db:
        try:
            all_docs = get_all_sources()
            if all_docs:
                st.write("Tích chọn tài liệu AI được phép dùng:")
                for doc in all_docs:
                    col_check, col_del = st.columns([4, 1])
                    with col_check:
                        # Hiển thị checkbox, mặc định là được chọn
                        if st.checkbox(doc, value=True, key=f"check_{doc}"):
                            selected_sources.append(doc)
                    with col_del:
                        # Nút xóa tài liệu
                        if st.button("🗑️", key=f"del_{doc}", help="Xóa hoàn toàn khỏi DB"):
                            if delete_source_from_db(doc):
                                st.success("Đã xóa!")
                                st.rerun() # Tải lại trang để cập nhật danh sách
            else:
                st.info("Chưa có tài liệu nào. Hãy upload bên tab kia.")
        except Exception as e:
            st.error(f"Lỗi tải danh sách: {e}")

    st.markdown("---")
    st.subheader("Thông số SLM")
    st.write("**Model:** Llama-3.1-8B-Instant")
    st.write("**DB:** Qdrant (Vector Engine)")
    st.write("**Optimization:** Groq LPU")
    
    st.markdown("---")
    st.subheader("Trích dẫn RAG")
    source_container = st.empty()

# --- KHUNG CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi đã sẵn sàng hỗ trợ bạn ôn tập Sinh học 12. Hãy tích chọn tài liệu ở bên trái và đặt câu hỏi nhé!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi về tài liệu đã chọn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy vấn kiến thức & phân tích..."):
            context = ""
            
            # --- BƯỚC 1: TRUY VẤN DỮ LIỆU CÓ BỘ LỌC TÙY CHỈNH ---
            if vector_db:
                if not selected_sources:
                    with source_container.container():
                        st.warning("⚠️ Bạn chưa chọn tài liệu nào để tham khảo.")
                else:
                    try:
                        # Tạo bộ lọc: Chỉ tìm trong các "source" nằm trong danh sách selected_sources
                        search_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="metadata.source",
                                    match=models.MatchAny(any=selected_sources),
                                )
                            ]
                        )
                        
                        # Thêm filter vào hàm search
                        docs = vector_db.similarity_search(prompt, k=3, filter=search_filter)
                        
                        if docs:
                            context = "\n\n---\n\n".join([d.page_content for d in docs])
                            
                            with source_container.container():
                                for i, d in enumerate(docs):
                                    source_name = d.metadata.get('source', 'Unknown')
                                    st.caption(f"Nguồn {i+1} ({source_name}):")
                                    st.info(d.page_content[:200] + "...")
                        else:
                            with source_container.container():
                                st.warning("Không tìm thấy dữ liệu liên quan trong các tài liệu đã chọn.")
                    except Exception as e:
                        st.error(f"Lỗi truy vấn Qdrant: {e}")

            # --- BƯỚC 2: GỌI GROQ API ---
            try:
                api_key = st.secrets["GROQ_API_KEY"] 
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                
                if context:
                    system_prompt = f"Bạn là trợ lý học tập Sinh học 12. Dựa CHỈ VÀO các thông tin sau đây để trả lời câu hỏi của học sinh. Nếu thông tin không có trong ngữ cảnh, hãy nói không biết:\n\n{context}"
                else:
                    system_prompt = "Bạn là trợ lý học tập. Hãy trả lời bằng kiến thức của bạn do học sinh không cung cấp tài liệu tham khảo."

                data = {
                    "model": "llama-3.1-8b-instant", 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3
                }
                
                response = requests.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    res_text = response.json()['choices'][0]['message']['content']
                    st.markdown(res_text)
                    st.session_state.messages.append({"role": "assistant", "content": res_text})
                else:
                    st.error(f"Lỗi API Groq: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Lỗi gọi API: {e}")