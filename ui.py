import streamlit as st
import requests
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CẤU HÌNH TRANG CHUYÊN NGHIỆP ---
st.set_page_config(
    page_title="Bio-SLM AI Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ĐỂ LÀM ĐẸP UI ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: 1px solid #30363d; }
    .stSidebar { background-color: #161b22; border-right: 1px solid #30363d; }
    h1 { color: #58a6ff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .status-box { padding: 10px; border-radius: 10px; border: 1px solid #30363d; background-color: #0d1117; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- TIÊU ĐỀ VÀ GIỚI THIỆU ---
col1, col2 = st.columns([1, 4])

with col2:
    st.title("Bio-SLM AI Assistant")
    st.markdown("*Hệ thống RAG hỗ trợ học tập Sinh học 12 dựa trên mô hình ngôn ngữ nhỏ (SLM)*")

st.divider()

# --- HÀM KHỞI TẠO RAG ---
@st.cache_resource
def init_knowledge_base():
    data_path = "./data"
    if not os.path.exists(data_path) or not os.listdir(data_path):
        return None
    
    documents = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
    
    if not documents: return None

    # Chia nhỏ văn bản thành các đoạn tri thức chuẩn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    return vector_db

# --- SIDEBAR: QUẢN LÝ HỆ THỐNG ---
with st.sidebar:
    st.header("⚙️ Cấu hình SLM")
    
    # Trạng thái RAG
    with st.container():
        try:
            vector_db = init_knowledge_base()
            if vector_db:
                st.success("● Kho kiến thức RAG: Sẵn sàng")
            else:
                st.warning("● Kho kiến thức RAG: Trống (data/)")
        except Exception as e:
            st.error(f"● Lỗi RAG: {e}")

    # Thông số hiệu năng SLM
    st.markdown("---")
    st.subheader("Thông số SLM")
    st.write(f"**Model:** Llama-3.1-8B-Instant")
    st.write(f"**Kiến trúc:** SLM (Small Language Model)")
    st.write(f"**Optimization:** Groq LPU Inference")
    
    # Nguồn trích dẫn
    st.markdown("---")
    st.subheader("Nguồn kiến thức RAG")
    source_container = st.empty()

# --- KHUNG CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi là trợ lý SLM đã được nạp kiến thức Sinh học 12. Bạn cần tìm hiểu về chủ đề nào?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi tôi về Di truyền, Tiến hóa, Sinh thái..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔄 SLM đang trích xuất tri thức..."):
            context = ""
            sources = []
            
            # 1. Tìm kiếm RAG từ file tri thức
            if vector_db:
                docs = vector_db.similarity_search(prompt, k=2)
                context = "\n\n".join([d.page_content for d in docs])
                sources = [d.page_content[:200] + "..." for d in docs]

            if sources:
                with source_container.container():
                    for i, s in enumerate(sources):
                        st.caption(f"Đoạn trích {i+1}:")
                        st.info(s)

            # 2. Gọi SLM qua Groq API
            try:
                api_key = st.secrets["GROQ_API_KEY"] 
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Cấu hình tham số chuẩn SLM
                data = {
                    "model": "llama-3.1-8b-instant", 
                    "messages": [
                        {
                            "role": "system", 
                            "content": f"Bạn là chuyên gia Sinh học 12 dạng SLM. Hãy trả lời ngắn gọn dựa trên tri thức: {context}"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4
                }
                
                response = requests.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    res_text = response.json()['choices'][0]['message']['content']
                    st.markdown(res_text)
                    st.session_state.messages.append({"role": "assistant", "content": res_text})
                else:
                    st.error(f"Lỗi API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Lỗi kết nối SLM Cloud: {e}")