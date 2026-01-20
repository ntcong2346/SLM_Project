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
with col1:
    st.markdown("")
with col2:
    st.title("Bio-SLM AI Assistant")
    st.markdown("*Hệ thống hỗ trợ học tập Sinh học 12 dựa trên mô hình ngôn ngữ nhỏ (SLM) & RAG*")

st.divider()

# --- HÀM KHỞI TẠO RAG (GIỮ NGUYÊN LOGIC NHƯNG THÊM THÔNG BÁO UI) ---
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings)
    return vector_db

# --- SIDEBAR: QUẢN LÝ HỆ THỐNG ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=80)
    st.header(" Cấu hình hệ thống")
    
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

    # Thông số hiệu năng từ Benchmark
    st.markdown("---")
    st.subheader("Hiệu năng thực tế")
    st.write(f"**Model:** Qwen2-1.5B")
    st.write(f"**Nén:** 4-bit GGUF (941MB)")
    st.write(f"**Tốc độ:** 6.18 tokens/s")
    
    # Nguồn trích dẫn (Chỉ hiện khi có kết quả tìm kiếm)
    st.markdown("---")
    st.subheader("Nguồn dữ liệu")
    source_container = st.empty()

# --- KHUNG CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi đã sẵn sàng hỗ trợ bạn ôn tập Sinh học 12. Bạn cần tìm hiểu về chủ đề nào?"}]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý nhập liệu
if prompt := st.chat_input("Hỏi tôi về Di truyền, Tiến hóa, Sinh thái..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất kiến thức..."):
            context = ""
            sources = []
            
            # Tìm kiếm RAG
            if vector_db:
                docs = vector_db.similarity_search(prompt, k=2)
                context = "\n\n".join([d.page_content for d in docs])
                sources = [d.page_content[:150] + "..." for d in docs]

            # Hiển thị nguồn bên sidebar để UI chính sạch sẽ
            if sources:
                with source_container.container():
                    for i, s in enumerate(sources):
                        st.caption(f"Trích đoạn {i+1}:")
                        st.info(s)
            else:
                source_container.write("Không có tài liệu phù hợp.")

          # --- GỌI MÔ HÌNH QUA API (DÀNH CHO DEPLOY CLOUD) ---
            try:
                # 1. Lấy API Key từ Streamlit Secrets (Cần thiết lập trên Cloud)
                # Nếu chạy local để test, bạn có thể thay bằng: api_key = "KEY_CỦA_BẠN"
                api_key = st.secrets["GROQ_API_KEY"] 
                
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # 2. Cấu hình tham số gọi mẫu (Dùng Qwen2 hoặc tương đương)
                data = {
                    "model": "qwen2-72b-4bit", # Groq hỗ trợ các bản nén siêu nhanh
                    "messages": [
                        {
                            "role": "system", 
                            "content": f"Bạn là chuyên gia Sinh học 12. Hãy trả lời dựa trên tri thức: {context}"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5
                }
                
                response = requests.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    res_text = response.json()['choices'][0]['message']['content']
                    st.markdown(res_text)
                    st.session_state.messages.append({"role": "assistant", "content": res_text})
                else:
                    st.error(f"Lỗi API: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"Lỗi kết nối Cloud API: {e}")