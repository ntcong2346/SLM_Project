import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_manager import init_vector_store, get_embeddings, get_sparse_embeddings, get_qdrant_client, COLLECTION_NAME, QDRANT_URL
from langchain_qdrant import QdrantVectorStore

# Kích thước chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def process_uploaded_files(uploaded_files, progress_bar, status_text, force_recreate=False):
    if not uploaded_files:
        return False, "Chưa có file nào được tải lên."

    documents = []
    total_files = len(uploaded_files)

    # BƯỚC 1: ĐỌC DỮ LIỆU TỪ FILE (DATA INGESTION)
    for i, file in enumerate(uploaded_files):
        if file.size == 0: continue
            
        status_text.caption(f"Đang đọc: {file.name}...")
        file_extension = os.path.splitext(file.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file.getbuffer())
            temp_filepath = temp_file.name

        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_filepath)
            elif file.name.endswith(".txt"):
                loader = TextLoader(temp_filepath, encoding="utf-8")
            elif file.name.endswith((".docx", ".doc")):
                loader = Docx2txtLoader(temp_filepath)
            else:
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = file.name
            documents.extend(loaded_docs)
            
        except Exception as e:
            os.remove(temp_filepath)
            return False, f"Lỗi khi đọc {file.name}: {str(e)}"
            
        os.remove(temp_filepath)
        progress_bar.progress((i + 1) / total_files * 0.4)

    if not documents:
        return False, "Không trích xuất được văn bản nào."

    # BƯỚC 2: XỬ LÝ CHUNKING
    status_text.caption("Đang chia nhỏ dữ liệu (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    progress_bar.progress(0.6)

    # BƯỚC 3: EMBEDDING & LƯU VÀO QDRANT (bio_col)
    status_text.caption(f"Đang nhúng (Embedding) {len(chunks)} chunks vào collection '{COLLECTION_NAME}'...")
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]

        # Xóa sạch và tạo mới collection nếu chọn force_recreate hoặc collection chưa tồn tại
        if COLLECTION_NAME not in collections or force_recreate:
            QdrantVectorStore.from_documents(
                chunks,
                embedding=get_embeddings(),
                sparse_embedding=get_sparse_embeddings(),
                retrieval_mode="hybrid",
                url = QDRANT_URL,
                collection_name=COLLECTION_NAME,
                force_recreate=True  # Khởi tạo lại Schema chuẩn cho bio_col
            )
        else:
            # Nếu collection đã tồn tại, chỉ cần thêm dữ liệu mới
            vector_store = init_vector_store()
            vector_store.add_documents(chunks)
            
        progress_bar.progress(1.0)
        return True, f"Thành công! Đã nhúng và lưu {len(chunks)} chunks vào '{COLLECTION_NAME}'."
    except Exception as e:
        return False, f"Lỗi Embedding: {str(e)}"

# --- GIAO DIỆN ---
def main():
    st.set_page_config(page_title="Data Ingestion", layout="centered")
    
    st.title("🧬 Data Ingestion - Bio SLM")
    st.write("Tải lên tài liệu Sinh học (.pdf, .txt, .docx)")

    uploaded_files = st.file_uploader(
        "Upload", 
        type=["txt", "pdf", "docx"], 
        accept_multiple_files=True,
        label_visibility="hidden"
    )

    # Nút check để tạo mới/xóa sạch collection bio_col
    force_recreate = st.checkbox(f" Xóa sạch DB cũ & Tạo lại Collection mới ({COLLECTION_NAME})", value=False)

    if st.button("Start Embedding", use_container_width=True):
        if not uploaded_files:
            st.warning("Vui lòng chọn file trước.")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success, msg = process_uploaded_files(uploaded_files, progress_bar, status_text, force_recreate)
            
        status_text.empty()
        progress_bar.empty()
        
        if success:
            st.success(msg)
        else:
            st.error(msg)

if __name__ == "__main__":
    main()