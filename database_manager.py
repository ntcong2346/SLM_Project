import streamlit as st
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.embeddings import JinaEmbeddings

# --- CẤU HÌNH CƠ BẢN ---
COLLECTION_NAME = "bio_col"  # Đã đổi tên collection thành bio_col
QDRANT_URL = "http://localhost:6333"

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    try:
        return JinaEmbeddings(
            jina_api_key=st.secrets["JINA_API_KEY"], 
            model_name="jina-embeddings-v3"
        )
    except KeyError:
        st.error("Missing 'JINA_API_KEY' in secrets.")
        st.stop()

@st.cache_resource(show_spinner=False)
def get_sparse_embeddings():
    return FastEmbedSparse(model_name="Qdrant/bm25")

def init_vector_store():
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode="hybrid"
    )

def get_all_sources():
    client = get_qdrant_client()
    try:
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            return []
        result = client.scroll(collection_name=COLLECTION_NAME, limit=10000, with_payload=True)
        points = result[0]
        sources = set()
        for p in points:
            if p.payload and "metadata" in p.payload:
                s = p.payload["metadata"].get("source")
                if s: sources.add(s)
            elif p.payload and "source" in p.payload:
                sources.add(p.payload.get("source"))
        return sorted(list(sources))
    except Exception:
        return []

def delete_source_from_db(source_name):
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                should=[
                    models.FieldCondition(key="metadata.source", match=models.MatchValue(value=source_name)),
                    models.FieldCondition(key="source", match=models.MatchValue(value=source_name))
                ]
            )
        )
        return True
    except Exception:
        return False