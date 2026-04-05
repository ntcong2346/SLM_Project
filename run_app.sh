#!/bin/bash
echo "Đang khởi động hệ thống Bio-SLM..."
streamlit run ui.py --server.port 8501 &
streamlit run embedding_ui.py --server.port 8502 &
wait