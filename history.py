import history_db
import streamlit as st

def build_context_for_llm():
    """Builds the exact message list required by Groq API."""
    if "current_node_id" not in st.session_state or not st.session_state.current_node_id:
        return []
        
    # Get the clean, unbranched path from the DB
    raw_branch = history_db.get_message_branch(st.session_state.current_node_id)
    
    # Format for the LLM
    formatted_context = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in raw_branch
    ]
    return formatted_context

def handle_user_message(prompt):
    """Processes a new message and updates the state pointer."""
    chat_id = st.session_state.current_chat_id
    parent_id = st.session_state.get("current_node_id", None)
    
    # 1. Save user message to DB and move the pointer to this new node
    new_user_node_id = history_db.add_message(chat_id, parent_id, "user", prompt)
    st.session_state.current_node_id = new_user_node_id
    
    # 2. (Here you would call Qdrant and Groq using build_context_for_llm)
    
    # 3. Save Assistant response to DB and move the pointer again
    # new_assistant_node_id = history_db.add_message(chat_id, new_user_node_id, "assistant", ai_response)
    # st.session_state.current_node_id = new_assistant_node_id