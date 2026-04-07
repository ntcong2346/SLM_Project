import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timezone

# --- CONNECTION MANAGER ---
@st.cache_resource
def get_db():
    """Establish connection to MongoDB and return the database object."""
    client = MongoClient(st.secrets["MONGO_URI"])
    return client["bioslm"] # Tên database là 'bioslm'

def init_db():
    """
    MongoDB is schema-less, so we don't need to CREATE TABLEs.
    However, we can create indexes here to make searching faster.
    """
    db = get_db()
    db.messages.create_index("chat_id")
    db.messages.create_index("parent_id")
    db.chats.create_index("created_at")

# --- CHAT CRUD ---
def create_chat(title="New Conversation"):
    """Creates a new chat session and returns the chat_id as a string."""
    db = get_db()
    result = db.chats.insert_one({
        "title": title,
        "created_at": datetime.now(timezone.utc)
    })
    return str(result.inserted_id)

def get_all_chats():
    """Retrieves all chat sessions for the sidebar UI."""
    db = get_db()
    chats = list(db.chats.find().sort("created_at", -1))
    
    # Đổi _id thành chat_id dạng string để UI dễ dùng
    for chat in chats:
        chat["chat_id"] = str(chat.pop("_id"))
    return chats

def get_chat(chat_id):
    """Fetches details of a specific chat session."""
    if not chat_id: return None
    db = get_db()
    chat = db.chats.find_one({"_id": ObjectId(chat_id)})
    if chat:
        chat["chat_id"] = str(chat.pop("_id"))
    return chat

def delete_chat(chat_id):
    """Deletes a chat and all its associated messages."""
    db = get_db()
    # Xóa chat
    db.chats.delete_one({"_id": ObjectId(chat_id)})
    # Xóa tất cả tin nhắn thuộc chat_id này (Thay thế cho CASCADE trong SQL)
    db.messages.delete_many({"chat_id": chat_id})

# --- MESSAGE CRUD & TREE LOGIC ---
def add_message(chat_id, parent_id, role, content):
    """Adds a single message to the tree and returns its message_id."""
    db = get_db()
    result = db.messages.insert_one({
        "chat_id": chat_id,
        "parent_id": parent_id, # Chứa string ID của tin nhắn trước
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc)
    })
    return str(result.inserted_id)

def get_latest_node_for_chat(chat_id):
    """Finds the most recent message in a chat to use as the starting point."""
    db = get_db()
    latest_msg = db.messages.find_one(
        {"chat_id": chat_id}, 
        sort=[("created_at", -1)]
    )
    return str(latest_msg["_id"]) if latest_msg else None

def get_message_branch(leaf_message_id):
    """
    Traces backwards from the leaf node up to the root to build the exact context.
    """
    if not leaf_message_id: return []
    
    db = get_db()
    branch = []
    current_id = leaf_message_id
    
    # Truy xuất ngược từ lá lên rễ
    while current_id:
        msg = db.messages.find_one({"_id": ObjectId(current_id)})
        if not msg:
            break
            
        # Chuẩn hóa data cho UI
        msg["message_id"] = str(msg.pop("_id"))
        branch.append(msg)
        current_id = msg.get("parent_id")
        
    # Đảo ngược list để có thứ tự từ rễ -> lá (theo thời gian)
    branch.reverse()
    return branch

# --- FORKING LOGIC ---
def fork_chat(original_chat_id, fork_node_id, new_title="Forked Conversation"):
    """
    Creates a new chat and clones the exact history path up to the fork_node_id.
    """
    # 1. Lấy nhánh gốc cần copy
    branch_to_clone = get_message_branch(fork_node_id)
    if not branch_to_clone:
        raise ValueError("Invalid fork_node_id. Message not found.")

    # 2. Tạo Chat mới
    new_chat_id = create_chat(title=new_title)
    
    # 3. Rebuild nhánh trong Chat mới
    db = get_db()
    id_mapping = {None: None}
    last_inserted_id = None
    
    for msg in branch_to_clone:
        old_id = msg['message_id']
        old_parent = msg['parent_id']
        new_parent_id = id_mapping.get(old_parent)
        
        # Insert tin nhắn mới
        result = db.messages.insert_one({
            "chat_id": new_chat_id,
            "parent_id": new_parent_id,
            "role": msg['role'],
            "content": msg['content'],
            "created_at": msg['created_at'] # Giữ nguyên timestamp cũ
        })
        
        new_id = str(result.inserted_id)
        id_mapping[old_id] = new_id
        last_inserted_id = new_id
        
    return new_chat_id, last_inserted_id