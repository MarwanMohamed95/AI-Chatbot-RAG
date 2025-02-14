# session_manager.py
from typing import Dict

# In-memory storage for session states
session_store: Dict[str, Dict] = {}

def get_session(session_id: str) -> Dict:
    if session_id not in session_store:
        session_store[session_id] = {}
    return session_store[session_id]
