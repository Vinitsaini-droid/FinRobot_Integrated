import streamlit as st
import sys
import time
from pathlib import Path
from typing import Literal

# --- Path Setup ---
# Ensure the project root is in sys.path so we can import modules
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# --- Internal Imports ---
from agent.meta_agent import MetaAgent
from memory.user_profile_store import UserProfileStore
from memory.memory_manager import MemoryManager
from agent.schemas import UserProfileSchema

# --- UI Configuration ---
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization & Caching ---

@st.cache_resource
def load_components():
    """
    Initialize the MetaAgent, Store, and MemoryManager only once to prevent 
    reloading heavy models on every interaction.
    """
    return MetaAgent(), UserProfileStore(), MemoryManager()

try:
    agent, profile_store, memory_manager = load_components()
except Exception as e:
    st.error(f"System Initialization Failed: {e}")
    st.stop()

# --- Session State Management ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "is_new_user" not in st.session_state:
    st.session_state.is_new_user = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---

def login_user(user_id: str, is_new_declared: bool):
    """
    Validates the user ID against the Profile Store.
    """
    if not user_id:
        st.error("Please enter a User ID.")
        return

    status = profile_store.check_user_status(user_id)
    
    # Logic: Mismatch Handling
    if is_new_declared and status == "old":
        st.error(f"❌ User ID '{user_id}' is already taken. Please choose a new ID.")
        return
    
    if not is_new_declared and status == "new":
        st.error(f"❌ Profile not found for '{user_id}'. Please select 'New User' to create an account.")
        return

    # If checks pass
    st.session_state.user_id = user_id
    
    if is_new_declared:
        # Proceed to Profile Setup
        st.session_state.is_new_user = True 
        # Don't set authenticated yet; wait for profile setup
    else:
        # Existing user, go straight to chat
        st.session_state.is_new_user = False
        st.session_state.authenticated = True
        st.rerun()

def save_profile(risk: str, depth: str, style: str):
    """
    Creates and saves the initial profile for a new user.
    """
    user_id = st.session_state.user_id
    
    # Create Schema Object
    new_profile = UserProfileSchema(
        user_id=user_id,
        risk_tolerance=risk,
        explanation_depth=depth,
        style_preference=style
    )
    
    # Persist to Pinecone via Store
    try:
        profile_store.update_profile(new_profile)
        st.success("Profile created successfully!")
        time.sleep(1)
        st.session_state.authenticated = True
        st.rerun()
    except Exception as e:
        st.error(f"Failed to save profile: {e}")

def chat_interface():
    """
    Main Chat Loop mimicking main_agent.py runtime.
    """
    st.sidebar.header(f"User: {st.session_state.user_id}")
    
    # --- Sidebar Actions ---
    
    # 1. LOGOUT with Memory Consolidation
    if st.sidebar.button("Logout"):
        with st.sidebar:
            with st.spinner("Saving session memories..."):
                user_id = st.session_state.user_id
                
                # Retrieve the active context from the AGENT'S memory instance
                # (The agent instance holds the RAM state for the current chat)
                raw_history = agent.memory._active_context.get(user_id, [])
                
                if raw_history:
                    # Format for Summarizer
                    conversation_log = []
                    for turn in raw_history:
                        role = "User" if turn['role'] == 'user' else "Agent"
                        conversation_log.append(f"{role}: {turn['content']}")
                    
                    try:
                        # Consolidate to Long-Term Facts (Pinecone)
                        agent.memory.consolidate_session(user_id, conversation_log)
                        # Clear Backend Context
                        agent.memory.clear_chat_history(user_id)
                        st.success("Memories saved.")
                    except Exception as e:
                        st.error(f"Failed to save memories: {e}")
                        time.sleep(2) # Give user time to see error
                
        # Clear Frontend State & Restart
        st.session_state.clear()
        st.rerun()

    # 2. DELETE PROFILE
    if st.sidebar.button("Delete Profile", type="primary"):
        with st.sidebar:
            with st.spinner("Deleting profile and memories..."):
                try:
                    success = memory_manager.reset_memory(st.session_state.user_id)
                    if success:
                        st.success("Profile deleted successfully.")
                        time.sleep(1)
                        st.session_state.clear()
                        st.rerun()
                    else:
                        st.error("Could not delete all data.")
                except Exception as e:
                    st.error(f"Error during deletion: {e}")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Handling
    if prompt := st.chat_input("How can I help you today?"):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Agent Generation
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Agent is reasoning..."):
                try:
                    # Call MetaAgent
                    response = agent.generate_response(
                        user_id=st.session_state.user_id, 
                        query=prompt
                    )
                    message_placeholder.markdown(response)
                    
                    # Store Response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Agent Error: {e}")

# --- Main App Logic ---

def main():
    if not st.session_state.authenticated:
        # === LOGIN SCREEN ===
        st.title("🤖 Agentic RAG System")
        st.markdown("### Authentication")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            user_input = st.text_input("Enter User ID", placeholder="e.g., alice_01")
            user_type = st.radio("Are you a new or existing user?", ["New User", "Existing User"])
            is_new = (user_type == "New User")
            
            if st.button("Start Session"):
                login_user(user_input.strip(), is_new)

        # === PROFILE SETUP (Conditional) ===
        if st.session_state.get("is_new_user") and not st.session_state.authenticated:
            st.divider()
            st.markdown("### 🛠️ Profile Setup")
            st.info("Please configure your AI assistant preferences before continuing.")
            
            with st.form("profile_form"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    risk = st.selectbox(
                        "Risk Tolerance", 
                        options=['low', 'medium', 'high'], 
                        index=1
                    )
                with c2:
                    depth = st.selectbox(
                        "Explanation Depth", 
                        options=['simple', 'detailed', 'technical'], 
                        index=1
                    )
                with c3:
                    style = st.selectbox(
                        "Style Preference", 
                        options=['formal', 'casual', 'concise'], 
                        index=0
                    )
                
                submitted = st.form_submit_button("Save & Start Chat")
                if submitted:
                    save_profile(risk, depth, style)

    else:
        # === CHAT SCREEN ===
        chat_interface()

if __name__ == "__main__":
    main()
