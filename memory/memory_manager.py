# memory/memory_manager.py
import time
import uuid
import sys
import json
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from agent.schemas import MemoryItem, MemoryType, UserProfileSchema
from utils.llm_client import get_embedding
from utils.json_utils import safe_json_load 
from utils.logger import get_logger
from memory.chat_summarizer import ChatSummarizer
from memory.user_profile_store import UserProfileStore
from retrieval.semantic_cache import clear_cache  

# --- Local DB Imports ---
try:
    from tinydb import TinyDB, Query
    import faiss
except ImportError:
    sys.exit("Critical: tinydb or faiss-cpu not installed. Run 'pip install tinydb faiss-cpu'")

logger = get_logger("MEMORY_MANAGER")

class MemoryManager:
    """
    Hybrid Memory System (Ported to TinyDB + FAISS):
    1. Short-Term (RAM): Linear context.
    2. Long-Term (TinyDB+FAISS): Episodic archival & Semantic Facts.
    3. Profile (TinyDB): User personality.
    """
    
    def __init__(self):
        self.summarizer = ChatSummarizer()
        self.profile_store = UserProfileStore()
        
        # Short-Term Linear Buffer: {user_id: [{"role": "user", "content": "..."}, ...]}
        self._active_context: Dict[str, List[Dict[str, str]]] = {}

        # --- Local Storage Setup ---
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize TinyDB for Metadata
        self.db_path = self.data_dir / "memory.json"
        self.db = TinyDB(str(self.db_path))
        self.MemQuery = Query()
        
        # 2. Initialize FAISS for Vectors (Using IndexIDMap for deletion support)
        # Using IndexFlatIP for Inner Product (Cosine Similarity if normalized)
        self.dimension =  getattr(settings, "EMBEDDING_DIMENSION", 768) 
        self.index_path = self.data_dir / "memory.index"
        
        if self.index_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.index_path))
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Creating new.")
                self._create_new_index()
        else:
            self._create_new_index()

        logger.info(f"MemoryManager initialized. DB: {self.db_path}")

    def _create_new_index(self):
        """Creates a new FAISS index with ID mapping support."""
        base_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index = faiss.IndexIDMap(base_index)

    def _save_faiss(self):
        """Helper to persist FAISS index to disk."""
        try:
            faiss.write_index(self.faiss_index, str(self.index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def get_profile(self, user_id: str) -> UserProfileSchema:
        """
        Retrieves the user profile. 
        If it does not exist (New User), creates a DEFAULT profile, 
        saves it to the DB, and returns it.
        """
        # FIX: Access the correctly named instance variable 'self.profile_store'
        profile = self.profile_store.get_profile(user_id)
        
        if profile:
            return profile
            
        # --- Auto-Create Default Profile for New Users ---
        logger.info(f"No profile found for {user_id}. Creating default profile.")
        
        default_profile = UserProfileSchema(
            user_id=user_id,
            explanation_depth='detailed',
            risk_tolerance="medium",
            style_preference= 'formal',
        )
        
        # Save it immediately so it persists
        # FIX: Access the correctly named instance variable 'self.profile_store'
        self.profile_store.update_profile(default_profile)
        
        return default_profile

    def get_immediate_context(self, user_id: str, window_size: int = 4) -> str:
        history = self._active_context.get(user_id, [])
        if not history:
            return ""
        
        recent_turns = history[-window_size:]
        formatted_context = []
        for msg in recent_turns:
            role = "User" if msg['role'] == "user" else "Agent"
            formatted_context.append(f"{role}: {msg['content']}")
            
        return "\n".join(formatted_context)

    def _delete_by_filter(self, filter_lambda):
        """Helper to emulate Pinecone delete-by-filter using TinyDB + FAISS."""
        try:
            # 1. Find matching records in TinyDB
            results = self.db.search(filter_lambda)
            if not results:
                return

            ids_to_delete = [r.doc_id for r in results]
            
            # 2. Remove from FAISS
            if ids_to_delete:
                ids_np = np.array(ids_to_delete, dtype='int64')
                self.faiss_index.remove_ids(ids_np)
                self._save_faiss()

            # 3. Remove from TinyDB
            self.db.remove(doc_ids=ids_to_delete)
            
        except Exception as e:
            logger.error(f"Delete by filter error: {e}")

    def clear_chat_history(self, user_id: str) -> bool:
        if user_id in self._active_context:
            del self._active_context[user_id]

        try:
            # Strictly scoped deletion: user_id AND type=episodic
            self._delete_by_filter(
                (self.MemQuery.user_id == user_id) & (self.MemQuery.type == "episodic")
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear episodic history: {e}")
            return False

    def reset_memory(self, user_id: str) -> bool:
        success = True
        if user_id in self._active_context:
            del self._active_context[user_id]

        try:
            # Strictly scoped deletion: user_id only (delete all types)
            self._delete_by_filter(self.MemQuery.user_id == user_id)
            logger.warning(f"Full memory reset performed for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to reset memory vectors: {e}")
            success = False

        # 2. Delete Profile from Persistence
        self.profile_store.delete_profile(user_id)
        
        return success

    def add_memory(self, content: str, memory_type: MemoryType, metadata: dict = None) -> Optional[str]:
        """
        Stores memory in TinyDB + FAISS.
        ENFORCES user isolation: Checks if user_id is missing for user-specific memory types.
        """
        if not content: return None
        
        safe_meta = metadata or {}
        
        # --- Strict Isolation Check ---
        if memory_type in [MemoryType.FACT, MemoryType.EPISODIC]:
            if "user_id" not in safe_meta:
                logger.error(f"Attempted to store {memory_type} without 'user_id'. Aborting.")
                return None

        try:
            vector = get_embedding(content)
            if not vector: return None
            
            # 1. Prepare Metadata for TinyDB
            mem_uuid = str(uuid.uuid4())
            record = {
                "uuid": mem_uuid, # Keep original UUID in metadata for reference
                "text": content,
                "type": memory_type.value,
                "timestamp": str(time.time()),
                "attributes_json": json.dumps(safe_meta)
            }
            
            # Lift user_id to top-level for querying
            if "user_id" in safe_meta:
                record["user_id"] = safe_meta["user_id"]
            
            # 2. Insert into TinyDB to get Integer ID (doc_id)
            doc_id = self.db.insert(record)
            
            # 3. Add to FAISS using doc_id
            vector_np = np.array([vector], dtype='float32')
            ids_np = np.array([doc_id], dtype='int64')
            
            self.faiss_index.add_with_ids(vector_np, ids_np)
            self._save_faiss()
            
            return mem_uuid # Return the UUID as per original signature
            
        except Exception as e:
            logger.error(f"Add memory failed: {e}")
            return None

    def retrieve_relevant(self, query: str, user_id: str, limit: int = 5, memory_type: Optional[MemoryType] = None, score_threshold: float = 0.70) -> List[MemoryItem]:
        """
        Semantic Retrieval with STRICT user_id filtering.
        """
        if not query: return []
        try:
            query_vector = get_embedding(query)
            if not query_vector: return []

            # 1. FAISS Search
            # We fetch more candidates because we post-filter
            search_k = limit * 10 
            query_np = np.array([query_vector], dtype='float32')
            
            scores, indices = self.faiss_index.search(query_np, search_k)
            
            # Flatten results
            scores = scores[0]
            indices = indices[0]

            memories = []
            
            for score, doc_id in zip(scores, indices):
                if doc_id == -1: continue # FAISS padding
                if score < score_threshold: continue

                # 2. Fetch Metadata from TinyDB
                # doc_id is int64, TinyDB expects int
                record = self.db.get(doc_id=int(doc_id))
                if not record: continue
                
                # 3. Apply Filters (User ID & Type)
                if record.get("user_id") != user_id:
                    continue
                
                if memory_type and record.get("type") != memory_type.value:
                    continue

                # Parse Attributes
                attr = safe_json_load(record.get("attributes_json", "{}")) or {}

                memories.append(MemoryItem(
                    id=record.get("uuid", str(doc_id)), # Use UUID if present, else ID
                    content=record.get("text", ""),
                    memory_type=MemoryType(record.get("type", "fact")),
                    timestamp=record.get("timestamp", ""),
                    metadata=attr
                ))
                
                if len(memories) >= limit:
                    break
            
            return memories

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def check_and_update_profile_pre_planning(self, user_id: str, user_query: str):
        if not user_query: return 

        try:
            current_profile = self.profile_store.get_profile(user_id)
            original_dump = current_profile.model_dump()

            updates = self.summarizer.analyze_interaction_delta(
                current_profile=current_profile, 
                last_user_msg=user_query, 
                last_agent_msg="[SYSTEM: PRE-RESPONSE CHECK]"
            )

            if updates:
                updated_data = original_dump.copy()
                updated_data.update(updates)
                new_profile = UserProfileSchema(**updated_data)
                
                changed = self.profile_store.sync_if_changed(current_profile, new_profile)
                if changed:
                    logger.info(f"Profile updated based on pre-planning check for {user_id}")

        except Exception as e:
            logger.warning(f"Pre-planning profile check failed: {e}")

    def process_realtime_interaction(self, user_id: str, user_msg: str, agent_msg: str):
        if not user_msg or not agent_msg: return

        # 1. Update Short-Term Context
        if user_id not in self._active_context:
            self._active_context[user_id] = []
        
        self._active_context[user_id].append({"role": "user", "content": user_msg})
        self._active_context[user_id].append({"role": "agent", "content": agent_msg})
        
        if len(self._active_context[user_id]) > 20:
             self._active_context[user_id] = self._active_context[user_id][-20:]

        # 2. Analyze Profile Delta
        try:
            current_profile = self.profile_store.get_profile(user_id)
            if current_profile:
                original_dump = current_profile.model_dump()

                updates = self.summarizer.analyze_interaction_delta(
                    current_profile=current_profile, 
                    last_user_msg=user_msg, 
                    last_agent_msg=agent_msg
                )

                if updates:
                    updated_data = original_dump.copy()
                    updated_data.update(updates)
                    new_profile = UserProfileSchema(**updated_data)
                    self.profile_store.sync_if_changed(current_profile, new_profile)

            # 3. Episodic Archival
            if len(user_msg) + len(agent_msg) > 50:
                self.add_memory(
                    content=f"User: {user_msg}\nAgent: {agent_msg}",
                    memory_type=MemoryType.EPISODIC,
                    metadata={"user_id": user_id}
                )

        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")

    def consolidate_session(self, user_id: str, conversation_history: List[str]):
        """End-of-session bulk summarization with STRICT 25-FACT LIMIT and USER SCOPING."""
        if not conversation_history: return
        
        try:
            # 1. Extract raw facts
            chat_data = self.summarizer.summarize(conversation_history)
            if not chat_data.key_facts: return

            # 2. Fetch existing facts (Strictly scoped to user_id)
            existing_memories = self.retrieve_relevant(
                query="general user facts", 
                user_id=user_id, 
                limit=50, 
                memory_type=MemoryType.FACT,
                score_threshold=0.0 # Fetch ALL candidates
            )
            existing_texts = [m.content for m in existing_memories]

            # 3. Smart Deduplication
            final_facts = self.summarizer.deduplicate_facts(existing_texts, chat_data.key_facts)

            # 4. Replace Strategy: Delete OLD facts (Scoped to User), Insert FINAL set
            # CRITICAL: Scope deletion to this user only
            self._delete_by_filter(
                (self.MemQuery.user_id == user_id) & (self.MemQuery.type == "fact")
            )
            
            for fact in final_facts:
                # add_memory now enforces user_id presence
                self.add_memory(fact, MemoryType.FACT, {"user_id": user_id})
            
            logger.info(f"Consolidated facts for {user_id}. Final count: {len(final_facts)}")

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")