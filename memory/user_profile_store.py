# memory/user_profile_store.py
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import UserProfileSchema
from config.settings import settings
from utils.json_utils import safe_json_load
from utils.logger import get_logger

# --- Local DB Imports ---
try:
    from tinydb import TinyDB, Query
    import faiss
except ImportError:
    sys.exit("Critical: tinydb or faiss-cpu not installed. Run 'pip install tinydb faiss-cpu'")

logger = get_logger("USER_PROFILE")

class UserProfileStore:
    """
    Manages persistence of the UserProfileSchema using TinyDB (Metadata) and FAISS (Vectors).
    Ported from Pinecone for local execution.
    """
    
    def __init__(self):
        # --- Local Storage Setup ---
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize TinyDB for JSON Metadata
        db_path = self.data_dir / "user_profiles.json"
        self.db = TinyDB(str(db_path))
        self.UserQuery = Query()
        
        # 2. Initialize FAISS for Vectors
        # Assuming standard dimension 1536 (OpenAI) or from settings. 
        # If your embeddings are different, adjust 'd'.
        self.dimension = getattr(settings, "EMBEDDING_DIMENSION", 768)
        self.index_path = self.data_dir / "user_profiles.index"
        
        if self.index_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.index_path))
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Creating new.")
                self.faiss_index = faiss.IndexFlatL2(self.dimension)
        else:
            self.faiss_index = faiss.IndexFlatL2(self.dimension)

        logger.info(f"UserProfileStore initialized. DB: {db_path}")

    def _save_faiss(self):
        """Helper to persist FAISS index to disk."""
        try:
            faiss.write_index(self.faiss_index, str(self.index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def get_profile(self, user_id: str) -> Optional[UserProfileSchema]:
        """
        Retrieves a user profile by ID from TinyDB.
        """
        try:
            # Exact match lookup in TinyDB
            results = self.db.search(self.UserQuery.user_id == user_id)
            
            if not results:
                return None
            
            # TinyDB returns a list of dicts. We take the first match.
            data = results[0]
            
            # Ensure we are extracting the profile_data correctly
            # In update_profile, we store it under 'profile_data' key as a JSON string 
            # or directly as fields depending on the original implementation's intent.
            # Looking at the original 'update_profile', it stored 'profile_data' string in metadata.
            
            if 'profile_data' in data:
                # If stored as JSON string (legacy Pinecone pattern port)
                profile_dict = json.loads(data['profile_data'])
                return UserProfileSchema(**profile_dict)
            else:
                # If stored directly
                return UserProfileSchema(**data)

        except Exception as e:
            logger.error(f"Error fetching profile for {user_id}: {e}")
            return None
    
    def check_user_status(self, user_id: str) -> str:
        """
        Determines if a user is 'new' or 'old' based on profile existence.
        """
        profile = self.get_profile(user_id)
        if profile is None:
            return "new"
        return "old" 

    def update_profile(self, profile: UserProfileSchema) -> None:
        """
        Updates or Creates a user profile.
        Writes metadata to TinyDB and (placeholder) vector to FAISS.
        """
        try:
            # 1. Prepare Data
            # The original code created a placeholder vector. We keep this logic.
            # If you have real embeddings in the profile, access them here.
            placeholder = [0.0] * self.dimension 
            profile_json = profile.model_dump_json()
            
            # 2. TinyDB Upsert
            # We store the user_id at the root for easier querying, 
            # and the full payload in profile_data to match original structure
            record = {
                'user_id': profile.user_id,
                'profile_data': profile_json,
                'type': 'user_profile'
            }
            
            # Upsert: Update if user_id exists, else Insert
            self.db.upsert(record, self.UserQuery.user_id == profile.user_id)
            
            # 3. FAISS Update
            # FAISS doesn't support easy updates/deletes by ID in simple IndexFlatL2.
            # For a Profile Store using placeholders, the vector is often irrelevant 
            # compared to the ID. 
            # However, to strictly follow "Port to FAISS", we add the vector.
            # In a production local setup, we would rebuild or use IDMap. 
            # Here we just append for safety/simplicity as profiles are rarely vector-searched.
            
            vector_np = np.array([placeholder], dtype='float32')
            self.faiss_index.add(vector_np)
            self._save_faiss()

            logger.info(f"Profile updated successfully for {profile.user_id}")

        except Exception as e:
            logger.error(f"Update profile error: {e}")

    def sync_if_changed(self, old_profile: UserProfileSchema, current_profile: UserProfileSchema) -> bool:
        """
        Smart Sync: Only writes if data actually changed.
        """
        # Logic remains exactly the same as original
        if old_profile.model_dump() != current_profile.model_dump():
            logger.info(f"Syncing profile changes for {current_profile.user_id}...")
            self.update_profile(current_profile)
            return True
        return False

    def delete_profile(self, user_id: str) -> None:
        """
        Hard Delete: Removes the user profile from the database.
        """
        try:
            logger.warning(f"DELETING PROFILE for {user_id}...")
            
            # 1. Remove from TinyDB
            deleted_ids = self.db.remove(self.UserQuery.user_id == user_id)
            
            # 2. Remove from FAISS
            # Note: Removing from standard FAISS without IDMap is complex.
            # Since this is a profile store, the TinyDB deletion effectively 
            # "hides" the data. We accept the orphan vector in FAISS 
            # to avoid index corruption risks in this simple port.
            
            if deleted_ids:
                logger.info(f"Profile deleted successfully for {user_id}")
            else:
                logger.warning(f"No profile found to delete for {user_id}")

        except Exception as e:
            logger.error(f"Delete profile error: {e}")

    def _parse_fetch_response(self, response: Any, user_id: str) -> Optional[Dict]:
        """
        Legacy helper kept for compatibility. 
        In TinyDB port, 'response' is the direct dictionary from DB.
        """
        if not response:
            return None
        return response