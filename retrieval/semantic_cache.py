# retrieval/semantic_cache.py
from typing import Optional, Dict, Any, Tuple
import time
import sys
import json
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from retrieval.pinecone_client import index
# Use RAG engine's embedding logic to ensure compatibility
try:
    from rag_engine.src.embeddings.embedding_provider import embed_query
except ImportError:
    # Fallback or error if RAG not available
    def embed_query(text): return []

from utils.logger import get_logger
from agent.schemas import PlanSchema, ThinkerOutput, VerificationReport

logger = get_logger("SEMANTIC_CACHE")

CACHE_NAMESPACE = "semantic_cache"
CACHE_THRESHOLD = 0.98  # High threshold for exact semantic matches

def clear_cache() -> bool:
    """Wipes the cache (use when docs are updated)."""
    if not index: return False
    try:
        index.delete(delete_all=True, namespace=CACHE_NAMESPACE)
        logger.info("Semantic cache cleared.")
        return True
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        return False

def store_cache(
    query: str, 
    plan: PlanSchema,
    thinker_output: ThinkerOutput,
    verification_report: VerificationReport
) -> None:
    """
    Stores reasoning artifacts (Plan, Thinker, Verifier) instead of final text.
    This allows the Explainer to re-render the response based on dynamic User Profile settings.
    """
    if not query or not index: 
        return

    try:
        # Use RAG embedding
        vector = embed_query(query)
        if not vector: return

        # Serialize artifacts to JSON strings (Pinecone metadata must be flat)
        metadata = {
            "query": query,
            "plan_json": plan.model_dump_json(),
            "thinker_output_json": thinker_output.model_dump_json(),
            "verification_report_json": verification_report.model_dump_json(),
            "timestamp": time.time()
        }

        # ID is hash of query ONLY. We do not stamp profile data anymore.
        import hashlib
        id_str = f"{query}"
        unique_id = hashlib.md5(id_str.encode()).hexdigest()

        index.upsert(
            vectors=[(unique_id, vector, metadata)],
            namespace=CACHE_NAMESPACE
        )
        logger.debug(f"Stored artifacts in cache: {unique_id[:8]}...")

    except Exception as e:
        logger.error(f"Cache store failed: {e}")

def retrieve_cache(query: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves cached reasoning artifacts if a semantic match is found.
    Returns a dictionary containing the reconstructed Pydantic models.
    """
    if not index or not query: return None

    try:
        vector = embed_query(query)
        if not vector: return

        # Query Pinecone without profile filters
        results = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            namespace=CACHE_NAMESPACE
        )

        matches = results.get('matches', [])
        if not matches:
            return None

        match = matches[0]
        if match.score >= CACHE_THRESHOLD:
            meta = match.metadata
            logger.info(f"CACHE HIT ({match.score:.4f}) | Retrieving Artifacts...")
            
            try:
                # Reconstruct Pydantic models from JSON strings
                return {
                    "plan": PlanSchema.model_validate_json(meta["plan_json"]),
                    "thinker_output": ThinkerOutput.model_validate_json(meta["thinker_output_json"]),
                    "verification_report": VerificationReport.model_validate_json(meta["verification_report_json"])
                }
            except Exception as parse_err:
                logger.error(f"Failed to deserialize cached artifacts: {parse_err}")
                return None
        
        return None

    except Exception as e:
        logger.error(f"Cache retrieval failed: {e}")
        return None