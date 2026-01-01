# retrieval/pinecone_client.py
from typing import List, Dict, Any, Optional, Set
from pinecone import Pinecone
import sys
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from utils.llm_client import get_embedding
from utils.logger import get_logger
from agent.schemas import Chunk

# --- Smart Retrieval Imports ---
try:
    from retrieval.query_refiner import refine_query
    from retrieval.context_compressor import compress_context
    SMART_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Smart components import failed: {e}")
    SMART_COMPONENTS_AVAILABLE = False

logger = get_logger("PINECONE_CLIENT")

# --- Initialization ---
pc: Optional[Pinecone] = None
index: Any = None

try:
    if settings.PINECONE_API_KEY and settings.PINECONE_INDEX_NAME:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
    else:
        logger.critical("Pinecone API Key or Index Name missing in settings.")
except Exception as e:
    logger.critical(f"Pinecone initialization failed: {e}")

# --- Core Functions ---

def retrieve(
    query: str, 
    filters: Optional[Dict[str, Any]] = None, 
    namespace: str = "", 
    top_k: int = settings.RAG_TOP_K
) -> List[Chunk]:
    """
    Performs a raw semantic vector search on the Pinecone index.
    Returns: List[Chunk]
    """
    if not index:
        logger.error("Pinecone index is not initialized. Returning empty results.")
        return []

    if not query:
        logger.warning("Empty query provided to retrieve().")
        return []

    # 1. Generate Embedding
    query_vector = get_embedding(query)
    if not query_vector:
        logger.error("Failed to generate embedding for retrieval query.")
        return []
    
    # 2. Query Pinecone
    try:
        search_args = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if filters:
            search_args["filter"] = filters
        
        if namespace:
            search_args["namespace"] = namespace

        results = index.query(**search_args)
        
        chunks = []
        for match in results.matches:
            meta = match.metadata if match.metadata else {}
            # Ensure text exists to avoid processing empty records
            text_content = meta.get("text", "")
            if not text_content:
                continue
                
            chunks.append(
                Chunk(
                    id=match.id,
                    text=text_content,
                    score=match.score if match.score else 0.0,
                    metadata=meta
                )
            )
        
        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:30]}...'")
        return chunks

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

def smart_retrieve(query: str, chat_history: str = "None") -> List[Chunk]:
    """
    Orchestrates Retrieval.
    1. Always includes the RAW query to ensure basic keyword/semantic matching works.
    2. Refines query using chat history.
    3. Fetches chunks.
    4. Compresses chunks based on the USER QUERY before returning.
    """
    # 1. Fallback if smart components are missing
    if not SMART_COMPONENTS_AVAILABLE:
        return retrieve(query)

    try:
        # 2. Prepare Query List
        # We start with the raw query to guarantee at least basic retrieval performance
        queries = [query]
        
        # Attempt refinement
        try:
            refined = refine_query(query, chat_history)
            if refined and isinstance(refined, list):
                queries.extend(refined)
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Proceeding with raw query only.")

        # 3. Execution (Multi-Query)
        all_chunks: List[Chunk] = []
        seen_ids: Set[str] = set()
        
        logger.info(f"Executing smart retrieval with {len(queries)} query variations.")
        
        for q in queries:
            # We use a slightly lower top_k for variations to keep context focused
            results = retrieve(q, top_k=3) 
            for chunk in results:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)
        
        logger.info(f"Smart Retrieve found {len(all_chunks)} unique chunks before compression.")
        
        # 4. Context Compression & Deduplication (Query Aware)
        # We pass the original user query to ensure compression preserves relevant info
        if all_chunks:
            return compress_context(all_chunks, query)
            
        return all_chunks

    except Exception as e:
        logger.error(f"Smart retrieval critical failure: {e}. Fallback to raw.")
        return retrieve(query)