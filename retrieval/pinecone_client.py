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
from utils.logger import get_logger
from agent.schemas import Chunk

# Import rag_engine components
try:
    from rag_engine.src.inference_plane.reader import InferenceReader, RetrievalResult
    from rag_engine.src.control_plane.manager import ControlPlaneManager, DataChecklist
    RAG_AVAILABLE = True
except ImportError as e:
    logger = get_logger("PINECONE_CLIENT")
    logger.critical(f"RAG Engine import failed: {e}")
    RAG_AVAILABLE = False

# --- Smart Retrieval Imports ---
try:
    from retrieval.query_refiner import refine_query
    from retrieval.context_compressor import compress_context
    SMART_COMPONENTS_AVAILABLE = True
except ImportError as e:
    # logger might not be init if imports fail above
    if 'logger' not in locals(): logger = get_logger("PINECONE_CLIENT")
    logger.warning(f"Smart components import failed: {e}")
    SMART_COMPONENTS_AVAILABLE = False

logger = get_logger("PINECONE_CLIENT")

# Remove direct Pinecone Init as rag_engine handles it
# But Semantic Cache needs an index object?
# The user said "in semantic_cache.py... pinecone will automatically produce embedding".
# The Semantic Cache likely needs to be updated to use the RAG Engine's mechanisms too or a separate index.
# For now, we expose the index from InferenceReader if needed, or initialized purely for Cache.
# Given RAG engine initializes its own index in InferenceReader, we might need a handle to it.
pc: Optional[Pinecone] = None
index: Any = None

try:
    reader = InferenceReader()
    index = reader.index # Expose index for cache if needed
    pc = reader.pc
except Exception as e:
    logger.warning(f"Could not initialize InferenceReader globally: {e}")

def extract_ticker(query: str) -> str:
    """
    Simple heuristic to find a ticker in the query.
    Defaults to AAPL if none found (Temporary for integration).
    """
    # Common tech tickers for demo
    # Ordered by length to match specific longer tickers first
    tickers = ["GOOGL", "MSFT", "AAPL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "TCS", "INFY"]
    query_upper = query.upper()
    
    # Check for Apple specifically as it's common
    if "APPLE" in query_upper:
        return "AAPL"
        
    for t in tickers:
        if t in query_upper:
            return t
            
    # Fallback to AAPL if no known ticker found
    return "AAPL"

def retrieve(
    query: str, 
    filters: Optional[Dict[str, Any]] = None, 
    namespace: str = "", 
    top_k: int = settings.RAG_TOP_K
) -> List[Chunk]:
    """
    Proxies the request to the Financial Analysis RAG Engine.
    """
    if not RAG_AVAILABLE:
        logger.error("RAG Engine not available.")
        return []

    if not query:
        logger.warning("Empty query provided to retrieve().")
        return []

    ticker = extract_ticker(query)
    logger.info(f"Delegating retrieval to RAG Engine for Ticker: {ticker}")

    try:
        # We need to Ensure Data Ready (Control Plane)
        # In a real high-throughput app, we might separate this, but for this Agent:
        manager = ControlPlaneManager()
        # Ensure fresh price (structured) and check unstructured
        checklist = DataChecklist(structured=["price"], unstructured=True)
        
        # We perform the check. This prints to stdout, which is fine.
        manager.ensure_data_ready(ticker, checklist)
        
        # Now Retrieve (Inference Plane)
        # We use a fresh reader to avoid stale state if any
        reader_instance = InferenceReader()
        
        # Call the RAG retrieval
        # Note: filters mapping might be needed if generic filters are passed
        result: RetrievalResult = reader_instance.retrieve(
            query=query,
            ticker=ticker,
            top_k=top_k,
            filter_dict=filters
        )
        
        chunks = []
        for match in result.matches:
            chunks.append(
                Chunk(
                    id=match.id,
                    text=match.text,
                    score=match.score,
                    metadata=match.metadata
                )
            )
        
        logger.info(f"RAG Engine returned {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logger.error(f"RAG Engine failure: {e}")
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
            results = retrieve(q, top_k=3) 
            for chunk in results:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)
        
        logger.info(f"Smart Retrieve found {len(all_chunks)} unique chunks before compression.")
        
        # 4. Context Compression & Deduplication (Query Aware)
        if all_chunks:
            return compress_context(all_chunks, query)
            
        return all_chunks

    except Exception as e:
        logger.error(f"Smart retrieval critical failure: {e}. Fallback to raw.")
        return retrieve(query)