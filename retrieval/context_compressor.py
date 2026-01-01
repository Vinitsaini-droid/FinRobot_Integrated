# retrieval/context_compressor.py
import sys
import json
from typing import List, Set
from pathlib import Path
from google.genai import types

# --- DYNAMIC PATH RESOLUTION ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# -------------------------------

# Internal Imports
from agent.schemas import Chunk
from utils.llm_client import generate_secondary, get_embedding, generate_schema_critique, repair_json_with_llm
from utils.similarity import cosine_similarity
from utils.json_utils import safe_json_load, validate_json_structure
from config.token_budgets import TOKEN_BUDGETS
from utils.logger import get_logger

logger = get_logger("CONTEXT_COMPRESSOR")

# Constants
MAX_RETRIES = 3
SIMILARITY_THRESHOLD = 0.9  # Chunks >90% similar are considered duplicates

# Updated Prompt to include Query Relevance
COMPRESSION_PROMPT = (
    "You are a Precision Editor for financial data. "
    "Your task is to COMPRESS the provided text based strictly on its relevance to the USER QUERY: '{query}'.\n"
    "Remove conversational filler, redundant adjectives, and information irrelevant to the query.\n\n"
    "CRITICAL PRESERVATION RULES:\n"
    "1. PRESERVE EXACTLY all numbers, percentages, dates, currency values, and proper nouns.\n"
    "2. PRESERVE all conditionality (if/then, unless, except).\n"
    "3. PRESERVE table structures if present.\n"
    "4. DO NOT round numbers or interpret legal clauses.\n"
    "5. Target length: roughly {max_tokens} tokens (but prioritize accuracy over length).\n\n"
    "Output strictly valid JSON with key 'compressed_text' (string)."
)

def deduplicate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Uses Vector Similarity to remove redundant context chunks.
    If Chunk A and Chunk B are >90% similar, we drop the redundant one.
    """
    if not chunks:
        return []

    logger.info(f"Deduplicating {len(chunks)} chunks...")
    unique_chunks = []
    
    # 1. Generate Embeddings for all chunks
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk.text)
        embeddings.append(emb)

    indices_to_drop: Set[int] = set()

    # 2. Compare O(N^2)
    for i in range(len(chunks)):
        if i in indices_to_drop:
            continue
            
        for j in range(i + 1, len(chunks)):
            if j in indices_to_drop:
                continue
            
            sim = cosine_similarity(embeddings[i], embeddings[j])
            
            if sim > SIMILARITY_THRESHOLD:
                logger.debug(f"Duplicate found: Chunk {chunks[i].id} vs {chunks[j].id} (Sim: {sim:.2f})")
                
                # Heuristic: Keep the longer text (more info)
                if len(chunks[i].text) >= len(chunks[j].text):
                    indices_to_drop.add(j)
                else:
                    indices_to_drop.add(i)
                    break 

    # 3. Reconstruct List
    for index, chunk in enumerate(chunks):
        if index not in indices_to_drop:
            unique_chunks.append(chunk)

    logger.info(f"Deduplication complete. Reduced {len(chunks)} -> {len(unique_chunks)} chunks.")
    return unique_chunks

def compress_context(chunks: List[Chunk], query: str) -> List[Chunk]:
    """
    Pipeline:
    1. Deduplicate chunks using cosine similarity.
    2. Perform abstractive summarization on remaining long chunks based on USER QUERY.
    """
    # Step 1: Deduplicate
    clean_chunks = deduplicate_chunks(chunks)
    
    compressed_chunks = []
    max_tokens = TOKEN_BUDGETS.get("COMPRESSION_TARGET_MAX", 500)
    
    # Standard Schema for validation
    compression_json_schema = {
        "type": "object",
        "properties": {
            "compressed_text": {"type": "string"}
        },
        "required": ["compressed_text"]
    }

    # Google Schema for generation
    compression_google_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "compressed_text": types.Schema(type=types.Type.STRING)
        },
        required=["compressed_text"]
    )
    
    # Format the base prompt with the user query once
    base_prompt_template = COMPRESSION_PROMPT.format(max_tokens=max_tokens, query=query)

    # Step 2: Compress if necessary
    for chunk in clean_chunks:
        # Simple heuristic: only compress documents that are likely "long docs"
        # 1 token ~= 4 chars. If text > 1.5x budget, compress.
        if len(chunk.text.split()) < (max_tokens * 1.5):
            compressed_chunks.append(chunk)
            continue

        logger.info(f"Compressing Chunk {chunk.id} (Length: {len(chunk.text)} chars) against query...")
        
        summary_text = None
        current_prompt = base_prompt_template
        
        # Retry Logic with Gatekeeper & Repair
        for attempt in range(MAX_RETRIES):
            prompt_content = f"Original Text:\n{chunk.text}"
            
            raw_response = generate_secondary(
                system_prompt=current_prompt,
                user_prompt=prompt_content,
                max_tokens=max_tokens,
                temperature=0.1,
                response_schema=compression_google_schema
            )
            
            # 1. Standard Validation
            is_valid = validate_json_structure(raw_response, json.dumps(compression_json_schema))
            
            # 2. LLM Repair Fallback
            if not is_valid:
                logger.info(f"Compression attempt {attempt+1} invalid. Trying LLM repair...")
                repaired_response = repair_json_with_llm(json.dumps(compression_json_schema), raw_response)
                
                if repaired_response and validate_json_structure(repaired_response, json.dumps(compression_json_schema)):
                    raw_response = repaired_response
                    is_valid = True
                    logger.info("Compression JSON repaired via LLM.")
            
            # 3. Success Handler
            if is_valid:
                data = safe_json_load(raw_response)
                if data and "compressed_text" in data and data["compressed_text"].strip():
                    summary_text = data["compressed_text"]
                    logger.debug(f"Chunk {chunk.id} compressed successfully on attempt {attempt + 1}.")
                    break
            
            # 4. Critique Fallback
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES}: Compression validation & repair failed. Retrying...")
            
            critique_obj = generate_schema_critique(
                expected_schema_str=json.dumps(compression_json_schema),
                received_output_str=raw_response
            )

            if critique_obj:
                 current_prompt = (
                    f"{base_prompt_template}\n\n"
                    f"### PREVIOUS ATTEMPT FAILED ###\n"
                    f"Critique: {critique_obj.critique}\n"
                    f"Fix Instructions: {critique_obj.suggestions}"
                )
            
        # Final Fallback
        if not summary_text:
            logger.error(f"Compression failed for chunk {chunk.id}. Retaining original.")
            compressed_chunks.append(chunk)
        else:
            # Create a new chunk object: Updates text, preserves metadata/score/id
            compressed_chunk = Chunk(
                id=chunk.id, 
                text=summary_text, 
                score=chunk.score,
                metadata={**chunk.metadata, "compression_status": "compressed_query_aware"}
            )
            compressed_chunks.append(compressed_chunk)
            logger.info(f"Chunk {chunk.id} compressed. New length: {len(summary_text)} chars.")
            
    return compressed_chunks