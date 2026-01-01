# retrieval/query_refiner.py
import sys
import json
from typing import List, Optional
from pathlib import Path
from google.genai import types 

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.llm_client import generate_secondary, generate_schema_critique, repair_json_with_llm
from utils.json_utils import safe_json_load, validate_json_structure
from utils.logger import get_logger

logger = get_logger("QUERY_REFINER")

MAX_RETRIES = 2

# --- Prompts ---

STRATEGY_SELECTOR_PROMPT = (
    "You are a Search Strategist. Analyze the User Query and Chat History to select the "
    "single best retrieval query refinement technique.\n"
    "OPTIONS:\n"
    "1. 'HyDE': Best for vague queries needing a hypothetical answer to ground the vector search.\n"
    "2. 'Decomposition': Best for complex, multi-part questions.\n"
    "3. 'HistoryContext': Best when the query relies heavily on previous chat context (e.g., 'What about him?').\n"
    "4. 'KeywordExpansion': Best for simple, specific technical lookups.\n\n"
    "Output valid JSON with key 'selected_strategy' (string) and 'reasoning' (string)."
)

QUERY_REFINER_PROMPT_TEMPLATE = (
    "You are a specialized Query Refiner using the '{STRATEGY}' technique.\n"
    "Generate 3 diverse search queries to maximize retrieval coverage.\n"
    "Context: {CHAT_HISTORY}\n"
    "User Query: {ORIGINAL_QUERY}\n\n"
    "Output strictly valid JSON with key 'refined_queries' (list of strings)."
)

PLANNER_REFINE_PROMPT = (
    "You are a Query Pre-processor for an AI Planner. Your goal is to rewrite the user's raw query "
    "into a clear, unambiguous objective statement that a Planner agent can easily break down.\n"
    "1. Resolve coreferences using Chat History (e.g., change 'it' to the actual subject).\n"
    "2. Remove conversational noise (e.g., 'Can you please tell me...').\n"
    "3. Make explicit any implied constraints.\n\n"
    "Chat History: {CHAT_HISTORY}\n"
    "Raw Query: {RAW_QUERY}\n\n"
    "Output strictly valid JSON with key 'refined_goal' (string)."
)

def refine_query(original_query: str, chat_history: str = "None") -> List[str]:
    """
    Expands a user query into multiple diverse queries.
    Uses strict schema enforcement and multi-stage repair to prevent format hallucinations.
    """
    if not original_query:
        return []

    # 1. Select Strategy
    strategy = "HyDE" # Default
    try:
        # Standard JSON Schema for Validation
        strategy_json_schema = {
            "type": "object",
            "properties": {
                "selected_strategy": {"type": "string"},
                "reasoning": {"type": "string"}
            },
            "required": ["selected_strategy"]
        }
        
        # Google Types Schema for Generation
        strategy_google_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "selected_strategy": types.Schema(type=types.Type.STRING),
                "reasoning": types.Schema(type=types.Type.STRING)
            },
            required=["selected_strategy"]
        )
        
        # --- Strategy Selection Loop ---
        strategy_resp = generate_secondary(
            system_prompt=STRATEGY_SELECTOR_PROMPT,
            user_prompt=f"Query: {original_query}\nHistory: {chat_history}",
            max_tokens=150,
            response_schema=strategy_google_schema,
            temperature=0.3
        )
        
        # Validation & Repair
        is_valid_strategy = validate_json_structure(strategy_resp, json.dumps(strategy_json_schema))
        
        if not is_valid_strategy:
            logger.info("Strategy selection JSON invalid. Attempting LLM repair...")
            repaired_strategy = repair_json_with_llm(json.dumps(strategy_json_schema), strategy_resp)
            if repaired_strategy and validate_json_structure(repaired_strategy, json.dumps(strategy_json_schema)):
                strategy_resp = repaired_strategy
                is_valid_strategy = True
                logger.info("Strategy selection JSON repaired via LLM.")

        if is_valid_strategy:
            strategy_data = safe_json_load(strategy_resp)
            if strategy_data and "selected_strategy" in strategy_data:
                strategy = strategy_data["selected_strategy"]
                logger.info(f"Selected Refinement Strategy: {strategy}")
        else:
            logger.warning("Strategy selection validation failed. Using default 'HyDE'.")

    except Exception as e:
        logger.warning(f"Strategy selection failed, defaulting to HyDE: {e}")

    # 2. Generate Queries using Strategy
    base_system_prompt = QUERY_REFINER_PROMPT_TEMPLATE.replace("{STRATEGY}", strategy)
    base_system_prompt = base_system_prompt.replace("{CHAT_HISTORY}", chat_history)
    base_system_prompt = base_system_prompt.replace("{ORIGINAL_QUERY}", original_query)

    # Standard Schema
    queries_json_schema = {
        "type": "object",
        "properties": {
            "refined_queries": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["refined_queries"]
    }

    # Google Schema
    queries_google_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "refined_queries": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING)
            )
        },
        required=["refined_queries"]
    )
    
    current_system_prompt = base_system_prompt

    # --- Robust Generation Loop ---
    for attempt in range(MAX_RETRIES):
        try:
            raw_response = generate_secondary(
                system_prompt=current_system_prompt,
                user_prompt="Generate Queries",
                max_tokens=256,
                response_schema=queries_google_schema,
                temperature=0.7 
            )
            
            # 1. Standard Validation (includes local repair)
            is_valid = validate_json_structure(raw_response, json.dumps(queries_json_schema))
            
            # 2. LLM Repair Fallback
            if not is_valid:
                logger.info(f"Refinement attempt {attempt+1} invalid. Trying LLM repair...")
                repaired_response = repair_json_with_llm(json.dumps(queries_json_schema), raw_response)
                
                if repaired_response and validate_json_structure(repaired_response, json.dumps(queries_json_schema)):
                    raw_response = repaired_response
                    is_valid = True
                    logger.info("Refinement JSON repaired via LLM.")

            # 3. Success Handler
            if is_valid:
                refined_data = safe_json_load(raw_response)
                if refined_data and "refined_queries" in refined_data:
                    generated = refined_data["refined_queries"]
                    final_set = list(set([original_query] + generated))
                    logger.info(f"Refined '{original_query}' into {len(final_set)} variations.")
                    return final_set
            
            # 4. Critique Fallback (if repair failed)
            logger.warning(f"Attempt {attempt+1} failed validation & repair. Generating critique...")
            critique_obj = generate_schema_critique(
                expected_schema_str=json.dumps(queries_json_schema),
                received_output_str=raw_response
            )
            
            if critique_obj:
                # Update prompt with specific feedback for the model
                current_system_prompt = (
                    f"{base_system_prompt}\n\n"
                    f"### PREVIOUS ATTEMPT FAILED ###\n"
                    f"Critique: {critique_obj.critique}\n"
                    f"Fix Instructions: {critique_obj.suggestions}\n"
                    f"Ensure strictly valid JSON."
                )
            
        except Exception as e:
            logger.warning(f"Refinement attempt {attempt+1} exception: {e}")
    
    logger.error("Query refinement failed after retries. Returning original query.")
    return [original_query]

def refine_query_for_planner(raw_query: str, chat_history: str = "None") -> str:
    """
    Refines a raw user query into a clean, standalone objective for the Planner.
    Resolves ambiguity using chat history with robust validation.
    """
    if not raw_query:
        return ""

    # Standard Schema
    goal_json_schema = {
        "type": "object",
        "properties": {
            "refined_goal": {"type": "string"}
        },
        "required": ["refined_goal"]
    }

    # Google Schema
    goal_google_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "refined_goal": types.Schema(type=types.Type.STRING)
        },
        required=["refined_goal"]
    )

    base_sys_prompt = PLANNER_REFINE_PROMPT.replace("{CHAT_HISTORY}", chat_history)
    base_sys_prompt = base_sys_prompt.replace("{RAW_QUERY}", raw_query)
    current_sys_prompt = base_sys_prompt

    for attempt in range(MAX_RETRIES):
        try:
            raw_response = generate_secondary(
                system_prompt=current_sys_prompt,
                user_prompt="REFINE GOAL",
                max_tokens=150,
                response_schema=goal_google_schema,
                temperature=0.2
            )
            
            # 1. Standard Validation (includes local repair)
            is_valid = validate_json_structure(raw_response, json.dumps(goal_json_schema))
            
            # 2. LLM Repair Fallback
            if not is_valid:
                logger.info(f"Planner refinement attempt {attempt+1} invalid. Trying LLM repair...")
                repaired_response = repair_json_with_llm(json.dumps(goal_json_schema), raw_response)
                
                if repaired_response and validate_json_structure(repaired_response, json.dumps(goal_json_schema)):
                    raw_response = repaired_response
                    is_valid = True
                    logger.info("Planner refinement JSON repaired via LLM.")

            # 3. Success Handler
            if is_valid:
                data = safe_json_load(raw_response)
                if data and "refined_goal" in data:
                    refined = data["refined_goal"]
                    logger.info(f"Planner Query Refined: '{raw_query}' -> '{refined}'")
                    return refined
            
            # 4. Critique Fallback
            logger.warning(f"Planner refinement attempt {attempt+1} failed validation & repair.")
            critique_obj = generate_schema_critique(
                expected_schema_str=json.dumps(goal_json_schema),
                received_output_str=raw_response
            )

            if critique_obj:
                 current_sys_prompt = (
                    f"{base_sys_prompt}\n\n"
                    f"### PREVIOUS ATTEMPT FAILED ###\n"
                    f"Critique: {critique_obj.critique}\n"
                    f"Fix Instructions: {critique_obj.suggestions}"
                )
            
        except Exception as e:
            logger.error(f"Planner query refinement exception: {e}")
    
    return raw_query