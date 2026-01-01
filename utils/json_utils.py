# utils/json_utils.py
import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import jsonschema
from jsonschema import ValidationError
import json_repair  # Added for robust repair

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.logger import get_logger

logger = get_logger("JSON_UTILS")

def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Extracts the first balanced JSON object or array from text.
    Returns the substring or None if no balanced structure is found.
    """
    stack = []
    start_index = None

    for i, ch in enumerate(text):
        if ch in "{[":
            if start_index is None:
                start_index = i
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or ch != stack[-1]:
                continue
            stack.pop()
            if not stack and start_index is not None:
                return text[start_index : i + 1]

    return None

def safe_json_load(raw_string: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Robustly parses a string into a JSON object or list.
    
    Strategy:
    1. Direct Parse
    2. Parse from markdown code blocks (```json ... ```)
    3. Balanced brace/bracket extraction from raw text
    """
    if not raw_string:
        return None

    # --- Strategy 1: Fast Path ---
    try:
        return json.loads(raw_string)
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: Markdown Code Blocks ---
    markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    blocks = re.findall(markdown_pattern, raw_string, re.DOTALL)

    for block in blocks:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # --- Strategy 3: Balanced Extraction ---
    candidate = _extract_balanced_json(raw_string)
    if not candidate:
        logger.warning(
            "No balanced JSON object or array found in text snippet: %s...",
            raw_string[:80],
        )
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        snippet = candidate[:120] + "..." if len(candidate) > 120 else candidate
        logger.warning(
            "JSON parsing failed after extraction. Error: %s | Candidate: %s",
            e,
            snippet,
        )
        return None
    except Exception as e:
        logger.error("Unexpected error in safe_json_load: %s", e)
        return None

def repair_json(raw_string: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Robustly attempts to repair and parse a malformed JSON string.
    Uses the 'json_repair' library to handle common LLM output errors 
    (e.g., missing quotes, unescaped characters, trailing commas).
    
    Args:
        raw_string: The malformed JSON string.
        
    Returns:
        The parsed Python dictionary/list if successful, or None if repair fails.
    """
    if not raw_string:
        return None

    try:
        # json_repair.loads attempts to decode and repair simultaneously
        decoded_object = json_repair.loads(raw_string)
        
        # Ensure we actually got a dict or list, not just a string/int
        if isinstance(decoded_object, (dict, list)):
             return decoded_object
        else:
             logger.warning("JSON Repair resulted in a primitive type, not an object/array.")
             return None
             
    except Exception as e:
        logger.warning(f"json_repair failed to salvage JSON: {e}")
        return None

def _enforce_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively modifies a JSON schema to ensure 'additionalProperties' is False
    for all objects. This forces the validation to reject any extra fields 
    (hallucinations) not explicitly defined in the schema.
    """
    if not isinstance(schema, dict):
        return schema

    # If it's an object type, forbid extra properties
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        
        # Recurse into properties
        if "properties" in schema:
            for prop, sub_schema in schema["properties"].items():
                schema["properties"][prop] = _enforce_strict_schema(sub_schema)

    # Recurse into array items
    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = _enforce_strict_schema(schema["items"])

    # Recurse into definitions/$defs if present
    for def_key in ["definitions", "$defs"]:
        if def_key in schema:
            for d_name, d_schema in schema[def_key].items():
                schema[def_key][d_name] = _enforce_strict_schema(d_schema)

    return schema

def validate_json_structure(raw_string: str, expected_schema_str: str) -> bool:
    """
    Boolean Gatekeeper: STRICTLY checks if the model's output matches the structure,
    types, and constraints of the expected schema using 'jsonschema'.
    
    UPGRADE: Automatically enforces no extra fields to prevent hallucinations.

    Args:
        raw_string: The raw string output from the model.
        expected_schema_str: The schema to validate against (as a JSON string).

    Returns:
        True if the model output is valid JSON and strictly conforms to the schema.
        False otherwise.
    """
    # 1. Parse Model Output
    model_data = safe_json_load(raw_string)
    
    # --- ADDED: Fallback to Repair if standard parse fails ---
    if model_data is None:
        logger.info("Standard parse failed. Attempting JSON repair...")
        model_data = repair_json(raw_string)

    if model_data is None:
        logger.warning("Schema Match Failed: Model output could not be parsed or repaired.")
        return False

    # 2. Parse Expected Schema
    try:
        base_schema = json.loads(expected_schema_str)
    except json.JSONDecodeError:
        logger.error("Schema Match Error: Provided 'expected_schema_str' is invalid JSON.")
        return False

    # 3. Apply Strictness (No Extra Fields)
    strict_schema = _enforce_strict_schema(base_schema)

    # 4. Strict Schema Validation
    try:
        jsonschema.validate(instance=model_data, schema=strict_schema)
        return True
    except ValidationError as e:
        # Log specific validation error to aid in the repair loop
        error_msg = e.message[:200] + "..." if len(e.message) > 200 else e.message
        path = " -> ".join([str(p) for p in e.path]) if e.path else "root"
        
        logger.warning(
            "Schema Validation Failed at '%s': %s. Got: %s", 
            path, 
            error_msg, 
            type(e.instance)
        )
        return False
    except Exception as e:
        logger.error("Unexpected error during schema validation: %s", e)
        return False