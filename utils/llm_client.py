# utils/llm_client.py
from google import genai
from google.genai import types
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Any

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from utils.logger import get_logger
from agent.schemas import SchemaCritique

logger = get_logger("LLM_CLIENT")

RPM_WAIT_SECONDS = 65  # hardcoded by design

# --- Client Initialization ---
try:
    if not settings.GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY is missing in settings.")
        client = None
    else:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
except Exception as e:
    logger.critical("Failed to initialize Gemini client: %s", e)
    client = None


def _execute_generation(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    response_schema: Optional[Any] = None,
) -> str:
    """
    Executes a single generation request with quota-aware handling.

    Behavior:
    - RPM quota → wait 60s, retry once
    - Daily quota → fail immediately
    """
    if not client:
        logger.error("Gemini client not initialized.")
        return "ERROR: Client initialization failure."

    contents = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)])
    ]

    config_args = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }

    if response_schema:
        config_args["response_mime_type"] = "application/json"
        config_args["response_schema"] = response_schema

    def _generate_once() -> Optional[str]:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                **config_args,
            ),
        )
        return response.text if response and response.text else None

    try:
        result = _generate_once()
        if not result:
            logger.warning("Model %s returned empty response.", model)
            return "ERROR: Empty response."
        return result

    except Exception as e:
        error_str = str(e).lower()

        is_quota_error = "429" in error_str or ("resource" in error_str and "exhausted" in error_str)

        if is_quota_error:
            # --- RPM Quota ---
            if "minute" in error_str or "rate" in error_str:
                logger.warning(
                    "RPM quota hit on model %s. Waiting %d seconds before retry.",
                    model,
                    RPM_WAIT_SECONDS,
                )
                time.sleep(RPM_WAIT_SECONDS)

                try:
                    logger.info("Retrying generation on model %s after RPM wait.", model)
                    retry_result = _generate_once()
                    if not retry_result:
                        logger.warning("Retry on model %s returned empty response.", model)
                        return "ERROR: Empty response after retry."
                    return retry_result
                except Exception as retry_e:
                    logger.error("Retry failed after RPM wait: %s", retry_e)
                    return "ERROR: RPM quota retry failed."

            # --- Daily / Hard Quota ---
            if "day" in error_str or "daily" in error_str or "quota" in error_str:
                logger.error("Daily quota exhausted for model %s: %s", model, e)
                return "ERROR: Daily quota limit hit."

        logger.error("Generation failed on model %s: %s", model, e)
        return "ERROR: Generation failure."


def generate_primary(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    response_schema: Optional[Any] = None,
) -> str:
    """Uses the PRIMARY (Pro) model."""
    return _execute_generation(
        model=settings.PRIMARY_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        response_schema=response_schema,
    )


def generate_secondary(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    response_schema: Optional[Any] = None,
) -> str:
    """Uses the SECONDARY (Flash) model."""
    return _execute_generation(
        model=settings.SECONDARY_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        response_schema=response_schema,
    )


def generate_schema_critique(
    expected_schema_str: str,
    received_output_str: str
) -> Optional[SchemaCritique]:
    """
    Uses the specialized JSON_VALIDATOR model to critique discrepancies between
    an expected schema and a received raw string output.
    
    Returns:
        SchemaCritique object containing 'critique' and 'suggestions',
        or None if generation fails.
    """
    system_prompt = (
        "You are an expert Data Schema Validator and JSON Repair Specialist. "
        "Your task is to strictly compare a 'Received Output' against an 'Expected Schema Definition'. "
        "Identify syntax errors, missing fields, type mismatches, or hallucinations. "
        "Both the Schemas should exactly match."
        "Be precise and technical." 
    )
    
    user_prompt = (
        f"### EXPECTED SCHEMA DEFINITION ###\n{expected_schema_str}\n\n"
        f"### RECEIVED OUTPUT ###\n{received_output_str}\n\n"
        "Analyze the discrepancies strictly. "
        "Return a JSON object with two fields:\n"
        "1. 'critique': A concise explanation of what is invalid.\n"
        "2. 'suggestions': Actionable instructions on how to correct the JSON to match the schema."
    )

    # We use a lower temperature for strict analysis
    response_str = _execute_generation(
        model=settings.JSON_VALIDATOR,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1024,
        temperature=0.1,
        response_schema=SchemaCritique
    )

    if "ERROR" in response_str:
        logger.error("Schema critique generation failed: %s", response_str)
        return None

    try:
        # Parse the JSON string into the Pydantic model
        return SchemaCritique.model_validate_json(response_str)
    except Exception as e:
        logger.error("Failed to parse SchemaCritique from model output: %s", e)
        # Fallback: try to return a basic error wrapper if parsing fails entirely
        # preventing the pipeline from crashing on a validator failure.
        return SchemaCritique(
            critique=f"Model output parsing failed: {str(e)}",
            suggestions="Ensure the model outputs valid JSON conforming to SchemaCritique."
        )


def repair_json_with_llm(
    expected_schema_str: str,
    received_output_str: str
) -> Optional[str]:
    """
    High-effort repair function that:
    1. Generates a critique of the malformed JSON using the SchemaCritique model.
    2. Uses that critique to guide the Validator model in generating a corrected JSON string.
    
    Returns:
        A string containing the repaired JSON (hopefully valid), or None if repair flow failed.
    """
    # 1. First, get the critique to understand WHY it failed
    critique = generate_schema_critique(expected_schema_str, received_output_str)
    if not critique:
        logger.warning("JSON Repair: Failed to generate critique. Aborting LLM repair.")
        return None

    # 2. Formulate the repair prompt with the critique context
    system_prompt = (
        "You are an expert JSON Repair Specialist. "
        "You will be given a malformed JSON string, the Schema it must adhere to, "
        "and a critique explaining exactly what is wrong. "
        "Your ONLY job is to output the Corrected JSON string. "
        "Do not output markdown code blocks. Do not output explanations. Just the raw JSON."
    )

    user_prompt = (
        f"### TARGET SCHEMA ###\n{expected_schema_str}\n\n"
        f"### MALFORMED INPUT ###\n{received_output_str}\n\n"
        f"### CRITIQUE ###\n{critique.critique}\n\n"
        f"### REPAIR INSTRUCTIONS ###\n{critique.suggestions}\n\n"
        "Generate the fully corrected JSON now."
    )

    # 3. Execute Repair using the JSON_VALIDATOR model
    return _execute_generation(
        model=settings.JSON_VALIDATOR,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=4096,  # Ensure enough space for the full object
        temperature=0.0,
        response_schema=None # We want raw string to avoid forced schema issues during repair
    )


def get_embedding(text: str) -> List[float]:
    """Generates an embedding vector using the configured embedding model."""
    if not client:
        logger.error("Gemini client not initialized. Cannot generate embeddings.")
        return []

    if not text:
        logger.warning("Empty text passed to embedding function.")
        return []

    try:
        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text,
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        return []