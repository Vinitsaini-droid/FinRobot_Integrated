# evaluation/aspect_critics.py
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from google.genai import types

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.llm_client import (
    generate_secondary, 
    generate_schema_critique, 
    repair_json_with_llm
)
from utils.json_utils import (
    safe_json_load, 
    validate_json_structure, 
    repair_json
)
from utils.logger import get_logger

logger = get_logger("ASPECT_CRITICS")

class CriticScore(BaseModel):
    score: int = Field(..., ge=0, le=10)
    reason: str

class AspectCritics:
    def __init__(self):
        # We still keep the Google Type Schema for the API call hint
        self.response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "score": types.Schema(type=types.Type.INTEGER),
                "reason": types.Schema(type=types.Type.STRING),
            },
            required=["score", "reason"]
        )

    def _robust_validate_and_parse(self, raw_text: str, schema_str: str) -> Optional[Dict[str, Any]]:
        """
        Helper: Executes the Multi-Stage Repair Pipeline.
        1. Validate Raw
        2. Repair Algorithmic (json_repair)
        3. Repair LLM (repair_json_with_llm)
        """
        # Stage 1: Standard Validation
        if validate_json_structure(raw_text, schema_str):
            return safe_json_load(raw_text)

        # Stage 2: Algorithmic Repair
        logger.debug("Stage 1 failed. Attempting algorithmic repair (repair_json)...")
        repaired_text = repair_json(raw_text)
        if repaired_text and validate_json_structure(repaired_text, schema_str):
            return safe_json_load(repaired_text)

        # Stage 3: LLM Repair
        logger.warning("Stage 2 failed. Attempting LLM-based repair (repair_json_with_llm)...")
        llm_repaired_text = repair_json_with_llm(schema_str, raw_text)
        if llm_repaired_text and validate_json_structure(llm_repaired_text, schema_str):
            return safe_json_load(llm_repaired_text)
        
        return None

    def _evaluate(self, prompt: str) -> CriticScore:
        """
        Evaluates a prompt with a self-healing JSON repair loop.
        """
        max_retries = 3
        # Use the Pydantic model to generate the strict expected schema string
        expected_schema_str = json.dumps(CriticScore.model_json_schema())
        
        current_prompt = prompt
        
        for attempt in range(max_retries):
            try:
                # 1. Generate Response
                raw_response = generate_secondary(
                    system_prompt="You are a financial auditor. Output valid JSON only.",
                    user_prompt=current_prompt,
                    response_schema=self.response_schema
                )

                # 2. Multi-Stage Validation & Repair
                data = self._robust_validate_and_parse(raw_response, expected_schema_str)

                if data:
                    return CriticScore.model_validate(data)
                
                # --- REPAIR LOOP START (Critique Fallback) ---
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: All repair stages failed. Initiating critique loop.")
                
                # Generate Critique
                critique_obj = generate_schema_critique(
                    expected_schema_str=expected_schema_str,
                    received_output_str=raw_response
                )
                
                if critique_obj:
                    # Update Prompt with Feedback
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"### PREVIOUS ATTEMPT FAILED ###\n"
                        f"Your previous JSON output was invalid.\n"
                        f"Critique: {critique_obj.critique}\n"
                        f"Correction Instructions: {critique_obj.suggestions}\n"
                        f"Please try again, strictly adhering to the schema."
                    )
                else:
                    # Fallback if critique generation fails
                    current_prompt = f"{prompt}\n\nPlease ensure strict JSON formatting."
                
                # --- REPAIR LOOP END ---

            except Exception as e:
                logger.error(f"Aspect Critic Logic Failure (Attempt {attempt}): {e}")
        
        # If all retries fail
        logger.error("Aspect Critics failed to produce valid JSON after retries.")
        return CriticScore(score=0, reason="Failed to parse LLM evaluation after multiple retries.")

    def critique_regulatory_compliance(self, answer: str) -> CriticScore:
        # Regulatory checks are often answer-standalone
        prompt = f"Rate Regulatory Compliance (0-10). Rules: No direct buy/sell advice, neutral tone. Answer: {answer}"
        return self._evaluate(prompt)

    def critique_tone(self, answer: str) -> CriticScore:
        prompt = f"Rate Professional Tone (0-10). 10 is standard banking English. Answer: {answer}"
        return self._evaluate(prompt)

    def critique_completeness(self, query: str, answer: str, chat_history: Optional[str] = None) -> CriticScore:
        context_str = f"Chat History: {chat_history}\n" if chat_history else ""
        prompt = (
            f"Given the conversation history and the latest query, rate Completeness (0-10).\n"
            f"{context_str}"
            f"Latest Query: {query}\n"
            f"Answer provided: {answer}\n"
            f"Did the answer address all parts of the user's intent?"
        )
        return self._evaluate(prompt)