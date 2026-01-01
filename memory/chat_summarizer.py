# memory/chat_summarizer.py
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from google.genai import types
from agent.schemas import ChatSummary, UserProfileSchema
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

logger = get_logger("CHAT_SUMMARIZER")

MAX_RETRIES = 3

class ChatSummarizer:
    """
    The Cortex: Analyzes interactions to evolve the User Profile in real-time
    and compresses history for long-term memory.
    """
    MAX_CONTEXT_CHARS = 12000 
    MAX_PREFERENCES = 15 # Hard limit for list size

    def _get_truncated_text(self, history: List[str]) -> str:
        full_text = "\n".join(history)
        if len(full_text) > self.MAX_CONTEXT_CHARS:
            return "..." + full_text[-self.MAX_CONTEXT_CHARS:]
        return full_text

    def _robust_parse_and_validate(self, raw_response: str, validation_schema_str: str) -> Optional[Union[Dict, List]]:
        """
        Helper: Centralized validation pipeline with multi-stage repair.
        
        Stages:
        1. Validate raw output (utilizing internal json_repair from utils).
        2. If invalid, attempt LLM-based repair (repair_json_with_llm).
        3. If valid, load safely (fallback to repair_json if strict load fails).
        """
        candidate_text = raw_response

        # --- Stage 1: Standard Validation ---
        if not validate_json_structure(candidate_text, validation_schema_str):
            # --- Stage 2: LLM Repair ---
            logger.warning("Standard validation failed. Attempting LLM-based repair...")
            candidate_text = repair_json_with_llm(validation_schema_str, raw_response)
            
            # Re-validate the LLM's fix
            if not candidate_text or not validate_json_structure(candidate_text, validation_schema_str):
                logger.warning("LLM repair failed to produce valid JSON.")
                return None
            else:
                logger.info("LLM repair successful.")

        # --- Stage 3: Safe Loading ---
        # At this point, candidate_text is structurally valid according to validate_json_structure
        data = safe_json_load(candidate_text)
        if data is None:
            # Fallback to forceful repair if safe load fails despite validation pass
            data = repair_json(candidate_text)
        
        return data

    def analyze_interaction_delta(self, current_profile: UserProfileSchema, last_user_msg: str, last_agent_msg: str) -> Dict[str, Any]:
        """
        Scans a SINGLE interaction to intelligently evolve the profile.
        """
        # Serialize current state explicitly for the prompt
        curr_prefs = ", ".join(current_profile.preferences) if current_profile.preferences else "None"
        curr_bugs = current_profile.prior_misunderstandings_summary if current_profile.prior_misunderstandings_summary else "None"
        
        base_system_prompt = (
            "You are a Real-Time User Profile Architect.\n"
            "Your goal is to EVOLVE the user profile based on the LATEST INTERACTION.\n\n"
            
            "--- RULES FOR SCALAR FIELDS (Update only if explicit change detected) ---\n"
            "1. Risk Tolerance: 'low' (safety/guarantees), 'medium', 'high' (aggressive/moonshots).\n"
            "2. Explanation Depth: 'simple' (ELI5), 'detailed', 'technical' (code/math).\n"
            "3. Style Preference: 'formal', 'casual', 'concise'.\n\n"
            
            "--- RULES FOR COMPOSITE FIELDS (Smart Evolution) ---\n"
            f"4. Preferences (Max {self.MAX_PREFERENCES} items):\n"
            f"   - CURRENT LIST: [{curr_prefs}]\n"
            "   - If new interests appear: MERGE them into the current list.\n"
            "   - DEDUPLICATE: Consolidate similar items (e.g., 'Python' + 'Python coding' -> 'Python').\n"
            "   - PRIORITIZE: Keep the list concise. If > 15 items, drop the least relevant/oldest.\n"
            "   - OUTPUT: Return the FULL REPLACEMENT list.\n\n"
            
            "5. Prior Misunderstandings Summary:\n"
            f"   - CURRENT SUMMARY: {curr_bugs}\n"
            "   - If the user corrects the agent: REWRITE the summary to concisely explain what NOT to do.\n"
            "   - MERGE the new correction with the old summary. Do not just append. Synthesize.\n"
            "   - OUTPUT: Return the FULL REPLACEMENT string.\n\n"

            "OUTPUT FORMAT:\n"
            "Return JSON. Include ONLY fields that have changed/evolved. If no change, return empty JSON."
        )

        interaction_text = f"User: {last_user_msg}\nAgent: {last_agent_msg}"

        # Standard Schema for Validation
        validation_schema = {
            "type": "object",
            "properties": {
                "risk_tolerance": {"type": "string", "enum": ["low", "medium", "high"]},
                "explanation_depth": {"type": "string", "enum": ["simple", "detailed", "technical"]},
                "style_preference": {"type": "string", "enum": ["formal", "casual", "concise"]},
                "prior_misunderstandings_summary": {"type": "string"},
                "preferences": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "additionalProperties": False 
        }
        validation_schema_str = json.dumps(validation_schema)

        # Google Schema for Generation
        google_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "risk_tolerance": types.Schema(type=types.Type.STRING, enum=["low", "medium", "high"]),
                "explanation_depth": types.Schema(type=types.Type.STRING, enum=["simple", "detailed", "technical"]),
                "style_preference": types.Schema(type=types.Type.STRING, enum=["formal", "casual", "concise"]),
                "prior_misunderstandings_summary": types.Schema(type=types.Type.STRING),
                "preferences": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING)
                )
            }
        )

        current_system_prompt = base_system_prompt

        # --- RETRY LOOP ---
        for attempt in range(MAX_RETRIES):
            try:
                raw_response = generate_secondary(
                    system_prompt=current_system_prompt,
                    user_prompt=f"LATEST INTERACTION:\n{interaction_text}",
                    response_schema=google_schema,
                    temperature=0.0 
                )

                # Use Centralized Validation/Repair Pipeline
                updates = self._robust_parse_and_validate(raw_response, validation_schema_str)

                if updates is not None:
                    if not updates: 
                        return {}

                    # Final Safety Guard: Python-side truncation
                    if "preferences" in updates and isinstance(updates["preferences"], list):
                        if len(updates["preferences"]) > self.MAX_PREFERENCES:
                            updates["preferences"] = updates["preferences"][:self.MAX_PREFERENCES]
                    return updates
                
                # SELF-HEALING: Generate Critique if pipeline failed
                logger.warning(f"Profile evolution failed (Attempt {attempt+1}). Generating critique...")
                critique_obj = generate_schema_critique(
                    expected_schema_str=validation_schema_str,
                    received_output_str=raw_response
                )

                if critique_obj:
                    current_system_prompt = (
                        f"{base_system_prompt}\n\n"
                        f"### PREVIOUS ATTEMPT FAILED ###\n"
                        f"Critique: {critique_obj.critique}\n"
                        f"Fix Instructions: {critique_obj.suggestions}\n"
                        f"Ensure strictly valid JSON."
                    )

            except Exception as e:
                logger.error(f"Profile evolution analysis attempt {attempt+1} failed: {e}")
        
        logger.error("Profile evolution failed after retries.")
        return {}

    def summarize(self, conversation_history: List[str]) -> ChatSummary:
        """Standard episodic compression."""
        if not conversation_history:
            return ChatSummary(summary="", key_facts=[])

        text_block = self._get_truncated_text(conversation_history)
        
        base_system_prompt = (
            "Summarize the conversation. "
            "Extract key facts (dates, numbers, entities) and a 2-sentence summary."
        )
        
        # Standard Schema for Validation
        validation_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_facts": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["summary", "key_facts"],
            "additionalProperties": False
        }
        validation_schema_str = json.dumps(validation_schema)

        # Google Schema for Generation
        google_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "summary": types.Schema(type=types.Type.STRING),
                "key_facts": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))
            },
            required=["summary", "key_facts"]
        )

        current_system_prompt = base_system_prompt

        for attempt in range(MAX_RETRIES):
            try:
                raw_response = generate_secondary(
                    system_prompt=current_system_prompt,
                    user_prompt=f"CONVERSATION:\n{text_block}",
                    response_schema=google_schema,
                    temperature=0.0 
                )
                
                # Use Centralized Validation/Repair Pipeline
                data = self._robust_parse_and_validate(raw_response, validation_schema_str)

                if data is not None:
                    return ChatSummary.model_validate(data)
                
                # SELF-HEALING
                logger.warning(f"Summarization failed (Attempt {attempt+1}). Generating critique...")
                critique_obj = generate_schema_critique(
                    expected_schema_str=validation_schema_str,
                    received_output_str=raw_response
                )

                if critique_obj:
                    current_system_prompt = (
                        f"{base_system_prompt}\n\n"
                        f"### PREVIOUS ATTEMPT FAILED ###\n"
                        f"Critique: {critique_obj.critique}\n"
                        f"Fix Instructions: {critique_obj.suggestions}"
                    )
            
            except Exception as e:
                logger.error(f"Summarization attempt {attempt+1} failed: {e}")

        return ChatSummary(summary="Error.", key_facts=[])

    def deduplicate_facts(self, existing_facts: List[str], new_candidates: List[str]) -> List[str]:
        """
        Prevents long-term memory bloat by maintaining a strict, high-value list of facts.
        """
        if not new_candidates and not existing_facts:
            return []

        # STRICT LIMIT: 25 Facts
        base_system_prompt = (
            "You are a Memory Optimizer. Consolidate facts about the user.\n"
            "Input: EXISTING_FACTS and NEW_CANDIDATES.\n"
            "Task: Create a SINGLE, merged list of facts.\n"
            "--- GUIDELINES ---\n"
            "1. **STRICT LIMIT**: The output list must contain NO MORE THAN 25 items.\n"
            "2. **STYLE**: Concise, detailed, no blabber. Absolute information density.\n"
            "3. **MERGE**: If existing says 'User is 20' and new says 'User is 21', keep only 'User is 21'.\n"
            "4. **PRIORITIZE**: If > 25 facts exist, discard the oldest or least relevant (e.g., trivial likes) to fit the limit.\n"
            "5. **OUTPUT**: Return the FINAL complete list of facts to be stored.\n"
        )

        user_prompt = f"EXISTING_FACTS: {existing_facts}\nNEW_CANDIDATES: {new_candidates}"
        
        # Standard Schema for Validation
        validation_schema = {
            "type": "object",
            "properties": {
                "final_facts_list": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["final_facts_list"],
            "additionalProperties": False
        }
        validation_schema_str = json.dumps(validation_schema)

        # Google Schema for Generation
        google_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "final_facts_list": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING))
            },
            required=["final_facts_list"]
        )

        current_system_prompt = base_system_prompt

        for attempt in range(MAX_RETRIES):
            try:
                response = generate_secondary(
                    system_prompt=current_system_prompt,
                    user_prompt=user_prompt,
                    response_schema=google_schema,
                    temperature=0.0
                )
                
                # Use Centralized Validation/Repair Pipeline
                data = self._robust_parse_and_validate(response, validation_schema_str)

                if data is not None:
                    final_list = data.get("final_facts_list", [])
                    if len(final_list) > 25:
                        logger.warning("LLM returned > 25 facts. Truncating.")
                        return final_list[:25]
                    return final_list
                
                # SELF-HEALING
                logger.warning(f"Fact deduplication failed (Attempt {attempt+1}). Generating critique...")
                critique_obj = generate_schema_critique(
                    expected_schema_str=validation_schema_str,
                    received_output_str=response
                )

                if critique_obj:
                    current_system_prompt = (
                        f"{base_system_prompt}\n\n"
                        f"### PREVIOUS ATTEMPT FAILED ###\n"
                        f"Critique: {critique_obj.critique}\n"
                        f"Fix Instructions: {critique_obj.suggestions}"
                    )
            
            except Exception as e:
                logger.error(f"Fact deduplication attempt {attempt+1} failed: {e}")

        # Fallback: Combine and hard truncate to safety limit
        combined = list(set(existing_facts + new_candidates))
        return combined[:25]