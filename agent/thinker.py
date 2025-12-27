# agent/thinker.py
import sys
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import (
    PlanSchema, 
    ThinkerOutput, 
    ActionType,
    Chunk
)
from config.token_budgets import TOKEN_BUDGETS
from utils.llm_client import generate_primary
from utils.json_utils import safe_json_load
from utils.logger import get_logger

# --- Smart Retrieval Layer Integration ---
try:
    from retrieval.pinecone_client import smart_retrieve
    RETRIEVAL_AVAILABLE = True
except ImportError as e:
    logger = get_logger("THINKER")
    logger.error(f"CRITICAL: Retrieval import failed. Thinker is blind. Error: {e}")
    RETRIEVAL_AVAILABLE = False
    def smart_retrieve(*args, **kwargs): return []

logger = get_logger("THINKER")

class Thinker:
    """
    The Executor: Handles Answer Synthesis and Verification Refinement Loops.
    Strictly executes the Plan provided by the Planner (RETRIEVE/REASON).
    Does NOT handle Clarification or Plan Refinement.
    """
    def __init__(self):
        self.prompt_path = project_root / "prompts" / "thinker_prompt.txt"
        self.base_prompt_path = project_root / "prompts" / "system_base.txt"
        self._load_prompt_template()

    def _load_prompt_template(self):
        # Load Base Prompt
        base_content = ""
        if self.base_prompt_path.exists():
             with open(self.base_prompt_path, "r", encoding="utf-8") as f:
                base_content = f.read() + "\n\n"
        else:
            logger.warning(f"System base prompt missing at: {self.base_prompt_path}")

        # Load Thinker Specific Prompt
        if not self.prompt_path.exists():
            # Fallback if file is missing, though file creation is preferred
            raise FileNotFoundError(f"Thinker prompt not found at {self.prompt_path}")
        
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            # Prepend base prompt to the thinker prompt
            self.system_prompt_template = base_content + f.read()

    def execute_plan(
        self, 
        user_query: str, 
        plan: PlanSchema, 
        chat_history: str = "None",
        existing_context: str = "None",
        previous_draft: str = "None",
        verifier_feedback: str = "None"
    ) -> ThinkerOutput:
        """
        Executes the plan by retrieving data (if needed) and synthesizing a draft.
        """
        logger.info(f"=== THINKER EXECUTION START ===")
        logger.info(f"Incoming User Query: '{user_query}'")
        logger.info(f"Plan Steps: {len(plan.steps)}")
        
        # Input Sanitization
        chat_history = str(chat_history) if chat_history is not None else "None"
        existing_context = str(existing_context) if existing_context is not None else "None"
        
        retrieved_chunks: List[Chunk] = []
        execution_log: List[str] = []
        
        # SYSTEM TRACE COLLECTOR: Captures hard evidence for the Verifier
        # Must strictly match ReasoningTrace schema
        system_traces: List[Dict[str, Any]] = []

        # --- 1. Execute Actions (Strict Execution) ---
        for i, step in enumerate(plan.steps):
            # 1. Normalize Action String for Comparison
            if hasattr(step.action, 'value'):
                action_str = str(step.action.value).upper()
            else:
                action_str = str(step.action).upper()
            
            # Constants for Enum comparison
            target_retrieve = str(ActionType.RETRIEVE.value).upper()
            target_reason = str(ActionType.REASON.value).upper()

            logger.info(f"[Step {i+1}] Action: '{action_str}' | Query: '{step.query}'")

            # 2. Execute based on ActionType
            if action_str == target_retrieve or action_str == "RETRIEVE":
                if RETRIEVAL_AVAILABLE:
                    logger.info(f"--> RETRIEVAL TRIGGERED | Query: '{step.query}'")
                    try:
                        results = smart_retrieve(step.query, chat_history)
                        
                        if results:
                            retrieved_chunks.extend(results)
                            msg = f"✓ Retrieved {len(results)} chunks for query: '{step.query}'"
                            execution_log.append(msg)
                            logger.info(msg)
                            
                            # --- EVIDENCE CAPTURE START ---
                            # Capture the actual content so Verifier can see it in reasoning_traces.
                            evidence_text = "\n".join([f"[Source {c.id}]: {c.text}" for c in results])
                            
                            # Create a valid ReasoningTrace dict
                            system_trace = {
                                "step_id": i + 1,  # Must be INT
                                "action": "retrieve", # Matches ActionType.RETRIEVE
                                "query": step.query,
                                "thought": f"System executed retrieval for query: {step.query}", # Required field
                                "observation": evidence_text
                            }
                            system_traces.append(system_trace)
                            # --- EVIDENCE CAPTURE END ---
                        else:
                            msg = f"⚠ Retrieval returned 0 results for query: '{step.query}'"
                            execution_log.append(msg)
                            logger.warning(msg)
                    except Exception as e:
                        err_msg = f"❌ Error inside smart_retrieve: {e}"
                        logger.error(err_msg)
                        execution_log.append(err_msg)
                else:
                    execution_log.append("⚠ Retrieval unavailable (Import Failed).")

            elif action_str == target_reason or action_str == "REASON":
                msg = f"ℹ Logic/Reasoning Step: {step.query}"
                execution_log.append(msg)
                logger.debug("Processing reasoning step (Internal Monologue).")
                
            else:
                logger.debug(f"Thinker skipping non-executable action: {action_str}")

        # --- 2. Context Construction ---
        # Deduplicate chunks by ID
        unique_chunks_map = {c.id: c for c in retrieved_chunks}
        unique_chunks = list(unique_chunks_map.values())
        
        logger.info(f"Total Unique Chunks passed to Context: {len(unique_chunks)}")

        new_evidence_text = "\n\n".join([f"Source {c.id}: {c.text}" for c in unique_chunks])
        
        context_blocks = []
        if new_evidence_text:
            context_blocks.append(f"=== NEWLY RETRIEVED EVIDENCE ===\n{new_evidence_text}")
        
        if existing_context and existing_context != "None":
            context_blocks.append(f"=== PREVIOUS CONTEXT ===\n{existing_context}")

        if not context_blocks:
            chunks_text = "NO EXTERNAL EVIDENCE FOUND."
            logger.warning("Thinker context is empty. No evidence found or passed.")
        else:
            chunks_text = "\n\n".join(context_blocks)

        # --- 3. Prompt Synthesis ---
        system_prompt = self.system_prompt_template
        replacements = {
            "{USER_QUERY}": user_query,
            "{CONTEXT_CHUNKS}": chunks_text,
            "{EXECUTION_LOG}": "\n".join(execution_log),
            "{PREVIOUS_DRAFT}": previous_draft or "None",
            "{VERIFIER_FEEDBACK}": verifier_feedback or "None"
        }
        
        for placeholder, value in replacements.items():
            system_prompt = system_prompt.replace(placeholder, str(value))

        try:
            response_text = generate_primary(
                system_prompt=system_prompt,
                user_prompt=f"Synthesize the answer for: {user_query}",
                response_schema=ThinkerOutput,
                max_tokens=TOKEN_BUDGETS.get("THINKER_DRAFT_MAX", 1500),
                temperature=0.1 
            )
            
            data = safe_json_load(response_text)
            if not data:
                raise ValueError("LLM returned empty JSON.")
            
            # --- CRITICAL FIX: Merge System Traces with LLM Traces ---
            if "reasoning_traces" not in data:
                data["reasoning_traces"] = []
            
            # Prepend system traces (Hard Evidence) to LLM traces (Reasoning)
            # This ensures the Verifier sees the actual retrieval data.
            data["reasoning_traces"] = system_traces + data["reasoning_traces"]
            
            # --- CRITICAL FIX: Sanitize Traces ---
            # Ensure every trace meets the Pydantic Schema requirements
            if isinstance(data["reasoning_traces"], list):
                valid_actions = {a.value for a in ActionType}
                
                for idx, trace in enumerate(data["reasoning_traces"]):
                    if isinstance(trace, dict):
                        # 1. Ensure step_id is an INT
                        if "step_id" not in trace or trace["step_id"] is None:
                            trace["step_id"] = idx + 1
                        else:
                            try:
                                trace["step_id"] = int(trace["step_id"])
                            except:
                                trace["step_id"] = idx + 1

                        # 2. Ensure Action is Valid
                        current_action = str(trace.get("action", "")).lower()
                        if current_action not in valid_actions:
                            # Map to safe fallback
                            trace["action"] = ActionType.REASON.value
                        
                        # 3. Ensure 'thought' exists (Required field)
                        if "thought" not in trace or not trace["thought"]:
                            trace["thought"] = "Processing step..."

            return ThinkerOutput.model_validate(data)

        except Exception as e:
            logger.error(f"Thinker Execution/Synthesis failed: {e}")
            return ThinkerOutput(
                draft_answer=f"I encountered an error during analysis: {str(e)}",
                key_facts_extracted=[],
                confidence_score=0.0,
                missing_information="System Error",
                reasoning_traces=[],
                xai_trace="Error trace"
            )