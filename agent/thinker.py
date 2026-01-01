import sys
import json
from typing import List, Optional, Dict, Any, Type
from pathlib import Path
from pydantic import BaseModel, Field

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import (
    PlanSchema, 
    ThinkerOutput, 
    ActionType,
    Chunk,
    ReasoningTrace
)
from utils.llm_client import (
    generate_primary, 
    generate_schema_critique,
    repair_json_with_llm 
)
from utils.json_utils import (
    safe_json_load, 
    validate_json_structure,
    repair_json 
)
from utils.logger import get_logger
from config.token_budgets import TOKEN_BUDGETS

# --- Smart Retrieval Layer Integration ---
try:
    from retrieval.pinecone_client import smart_retrieve
    RETRIEVAL_AVAILABLE = True
except ImportError:
    # Minimal fallback logger setup if imports fail
    logger = get_logger("THINKER")
    logger.warning("Retrieval import failed. Thinker is operating without external memory.")
    RETRIEVAL_AVAILABLE = False
    def smart_retrieve(*args, **kwargs): return []

logger = get_logger("THINKER")

# --- Internal Schema for LLM Generation ---
# Separates reasoning (LLM side) from serialization (System side).
class ThinkerSynthesis(BaseModel):
    thought_process: str = Field(..., description="Acts as a thought space for reasoning models")
    draft_answer: str
    key_facts_extracted: List[str]
    confidence_score: float
    missing_information: Optional[str]
    xai_trace: str

class Thinker:
    """
    The Executor: Handles Answer Synthesis and Verification Refinement Loops.
    Strictly executes the Plan provided by the Planner (RETRIEVE/REASON).
    """
    def __init__(self):
        self.prompt_path = project_root / "prompts" / "thinker_prompt.txt"
        self.base_prompt_path = project_root / "prompts" / "system_base.txt"
        self._load_prompt_template()
        logger.info("Thinker Initialized")

    def _load_prompt_template(self):
        try:
            self.base_system_prompt = self.base_prompt_path.read_text(encoding='utf-8')
            self.thinker_prompt_template = self.prompt_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            self.base_system_prompt = "You are the Thinker Agent."
            self.thinker_prompt_template = "Context: {CONTEXT_CHUNKS}\nQuery: {USER_QUERY}"

    def _robust_generation(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Gatekeeper Layer: Enforces strict JSON schema compliance.
        Pipeline: Validate -> Library Repair -> LLM Repair -> Critique Loop.
        """
        current_prompt = user_prompt
        expected_schema_dict = response_model.model_json_schema()
        expected_schema_json = json.dumps(expected_schema_dict, indent=2)

        for attempt in range(max_retries + 1):
            # 1. Generate Raw Output
            raw_response = generate_primary(
                system_prompt=system_prompt,
                user_prompt=current_prompt,
                response_schema=response_model 
            )

            # --- Stage 1: Standard Validation ---
            if validate_json_structure(raw_response, expected_schema_json):
                parsed_json = safe_json_load(raw_response)
                if parsed_json:
                    return parsed_json

            # --- Stage 2: Library Repair (json_repair) ---
            logger.warning(f"Stage 1 Failed. Attempting Library Repair (Attempt {attempt + 1})")
            repaired_obj = repair_json(raw_response)
            if repaired_obj:
                # Validate the repaired object against schema
                if validate_json_structure(json.dumps(repaired_obj), expected_schema_json):
                    logger.info("Stage 2 Success: JSON repaired via library.")
                    return repaired_obj

            # --- Stage 3: LLM One-Shot Repair ---
            logger.warning("Stage 2 Failed. Attempting LLM Repair.")
            try:
                repaired_text = repair_json_with_llm(expected_schema_json, raw_response)
                
                if repaired_text and validate_json_structure(repaired_text, expected_schema_json):
                    logger.info("Stage 3 Success: JSON repaired via LLM.")
                    return safe_json_load(repaired_text)
            except Exception as e:
                logger.error(f"Stage 3 error: {e}")

            # --- Stage 4: Critique & Loop (Fallback) ---
            logger.warning("Stage 3 Failed. Entering Critique Loop.")
            
            if attempt < max_retries:
                critique = generate_schema_critique(
                    expected_schema_str=expected_schema_json,
                    received_output_str=raw_response
                )
                
                if critique:
                    feedback = (
                        f"\n\n<SYSTEM_FEEDBACK>\n"
                        f"CRITIQUE: {critique.critique}\n"
                        f"SUGGESTION: {critique.suggestions}\n"
                        f"Ensure strict adherence to the schema:\n{expected_schema_json}\n"
                        f"</SYSTEM_FEEDBACK>"
                    )
                    current_prompt += feedback
                else:
                    current_prompt += f"\n\nError parsing JSON. Ensure output matches:\n{expected_schema_json}"
        
        logger.error("Thinker failed to generate valid JSON after all repair stages.")
        # Fail-safe fallback matching ThinkerSynthesis
        return {
            "thought_process": "System Failure: Unable to generate valid JSON.",
            "draft_answer": "I encountered an internal error while processing the data.",
            "key_facts_extracted": [],
            "confidence_score": 0.0,
            "missing_information": "JSON Schema Validation Failed",
            "xai_trace": "Error during generation."
        }

    def execute_plan(
        self,
        user_query: str,
        plan: PlanSchema,
        chat_history: str = "None",
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
        previous_draft = str(previous_draft) if previous_draft is not None else "None"
        verifier_feedback = str(verifier_feedback) if verifier_feedback is not None else "None"
        
        retrieved_chunks: List[Chunk] = []
        execution_log: List[str] = []
        
        # SYSTEM TRACE COLLECTOR: Captures hard evidence for the Verifier manually
        system_traces: List[ReasoningTrace] = []

        # --- 1. Execute Actions (Strict Execution) ---
        target_retrieve = ActionType.RETRIEVE.value.upper()
        target_reason = ActionType.REASON.value.upper()

        for step in plan.steps:
            try:
                # Handle Enum or String input for Action
                action_str = step.action.value.upper() if hasattr(step.action, 'value') else str(step.action).upper()
                logger.info(f"Step {step.step_id}: {action_str} - {step.query}")

                if action_str == target_retrieve:
                    # Execute Retrieval
                    if RETRIEVAL_AVAILABLE:
                        chunks = smart_retrieve(step.query)
                        count = len(chunks) if chunks else 0
                        
                        if count > 0:
                            retrieved_chunks.extend(chunks)
                            execution_log.append(f"Step {step.step_id} (RETRIEVE): Retrieved {count} chunks.")
                            system_traces.append(ReasoningTrace(
                                step_id=step.step_id,
                                action=ActionType.RETRIEVE,
                                query=step.query,
                                thought="Executing Retrieval via Pinecone",
                                observation=f"Found {count} documents."
                            ))
                        else:
                            execution_log.append(f"Step {step.step_id} (RETRIEVE): No results found.")
                            system_traces.append(ReasoningTrace(
                                step_id=step.step_id,
                                action=ActionType.RETRIEVE,
                                query=step.query,
                                thought="Executing Retrieval via Pinecone",
                                observation="No documents found."
                            ))
                    else:
                        execution_log.append(f"Step {step.step_id} (RETRIEVE): Skipped (Retrieval Unavailable).")

                elif action_str == target_reason:
                    # Log Reasoning intent (actual reasoning happens in synthesis)
                    execution_log.append(f"Step {step.step_id} (REASON): Planned logic - {step.query}")
                    system_traces.append(ReasoningTrace(
                        step_id=step.step_id,
                        action=ActionType.REASON,
                        query=step.query,
                        thought="Planned internal reasoning step",
                        observation="Pending synthesis..."
                    ))
                
                else:
                    logger.warning(f"Step {step.step_id}: Unknown or unsupported action '{action_str}' encountered. Skipping.")

            except Exception as e:
                logger.error(f"Error executing step {step.step_id}: {e}")
                execution_log.append(f"Step {step.step_id}: FAILED ({str(e)})")

        # --- 2. Context Preparation & Serialization ---
        # Serialize Plan for the Prompt
        plan_json = plan.model_dump_json() if hasattr(plan, 'model_dump_json') else json.dumps(plan, default=str)

        # Merge Retrieved Chunks into {CONTEXT_CHUNKS}
        context_parts = []
        
        if retrieved_chunks:
            # Simple dedup based on ID if available
            seen_ids = set()
            unique_chunks = []
            for c in retrieved_chunks:
                if c.id not in seen_ids:
                    unique_chunks.append(c)
                    seen_ids.add(c.id)
            
            chunk_text = "\n".join([f"[{c.id}] {c.text}" for c in unique_chunks])
            context_parts.append(f"--- RETRIEVED EVIDENCE ---\n{chunk_text}")
        else:
            context_parts.append("No external evidence found.")
        
        final_context_str = "\n\n".join(context_parts)
        final_exec_log_str = "\n".join(execution_log)
        token_limit = TOKEN_BUDGETS.get("THINKER_DRAFT_MAX", 1500)

        # --- 3. Prompt Engineering ---
        # Strictly replacing placeholders defined in thinker_prompt.txt
        user_prompt = self.thinker_prompt_template.replace("{USER_QUERY}", user_query)
        user_prompt = user_prompt.replace("{CHAT_HISTORY}", chat_history)
        user_prompt = user_prompt.replace("{PLAN_JSON}", plan_json)
        user_prompt = user_prompt.replace("{CONTEXT_CHUNKS}", final_context_str)
        user_prompt = user_prompt.replace("{EXECUTION_LOG}", final_exec_log_str)
        user_prompt = user_prompt.replace("{PREVIOUS_DRAFT}", previous_draft)
        user_prompt = user_prompt.replace("{VERIFIER_FEEDBACK}", verifier_feedback)
        user_prompt += f"\n\nCONSTRAINT: Keep the draft_answer concise (approx {token_limit} tokens)."

        # --- 4. Robust Generation ---
        synthesis_data = self._robust_generation(
            system_prompt=self.base_system_prompt,
            user_prompt=user_prompt,
            response_model=ThinkerSynthesis 
        )

        # --- 5. Final Assembly (Manual Trace Injection & Score Normalization) ---
        try:
            # Normalize Confidence Score (0.0 - 1.0)
            raw_score = synthesis_data.get("confidence_score", 0.0)
            if raw_score > 1.0:
                raw_score = raw_score / 100.0
            
            # Clamp strictly between 0 and 1
            final_score = max(0.0, min(1.0, raw_score))
            synthesis_data["confidence_score"] = final_score

            # Convert dict back to model to validate fields
            synthesis_model = ThinkerSynthesis(**synthesis_data)
            
            # Construct Final Output by merging LLM thoughts with Manual System Traces
            final_output = ThinkerOutput(
                thought_process=synthesis_model.thought_process,
                draft_answer=synthesis_model.draft_answer,
                key_facts_extracted=synthesis_model.key_facts_extracted,
                confidence_score=synthesis_model.confidence_score,
                retrieved_context=final_context_str,
                missing_information=synthesis_model.missing_information,
                reasoning_traces=system_traces, # INJECTED MANUALLY from execution loop
                xai_trace=synthesis_model.xai_trace
            )
            
            logger.info("Thinker Execution Complete. Traces merged and score normalized.")
            return final_output

        except Exception as e:
            logger.error(f"Failed to assemble final ThinkerOutput: {e}")
            return ThinkerOutput(
                thought_process="Assembly Error",
                draft_answer="Error assembling final response.",
                key_facts_extracted=[],
                confidence_score=0.0,
                missing_information=f"System Error: {str(e)}",
                reasoning_traces=[],
                xai_trace="Error"
            )