# agent/planner.py
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import (
    PlanSchema, 
    ActionType, 
    PlanCritique,
    PlanValidity
)
from config.token_budgets import TOKEN_BUDGETS
from utils.llm_client import generate_secondary, generate_primary
from utils.json_utils import safe_json_load
from utils.logger import get_logger

try:
    from retrieval.query_refiner import refine_query_for_planner
    REFINER_AVAILABLE = True
except ImportError:
    REFINER_AVAILABLE = False

logger = get_logger("PLANNER")

class Planner:
    """
    The Architect: Converts user queries into a structured reasoning plan.
    Self-Improving: Critiques and refines its own plans before outputting.
    """
    def __init__(self):
        self.prompt_path = project_root / "prompts" / "planner_prompt.txt"
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

        # Load Planner Specific Prompt
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Planner prompt missing at: {self.prompt_path}")
        
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            # Prepend base prompt to the planner prompt
            self.system_prompt_template = base_content + f.read()

    def generate_plan(
        self, 
        user_query: str, 
        chat_history: str = "None",
        external_feedback: Optional[str] = None,
        max_loops: int = 2 
    ) -> PlanSchema:
        """
        Generates a Plan with an internal self-correction loop.
        Ensures the final plan is executable (RETRIEVE/REASON) rather than passive (CLARIFY).
        """
        # --- Input Sanitization ---
        chat_history = chat_history if chat_history is not None else "None"
       
        # 1. Refine Query 
        active_query = user_query
        if REFINER_AVAILABLE and not external_feedback:
            logger.info("Refining query for Planner context...")
            active_query = refine_query_for_planner(user_query, chat_history)

        current_feedback = external_feedback if external_feedback else "None"
        candidate_plan = None
        
        # Self-Correction Loop
        for i in range(max_loops):
            is_refinement = (i > 0) or (current_feedback != "None")
            mode = "REFINEMENT" if is_refinement else "DRAFTING"
            
            logger.info(f"Planner Iteration {i+1}/{max_loops}: {mode}")

            # A. Generate Draft / Refinement
            candidate_plan = self._draft_plan(
                query=active_query, 
                history=chat_history, 
                feedback=current_feedback,
                prev_plan_json=candidate_plan.model_dump_json() if candidate_plan else "None"
            )

            # B. Internal Critique (Self-Reflection)
            if i == max_loops - 1:
                logger.info("Max iterations reached. Returning current plan.")
                return candidate_plan

            # Check if Plan lazily asks for clarification
            has_clarify = any(
                (step.action == ActionType.CLARIFY or str(step.action).lower() == "clarify") 
                for step in candidate_plan.steps
            )

            # If the plan relies on CLARIFY, force the critique to address it
            critique_instruction = "None"
            if has_clarify:
                critique_instruction = (
                    "CRITICAL CHECK: The plan contains CLARIFY steps. "
                    "The Planner must resolve ambiguity internally. "
                    "Critique should demand conversion of CLARIFY into specific RETRIEVE queries "
                    "unless the query is totally incoherent."
                )

            critique = self._generate_critique(
                active_query, 
                candidate_plan, 
                chat_history, 
                override_instruction=critique_instruction
            )
            
            if critique.validity == PlanValidity.VALID and not has_clarify:
                logger.info("Internal Critique passed. Plan is valid.")
                return candidate_plan
            else:
                reason = critique.critique
                if has_clarify:
                    reason = f"Plan contained lazy CLARIFY steps. {critique.critique}"
                
                logger.warning(f"Internal Critique failed: {reason}")
                
                # Update feedback for the next loop
                current_feedback = (
                    f"Critique: {reason}\n"
                    f"Required Fix: {critique.suggestions}\n"
                    f"Instruction: Replace CLARIFY with specific RETRIEVE steps to find the answer."
                )

        return candidate_plan

    def _draft_plan(self, query: str, history: str, feedback: str, prev_plan_json: str) -> PlanSchema:
        """Helper to call the LLM for plan generation/refinement."""
        # Prepare Prompt
        system_prompt = self.system_prompt_template.replace("{USER_QUERY}", query)
        system_prompt = system_prompt.replace("{CHAT_HISTORY}", history)
        system_prompt = system_prompt.replace("{CANDIDATE_PLAN}", prev_plan_json) 
        system_prompt = system_prompt.replace("{FEEDBACK}", feedback)

        try:
            temp = 0.4 if feedback != "None" else 0.2
            user_prompt_str = "GENERATE EXECUTABLE PLAN (Prioritize RETRIEVE/REASON over CLARIFY)" if feedback == "None" else "REFINE PLAN"
            
            response_text = generate_secondary(
                system_prompt=system_prompt,
                user_prompt=user_prompt_str,
                response_schema=PlanSchema,
                max_tokens=TOKEN_BUDGETS.get("PLANNER_OUTPUT_MAX", 400),
                temperature=temp
            )
            
            data = safe_json_load(response_text)
            if not data: raise ValueError("Empty model response")

            # --- CRITICAL FIX: Sanitize Action Enums ---
            # Ensures LLM output matches ActionType(str, Enum) values exactly
            if "steps" in data and isinstance(data["steps"], list):
                for step in data["steps"]:
                    if "action" in step and isinstance(step["action"], str):
                        raw_action = step["action"].lower().strip()
                        
                        # Direct Map
                        if raw_action in ["retrieve", "search", "lookup", "find"]:
                            step["action"] = "retrieve"
                        elif raw_action in ["reason", "analyze", "think", "calc"]:
                            step["action"] = "reason"
                        elif raw_action in ["clarify", "ask", "question"]:
                            step["action"] = "clarify"
                        elif raw_action in ["verify", "check", "audit"]:
                            step["action"] = "verify"
                        elif raw_action in ["refuse", "deny", "reject"]:
                            step["action"] = "refuse"
                        else:
                            # Smart Fallback: If totally unknown, default to RETRIEVE
                            # This prevents the agent from crashing on a made-up action
                            logger.warning(f"Sanitizing unknown action '{raw_action}' to 'retrieve'")
                            step["action"] = "retrieve"
            # ---------------------------------------------
            
            return PlanSchema.model_validate(data)
            
        except Exception as e:
            logger.error(f"Planning draft failed: {e}")
            return self._fallback_plan(query)

    def _generate_critique(self, query: str, plan: PlanSchema, history: str, override_instruction: str = "None") -> PlanCritique:
        """
        Uses the Primary Model to judge the plan. 
        Enforces that the plan must be executable by the Thinker (RETRIEVE/REASON).
        """
        logger.info("Planner requesting Internal Critique...")
        
        plan_json = plan.model_dump_json(indent=2)
        
        system_prompt = self.system_prompt_template.replace("{USER_QUERY}", query)
        system_prompt = system_prompt.replace("{CHAT_HISTORY}", history)
        system_prompt = system_prompt.replace("{CANDIDATE_PLAN}", plan_json)
        
        # If we have a specific instruction (like catching CLARIFY), inject it into the feedback slot
        # to guide the Critic's prompt context
        context_instruction = "None"
        if override_instruction != "None":
            context_instruction = override_instruction

        system_prompt = system_prompt.replace("{FEEDBACK}", context_instruction)

        try:
            response_text = generate_primary(
                system_prompt=system_prompt,
                user_prompt="PERFORM MODE B: CRITIQUE. Ensure steps are strictly RETRIEVE or REASON.",
                response_schema=PlanCritique,
                max_tokens=300,
                temperature=0.1 
            )
            
            data = safe_json_load(response_text)
            if not data: raise ValueError("Empty critique response")
            return PlanCritique.model_validate(data)

        except Exception as e:
            logger.error(f"Critique generation failed: {e}")
            return PlanCritique(
                validity=PlanValidity.VALID, 
                critique="Critique generation failed.", 
                suggestions="None"
            )

    def _fallback_plan(self, query: str) -> PlanSchema:
        logger.warning("Triggering Fallback Plan.")
        return PlanSchema(
            steps=[
                {
                    "step_id": 1, 
                    "action": ActionType.RETRIEVE, 
                    "query": f"{query}", 
                    "status": "pending"
                },
                {
                    "step_id": 2, 
                    "action": ActionType.REASON, 
                    "query": "Synthesize findings", 
                    "status": "pending"
                }
            ],
            risk_level="low",
            requires_compliance=False,
            xai_notes="Generated via fallback mechanism."
        )