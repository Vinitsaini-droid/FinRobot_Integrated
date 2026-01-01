import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union, Type, List
from enum import Enum
from pydantic import BaseModel, Field

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import (
    PlanSchema, 
    PlanStep,
    ActionType, 
    PlanCritique,
    PlanValidity,
    SchemaCritique
)
from utils.llm_client import (
    generate_secondary, 
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
from retrieval.query_refiner import refine_query_for_planner
from config.token_budgets import TOKEN_BUDGETS

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
        
        logger.info("Planner Loaded")

    def _load_prompt_template(self):
        try:
            self.planner_prompt_template = self.prompt_path.read_text(encoding='utf-8')
            self.base_system_prompt = self.base_prompt_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            self.planner_prompt_template = "Draft a plan for: {USER_QUERY}"
            self.base_system_prompt = "You are an AI planner."

    def generate_plan(
        self, 
        user_query: str, 
        chat_history: str = "",
        external_feedback: Optional[str] = None
    ) -> PlanSchema:
        """
        Public entry point: Orchestrates Drafting -> Critiquing -> Refinement.
        Includes a final sanitization layer to remove unsupported actions.
        """
        logger.info("=== Planner Started ===")
        
        try:
            # 0. Refine Query (Added Step)
            # Clarify the user's intent using history before planning
            refined_query = refine_query_for_planner(user_query, chat_history) + "\n" + user_query
            logger.info(f"Planner processing refined query: {refined_query}")

            # 1. Draft Candidate Plan (Using Refined Query)
            candidate_plan = self._generate_plan(
                refined_query, 
                chat_history, 
                external_feedback
            )

            # 2. Critique the Plan
            critique = self._critique_plan(refined_query, candidate_plan)
            
            final_plan = candidate_plan
            
            # 3. Refine if Invalid
            if critique.validity != PlanValidity.VALID:
                logger.info(f"Refining plan. Critique: {critique.critique}")
                final_plan = self._refine_plan(refined_query, candidate_plan, critique, chat_history)
            
            # 4. Sanitization & Re-indexing
            # STRICT FILTER: Ensure only RETRIEVE or REASON actions exist.
            valid_steps = []
            for step in final_plan.steps:
                if step.action in [ActionType.RETRIEVE, ActionType.REASON]:
                    valid_steps.append(step)
                else:
                    logger.warning(f"Sanitizing plan: Removing unsupported action '{step.action}'")

            final_plan.steps = valid_steps

            # Re-index step IDs to be sequential after filtering
            for i, step in enumerate(final_plan.steps):
                step.step_id = i + 1

            if not final_plan.steps:
                logger.warning("Plan was empty after sanitization. Injecting fallback retrieval step.")
                final_plan.steps = [
                    PlanStep(
                        step_id=1, 
                        action=ActionType.RETRIEVE, 
                        query=f"{user_query}", 
                        status="pending"
                    ),
                    PlanStep(
                        step_id=2, 
                        action=ActionType.REASON, 
                        query="Synthesize findings based on retrieved context.", 
                        status="pending"
                    )
                ]

            return final_plan

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._fallback_plan(user_query)

    def _generate_plan(self, query: str, history: str, feedback: Optional[str]) -> PlanSchema:
        """Internal drafting method."""
        token_limit = TOKEN_BUDGETS.get("PLANNER_OUTPUT_MAX", 400)
        user_prompt = self.planner_prompt_template.replace("{USER_QUERY}", query)
        user_prompt = user_prompt.replace("{CANDIDATE_PLAN}", "None") 
        user_prompt = user_prompt.replace("{CHAT_HISTORY}", history if history else "None")
        user_prompt = user_prompt.replace("{FEEDBACK}", feedback if feedback else "None")
        
        # INJECT STRICT RETRIEVAL MANDATE
        user_prompt += (
            "\n\nCONSTRAINT: You must strictly use ONLY 'retrieve' or 'reason' actions.\n"
            "PRIMARY RULE: If the user query asks for ANY facts, figures, financial results, dates, or external information, "
            "your FIRST step MUST be 'retrieve'.\n"
            "Do NOT use 'reason' to assume data is unavailable or to explain why you can't answer.\n"
            "ALWAYS attempt to 'retrieve' first.\n"
            "Only use 'reason' as the first step for purely creative tasks (e.g., 'write a poem' with no data requirements).\n"
            f"EFFICIENCY: Keep the plan concise. You have a strict limit of {token_limit} tokens for your output. "
            "Be direct and avoid verbose thought processes."
        )

        return self._robust_generation(
            func=generate_primary, # Planner uses Primary Model
            system_prompt=self.base_system_prompt,
            user_prompt=user_prompt,
            response_model=PlanSchema
        )

    def _critique_plan(self, query: str, plan: PlanSchema) -> PlanCritique:
        """Self-Correction Loop."""
        system_prompt = (
            "You are a Senior QC Agent. Critique the following plan.\n"
            "FAIL the plan if the user asked for facts/data but the first step is 'reason' instead of 'retrieve'.\n"
            "FAIL the plan if it assumes data is unavailable without checking.\n"
            "Ensure it is logical, efficient, and uses 'thought_process' to explain validation."
        )
        
        plan_json = plan.model_dump_json() if hasattr(plan, 'model_dump_json') else json.dumps(plan)
        
        user_prompt = f"Query: {query}\nPlan: {plan_json}\nEvaluate validity."

        return self._robust_generation(
            func=generate_secondary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=PlanCritique
        )

    def _refine_plan(self, query: str, plan: PlanSchema, critique: PlanCritique, history: str) -> PlanSchema:
        """
        Refinement Step.
        """
        plan_json = plan.model_dump_json() if hasattr(plan, 'model_dump_json') else json.dumps(plan)
        
        full_feedback = (
            f"CRITIQUE: {critique.critique}\n"
            f"REQUIRED FIXES: {critique.suggestions}\n"
            "REMINDER: If this is a factual query, Step 1 MUST be 'retrieve'."
        )

        user_prompt = self.planner_prompt_template.replace("{USER_QUERY}", query)
        user_prompt = user_prompt.replace("{CANDIDATE_PLAN}", plan_json)
        user_prompt = user_prompt.replace("{CHAT_HISTORY}", history if history else "Refinement Mode")
        user_prompt = user_prompt.replace("{FEEDBACK}", full_feedback)

        user_prompt += (
            "\n\nMODE: REFINEMENT. Fix the plan based on the feedback above.\n"
            "CONSTRAINT: Force 'retrieve' as the first step for data queries."
        )

        return self._robust_generation(
            func=generate_primary,
            system_prompt=self.base_system_prompt,
            user_prompt=user_prompt,
            response_model=PlanSchema
        )

    def _robust_generation(
        self, 
        func: callable, 
        system_prompt: str, 
        user_prompt: str, 
        response_model: Type[Any], 
        max_retries: int = 2
    ) -> Any:
        """
        Gatekeeper: Enforces strict JSON compliance via a 4-Stage Repair Pipeline.
        """
        expected_schema_dict = response_model.model_json_schema()
        expected_schema_str = json.dumps(expected_schema_dict, indent=2)
        
        # --- Stage 1: Generation & Standard Validation ---
        raw_output = func(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_model
        )

        if validate_json_structure(raw_output, expected_schema_str):
            data = safe_json_load(raw_output)
            return response_model.model_validate(data)
        
        logger.warning("Standard parse failed. Entering 4-Stage Repair Pipeline.")

        # --- Stage 2: Library Repair ---
        logger.info("Stage 2: Attempting 'json_repair'...")
        repaired_obj = repair_json(raw_output)
        
        if repaired_obj:
            try:
                # Re-verify schema compliance
                if validate_json_structure(json.dumps(repaired_obj), expected_schema_str):
                    logger.info("Stage 2 Success: JSON repaired via library.")
                    return response_model.model_validate(repaired_obj)
            except Exception:
                pass

        # --- Stage 3: LLM One-Shot Repair ---
        logger.info("Stage 3: Attempting 'repair_json_with_llm'...")
        try:
            repaired_llm_text = repair_json_with_llm(expected_schema_str, raw_output)
            if validate_json_structure(repaired_llm_text, expected_schema_str):
                 logger.info("Stage 3 Success: JSON repaired via LLM.")
                 data = safe_json_load(repaired_llm_text)
                 return response_model.model_validate(data)
        except Exception as e:
            logger.warning(f"Stage 3 failed: {e}")

        # --- Stage 4: Iterative Critique Loop ---
        logger.warning("Stage 3 failed. Entering Stage 4: Interactive Critique Loop.")
        
        current_output = raw_output
        for attempt in range(max_retries):
            critique = generate_schema_critique(expected_schema_str, current_output)
            
            if not critique:
                logger.error("Critique generation failed. Aborting loop.")
                break

            logger.info(f"Critique Loop {attempt+1}/{max_retries}: {critique.critique[:100]}...")

            repair_prompt = (
                f"### TARGET SCHEMA ###\n{expected_schema_str}\n\n"
                f"### MALFORMED INPUT ###\n{current_output}\n\n"
                f"### CRITIQUE ###\n{critique.critique}\n\n"
                f"### REPAIR INSTRUCTIONS ###\n{critique.suggestions}\n\n"
                "Generate the fully corrected JSON now. No markdown. No text."
            )
            
            # Using generate_secondary (Flash) for fast repairs
            current_output = generate_secondary(
                system_prompt="You are a JSON Repair Agent.",
                user_prompt=repair_prompt,
                response_schema=response_model 
            )

            if validate_json_structure(current_output, expected_schema_str):
                logger.info("Stage 4 Success: JSON repaired via Critique Loop.")
                data = safe_json_load(current_output)
                return response_model.model_validate(data)

        logger.error("All repair stages failed. Returning None.")
        return None

    def _fallback_plan(self, query: str) -> PlanSchema:
        logger.warning("Triggering Fallback Plan.")
        return PlanSchema(
            thought_process="Fallback mechanism triggered due to planning failure.",
            steps=[
                PlanStep(
                    step_id=1, 
                    action=ActionType.RETRIEVE, 
                    query=f"{query}", 
                    status="pending"
                ),
                PlanStep(
                    step_id=2, 
                    action=ActionType.REASON, 
                    query="Synthesize findings", 
                    status="pending"
                )
            ],
            risk_level="low",
            requires_compliance=False,
            xai_notes="Generated via fallback mechanism."
        )