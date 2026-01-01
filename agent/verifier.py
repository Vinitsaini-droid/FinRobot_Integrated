import sys
import json
from pathlib import Path
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import (
    VerificationReport, 
    VerificationStatus, 
    ThinkerOutput,
    PlanSchema
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
from config.compliance_rules import COMPLIANCE_RULES
from config.token_budgets import TOKEN_BUDGETS

logger = get_logger("VERIFIER")

class Verifier:
    """
    The Auditor: Validates Draft Answer for truth, safety, and logic.
    Strictly enforces authoritative COMPLIANCE_RULES injected from config (via prompts).
    Checks adherence to the original Plan.
    """
    def __init__(self):
        self.prompt_path = project_root / "prompts" / "verifier_prompt.txt"
        self.base_prompt_path = project_root / "prompts" / "system_base.txt"
        self.base_system_prompt = "You are the Verifier. Audit the answer against the plan and context."
        self.verifier_template = ""
        self._load_prompt_template()
        logger.info("Verifier Initialized")

    def _load_prompt_template(self):
        # Load Base Prompt
        if self.base_prompt_path.exists():
             with open(self.base_prompt_path, "r", encoding="utf-8") as f:
                self.base_system_prompt = f.read()
        else:
            logger.warning(f"System base prompt missing at: {self.base_prompt_path}")

        # Load Verifier Specific Prompt Template
        if self.prompt_path.exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.verifier_template = f.read()
        else:
            logger.error(f"Verifier prompt missing at: {self.prompt_path}")
            self.verifier_template = (
                "Audit the following:\n"
                "Query: {USER_QUERY}\n"
                "Draft: {DRAFT_ANSWER}\n"
                "Evidence: {EVIDENCE_CONTEXT}\n"
                "Plan: {PLAN}\n"
                "Logic: {EXECUTION_LOG}\n"
                "Rules: {COMPLIANCE_RULES}"
            )

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
                # FIX: Swapped arguments to match definition: (expected_schema, raw_output)
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
        
        logger.error("Verifier failed to generate valid JSON after all repair stages.")
        # Fail-safe fallback matching VerificationReport
        return {
            "thought_process": "System Failure: Unable to generate valid JSON.",
            "verification_status": VerificationStatus.FAIL,
            "critique": "JSON Schema Validation Failed",
            "suggested_correction": "Please retry generation.",
            "confidence_score": 0.0,
            "xai_citations": []
        }

    def verify_response(
        self,
        user_query: str,
        plan: PlanSchema,
        thinker_output: ThinkerOutput,
        evidence_context: str = "" 
    ) -> VerificationReport:
        """
        Audits the Thinker's draft against the original Plan and User Query.
        """
        logger.info("=== VERIFIER AUDIT START ===")
        
        # Serialize inputs for prompt injection
        plan_steps_str = "\n".join([f"{step.step_id}. {step.action}: {step.query}" for step in plan.steps])
        
        # Format reasoning traces from Thinker for audit
        traces_str = "No reasoning traces provided."
        
        if thinker_output.reasoning_traces:
            traces_str = "\n".join([
                f"- [Step {t.step_id}] Action: {t.action} | Thought: {t.thought} | Observation: {t.observation}"
                for t in thinker_output.reasoning_traces
            ])

        # --- PROCESS COMPLIANCE RULES ---
        # Convert the list from compliance_rules.py into a clean, numbered string
        formatted_rules = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(COMPLIANCE_RULES)])

        # Construct the Prompt Variables
        # Mapping strictly to verifier_prompt.txt placeholders:
        # {USER_QUERY}, {DRAFT_ANSWER}, {EVIDENCE_CONTEXT}, {EXECUTION_LOG}, {PLAN}, {COMPLIANCE_RULES}
        prompt_variables = {
            "{USER_QUERY}": user_query,
            "{DRAFT_ANSWER}": thinker_output.draft_answer,
            "{EVIDENCE_CONTEXT}": evidence_context,
            "{EXECUTION_LOG}": traces_str,
            "{PLAN}": plan_steps_str,
            "{COMPLIANCE_RULES}": formatted_rules 
        }

        user_prompt_str = self.verifier_template
        for key, value in prompt_variables.items():
            user_prompt_str = user_prompt_str.replace(key, str(value))
            
        # Fallback: If prompt template didn't have {EVIDENCE_CONTEXT} (e.g. malformed file), append it manually
        if evidence_context and evidence_context not in user_prompt_str and "{EVIDENCE_CONTEXT}" not in self.verifier_template:
             user_prompt_str += f"\n\n### ADDITIONAL EVIDENCE CONTEXT ###\n{evidence_context}"
        token_limit = TOKEN_BUDGETS.get("VERIFIER_REPORT_MAX", 500)
        # Execute Generation
        # NOTE: passing base_system_prompt as system instructions, and the filled template as user_prompt
        user_prompt_str += f"\n\nCONSTRAINT: Keep the final_report concise (approx {token_limit} tokens)."
        report_data = self._robust_generation(
            system_prompt=self.base_system_prompt,
            user_prompt=user_prompt_str,
            response_model=VerificationReport
        )

        try:
            # Final Object Construction
            report = VerificationReport(**report_data)
            
            # Post-processing Safety Check: Ensure valid Enum status
            if report.verification_status not in [e.value for e in VerificationStatus]:
                logger.warning(f"Invalid status '{report.verification_status}' detected. Defaulting to FAIL.")
                report.verification_status = VerificationStatus.FAIL
            
            logger.info(f"Verification Result: {report.verification_status}")
            return report

        except Exception as e:
            logger.error(f"Failed to assemble VerificationReport: {e}")
            return VerificationReport(
                thought_process="Assembly Error",
                verification_status=VerificationStatus.FAIL,
                critique=f"Error constructing report: {str(e)}",
                suggested_correction="System error.",
                confidence_score=0.0,
                xai_citations=[]
            )