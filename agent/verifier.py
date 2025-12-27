# agent/verifier.py
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

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
from config.token_budgets import TOKEN_BUDGETS
from config.compliance_rules import COMPLIANCE_RULES 
from utils.llm_client import generate_primary
from utils.json_utils import safe_json_load
from utils.logger import get_logger

logger = get_logger("VERIFIER")

class Verifier:
    """
    The Auditor: Validates Draft Answer for truth, safety, and logic.
    Strictly enforces authoritative COMPLIANCE_RULES injected from config.
    Checks adherence to the original Plan.
    """
    def __init__(self):
        self.prompts_dir = project_root / "prompts"
        self.prompt_path = self.prompts_dir / "verifier_prompt.txt"
        self.system_base_path = self.prompts_dir / "system_base.txt"
        
        self.system_base = ""
        self.template = ""
        
        self._load_prompt_template()

    def _load_prompt_template(self):
        # 1. Load System Base
        if self.system_base_path.exists():
            with open(self.system_base_path, "r", encoding="utf-8") as f:
                self.system_base = f.read().strip()
            logger.info("Verifier loaded system_base.txt")
        else:
            logger.warning("system_base.txt not found. Verifier running without global directives.")

        # 2. Load Verifier Template
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Verifier prompt missing at: {self.prompt_path}")
        
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.template = f.read()

    def verify_response(self, 
                       user_query: str, 
                       thinker_output: ThinkerOutput,
                       plan: PlanSchema) -> VerificationReport:
        """
        Runs the verification loop using the Primary Model (High Intelligence).
        """
        logger.info("Initiating Verification Cycle...")

        # 1. Prepare Data
        draft_answer = thinker_output.draft_answer
        
        # Serialize Context Evidence (XAI Trace source)
        evidence_context = json.dumps([
            t.observation for t in thinker_output.reasoning_traces 
            if t.action == "retrieve"
        ], indent=2)

        # Serialize Logic Path
        traces_json = json.dumps([
            t.model_dump() for t in thinker_output.reasoning_traces
        ], indent=2)

        # Serialize Plan (STRATEGIC CHANGE)
        plan_json = plan.model_dump_json(indent=2)

        # Serialize Compliance Rules
        compliance_section = json.dumps(COMPLIANCE_RULES, indent=2)

        # 2. Inject into Prompt
        # Prepend System Base to the Verifier Template
        combined_template = f"{self.system_base}\n\n{self.template}"
        system_prompt = combined_template
        
        system_prompt = system_prompt.replace("{USER_QUERY}", user_query)
        system_prompt = system_prompt.replace("{DRAFT_ANSWER}", draft_answer)
        system_prompt = system_prompt.replace("{EVIDENCE_CONTEXT}", evidence_context)
        system_prompt = system_prompt.replace("{EXECUTION_LOG}", traces_json)
        system_prompt = system_prompt.replace("{PLAN}", plan_json) # <-- NEW: Plan Injection
        system_prompt = system_prompt.replace("{COMPLIANCE_RULES}", compliance_section) 
        # -----------------------------

        try:
            response_text = generate_primary(
                system_prompt=system_prompt,
                user_prompt="VERIFY OUTPUT",
                response_schema=VerificationReport,
                max_tokens=TOKEN_BUDGETS.get("VERIFIER_REPORT_MAX", 600),
                temperature=0.1 # High rigor for auditing
            )
            
            data = safe_json_load(response_text)
            if not data:
                raise ValueError("Verifier returned empty JSON.")

            # --- CRITICAL FIX: Sanitize Verification Status ---
            # Maps inconsistent LLM capitalization/synonyms to strict Enum values
            if "verification_status" in data and isinstance(data["verification_status"], str):
                raw_status = data["verification_status"].upper().strip()
                
                # Standard Mappings
                if raw_status in ["PASS", "VALID", "TRUE", "CORRECT", "APPROVED"]:
                    data["verification_status"] = "PASS"
                elif raw_status in ["FAIL", "INVALID", "FALSE", "INCORRECT", "REJECTED"]:
                    data["verification_status"] = "FAIL"
                elif raw_status in ["FLAG", "UNCERTAIN", "WARN", "REVIEW"]:
                    data["verification_status"] = "FLAG"
                else:
                    # Smart Fallback: If status is unknown, it requires human/safety review -> FLAG
                    logger.warning(f"Sanitizing unknown status '{raw_status}' to 'FLAG'")
                    data["verification_status"] = "FLAG"
            # --------------------------------------------------

            report = VerificationReport(**data)
            
            # Log Result
            logger.info(f"Verification Complete. Status: {report.verification_status}")
            if report.verification_status != VerificationStatus.PASS:
                logger.warning(f"Critique: {report.critique}")

            return report

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            # Fallback report matching schemas.py exactly
            return VerificationReport(
                verification_status=VerificationStatus.FAIL,
                critique=f"Internal Verifier Error: {str(e)}",
                suggested_correction="System error. Please retry generation.",
                confidence_score=0.0,
                xai_citations=[] # Empty list as no valid verification occurred
            )