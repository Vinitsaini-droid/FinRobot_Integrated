# agent/explainer.py
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Type, List
from pydantic import BaseModel, Field, ConfigDict

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agent.schemas import (
    VerificationReport, 
    UserProfileSchema, 
    ThinkerOutput, 
    PlanSchema,
    ExplainerOutput,
    ReasoningTrace,
    XAICitation
)
from config.token_budgets import TOKEN_BUDGETS
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
from config.token_budgets import TOKEN_BUDGETS

logger = get_logger("EXPLAINER")

# --- Internal Schemas for LLM Generation Only ---
class ExplainerMetadata(BaseModel):
    """
    Strict definition of metadata to prevent validation errors 
    when the LLM adds fields like 'risk_assessment' or 'depth'.
    """
    model_config = ConfigDict(extra='allow') # Allow LLM to add extra metadata fields without crashing
    
    tone_used: str = Field(..., description="The tone adopted (e.g., formal, casual)")
    depth_mode: str = Field(..., description="The explanation depth (e.g., detailed, simple)")
    risk_level: str = Field(..., description="Risk assessment of the answer")

class ExplainerLLMResponse(BaseModel):
    """
    Subset of ExplainerOutput for the LLM to generate.
    We exclude 'plan', 'reasoning_traces', and 'citations' to prevent hallucinations.
    """
    thought_process: str = Field(..., description="Brief thought on how to frame the answer")
    explanation: str = Field(..., description="The final response text for the user.")
    meta_data: ExplainerMetadata = Field(..., description="Meta information about the response style")

# ------------------------------------------------

class Explainer:
    """
    The Spokesperson: Synthesizes the final response based on User Profile.
    Adapts explanation depth, tone, and addresses prior misunderstandings.
    """
    def __init__(self):
        self.prompts_dir = project_root / "prompts"
        self.prompt_path = self.prompts_dir / "explainer_prompt.txt"
        self.system_base_path = self.prompts_dir / "system_base.txt"
        
        self.system_base = ""
        self.system_prompt_template = ""
        
        self._load_prompt_template()
        logger.info("Explainer Loaded")

    def _load_prompt_template(self):
        # 1. Load System Base
        if self.system_base_path.exists():
            with open(self.system_base_path, "r", encoding="utf-8") as f:
                self.system_base = f.read().strip()
        else:
            logger.warning("system_base.txt not found. Explainer running without global directives.")

        # 2. Load Explainer Template
        if not self.prompt_path.exists():
            self.system_prompt_template = (
                "You are the Explainer Agent. Synthesize the provided context into a helpful response."
            )
        else:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt_template = f.read()

    def generate_explanation(
        self,
        user_query: str,
        user_profile: UserProfileSchema,
        plan: PlanSchema,
        thinker_output: ThinkerOutput,
        verification_report: VerificationReport
    ) -> ExplainerOutput:
        """
        Generates the ExplainerOutput object (Schema Generation).
        """
        logger.info("=== EXPLAINER STARTED ===")
        
        # 1. Prepare Prompt Context
        system_prompt = self._construct_system_prompt(
            user_query, user_profile, plan, thinker_output, verification_report
        )
        token_limit = TOKEN_BUDGETS.get("EXPLAINER_FINAL_MAX", 1200)
        user_prompt="GENERATE PERSONALIZED RESPONSE JSON"
        user_prompt += f"\n\nCONSTRAINT: Keep the final_answer concise (approx {token_limit} tokens)."
        # 2. Gatekeeper Loop: Generate & Validate JSON
        # We use ExplainerLLMResponse to force the LLM to focus ONLY on text generation
        llm_response_data = self._generate_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ExplainerLLMResponse, # Pass the restricted schema
            max_retries=3
        )

        if not llm_response_data:
            return self._create_fallback_output(thinker_output, verification_report, plan)

        # 3. Construct Complete Explainer Output Schema
        # Merge LLM text output with trustworthy artifacts from upstream agents
        try:
            # Extract dict if it's not already
            if hasattr(llm_response_data, "model_dump"):
                data_dict = llm_response_data.model_dump()
            else:
                data_dict = llm_response_data

            final_output = ExplainerOutput(
                thought_process=data_dict.get("thought_process", "No thought process"),
                explanation=data_dict.get("explanation", "Error in generation"),
                # ARTIFACT INJECTION: Directly map upstream sources
                citations=verification_report.xai_citations,
                plan=plan,
                reasoning_traces=thinker_output.reasoning_traces,
                # Metadata from LLM
                meta_data=data_dict.get("meta_data", {})
            )
            return final_output
        except Exception as e:
            logger.error(f"Error constructing ExplainerOutput: {e}")
            return self._create_fallback_output(thinker_output, verification_report, plan)

    def render_explanation(self, output: ExplainerOutput, depth_mode: str) -> str:
        """
        Formatting Layer: Converts the structured ExplainerOutput into the final string.
        """
        buffer = [output.explanation] 
        buffer.append("\n\n---") 

        # Filter Logic
        show_plan = depth_mode in ['detailed', 'technical']
        show_traces = depth_mode == 'technical'

        if show_plan and output.plan:
            buffer.append("\n**Execution Plan Summary:**")
            for step in output.plan.steps:
                status_icon = "✓" if step.status == "completed" else "○"
                buffer.append(f"- [{status_icon}] {step.query}")

        if show_traces and output.reasoning_traces:
            buffer.append("\n**Technical Reasoning Traces:**")
            buffer.append("```json")
            # Convert Pydantic models to dicts for clean dumping
            traces_dump = [t.model_dump(mode='json') for t in output.reasoning_traces]
            buffer.append(json.dumps(traces_dump, indent=2))
            buffer.append("```")
            
        return "\n".join(buffer)

    def _generate_with_retry(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        response_model: Type[BaseModel],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Executes generation with a multi-stage self-healing loop.
        Uses response_model.model_json_schema() to drive validation.
        """
        # Generate the strict schema string for validation
        expected_schema_dict = response_model.model_json_schema()
        expected_schema_str = json.dumps(expected_schema_dict, indent=2)
        
        current_user_prompt = user_prompt + f"\n\n<JSON_SCHEMA>\n{expected_schema_str}\n</JSON_SCHEMA>"

        for attempt in range(max_retries + 1):
            logger.info(f"Explainer Generation Attempt {attempt + 1}/{max_retries + 1}")
            
            # A. Generate Raw Text
            raw_response = generate_secondary(
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                max_tokens=TOKEN_BUDGETS.get("EXPLAINER_FINAL_MAX", 2000),
                temperature=0.3
            )

            # --- REPAIR & VALIDATION WATERFALL ---
            
            # 1. Primary Check: Validate JSON Structure
            if validate_json_structure(raw_response, expected_schema_str):
                data = safe_json_load(raw_response)
                if data: return data

            # 2. Secondary Check: Heuristic Repair (json_repair)
            logger.warning("Standard validation failed. Attempting Heuristic Repair...")
            repaired_heuristic = repair_json(raw_response)
            if repaired_heuristic:
                # Convert back to string to validate against strict schema
                repaired_str = json.dumps(repaired_heuristic)
                if validate_json_structure(repaired_str, expected_schema_str):
                    logger.info("Heuristic Repair Successful.")
                    return repaired_heuristic

            # 3. Tertiary Check: LLM-based Repair
            logger.warning("Heuristic Repair failed. Attempting LLM Repair...")
            repaired_llm_str = repair_json_with_llm(expected_schema_str, raw_response)
            if repaired_llm_str and validate_json_structure(repaired_llm_str, expected_schema_str):
                logger.info("LLM Repair Successful.")
                return safe_json_load(repaired_llm_str)

            # 4. Critique & Loop (Constraint Tightening)
            logger.warning(f"Attempt {attempt + 1} failed completely. Generating Critique...")
            critique = generate_schema_critique(expected_schema_str, raw_response)
            
            if critique and attempt < max_retries:
                feedback_prompt = (
                    f"\n\n### PREVIOUS ATTEMPT FAILED ###\n"
                    f"Critique: {critique.critique}\n"
                    f"Fix Instructions: {critique.suggestions}\n"
                    f"Ensure you STRICTLY follow the schema:\n{expected_schema_str}\n"
                    f"Respond ONLY with the corrected JSON."
                )
                current_user_prompt += feedback_prompt
            else:
                logger.error("All repairs and retries failed.")

        return None

    def _construct_system_prompt(
        self,
        query: str,
        profile: UserProfileSchema,
        plan: PlanSchema,
        thinker: ThinkerOutput,
        verification: VerificationReport
    ) -> str:
        """
        Dynamically assembles the system prompt.
        """
        # Serialize objects for prompt injection
        profile_json = profile.model_dump_json(indent=2)
        plan_json = plan.model_dump_json(include={'risk_level', 'xai_notes', 'steps'}, indent=2)
        
        # We only pass text traces to prompt context, not ask for them back
        traces_json = json.dumps([t.model_dump() for t in thinker.reasoning_traces], indent=2)

        verification_summary = f"Status: {verification.verification_status}\nNotes: {verification.critique}"

        prompt = self.system_prompt_template
        
        # 1. User & Profile Context
        prompt = prompt.replace("{USER_QUERY}", query)
        prompt = prompt.replace("{PROFILE_CONTEXT}", profile_json)
        
        # 2. Answers & Verification
        prompt = prompt.replace("{VERIFIED_ANSWER}", thinker.draft_answer)
        prompt = prompt.replace("{VERIFICATION_NOTE}", verification_summary)
        
        # 3. Optional Context
        prompt = prompt.replace("{PLAN_SUMMARY}", plan_json)
        prompt = prompt.replace("{REASONING_TRACES}", traces_json)
        
        # 4. Teaching Context
        misunderstandings = profile.prior_misunderstandings_summary if profile.prior_misunderstandings_summary else "None"
        prompt = prompt.replace("{MISUNDERSTANDINGS}", misunderstandings)

        # 5. Inject Tone Instructions
        # CRITICAL FIX: Explicit instruction to avoid verbatim copying
        tone_instr = (
            f"Tone: {profile.style_preference}. Depth: {profile.explanation_depth}.\n"
            f"IMPORTANT: You MUST rewrite the 'Verified Answer' to match the specific tone and depth above. "
            f"Do NOT copy the answer verbatim."
        )
        prompt = prompt.replace("{TONE_INSTRUCTION}", tone_instr)

        # 6. Inject Risk Instructions
        if verification.verification_status == "RISKY" or plan.risk_level == "high":
            risk_instr = "WARNING: Topic is HIGH RISK. You must add standard financial disclaimers."
        else:
            risk_instr = "Topic is standard. Be helpful and accurate."
        prompt = prompt.replace("{RISK_INSTRUCTION}", risk_instr)

        return prompt + "\n\n" + self.system_base

    def _create_fallback_output(
        self, 
        thinker_output: ThinkerOutput, 
        verification_report: VerificationReport,
        plan: PlanSchema
    ) -> ExplainerOutput:
        """
        Robust fallback if personalization fails entirely. Returns a valid ExplainerOutput.
        """
        logger.error("Explainer falling back to raw Thinker output.")
        
        fallback_text = (
            f"**System Note:** Personalization failed due to technical constraints. "
            f"Here is the verified answer:\n\n"
            f"{thinker_output.draft_answer}"
        )
        
        return ExplainerOutput(
            thought_process="Fallback triggered due to JSON validation failures.",
            explanation=fallback_text,
            citations=verification_report.xai_citations,
            meta_data={
                "tone_used": "formal", 
                "depth_mode": "detailed", 
                "risk_level": "medium"
            },
            plan=plan,
            reasoning_traces=thinker_output.reasoning_traces
        )