# agent/explainer.py
import sys
import json
from pathlib import Path
from typing import Optional, List

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
    VerificationStatus
)
from config.token_budgets import TOKEN_BUDGETS
from utils.llm_client import generate_secondary
from utils.logger import get_logger

logger = get_logger("EXPLAINER")

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

    def _load_prompt_template(self):
        # 1. Load System Base
        if self.system_base_path.exists():
            with open(self.system_base_path, "r", encoding="utf-8") as f:
                self.system_base = f.read().strip()
            logger.info("Explainer loaded system_base.txt")
        else:
            logger.warning("system_base.txt not found. Explainer running without global directives.")

        # 2. Load Explainer Template
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Explainer prompt missing at: {self.prompt_path}")
        
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def explain(
        self,
        user_query: str,
        user_profile: UserProfileSchema,
        plan: PlanSchema,
        thinker_output: ThinkerOutput,
        verification_report: VerificationReport
    ) -> str:
        """
        Generates the final user-facing response.
        """
        # 1. Prepare XAI Citations (Append to answer so Explainer sees the sources)
        citations_text = ""
        if verification_report.xai_citations:
            # Format: [Source ID] Snippet...
            citations_list = [f"[{c.evidence_ids}] {c.claim}..." for c in verification_report.xai_citations]
            citations_text = "\n\n**Verified Sources:**\n" + "\n".join(citations_list)
        
        # Combine Draft Answer with Citations for the prompt context
        verified_answer_context = thinker_output.draft_answer + citations_text

        # 2. Smart Depth Logic (Conditional Context Injection)
        # We filter the context *before* the prompt to strictly enforce depth preferences and save tokens.
        depth_mode = user_profile.explanation_depth # 'simple', 'detailed', 'technical'
        
        plan_summary_str = "Not provided (User requested Simple mode)."
        reasoning_traces_str = "Not provided (User requested non-Technical mode)."

        if depth_mode in ['detailed', 'technical']:
            # 'detailed' and 'technical' both see the Plan
            plan_summary_str = "\n".join([f"{idx+1}. {step.query}" for idx, step in enumerate(plan.steps)])
        
        if depth_mode == 'technical':
            # Only 'technical' sees the raw Reasoning Traces
            reasoning_traces_str = json.dumps([t.model_dump() for t in thinker_output.reasoning_traces], indent=2)

        # 3. Smart Misunderstandings Handling
        # Inject constraint: Only address if relevant to THIS query.
        misunderstandings_str = "None."
        if user_profile.prior_misunderstandings_summary:
            misunderstandings_str = (
                f"[User History: {user_profile.prior_misunderstandings_summary}]\n"
                f"[Constraint: Explicitly address this history ONLY if it is directly relevant to the query: '{user_query}'. "
                f"If unrelated, ignore it entirely.]"
            )

        # 4. Personalization Context & Mapping
        # Format general free-text preferences
        custom_prefs_str = "None"
        if user_profile.preferences:
            # Join list into a comma-separated string for the prompt
            custom_prefs_str = ", ".join(user_profile.preferences)

        profile_context_str = (
            f"Explanation Depth: {depth_mode.upper()}\n"
            f"Tone Preference: {user_profile.style_preference.upper()}\n"
            f"Risk Tolerance: {user_profile.risk_tolerance.upper()}\n"
            f"Specific User Constraints: {custom_prefs_str}"
        )

        # Map Enums to Natural Language Instructions
        tone_map = {
            'formal': "Professional, objective, and precise banking language.",
            'casual': "Friendly, conversational, and easy to understand.",
            'concise': "Extremely brief. Bullet points preferred. No fluff."
        }
        risk_map = {
            'low': "Strictly emphasize capital preservation and risks. Be conservative.",
            'medium': "Balance growth potential with clear risk disclosures.",
            'high': "Focus on growth potential, but briefly mention volatility."
        }
        
        tone_instr = tone_map.get(user_profile.style_preference, tone_map['formal'])
        risk_instr = risk_map.get(user_profile.risk_tolerance, risk_map['medium'])

        # --- Prompt Population ---
        # Prepend System Base to the Explainer Template
        combined_template = f"{self.system_base}\n\n{self.system_prompt_template}"
        system_prompt = combined_template
        
        system_prompt = system_prompt.replace("{USER_QUERY}", user_query)
        system_prompt = system_prompt.replace("{VERIFIED_ANSWER}", verified_answer_context)
        system_prompt = system_prompt.replace("{VERIFICATION_NOTE}", verification_report.critique)
        
        system_prompt = system_prompt.replace("{PROFILE_CONTEXT}", profile_context_str)
        system_prompt = system_prompt.replace("{MISUNDERSTANDINGS}", misunderstandings_str)
        
        # These will contain actual data OR placeholder text based on Depth Logic
        system_prompt = system_prompt.replace("{PLAN_SUMMARY}", plan_summary_str)
        system_prompt = system_prompt.replace("{REASONING_TRACES}", reasoning_traces_str)
        
        system_prompt = system_prompt.replace("{TONE_INSTRUCTION}", tone_instr)
        system_prompt = system_prompt.replace("{RISK_INSTRUCTION}", risk_instr)

        try:
            # --- Generate Final Response ---
            response_text = generate_secondary(
                system_prompt=system_prompt,
                user_prompt="GENERATE PERSONALIZED RESPONSE",
                max_tokens=TOKEN_BUDGETS.get("EXPLAINER_FINAL_MAX", 1500),
                temperature=0.3 # Balanced for fluency and accuracy
            )
            return response_text

        except Exception as e:
            logger.error(f"Explainer synthesis failed: {e}")
            # Robust fallback: Return the verified answer directly if synthesis fails
            return (
                f"**System Note:** Personalization failed. Here is the raw verified answer:\n\n"
                f"{thinker_output.draft_answer}\n\n"
                f"Sources:\n{citations_text}"
            )