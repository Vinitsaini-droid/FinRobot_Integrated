# agent/meta_agent.py
import sys
import time
from pathlib import Path
from typing import Optional, List, Literal

# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Internal Imports ---
from agent.schemas import (
    VerificationStatus, 
    ThinkerOutput,
    VerificationReport,
    GlobalState,
    AgentStatus,
    IterationHistory,
    PlanSchema
)
from agent.planner import Planner
from agent.thinker import Thinker
from agent.verifier import Verifier
from agent.explainer import Explainer
from memory.memory_manager import MemoryManager
from retrieval.semantic_cache import retrieve_cache, store_cache
from evaluation.trace_logger import TraceLogger
from utils.logger import get_logger

logger = get_logger("META_AGENT")

class MetaAgent:
    """
    The 'Brain' of the system.
    Orchestrates the cognitive loop: Plan -> Think -> Verify -> Explain.
    
    Features Smart Fault Attribution:
    - If Verification fails, it determines if the Plan was bad or the Execution was bad.
    - Routes feedback strictly to the responsible agent to save tokens and improve stability.
    """
    
    MAX_LOOP_RETRIES = 3

    def __init__(self):
        # 1. Initialize Cognitive Organs
        self.planner = Planner()
        self.thinker = Thinker()
        self.verifier = Verifier()
        self.explainer = Explainer()
        self.memory = MemoryManager()
        
        # 2. Initialize Auditor
        self.trace_logger = TraceLogger()
        
        # 3. Load System Base Instructions
        self.system_base = ""
        try:
            # Assuming system_base.txt is in the project root based on context
            base_path = project_root / "system_base.txt"
            if base_path.exists():
                with open(base_path, "r", encoding="utf-8") as f:
                    self.system_base = f.read().strip()
                logger.info("Loaded system_base.txt directives successfully.")
            else:
                logger.warning(f"system_base.txt not found at {base_path}. Agents will run without global directives.")
        except Exception as e:
            logger.error(f"Failed to load system_base.txt: {e}")
        
        logger.info("MetaAgent initialized with Smart P-T-V-E Architecture.")

    def generate_response(self, user_id: str, query: str) -> str:
        """
        Main entry point for the agentic reasoning loop.
        """
        start_time = time.time()
        
        # --- 1. User Profile Update ---
        self.memory.check_and_update_profile_pre_planning(user_id, query)
        user_profile = self.memory.get_profile(user_id)

        # --- 2. Semantic Cache Check ---
        # Note: Cache uses the raw query to ensure hits match user intent, not system prompt variations.
        cached_response = retrieve_cache(query, user_profile)
        if cached_response:
            logger.info("Semantic Cache HIT.")
            self.memory.process_realtime_interaction(user_id, query, cached_response)
            return cached_response

        # --- 3. Context Retrieval ---
        chat_history = self.memory.get_immediate_context(user_id)
        
        # --- 4. Prepare System-Aware Context ---
        # Interject system_base before the prompt for all agents to enforce Truth Gate/Core Directives.
        system_aware_query = f"{self.system_base}\n\n[USER QUERY START]\n{query}\n[USER QUERY END]" if self.system_base else query
        
        # --- 5. Cognitive Loop (Smart Retry) ---
        loop_count = 0
        current_plan: Optional[PlanSchema] = None
        
        # Feedback Containers
        planner_feedback: Optional[str] = None
        thinker_feedback: Optional[str] = None
        
        # Fault Tracking
        fault_source: Literal["planner", "thinker", "none"] = "planner" # Start by needing a plan
        
        # Artifacts for final output
        final_thinker_output: Optional[ThinkerOutput] = None
        final_verification_report: Optional[VerificationReport] = None
        iteration_log: List[IterationHistory] = []

        while loop_count < self.MAX_LOOP_RETRIES:
            logger.info(f"--- Meta-Agent Cycle {loop_count + 1} (Fault Focus: {fault_source.upper()}) ---")
            
            # A. PLANNING PHASE
            # We only re-plan if the Planner is at fault or we don't have a plan yet.
            if fault_source == "planner" or current_plan is None:
                logger.info("Calling Planner...")
                current_plan = self.planner.generate_plan(
                    user_query=system_aware_query, # Inject System Base
                    chat_history=chat_history,
                    external_feedback=planner_feedback
                )
                
                iteration_log.append(IterationHistory(
                    iteration_number=loop_count + 1,
                    agent_name="Planner",
                    output_snapshot=current_plan.model_dump(),
                    feedback_received=planner_feedback
                ))
                # Reset Planner feedback after use
                planner_feedback = None

            # B. THINKING PHASE
            # Thinker always runs if we are in the loop, either to execute new plan or fix old draft.
            logger.info("Calling Thinker...")
            
            # If we are strictly fixing a Thinker error, we pass the previous draft to help it diff.
            previous_draft = final_thinker_output.draft_answer if final_thinker_output else None

            thinker_output = self.thinker.execute_plan(
                user_query=system_aware_query, # Inject System Base
                plan=current_plan,
                chat_history=chat_history,
                previous_draft=previous_draft,
                verifier_feedback=thinker_feedback
            )
            final_thinker_output = thinker_output
            
            iteration_log.append(IterationHistory(
                iteration_number=loop_count + 1,
                agent_name="Thinker",
                output_snapshot=thinker_output.model_dump(),
                feedback_received=thinker_feedback
            ))
            
            # Reset Thinker feedback after use
            thinker_feedback = None

            # C. VERIFICATION PHASE
            logger.info("Calling Verifier...")
            report = self.verifier.verify_response(
                user_query=system_aware_query, # Inject System Base
                thinker_output=thinker_output,
                plan=current_plan
            )
            final_verification_report = report
            
            iteration_log.append(IterationHistory(
                iteration_number=loop_count + 1,
                agent_name="Verifier",
                output_snapshot=report.model_dump(),
                feedback_received=None
            ))

            # D. DECISION GATE & SMART FAULT ATTRIBUTION
            if report.verification_status == VerificationStatus.PASS:
                logger.info("✅ Verification PASSED.")
                break
            
            elif loop_count < self.MAX_LOOP_RETRIES - 1:
                logger.warning(f"❌ Verification {report.verification_status}. Analyzing Fault...")
                
                # Determine who failed: Planner or Thinker?
                fault_source = self._determine_fault_source(report.critique)
                combined_feedback = f"Critique: {report.critique}. Fix: {report.suggested_correction}"
                
                if fault_source == "planner":
                    logger.info("🔎 Fault identified in PLAN. Feedback routed to Planner.")
                    planner_feedback = combined_feedback
                    # current_plan remains, but will be overwritten in next loop
                else:
                    logger.info("🧠 Fault identified in THINKING. Feedback routed to Thinker.")
                    thinker_feedback = combined_feedback
                    # We do NOT wipe current_plan; we keep it for the Thinker to retry execution
                
                loop_count += 1
            else:
                logger.error("Max Retries Reached. Proceeding with best effort.")
                break

        # --- 6. Final Output Generation ---
        response_text = self.explainer.explain(
            user_query=system_aware_query, # Inject System Base
            user_profile=user_profile,
            plan=current_plan,
            thinker_output=final_thinker_output,
            verification_report=final_verification_report
        )

        # --- 7. Memory & Cache ---
        # Store using the original RAW query to maintain clean user history without system injection artifacts.
        self.memory.process_realtime_interaction(user_id, query, response_text)
        store_cache(query, response_text, user_profile)
        
        # --- 8. Trace Logging ---
        self._log_trace(user_id, query, user_profile, current_plan, final_thinker_output, iteration_log)
        
        elapsed = time.time() - start_time
        logger.info(f"Response generated in {elapsed:.2f}s")
        return response_text

    def _determine_fault_source(self, critique: str) -> Literal["planner", "thinker"]:
        """
        Heuristic to decide if the feedback should go to Planner or Thinker.
        If the critique attacks the *strategy*, *steps*, or *missed intent*, it's Planner.
        If the critique attacks the *facts*, *hallucination*, or *detail*, it's Thinker.
        """
        critique_lower = critique.lower()
        
        # Keywords suggesting the Plan itself is flawed
        planner_keywords = [
            "plan", "strategy", "steps", "missing step", "order", 
            "didn't ask", "irrelevant approach", "user intent"
        ]
        
        for kw in planner_keywords:
            if kw in critique_lower:
                return "planner"
        
        # Default to Thinker (Execution error) for factual/synthesis issues
        return "thinker"

    def _log_trace(self, user_id, query, profile, plan, thinker_output, history):
        """Safe wrapper for trace logging to prevent main loop crash."""
        try:
            state_snapshot = GlobalState(
                user_id=user_id,
                query=query,
                status=AgentStatus.COMPLETED,
                user_profile=profile,
                plan=plan,
                thinker_output=thinker_output,
                iteration_log=history,
                memory_summary="See MemoryManager"
            )
            self.trace_logger.log_cycle(state_snapshot)
        except Exception as e:
            logger.error(f"Failed to log trace: {e}")