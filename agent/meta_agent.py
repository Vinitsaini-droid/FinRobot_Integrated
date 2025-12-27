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
    PlanSchema,
    MemoryType 
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
        
        # 3. System Base Instructions Removed
        # Logic for loading system_base.txt has been deprecated as per architectural update.
        
        logger.info("MetaAgent initialized with Smart P-T-V-E Architecture (System Base Disabled).")

    def generate_response(self, user_id: str, query: str) -> str:
        """
        Main entry point for the agentic reasoning loop.
        """
        start_time = time.time()
        
        # --- 1. User Profile Update ---
        self.memory.check_and_update_profile_pre_planning(user_id, query)
        user_profile = self.memory.get_profile(user_id)

        # --- 2. Semantic Cache Check ---
        # Note: Cache uses the raw query to ensure hits match user intent.
        cached_response = retrieve_cache(query, user_profile)
        if cached_response:
            logger.info("Semantic Cache HIT.")
            self.memory.process_realtime_interaction(user_id, query, cached_response)
            return cached_response

        # --- 3. Context Retrieval ---
        chat_history = self.memory.get_immediate_context(user_id)
        
        # --- 3.5. Long-Term Memory Retrieval (Smart Injection) ---
        # Retrieve relevant facts explicitly to bridge the episodic gap.
        relevant_memories = self.memory.retrieve_relevant(
            query=query,
            user_id=user_id,
            limit=5,
            memory_type=MemoryType.FACT
        )
        
        # Format facts for injection
        facts_block = ""
        if relevant_memories:
            fact_list = [f"- {m.content}" for m in relevant_memories]
            facts_block = "RELEVANT LONG-TERM MEMORY:\n" + "\n".join(fact_list) + "\n\n"
            logger.info(f"Injected {len(relevant_memories)} long-term facts into context.")

        # --- 4. Prepare System-Aware Context ---
        # Structure: [Long-Term Facts] -> [User Query]
        # Removed system_base injection.
        system_aware_query = f"{facts_block}[USER QUERY START]\n{query}\n[USER QUERY END]"
        
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
            if fault_source == "planner" or current_plan is None:
                logger.info("Calling Planner...")
                current_plan = self.planner.generate_plan(
                    user_query=system_aware_query, # Inject Facts + Query
                    chat_history=chat_history,
                    external_feedback=planner_feedback
                )
                
                iteration_log.append(IterationHistory(
                    iteration_number=loop_count + 1,
                    agent_name="Planner",
                    output_snapshot=current_plan.model_dump(),
                    feedback_received=planner_feedback
                ))
                planner_feedback = None

            # B. THINKING PHASE
            logger.info("Calling Thinker...")
            previous_draft = final_thinker_output.draft_answer if final_thinker_output else None

            thinker_output = self.thinker.execute_plan(
                user_query=system_aware_query, # Inject Facts + Query
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
            thinker_feedback = None

            # C. VERIFICATION PHASE
            logger.info("Calling Verifier...")
            report = self.verifier.verify_response(
                user_query=system_aware_query, # Inject Facts + Query
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
                
                fault_source = self._determine_fault_source(report.critique)
                combined_feedback = f"Critique: {report.critique}. Fix: {report.suggested_correction}"
                
                if fault_source == "planner":
                    logger.info("🔎 Fault identified in PLAN. Feedback routed to Planner.")
                    planner_feedback = combined_feedback
                else:
                    logger.info("🧠 Fault identified in THINKING. Feedback routed to Thinker.")
                    thinker_feedback = combined_feedback
                
                loop_count += 1
            else:
                logger.error("Max Retries Reached. Proceeding with best effort.")
                break

        # --- 6. Final Output Generation ---
        response_text = self.explainer.explain(
            user_query=system_aware_query, 
            user_profile=user_profile,
            plan=current_plan,
            thinker_output=final_thinker_output,
            verification_report=final_verification_report
        )

        # --- 7. Memory & Cache ---
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
        """
        critique_lower = critique.lower()
        planner_keywords = [
            "plan", "strategy", "steps", "missing step", "order", 
            "didn't ask", "irrelevant approach", "user intent"
        ]
        
        for kw in planner_keywords:
            if kw in critique_lower:
                return "planner"
        
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