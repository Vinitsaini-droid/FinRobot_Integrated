# agent/meta_agent.py
import sys
import time
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

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
    MemoryType,
    ExplainerOutput
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
        
        logger.info("MetaAgent initialized with Smart P-T-V-E Architecture.")

    def generate_response(self, user_id: str, query: str) -> str:
        """
        Main entry point for the agentic reasoning loop.
        """
        start_time = time.time()
        
        # --- 1. User Profile Update ---
        self.memory.check_and_update_profile_pre_planning(user_id, query)
        user_profile = self.memory.get_profile(user_id)

        # --- 2. Semantic Cache Check (Artifact Retrieval) ---
        cached_artifacts = retrieve_cache(query)
        
        # Variables to hold final state (either from cache or computation)
        current_plan: Optional[PlanSchema] = None
        final_thinker_output: Optional[ThinkerOutput] = None
        final_verification_report: Optional[VerificationReport] = None
        iteration_log: List[IterationHistory] = []
        is_cache_hit = False

        if cached_artifacts:
            logger.info("Semantic Cache HIT - Retrieving Reasoning Artifacts.")
            current_plan = cached_artifacts["plan"]
            final_thinker_output = cached_artifacts["thinker_output"]
            final_verification_report = cached_artifacts["verification_report"]
            is_cache_hit = True

        # --- 3. Context Retrieval & Formatting ---
        # 3a. Get raw chat history
        raw_history = self.memory.get_immediate_context(user_id)
        
        # Convert List[Dict] to String representation
        chat_history_str = ""
        if isinstance(raw_history, list):
            for msg in raw_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                chat_history_str += f"{role.upper()}: {content}\n"
        else:
            chat_history_str = str(raw_history)
        
        # 3b. Long-Term Memory Retrieval
        relevant_memories = self.memory.retrieve_relevant(
            query=query,
            user_id=user_id,
            limit=5,
            memory_type=MemoryType.FACT,
            score_threshold=0.70
        )
        
        # Format facts for injection
        facts_block = ""
        if relevant_memories:
            fact_list = [f"- {m.content}" for m in relevant_memories]
            facts_block = "RELEVANT LONG-TERM MEMORY (BACKGROUND FACTS):\n" + "\n".join(fact_list) + "\n\n"
            logger.info(f"Retrieved {len(relevant_memories)} long-term facts.")

        # --- 4. Prepare Consolidated Context (String) ---
        full_context_str = f"{facts_block}CHAT HISTORY:\n{chat_history_str}"

        # --- 5. Cognitive Loop (Only if NOT a Cache Hit) ---
        if not is_cache_hit:
            loop_count = 0
            
            # Feedback Containers
            planner_feedback: Optional[str] = None
            thinker_feedback: Optional[str] = None
            
            # Fault Tracking
            fault_source: Literal["planner", "thinker", "none"] = "planner" 
            
            while loop_count < self.MAX_LOOP_RETRIES:
                logger.info(f"--- Meta-Agent Cycle {loop_count + 1} (Fault Focus: {fault_source.upper()}) ---")
                
                # A. PLANNING PHASE
                if fault_source == "planner" or current_plan is None:
                    logger.info("Calling Planner...")
                    current_plan = self.planner.generate_plan(
                        user_query=query,          
                        chat_history=full_context_str, # Passed as str
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
                    user_query=query,          
                    plan=current_plan,
                    chat_history=full_context_str, # Passed as str
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
                rag_evidence = thinker_output.retrieved_context if thinker_output.retrieved_context else ""
                combined_evidence = f"{facts_block}\n### RAG EVIDENCE ###\n{rag_evidence}"
                report = self.verifier.verify_response(
                    user_query=query, 
                    thinker_output=thinker_output,
                    plan=current_plan,
                    evidence_context=combined_evidence
                )
                final_verification_report = report
                
                iteration_log.append(IterationHistory(
                    iteration_number=loop_count + 1,
                    agent_name="Verifier",
                    output_snapshot=report.model_dump(),
                    feedback_received=None
                ))

                # D. DECISION GATE
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

        # --- 6. Final Output Generation (Explainer) ---
        # The Explainer now runs for BOTH fresh generations and cache hits.
        # It takes the artifacts (Plan, Thinking, Report) and renders them according to the CURRENT user profile.
        final_explainer_output = self.explainer.generate_explanation(
            user_query=query,
            user_profile=user_profile,
            plan=current_plan,
            thinker_output=final_thinker_output,
            verification_report=final_verification_report
        )

        depth_mode = user_profile.explanation_depth if user_profile else "detailed"
        response_text = self.explainer.render_explanation(final_explainer_output, depth_mode)

        # --- 7. Memory & Cache ---
        self.memory.process_realtime_interaction(user_id, query, response_text)
        
        # Only store in cache if this was a fresh generation (to avoid redundant writes)
        if not is_cache_hit and final_verification_report.verification_status == VerificationStatus.PASS:
            store_cache(
                query=query, 
                plan=current_plan,
                thinker_output=final_thinker_output,
                verification_report=final_verification_report
            )
        
        # --- 8. Trace Logging ---
        self._log_trace(
            user_id=user_id, 
            query=query, 
            profile=user_profile, 
            plan=current_plan, 
            thinker_output=final_thinker_output, 
            verification_report=final_verification_report,
            history=iteration_log,
            final_explainer_output=final_explainer_output
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Response generated in {elapsed:.2f}s (Cache Hit: {is_cache_hit})")
        return response_text

    def _determine_fault_source(self, critique: str) -> Literal["planner", "thinker"]:
        """Heuristic to decide if the feedback should go to Planner or Thinker."""
        critique_lower = critique.lower()
        planner_keywords = [
            "plan", "strategy", "steps", "missing step", "order", 
            "didn't ask", "irrelevant approach", "user intent"
        ]
        for kw in planner_keywords:
            if kw in critique_lower:
                return "planner"
        return "thinker"

    def _log_trace(
        self, user_id: str, query: str, profile, plan, thinker_output, 
        verification_report, history, final_explainer_output: Optional[ExplainerOutput]
    ):
        try:
            state_snapshot = GlobalState(
                user_id=user_id,
                query=query,
                status=AgentStatus.COMPLETED,
                user_profile=profile,
                plan=plan,
                thinker_output=thinker_output,
                verification_report=verification_report,
                iteration_log=history,
                final_response=final_explainer_output,
                memory_summary="See MemoryManager"
            )
            self.trace_logger.log_cycle(state_snapshot)
        except Exception as e:
            logger.error(f"Failed to log trace: {e}")