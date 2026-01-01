# evaluation/ragas_runner.py
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from google.genai import types

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from utils.llm_client import (
    generate_secondary, 
    get_embedding, 
    generate_schema_critique, 
    repair_json_with_llm
)
from utils.json_utils import (
    safe_json_load, 
    validate_json_structure, 
    repair_json
)
from utils.similarity import cosine_similarity
from utils.logger import get_logger

logger = get_logger("RAGAS_RUNNER")

class MetricResult(BaseModel):
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

class RagasRunner:
    def __init__(self):
        # Google Type Schema for the API call hint
        self.schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "metric_name": types.Schema(type=types.Type.STRING),
                "score": types.Schema(type=types.Type.NUMBER),
                "reasoning": types.Schema(type=types.Type.STRING),
            },
            required=["metric_name", "score", "reasoning"]
        )

    def _robust_validate_and_parse(self, raw_text: str, schema_str: str) -> Optional[Dict[str, Any]]:
        """
        Helper: Executes the Multi-Stage Repair Pipeline.
        1. Validate Raw
        2. Repair Algorithmic (json_repair)
        3. Repair LLM (repair_json_with_llm)
        """
        # Stage 1: Standard Validation
        if validate_json_structure(raw_text, schema_str):
            return safe_json_load(raw_text)

        # Stage 2: Algorithmic Repair
        logger.debug("Stage 1 failed. Attempting algorithmic repair (repair_json)...")
        repaired_text = repair_json(raw_text)
        if repaired_text and validate_json_structure(repaired_text, schema_str):
            return safe_json_load(repaired_text)

        # Stage 3: LLM Repair
        logger.warning("Stage 2 failed. Attempting LLM-based repair (repair_json_with_llm)...")
        llm_repaired_text = repair_json_with_llm(schema_str, raw_text)
        if llm_repaired_text and validate_json_structure(llm_repaired_text, schema_str):
            return safe_json_load(llm_repaired_text)
        
        return None

    def _call_metric(self, sys_p: str, usr_p: str) -> MetricResult:
        """
        Executes a metric evaluation with self-healing JSON enforcement.
        """
        max_retries = 3
        # Generate strict JSON schema string from Pydantic model
        expected_schema_str = json.dumps(MetricResult.model_json_schema())
        
        current_usr_p = usr_p

        for attempt in range(max_retries):
            try:
                # 1. Generate Response
                res = generate_secondary(sys_p, current_usr_p, response_schema=self.schema)
                
                # 2. Multi-Stage Validation & Repair
                data = self._robust_validate_and_parse(res, expected_schema_str)

                if data:
                    return MetricResult.model_validate(data)
                
                # --- REPAIR LOOP START (Critique Fallback) ---
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: All repair stages failed. Initiating critique loop.")
                
                # Generate Critique
                critique_obj = generate_schema_critique(
                    expected_schema_str=expected_schema_str,
                    received_output_str=res
                )
                
                if critique_obj:
                    # Update Prompt with Feedback
                    current_usr_p = (
                        f"{usr_p}\n\n"
                        f"### PREVIOUS ATTEMPT FAILED ###\n"
                        f"Your previous JSON output was invalid.\n"
                        f"Critique: {critique_obj.critique}\n"
                        f"Correction Instructions: {critique_obj.suggestions}\n"
                        f"Please try again, strictly adhering to the schema."
                    )
                else:
                    current_usr_p = f"{usr_p}\n\nPlease ensure strict JSON formatting."
                
                # --- REPAIR LOOP END ---
                
            except Exception as e:
                logger.error(f"Ragas Metric Failure (Attempt {attempt}): {e}")

        return MetricResult(metric_name="error", score=0.0, reasoning="Parsing failure after retries.")

    def evaluate_faithfulness(self, context: List[str], answer: str) -> MetricResult:
        prompt = f"Is the answer grounded in context? Context: {' '.join(context)} | Answer: {answer}"
        return self._call_metric("Evaluation: Faithfulness (0-1)", prompt)

    def evaluate_answer_relevance(self, query: str, answer: str, chat_history: Optional[str] = None) -> MetricResult:
        context_str = f"Chat History: {chat_history}\n" if chat_history else ""
        prompt = (
            f"Determine if the answer is relevant to the query given the history.\n"
            f"{context_str}"
            f"Query: {query} | Answer: {answer}"
        )
        return self._call_metric("Evaluation: Relevance (0-1)", prompt)

    def evaluate_semantic_similarity(self, query: str, answer: str) -> MetricResult:
        try:
            v1, v2 = get_embedding(query), get_embedding(answer)
            score = cosine_similarity(v1, v2) if v1 and v2 else 0.0
            return MetricResult(metric_name="semantic_similarity", score=score, reasoning="Vector cosine distance.")
        except Exception as e:
            logger.error(f"Sim-Eval Error: {e}")
            return MetricResult(metric_name="semantic_similarity", score=0.0, reasoning="Vector error.")

    def run_full_evaluation(self, query: str, context: List[str], answer: str, chat_history: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Initiating RAG evaluation cycle...")
        
        f = self.evaluate_faithfulness(context, answer)
        r = self.evaluate_answer_relevance(query, answer, chat_history)
        s = self.evaluate_semantic_similarity(query, answer)
        
        overall = (f.score * 0.5) + (r.score * 0.3) + (s.score * 0.2)
        
        return {
            "metrics": {"faithfulness": f.dict(), "relevance": r.dict(), "semantic": s.dict()},
            "overall_score": round(overall, 4)
        }