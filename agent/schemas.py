# agent/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Any, Dict, Union
from enum import Enum
import datetime

# --- CONFIGURATION ---
class BaseSchema(BaseModel):
    """Base class to ensure production-grade configuration across all models."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True, # Essential for snapshots of complex objects
        populate_by_name=True,
        use_enum_values=True # Ensures JSON serialization uses the string value
    )

# --- ENUMS ---

class AgentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    PLAN_REVIEW = "plan_review"
    THINKING = "thinking"
    VERIFYING = "verifying"
    EXPLAINING = "explaining"
    COMPLETED = "completed"
    ERROR = "error"

class ActionType(str, Enum):
    RETRIEVE = "retrieve"
    REASON = "reason"
    VERIFY = "verify"

class VerificationStatus(str, Enum):
    PASS = "PASS"
    RISKY = "RISKY"
    FAIL = "FAIL"

class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    GOAL = "goal"
    CRITIQUE = "critique"
    EPISODIC = "episodic"

class PlanValidity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"

# --- DATA MODELS (DTOs) ---

class Chunk(BaseModel):
    """
    Represents a single retrieved chunk of data from the vector DB.
    """
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryItem(BaseSchema):
    id: str
    content: str
    memory_type: MemoryType
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatSummary(BaseSchema):
    summary: str
    key_facts: List[str]

class SchemaCritique(BaseSchema):
    critique: str = Field(..., description="Specific feedback on what is wrong or different with the JSON.")
    suggestions: str = Field(..., description="How the Model should fix it.")


# --- PLANNER OUTPUTS ---

class PlanStep(BaseSchema):
    step_id: int
    action: ActionType
    query: str = Field(..., description="Exact purpose of this step")
    status: Literal["failed", "pending", "completed"]

class PlanSchema(BaseSchema):
    thought_process: str = Field(..., description="Acts as a thought space for reasoning models")
    steps: List[PlanStep]
    risk_level: Literal["low", "medium", "high"]
    requires_compliance: bool
    xai_notes: str = Field(
        None, 
        description="Explanation of why this plan was chosen"
    )

class PlanCritique(BaseSchema):
    validity: PlanValidity
    critique: str = Field(..., description="Specific feedback on what is wrong with the plan.")
    suggestions: str = Field(..., description="How the planner should fix it.")

# --- THINKER OUTPUTS ---

class ReasoningTrace(BaseSchema):
    step_id: int
    action: ActionType
    query: str
    thought: str = Field(..., description="Internal monologue or reasoning logic")
    observation: Optional[str] = Field(None, description="Result of the action or retrieval")

class ThinkerOutput(BaseSchema):
    thought_process: str = Field(..., description="Acts as a thought space for reasoning models")
    draft_answer: str = Field(..., description="Concise answer with inline citations")
    key_facts_extracted: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    retrieved_context: Optional[str] = Field(None, description="The raw evidence text used to generate the answer")
    missing_information: Optional[str] = Field(
        None, 
        description="Explicit statement of what is missing or if evidence was insufficient"
    )
    reasoning_traces: List[ReasoningTrace] = Field(
        default_factory=list, 
        description="Structured execution trace (Thought -> Observation) creating the answer."
    )
    xai_trace: str = Field(
        None, 
        description="Brief reasoning of how evidence was combined"
    )

# --- VERIFIER OUTPUTS ---

class XAICitation(BaseSchema):
    claim: str = Field(..., description="Exact claim or sentence being evaluated")
    evidence_ids: List[str] = Field(..., description="Source IDs used for verification")
    verdict: str = Field(..., description="SUPPORTED | WEAK | UNSUPPORTED | CONTRADICTED")

class VerificationReport(BaseSchema):
    thought_process: str = Field(..., description="Acts as a thought space for reasoning models")
    verification_status: VerificationStatus
    critique: str = Field(..., description="Exact explanation of errors or confirmation")
    suggested_correction: Optional[str] = Field(None, description="Corrected sentence or logic")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    xai_citations: List[XAICitation] = Field(
        default_factory=list, 
        description="Evidence grounding for specific claims."
    )

# --- EXPLAINER OUTPUTS ---

class ExplainerOutput(BaseSchema):
    thought_process: str = Field(..., description="Acts as a thought space for reasoning models")
    explanation: str = Field(..., description="The final, personalized response text with inline citations for the user.")
    plan: PlanSchema = Field(None, description="The plan executed to derive this answer.")
    reasoning_traces: List[ReasoningTrace] = Field(
        default_factory=list, 
        description="The structured chain of thought and observations."
    )
    citations: List[XAICitation] = Field(
        default_factory=list, 
        description="Verified evidence grounding the explanation."
    )
    meta_data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata including tone, depth, and risk checks."
    )
  
# --- USER PROFILE ---

class UserProfileSchema(BaseSchema):
    user_id: str
    risk_tolerance: Literal['low', 'medium', 'high'] = 'medium'
    explanation_depth: Literal['simple', 'detailed', 'technical'] = 'detailed'
    preferences: List[str] = Field(default_factory=list)
    prior_misunderstandings_summary: Optional[str] = None
    style_preference: Literal['formal', 'casual', 'concise'] = 'formal'

# --- Orchestration & Logging ---

class IterationHistory(BaseSchema):
    iteration_number: int
    agent_name: str # "Planner" or "Thinker"
    output_snapshot: Any 
    feedback_received: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

class GlobalState(BaseSchema):
    user_id: str
    query: str
    status: AgentStatus = AgentStatus.IDLE
    
    # Context
    user_profile: Optional[UserProfileSchema] = None
    memory_summary: Optional[str] = None
    iteration_log: List[IterationHistory] = Field(
        default_factory=list, 
        description="Chronological log of all agent iterations and critiques"
    )
    # Execution Artifacts
    plan: Optional[PlanSchema] = None
    thinker_output: Optional[ThinkerOutput] = None
    verification_report: Optional[VerificationReport] = None
    
    # Final Output
    final_response: Optional[ExplainerOutput] = None