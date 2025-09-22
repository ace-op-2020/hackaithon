"""
MCP Protocol Models
Defines the data structures for Model Context Protocol communication
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class ActionType(str, Enum):
    """Types of actions that can be performed in the React app"""
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    WAIT = "wait"
    VALIDATE = "validate"
    EXTRACT = "extract"
    CUSTOM = "custom"

class Priority(str, Enum):
    """Priority levels for action steps"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionStep(BaseModel):
    """Individual action step in an automation sequence"""
    id: str = Field(..., description="Unique identifier for the step")
    action_type: ActionType = Field(..., description="Type of action to perform")
    description: str = Field(..., description="Human-readable description of the action")
    target: str = Field(..., description="CSS selector or element identifier")
    value: Optional[str] = Field(None, description="Value to input (for type actions)")
    priority: Priority = Field(Priority.MEDIUM, description="Priority level")
    estimated_duration: Optional[int] = Field(None, description="Estimated time in milliseconds")
    depends_on: Optional[List[str]] = Field(None, description="IDs of steps this depends on")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Conditions that must be met")

class MCPRequest(BaseModel):
    """Incoming request from the React chatbot"""
    user_request: str = Field(..., description="Natural language request from user")
    context: Optional[Dict[str, Any]] = Field(None, description="Current application context")
    current_page: Optional[str] = Field(None, description="Current page URL or identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences and settings")

class MCPResponse(BaseModel):
    """Response from the MCP server"""
    success: bool = Field(..., description="Whether the request was processed successfully")
    message: str = Field(..., description="Human-readable response message")
    action_steps: List[ActionStep] = Field(default_factory=list, description="List of action steps to execute")
    estimated_total_time: Optional[int] = Field(None, description="Total estimated time in milliseconds")
    confidence_score: Optional[float] = Field(None, description="Confidence in the action plan (0-1)")
    alternatives: Optional[List[str]] = Field(None, description="Alternative approaches")
    warnings: Optional[List[str]] = Field(None, description="Potential issues or warnings")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ValidationResult(BaseModel):
    """Result of action step validation"""
    valid: bool = Field(..., description="Whether the action steps are valid")
    feedback: str = Field(..., description="Feedback on the validation")
    issues: Optional[List[str]] = Field(None, description="List of identified issues")
    suggestions: Optional[List[str]] = Field(None, description="Suggested improvements")

class ExecutionStatus(BaseModel):
    """Status of action execution"""
    step_id: str = Field(..., description="ID of the executed step")
    status: str = Field(..., description="Status: success, failed, in_progress")
    message: Optional[str] = Field(None, description="Status message")
    timestamp: str = Field(..., description="Execution timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
