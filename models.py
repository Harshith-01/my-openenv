from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator

class Action(BaseModel):
    action_type: Literal[
        "query_database",
        "search_knowledge_base",
        "categorize_ticket",
        "reply_to_user",
        "escalate_ticket",
        "end_turn"
    ] = Field(description="The type of action to perform.")
    
    # Parameters for query_database
    query: Optional[str] = Field(None, max_length=300, description="SQL-like or semantic query to search the user database.")
    
    # Parameters for search_knowledge_base
    topic: Optional[str] = Field(None, max_length=300, description="Topic or keywords to search in the knowledge base.")
    
    # Parameters for categorize_ticket
    category: Optional[str] = Field(None, max_length=32, description="The category to assign to the ticket (e.g., 'account', 'hardware', 'software').")
    tags: Optional[List[str]] = Field(None, description="Tags to assign to the ticket.")
    
    # Parameters for reply_to_user
    message: Optional[str] = Field(None, max_length=800, description="The message to send to the user.")
    language: Optional[str] = Field(None, max_length=5, description="The language of the message (e.g., 'en', 'ja').")
    
    # Parameters for escalate_ticket
    engineering_notes: Optional[str] = Field(None, max_length=800, description="Notes to provide to the engineering department upon escalation.")

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().lower()
        if normalized not in {"en", "ja"}:
            raise ValueError("language must be 'en' or 'ja'")
        return normalized

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value

        normalized_tags: List[str] = []
        for raw_tag in value:
            tag = str(raw_tag).strip().lower()
            if not tag:
                continue
            if len(tag) > 40:
                raise ValueError("tag length must be <= 40 characters")
            if tag not in normalized_tags:
                normalized_tags.append(tag)

        if len(normalized_tags) > 5:
            raise ValueError("no more than 5 tags are allowed")
        return normalized_tags

class Observation(BaseModel):
    current_ticket: Optional[Dict[str, Any]] = Field(None, description="Details of the current active ticket.")
    last_action_result: Optional[str] = Field(None, description="Result or feedback from the last action performed.")
    ticket_history: List[Dict[str, Any]] = Field(default_factory=list, description="Action and system event history for the current ticket.")
    system_tags: List[str] = Field(default_factory=list, description="Tags currently applied to the ticket.")
    system_category: Optional[str] = Field(None, description="Category currently applied to the ticket.")

class State(BaseModel):
    task_name: str = Field(description="The active task name.")
    ticket_data: Dict[str, Any] = Field(description="Underlying mock database representation of the ticket.")
    observation: Observation = Field(description="The current observation seen by the agent.")
    step_count: int = Field(0, description="Number of steps taken in the current episode.")
    done: bool = Field(False, description="Whether the episode is finished.")
    reward: float = Field(0.0, description="Accumulated reward so far.")
