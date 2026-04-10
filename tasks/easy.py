from typing import Dict, Any
from tasks.scoring import to_open_interval

def easy_task() -> Dict[str, Any]:
    task_definition = {
        "name": "easy",
        "description": "User requests a password reset in English. Categorize correctly and assign appropriate tags.",
        "initial_ticket": {
            "ticket_id": "T-100",
            "user_id": "u100",
            "message": "Hi, I forgot my password and cannot log into my account. Please help me reset it.",
            "language": "en",
            "priority": "low"
        }
    }
    return task_definition

def grade_easy(state: Dict[str, Any]) -> float:
    """
    Deterministic dense grading for easy task.
    Goal: quick and correct triage for password reset tickets.
    """
    reward = 0.0
    history = state.get("ticket_history", [])
    agent_actions = [entry for entry in history if entry.get("role") == "agent"]
    system_events = [entry for entry in history if entry.get("role") == "system"]

    if state.get("system_category") == "account":
        reward += 0.45

    tags = [str(tag).lower() for tag in state.get("system_tags", [])]
    if "password_reset" in tags:
        reward += 0.35

    used_categorize = any(action.get("action_type") == "categorize_ticket" for action in agent_actions)
    if used_categorize:
        reward += 0.1

    off_topic_actions = sum(
        1
        for action in agent_actions
        if action.get("action_type") in {"query_database", "search_knowledge_base", "escalate_ticket"}
    )
    invalid_actions = sum(
        1
        for event in system_events
        if "failed" in str(event.get("content", "")).lower()
    )
    reward -= min(0.2, off_topic_actions * 0.05)
    reward -= min(0.12, invalid_actions * 0.04)

    if off_topic_actions == 0 and len(agent_actions) <= 2 and used_categorize:
        reward += 0.1

    return to_open_interval(max(0.0, min(1.0, reward)))
