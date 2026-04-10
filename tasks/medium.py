from typing import Dict, Any
from tasks.scoring import to_open_interval

def medium_task() -> Dict[str, Any]:
    task_definition = {
        "name": "medium",
        "description": "User complains about a broken hardware part. Query database to find recent orders, identify 'GPU', and reply offering a standard refund policy with the correct serial number.",
        "initial_ticket": {
            "ticket_id": "T-101",
            "user_id": "u100",
            "message": "My screen is flickering uncontrollably since I installed the new part I ordered. Can I get a replacement or what's the deal?",
            "language": "en",
            "priority": "medium"
        }
    }
    return task_definition

def grade_medium(state: Dict[str, Any]) -> float:
    """
    Deterministic dense grading for medium task.
    Goal: investigate order data and provide a resolution-backed customer reply.
    """
    reward = 0.0
    history = state.get("ticket_history", [])
    agent_actions = [entry for entry in history if entry.get("role") == "agent"]
    system_events = [entry for entry in history if entry.get("role") == "system"]

    db_queried = any("sn-gpu-8819" in str(event.get("content", "")).lower() for event in system_events)
    if db_queried:
        reward += 0.35

    query_quality = any(
        action.get("action_type") == "query_database"
        and any(token in str(action.get("query", "")).lower() for token in ("order", "gpu", "serial", "u100", "alice"))
        for action in agent_actions
    )
    if query_quality:
        reward += 0.1

    if state.get("system_category") in {"hardware", "technical"}:
        reward += 0.1

    reply_messages = [
        action for action in agent_actions
        if action.get("action_type") == "reply_to_user"
    ]
    if reply_messages:
        reply = reply_messages[-1]
        content = str(reply.get("message", "")).lower()
        language = str(reply.get("language", "")).lower()
        if language == "en":
            reward += 0.1
        if "sn-gpu-8819" in content:
            reward += 0.15
        if "refund" in content or "replacement" in content or "replace" in content:
            reward += 0.15
        if "sorry" in content or "apolog" in content:
            reward += 0.05

    escalations = sum(1 for action in agent_actions if action.get("action_type") == "escalate_ticket")
    invalid_actions = sum(
        1
        for event in system_events
        if "failed" in str(event.get("content", "")).lower()
    )
    reward -= min(0.1, escalations * 0.05)
    reward -= min(0.12, invalid_actions * 0.04)

    if db_queried and reply_messages and len(agent_actions) <= 5:
        reward += 0.05

    return to_open_interval(max(0.0, min(1.0, reward)))
