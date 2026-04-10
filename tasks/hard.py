from typing import Dict, Any
from tasks.scoring import to_open_interval


def hard_task() -> Dict[str, Any]:
    task_definition = {
        "name": "hard",
        "description": "Japanese enterprise client reports a complex API 500 error. Search KB, escalate with Kaizen report structure including bug ID #ERR-7782, and reply to the user using formal Japanese Keigo.",
        "initial_ticket": {
            "ticket_id": "T-102",
            "user_id": "u101",
            "message": "API\u3092\u547c\u3073\u51fa\u3059\u3068500\u30a8\u30e9\u30fc\u304c\u767a\u751f\u3057\u7d9a\u3051\u3066\u3044\u307e\u3059\u3002\u30d0\u30c3\u30c1\u51e6\u7406\u304c\u5b8c\u5168\u306b\u6b62\u307e\u3063\u3066\u3044\u307e\u3059\u3002\u81f3\u6025\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
            "language": "ja",
            "priority": "high",
        },
    }
    return task_definition


def grade_hard(state: Dict[str, Any]) -> float:
    """
    Deterministic dense grading for hard task.
    Goal: identify known incident, escalate with structured notes, respond in formal Japanese.
    """
    reward = 0.0
    history = state.get("ticket_history", [])
    agent_actions = [entry for entry in history if entry.get("role") == "agent"]
    system_events = [entry for entry in history if entry.get("role") == "system"]

    kb_searched = any("err-7782" in str(event.get("content", "")).lower() for event in system_events)
    if kb_searched:
        reward += 0.25

    kb_action_quality = any(
        action.get("action_type") == "search_knowledge_base"
        and any(token in str(action.get("topic", "")).lower() for token in ("api 500", "err-7782", "kaizen", "enterprise"))
        for action in agent_actions
    )
    if kb_action_quality:
        reward += 0.1

    escalation_actions = [
        action for action in agent_actions
        if action.get("action_type") == "escalate_ticket"
    ]
    if escalation_actions:
        notes = str(escalation_actions[-1].get("engineering_notes", "")).lower()
        if "err-7782" in notes:
            reward += 0.15
        if "root cause" in notes:
            reward += 0.08
        if "impact" in notes:
            reward += 0.06
        if "mitigation" in notes or "workaround" in notes:
            reward += 0.04
        if "eta" in notes:
            reward += 0.02

    reply_actions = [
        action for action in agent_actions
        if action.get("action_type") == "reply_to_user"
    ]
    if reply_actions:
        reply = reply_actions[-1]
        content = str(reply.get("message", ""))
        lower_content = content.lower()
        lang = str(reply.get("language", "")).lower()

        if lang == "ja":
            reward += 0.08
        if "\u7533\u3057\u8a33\u3054\u3056\u3044\u307e\u305b\u3093" in content:
            reward += 0.1
        if "\u5f0a\u793e" in content:
            reward += 0.06
        if "\u78ba\u8a8d" in content or "confirm" in lower_content:
            reward += 0.03
        if "err-7782" in lower_content:
            reward += 0.03

    action_types = [action.get("action_type") for action in agent_actions]
    if all(required in action_types for required in ("search_knowledge_base", "escalate_ticket", "reply_to_user")):
        kb_index = action_types.index("search_knowledge_base")
        escalate_index = action_types.index("escalate_ticket")
        reply_index = action_types.index("reply_to_user")
        if kb_index <= escalate_index <= reply_index:
            reward += 0.07

    invalid_actions = sum(
        1
        for event in system_events
        if "failed" in str(event.get("content", "")).lower()
    )
    reward -= min(0.12, invalid_actions * 0.04)

    return to_open_interval(max(0.0, min(1.0, reward)))
