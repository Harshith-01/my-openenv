from typing import Dict, Any

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
    # Deterministic grading for easy task
    reward = 0.0
    
    # Needs to end turn with categorization
    if state.get("system_category") == "account":
        reward += 0.5
        
    tags = state.get("system_tags", [])
    if "password_reset" in tags:
        reward += 0.5
        
    return reward
