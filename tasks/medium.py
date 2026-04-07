from typing import Dict, Any

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
    # Deterministic grading for medium task
    reward = 0.0
    history = state.get("ticket_history", [])
    
    # Check if they queried the DB for the user's order to get the Serial SN-GPU-8819
    db_queried = False
    for msg in history:
        if msg.get("role") == "system" and "SN-GPU-8819" in msg.get("content", ""):
            db_queried = True
            break
            
    if db_queried:
        reward += 0.4
        
    replied_correctly = False
    for msg in history:
        if msg.get("role") == "agent" and msg.get("action_type") == "reply_to_user":
            content = str(msg.get("message", "")).lower()
            language = str(msg.get("language", "")).lower()
            if language == "en" and "sn-gpu-8819" in content and ("refund" in content or "replace" in content):
                replied_correctly = True
                break

    if replied_correctly:
        reward += 0.6
        
    return reward
