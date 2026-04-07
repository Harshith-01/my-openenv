from typing import Dict, Any

def hard_task() -> Dict[str, Any]:
    task_definition = {
        "name": "hard",
        "description": "Japanese enterprise client reports a complex API 500 error. Search KB, escalate with Kaizen report structure including bug ID #ERR-7782, and reply to the user using formal Japanese Keigo.",
        "initial_ticket": {
            "ticket_id": "T-102",
            "user_id": "u101",
            "message": "APIを呼び出すと500エラーが発生し続けています。バッチ処理が完全に止まっています。至急確認してください。",  # "API calls keep resulting in 500 errors. Batch processing is completely halted. Please check urgently."
            "language": "ja",
            "priority": "high"
        }
    }
    return task_definition

def grade_hard(state: Dict[str, Any]) -> float:
    # Deterministic grading for hard task
    reward = 0.0
    history = state.get("ticket_history", [])
    
    # 1. Did they search the KB and find the ERR-7782 bug?
    kb_searched = False
    for msg in history:
        if msg.get("role") == "system" and "ERR-7782" in msg.get("content", ""):
            kb_searched = True
            break
            
    if kb_searched:
        reward += 0.3
        
    # 2. Did they escalate cleanly using correct bug ID?
    escalated_correctly = False
    for msg in history:
        if msg.get("role") == "agent" and msg.get("action_type") == "escalate_ticket":
            notes = str(msg.get("engineering_notes", ""))
            if "ERR-7782" in notes and ("root cause" in notes.lower() or "impact" in notes.lower()):
                escalated_correctly = True
                break
                
    if escalated_correctly:
        reward += 0.3
        
    # 3. Did they reply in Japanese using formal honorifics (Keigo)?
    replied_correctly = False
    for msg in history:
        if msg.get("role") == "agent" and msg.get("action_type") == "reply_to_user":
            content = str(msg.get("message", ""))
            lang = str(msg.get("language", ""))
            # Check for standard polite apology in Japanese and proper self-reference
            if lang == "ja" and "申し訳ございません" in content and "弊社" in content:
                replied_correctly = True
                break
                
    if replied_correctly:
        reward += 0.4
        
    return reward
