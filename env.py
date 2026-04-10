from typing import Tuple, Dict, Any, Optional
from models import Action, Observation, State
from db import customer_db, knowledge_base
from tasks import get_task
from tasks.easy import grade_easy
from tasks.medium import grade_medium
from tasks.hard import grade_hard


SUSPICIOUS_PAYLOAD_PATTERNS = (
    "rm -rf",
    "del /f /q",
    "format c:",
    "powershell",
    "cmd /c",
    "bash -c",
    "os.system",
    "subprocess",
    "__import__(",
    "drop table",
    "truncate table",
    "../",
    "..\\",
    "<script",
    "wget ",
    "curl ",
)

MAX_STEPS = 8
INVALID_ACTION_PENALTY = 0.08
REPEATED_ACTION_PENALTY = 0.05
STRICT_MIN_SCORE = 0.01
STRICT_MAX_SCORE = 0.99
TASK_COMPLETION_THRESHOLDS = {
    "easy": 0.92,
    "medium": 0.9,
    "hard": 0.9,
}


class SupportEnv:
    """
    OpenEnv-compatible environment for Multi-lingual Support Ticket Escalation.
    """
    def __init__(self):
        self._state: Optional[State] = None

    def _contains_suspicious_payload(self, action: Action) -> bool:
        payload = " ".join(
            [
                action.query or "",
                action.topic or "",
                action.category or "",
                action.message or "",
                action.engineering_notes or "",
                " ".join(action.tags or []),
            ]
        ).lower()
        return any(pattern in payload for pattern in SUSPICIOUS_PAYLOAD_PATTERNS)

    def _to_open_interval_score(self, value: float) -> float:
        clamped = max(0.0, min(1.0, float(value)))
        if clamped <= 0.0:
            return STRICT_MIN_SCORE
        if clamped >= 1.0:
            return STRICT_MAX_SCORE
        return clamped

    def _is_repeated_action(self, action: Action) -> bool:
        history = self._state.observation.ticket_history if self._state else []
        for event in reversed(history):
            if event.get("role") == "agent":
                return (
                    event.get("action_type") == action.action_type
                    and event.get("query", "") == (action.query or "")
                    and event.get("topic", "") == (action.topic or "")
                    and event.get("category", "") == (action.category or "")
                    and event.get("message", "") == (action.message or "")
                    and event.get("language", "") == (action.language or "")
                    and event.get("engineering_notes", "") == (action.engineering_notes or "")
                    and event.get("tags", []) == (action.tags or [])
                )
        return False

    def reset(self, task_name: str = "easy") -> Observation:
        task_def = get_task(task_name)
        obs = Observation(
            current_ticket=task_def["initial_ticket"],
            last_action_result="Environment reset successfully.",
            ticket_history=[],
            system_tags=[],
            system_category=None
        )
        self._state = State(
            task_name=task_name,
            ticket_data=task_def["initial_ticket"],
            observation=obs,
            step_count=0,
            done=False,
            reward=STRICT_MIN_SCORE
        )
        return obs

    def close(self) -> None:
        """No-op close for inference script compatibility."""
        return None

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self._state.done:
            return self._state.observation, self._state.reward, self._state.done, {"last_action_error": "Episode already done."}

        self._state.step_count += 1
        last_result = ""
        done = False
        last_action_error = None
        security_violation = False
        invalid_action = False
        repeated_action = self._is_repeated_action(action)
        security_penalty = 0.0
        
        self._state.observation.ticket_history.append({
            "role": "agent",
            "action_type": action.action_type,
            "query": action.query or "",
            "topic": action.topic or "",
            "category": action.category or "",
            "tags": action.tags or [],
            "message": action.message or "",
            "language": action.language or "",
            "engineering_notes": action.engineering_notes or ""
        })

        if self._contains_suspicious_payload(action):
            last_result = "Security policy triggered: potentially unsafe payload detected."
            last_action_error = "unsafe payload detected"
            security_violation = True
            security_penalty = 0.5
            done = True

        elif action.action_type == "query_database":
            if action.query:
                result = customer_db.query(action.query)
                last_result = f"DB Search Result: {result}"
            else:
                last_result = "DB Search Failed: No query provided."
                last_action_error = "query_database requires query"
                invalid_action = True

        elif action.action_type == "search_knowledge_base":
            if action.topic:
                results = knowledge_base.search(action.topic, top_k=1)
                last_result = f"KB Search Result: {results[0]['chunk']}"
            else:
                last_result = "KB Search Failed: No topic provided."
                last_action_error = "search_knowledge_base requires topic"
                invalid_action = True

        elif action.action_type == "categorize_ticket":
            if action.category:
                self._state.observation.system_category = action.category
            if action.tags:
                for tag in action.tags:
                    if tag not in self._state.observation.system_tags:
                        self._state.observation.system_tags.append(tag)
            last_result = f"Ticket categorized as {action.category} with tags {action.tags}"
            if not action.category and not action.tags:
                last_action_error = "categorize_ticket requires category or tags"
                invalid_action = True

        elif action.action_type == "reply_to_user":
            if action.message and action.language:
                last_result = f"Sent message to user in {action.language}."
            else:
                last_result = "Reply failed: message or language missing."
                last_action_error = "reply_to_user requires message and language"
                invalid_action = True

        elif action.action_type == "escalate_ticket":
            if action.engineering_notes:
                last_result = f"Ticket escalated to engineering with notes: {action.engineering_notes}"
            else:
                last_result = "Escalation failed: engineering notes missing."
                last_action_error = "escalate_ticket requires engineering_notes"
                invalid_action = True

        elif action.action_type == "end_turn":
            last_result = "Ended turn. Episode finishing."
            done = True

        else:
            last_result = f"Unknown action type: {action.action_type}"
            last_action_error = f"unknown action type {action.action_type}"
            invalid_action = True

        if self._state.step_count >= MAX_STEPS:
            last_result = "Max steps reached. Episode finishing."
            done = True

        self._state.observation.last_action_result = last_result
        self._state.observation.ticket_history.append({"role": "system", "content": last_result})

        raw_reward = float(self._compute_reward())
        if invalid_action:
            raw_reward -= INVALID_ACTION_PENALTY
        if repeated_action:
            raw_reward -= REPEATED_ACTION_PENALTY
        raw_reward -= security_penalty
        raw_reward = max(0.0, min(1.0, raw_reward))

        completion_threshold = TASK_COMPLETION_THRESHOLDS.get(self._state.task_name, 0.9)
        if (
            not done
            and raw_reward >= completion_threshold
            and action.action_type in {"categorize_ticket", "reply_to_user", "escalate_ticket", "end_turn"}
        ):
            done = True
            completion_note = "Task objective satisfied. Episode finishing."
            self._state.observation.last_action_result = f"{last_result} {completion_note}"
            self._state.observation.ticket_history.append({"role": "system", "content": completion_note})

        reward = self._to_open_interval_score(raw_reward)
        self._state.reward = reward
        self._state.done = done

        return self._state.observation, reward, done, {
            "last_action_error": last_action_error,
            "security_violation": security_violation,
            "invalid_action": invalid_action,
            "repeated_action": repeated_action,
            "raw_reward": raw_reward,
        }

    def state(self) -> State:
        if self._state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self._state

    def _compute_reward(self) -> float:
        state_dict = self._state.model_dump()
        obs_dict = state_dict.get("observation", {})
        if self._state.task_name == "easy":
            return grade_easy(obs_dict)
        elif self._state.task_name == "medium":
            return grade_medium(obs_dict)
        elif self._state.task_name == "hard":
            return grade_hard(obs_dict)
        return 0.0
