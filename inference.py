import os
import json
from openai import OpenAI
from typing import List

from env import SupportEnv
from models import Action

HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("SUPPORT_ENV_BENCHMARK", "support_env")
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.9"))
STRICT_MIN_SCORE = 0.01

def run_inference(task_name: str):
    if not API_KEY:
        raise RuntimeError("Missing API key. Set HF_TOKEN (or OPENAI_API_KEY).")

    env = SupportEnv()
    rewards: List[float] = []
    done = False
    step = 0
    final_score = 0.0
    success = False
    error_msg = "null"

    obs = env.reset(task_name)
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    system_prompt = (
        "You are an AI support agent. The available actions are defined in the JSON schema.\n"
        "Your goal is to solve the support ticket efficiently."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"New Ticket: {json.dumps(obs.current_ticket)}\n"
                f"Action JSON schema: {json.dumps(Action.model_json_schema())}"
            )
        }
    ]
    
    while not done and step < MAX_STEPS:
        step += 1
        
        try:
            # We enforce structured output (or just parse standard JSON for broad model compatibility)
            # For hackathon robust baseline, we use traditional function calling or JSON mode.
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300
            )
            
            raw_content = response.choices[0].message.content
            action_dict = json.loads(raw_content)
            action = Action(**action_dict)
            
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            step_error = info.get("last_action_error") if isinstance(info, dict) else None
            error_msg = step_error if step_error else "null"
            
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({"role": "user", "content": f"Result: {obs.last_action_result}"})
            
            action_str = f"{action.action_type}(...)"
            
        except Exception as e:
            # Error handling to prevent script crash and log exactly the error
            error_msg = str(e).replace('\n', ' ')
            action_str = "error"
            reward = 0.0
            done = True
            rewards.append(reward)
            
        print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")

    final_score = rewards[-1] if rewards else STRICT_MIN_SCORE
    success = final_score >= SUCCESS_SCORE_THRESHOLD
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={step} score={final_score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task)
