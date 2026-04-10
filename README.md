---
title: Cross-Cultural Customer Support OpenEnv
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Cross-Cultural Customer Support Escalation (OpenEnv)

## Description and Motivation
This environment simulates a real customer support workflow where an agent must triage, investigate, and resolve multilingual technical tickets.

It is designed for realistic enterprise support behavior:
- English and Japanese ticket handling
- Structured escalation to engineering
- Retrieval-augmented support via a semantic knowledge base
- Deterministic grading with strict open-interval rewards `(0,1)`

## Action Space
The agent uses the `Action` Pydantic model with these `action_type` values:
- `categorize_ticket` with `category`, `tags`
- `search_knowledge_base` with `topic`
- `query_database` with `query`
- `reply_to_user` with `message`, `language`
- `escalate_ticket` with `engineering_notes`
- `end_turn`

## Observation Space
The `Observation` model returns:
- `current_ticket`: active ticket payload (`message`, `language`, `priority`, etc.)
- `last_action_result`: deterministic feedback from the previous step
- `ticket_history`: ordered agent/system event trace
- `system_tags`: tags assigned by the agent
- `system_category`: category assigned by the agent

## Tasks and Difficulty
1. `easy`: English password-reset ticket triage.
2. `medium`: Hardware incident requiring DB lookup and a policy-aligned user reply.
3. `hard`: Japanese enterprise API incident requiring KB lookup, Kaizen-style escalation, and formal Keigo response.

## Reward and Grader Design
- Graders are deterministic and return strict open-interval scores `(0,1)`.
- Rewards provide partial progress credit throughout the trajectory.
- Penalties are applied for invalid/redundant actions.
- Episode completion uses sensible boundaries (explicit end-turn, step limit, or objective completion threshold).

## Local Setup
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Server
```bash
python server/app.py
```

Quick smoke checks:
```bash
curl http://127.0.0.1:7860/
curl -X POST "http://127.0.0.1:7860/reset?task_name=easy"
curl -X POST "http://127.0.0.1:7860/step" -H "Content-Type: application/json" -d '{"action_type":"categorize_ticket","category":"account","tags":["password_reset"]}'
curl http://127.0.0.1:7860/state
```

## Docker
```bash
docker build -t openenv-support .
docker run -p 7860:7860 openenv-support
```

## Inference Baseline
Set required variables before running:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- Optional when using Docker image mode: `LOCAL_IMAGE_NAME`

Then run:
```bash
python inference.py
```

The script:
- uses the OpenAI client for model calls
- emits strict structured logs (`[START]`, `[STEP]`, `[END]`)
- includes deterministic fallback actions for robustness if model output is malformed

Compliance notes:
- Defaults are set only for `API_BASE_URL` and `MODEL_NAME`.
- `HF_TOKEN` has no default and must be set by the runner.
- `[STEP] error=` prints only `last_action_error` or `null`.

## Validation Checklist
1. `openenv validate` passes
2. `docker build -t openenv-support .` passes
3. Endpoints `/`, `/reset`, `/step`, `/state` respond correctly
4. `inference.py` completes and prints strict evaluator log format

Helper scripts:
- PowerShell: `./scripts/check_submission.ps1`
- Bash: `./scripts/check_submission.sh`
