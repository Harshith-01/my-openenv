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

## Description & Motivation
This environment simulates a multilingual customer support helpdesk where an AI agent must process, troubleshoot, and escalate technical issues based on a set of logical operations. The agent operates in English and Japanese.
Multilingual workflow automation is highly sought after by global companies. Demonstrating an AI that can handle both technical troubleshooting (database querying, KB referencing) and regional business etiquette (Japanese Keigo) assesses true frontier model capabilities in practical applications.

## Action Space
The agent can invoke the `Action` pydantic model with these `action_type` values:
- `categorize_ticket` (category, tags)
- `search_knowledge_base` (topic)
- `query_database` (query)
- `reply_to_user` (message, language)
- `escalate_ticket` (engineering_notes)
- `end_turn` ()

## Observation Space
At each step, the `Observation` model returns:
- `current_ticket`: Dict (message, language, priority)
- `last_action_result`: String providing the feedback loop for the agent.
- `ticket_history`: List tracking all actions taken and their results.
- `system_tags`: List
- `system_category`: String

## Tasks
1. **Easy**: `Standard English Classification`. Agent categorizes a password reset email correctly.
2. **Medium**: `DB Reasoning & Standard Reply`. User complains about a broken GPU. Agent must query DB for the serial number and issue standard refund text.
3. **Hard**: `Multi-turn Japanese Escalation`. Japanese enterprise client reports an API 500. Agent must search KB to identify the known bug (#ERR-7782), escalate using the Kaizen method, and reply to the user using formal Japanese Keigo (honourifics).

## Setup & Usage Instructions

### Create and Activate Environment
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

Create your local environment file from the template:
Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

Linux/macOS:
```bash
cp .env.example .env
```

### Run via Docker (Recommended for HF Spaces)
```bash
docker build -t openenv-support .
docker run -p 7860:7860 openenv-support
```
The FastAPI instance exposes the endpoints expected by OpenEnv on `http://127.0.0.1:7860/`.

Quick runtime checks:
```bash
curl http://127.0.0.1:7860/
curl -X POST "http://127.0.0.1:7860/reset?task_name=easy"
curl -X POST "http://127.0.0.1:7860/step" -H "Content-Type: application/json" -d '{"action_type":"categorize_ticket","category":"account","tags":["password_reset"]}'
curl http://127.0.0.1:7860/state
```

### Run Server Locally (without Docker)
```bash
python server/app.py
```

UI endpoint (if static files exist):
- `http://127.0.0.1:7860/ui/index.html`

### Run Inference Baseline
Windows PowerShell:
```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="your-huggingface-token"
python inference.py
```

Linux/macOS:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token"
python inference.py
```

The script prints strict structured logs in this format:
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> score=<0.00-1.00> rewards=<r1,r2,...>

## Baseline Scores (Deterministic grading)
- **Easy**: 0.99 (Task completed perfectly using deterministic checks)
- **Medium**: 0.99 (DB queried successfully, serial found and included in response)
- **Hard**: 0.99 (KB searched, escalated with bug ID, drafted formal Japanese Keigo response)

## Pre-Submission Checklist
1. `openenv validate` passes.
2. `docker build -t openenv-support .` succeeds.
3. Docker container responds on `/`, `/reset`, `/step`, `/state`.
4. `inference.py` emits strict `[START]`, `[STEP]`, `[END]` logs.

Automated helper scripts:
- PowerShell: `./scripts/check_submission.ps1`
- Bash: `./scripts/check_submission.sh`
