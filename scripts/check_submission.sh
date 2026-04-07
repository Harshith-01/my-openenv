#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ".venv/bin/python" ]]; then
	echo "Missing .venv/bin/python. Create and activate a Linux/macOS venv first."
	exit 1
fi

echo "== OpenEnv Validate =="
.venv/bin/openenv validate

echo "== Docker Build =="
docker build -t openenv-support:local .

echo "== Local API Smoke Test =="
.venv/bin/python -c "from fastapi.testclient import TestClient; from server.app import app; c=TestClient(app); assert c.post('/reset', params={'task_name':'easy'}).status_code==200; assert c.post('/step', json={'action_type':'categorize_ticket','category':'account','tags':['password_reset']}).status_code==200; assert c.get('/state').status_code==200; print('Smoke test passed')"

echo "All checks completed."
