$ErrorActionPreference = 'Stop'

Set-Location "$PSScriptRoot\.."

Write-Host '== OpenEnv Validate ==' -ForegroundColor Cyan
.\.venv\Scripts\openenv.exe validate

Write-Host '== Docker Build ==' -ForegroundColor Cyan
docker build -t openenv-support:local .

Write-Host '== Local API Smoke Test ==' -ForegroundColor Cyan
.\.venv\Scripts\python.exe -c "from fastapi.testclient import TestClient; from server.app import app; c=TestClient(app); assert c.post('/reset', params={'task_name':'easy'}).status_code==200; assert c.post('/step', json={'action_type':'categorize_ticket','category':'account','tags':['password_reset']}).status_code==200; assert c.get('/state').status_code==200; print('Smoke test passed')"

Write-Host 'All checks completed.' -ForegroundColor Green
