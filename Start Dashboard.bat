@echo off
cd /d "%~dp0"
if not exist "frontend\dist\index.html" (
  echo Building the React dashboard once - needs Node.js from https://nodejs.org
  pushd frontend
  call npm install
  call npm run build
  popd
)
echo Starting the water quality monitoring dashboard (React + FastAPI) with the live system:
echo built-in MQTT broker, sensor feeder agent and ingestion worker - no CSV file needed.
echo Loading the models takes about a minute; then open http://127.0.0.1:8765
start "" cmd /c "timeout /t 45 /nobreak >nul & start http://127.0.0.1:8765"
python -m uvicorn api:app --port 8765
pause
