@echo off
cd /d "%~dp0"
if not exist "frontend\dist\index.html" (
  echo Building the React dashboard once - needs Node.js from https://nodejs.org
  pushd frontend
  call npm install
  call npm run build
  popd
)
echo Starting the live system as separate processes: MQTT broker, ingestion service, sensor feeder agent,
echo and the API with the React dashboard. Close the four windows to stop it.
echo (Start Dashboard.bat alone also runs the whole live system, inside the API.)
start "MQTT broker" cmd /k python broker.py
timeout /t 3 /nobreak >nul
start "Ingestion service" cmd /k python ingest_service.py
timeout /t 8 /nobreak >nul
start "Sensor feeder agent" cmd /k python sensor_publisher.py --interval 1
start "API + dashboard - http://127.0.0.1:8765" cmd /k "set LIVE_AGENT=0&& python -m uvicorn api:app --port 8765"
echo.
echo Dashboard: http://127.0.0.1:8765   (opens on Live (MQTT))
echo REST API:  http://127.0.0.1:8765/docs
pause
