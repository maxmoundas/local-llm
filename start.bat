@echo off
setlocal

echo Starting Local LLM...

:: Backend
start "Local LLM Backend" cmd /k "cd /d %~dp0backend && python serve.py"

:: Give the backend a moment to bind before opening the browser hint
timeout /t 2 /nobreak >nul

:: Frontend
start "Local LLM Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Both services are starting in separate windows.
echo Close those windows to stop the servers.
