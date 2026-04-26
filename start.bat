@echo off
title DevastatorAI

echo.
echo  ====================================
echo   DevastatorAI - Agent Starter Kit
echo  ====================================
echo.

:: Check Node.js is installed
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo         Download it from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('node --version') do set NODE_VER=%%v
echo [OK] Node.js %NODE_VER% detected.

:: Copy .env.example to .env if .env does not exist
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [OK] .env created from .env.example. Edit it before running agents.
        echo.
        echo      Open .env and set your model and API key, then re-run start.bat.
        echo.
        pause
        exit /b 0
    ) else (
        echo [WARN] .env.example not found. Skipping .env creation.
    )
) else (
    echo [OK] .env found.
)

echo.
echo  DevastatorAI is ready.
echo.
echo  Usage:
echo    node core\agent_runner.js --agent rachel --prompt "Your question here"
echo.
echo  Agents available:
echo    rachel   - Research Agent
echo    winter   - Writing Agent
echo    charlie  - Coding Agent
echo    chief    - Orchestration Agent
echo    sentinel - Security Agent
echo.

cmd /k
