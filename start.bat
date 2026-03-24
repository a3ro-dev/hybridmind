@echo off
echo =======================================================
echo               Starting HybridMind
echo =======================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run the setup first.
    pause
    exit /b 1
)

echo [1/2] Launching Streamlit UI in a new window...
start "HybridMind UI" cmd /c ".\.venv\Scripts\activate.bat && streamlit run ui\app.py"

echo [2/2] Starting FastAPI Backend on port 8000...
.\.venv\Scripts\activate.bat && uvicorn main:app --reload --port 8000
