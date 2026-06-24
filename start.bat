@echo off
echo ========================================
echo Kronos Trading Signal Generator
echo ========================================
echo.

cd /d "%~dp0"

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Web UI...
echo Access: http://localhost:8888
echo.

python main.py --web --no-kronos

pause
