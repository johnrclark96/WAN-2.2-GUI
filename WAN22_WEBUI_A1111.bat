@echo off
setlocal
chcp 65001 >NUL
@echo off
setlocal

set ROOT=D:\wan22
set PY=%ROOT%\venv\Scripts\python.exe
set UI=%ROOT%\wan22_webui_a1111.py
set LOG=%ROOT%\webui_launch.log

echo [BOOT] WAN 2.2 UI from %ROOT%
cd /d "%ROOT%" || (echo [ERROR] Could not cd into %ROOT% & pause & exit /b 1)

REM Activate venv in this shell
call "%ROOT%\venv\Scripts\activate.bat"

REM Launch the UI (auto-switch to 7861 if 7860 is busy), mirror output to console and log
echo [RUN] Starting Gradio UI...
powershell -NoProfile -Command ^
  "& { & '%PY%' '%UI%' --listen --port 7860 2>&1 | Tee-Object -FilePath '%LOG%' }"

echo [DONE] Exit code %ERRORLEVEL%
pause

