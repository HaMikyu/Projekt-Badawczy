@echo off
rem Buduje DetektorAI.exe w glownym folderze projektu (wymaga: python -m pip install pyinstaller)
cd /d "%~dp0.."
set BUILD=%TEMP%\detektor_exe_build
python -m PyInstaller --noconfirm --clean --onefile --noconsole --name DetektorAI ^
  --distpath "%CD%" --workpath "%BUILD%\work" --specpath "%BUILD%" app\launcher.py
if errorlevel 1 (
  echo Budowanie nie powiodlo sie.
  exit /b 1
)
echo Gotowe: %CD%\DetektorAI.exe
