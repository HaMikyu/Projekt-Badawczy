"""Zrodlo DetektorAI.exe (budowane przez app/build_exe.bat).

Exe niczego nie trzyma otwartego: uruchamia app/server.py w tle (pythonw)
i od razu sie konczy. Serwer otwiera okno aplikacji i wylacza sie sam,
gdy to okno zostanie zamkniete.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def find_python(root):
    override = os.environ.get("DETEKTOR_PYTHON")
    candidates = [
        override,
        root / ".venv" / "Scripts" / "pythonw.exe",
        shutil.which("pythonw"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return str(path)
    return None


def show_error(msg):
    import ctypes
    ctypes.windll.user32.MessageBoxW(0, msg, "Detektor AI", 0x10)


def main():
    root = project_root()
    server = root / "app" / "server.py"
    if not server.is_file():
        show_error(f"Nie znaleziono {server}.\nDetektorAI.exe musi lezec w glownym folderze projektu.")
        return 1
    python = find_python(root)
    if python is None:
        show_error("Nie znaleziono pythonw.exe.\nZainstaluj Pythona z pakietami z requirements.txt "
                   "albo ustaw zmienna DETEKTOR_PYTHON.")
        return 1
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([python, str(server)], cwd=str(root), creationflags=flags, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
