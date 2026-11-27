"""Detektor AI jako aplikacja okienkowa (DetektorAI.exe).

Jeden proces: serwer HTTP aplikacji dziala w watku w tle na losowym wolnym porcie 127.0.0.1, a interfejs
(app/index.html) wyswietla natywne okno Windows z silnikiem Edge WebView2 (pywebview). Zamkniecie okna
konczy caly proces, wiec serwer nie moze zostac w tle. Przy starcie okno pokazuje ekran ladowania,
a po wczytaniu modeli przechodzi do aplikacji.

    python app\\desktop.py            # z repo
    dist\\DetektorAI\\DetektorAI.exe   # po zbudowaniu: build_exe.ps1
"""
import os
import sys
import threading
import traceback
from pathlib import Path


def find_root():
    """Katalog z models/ i statistical/runs/: DETEKTOR_ROOT, folder exe (wersja przenosna) albo repo powyzej."""
    if os.environ.get("DETEKTOR_ROOT"):
        return Path(os.environ["DETEKTOR_ROOT"])
    start = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    for d in [start, *start.parents]:
        if (d / "models" / "system-one-final" / "system_one.pt").is_file():
            return d
    return start


ROOT = find_root()
os.environ["DETEKTOR_ROOT"] = str(ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # modele sa lokalne, bez zapytan do Hugging Face
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys._MEIPASS) / "app"
else:
    APP_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(APP_DIR))
    sys.path.insert(0, str(APP_DIR.parent))

import webview  # noqa: E402

TITLE = "Detektor AI — eseje"

LOADING_HTML = """<!doctype html><html lang="pl"><head><meta charset="utf-8"><style>
html,body{height:100%;margin:0}
body{display:flex;align-items:center;justify-content:center;font-family:'Segoe UI',system-ui,sans-serif;
background:#f6f4ef;color:#1f2328}
.box{text-align:center}.t{font-size:22px;font-weight:600;margin-bottom:6px}.s{font-size:14px;color:#6b6f76}
.bar{width:220px;height:4px;border-radius:2px;background:#e3e0d8;margin:22px auto 0;overflow:hidden}
.bar i{display:block;width:40%;height:100%;background:#3b5bdb;animation:m 1.2s ease-in-out infinite}
@keyframes m{0%{transform:translateX(-100%)}100%{transform:translateX(250%)}}
@media (prefers-color-scheme:dark){body{background:#1b1c1e;color:#e8e6e1}.s{color:#9a9ca1}.bar{background:#333}}
</style></head><body><div class="box"><div class="t">Detektor AI</div>
<div class="s" id="s">Ładowanie modeli… (pierwsze uruchomienie może potrwać do minuty)</div>
<div class="bar"><i></i></div></div></body></html>"""


def error_html(msg):
    import html
    return ("<!doctype html><html lang='pl'><head><meta charset='utf-8'></head>"
            "<body style=\"font-family:'Segoe UI',sans-serif;padding:32px;background:#f6f4ef;color:#1f2328\">"
            "<h2>Nie udało się uruchomić Detektora</h2><pre style='white-space:pre-wrap;font-size:12px'>"
            + html.escape(msg) + "</pre></body></html>")


LOG_PATH = Path(os.environ.get("LOCALAPPDATA", ROOT)) / "DetektorAI" / "server.log"


def start_server(window, server):
    """Watek startowy: modele -> serwer na wolnym porcie -> okno przechodzi do aplikacji."""
    try:
        from http.server import ThreadingHTTPServer
        server.log("Ladowanie modeli...")
        server.MODELS = server.load_models()
        server.log("Modele zaladowane")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.DetectorHandler)
        httpd.daemon_threads = True
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        server.log(f"Listening on http://127.0.0.1:{port}")
        window.load_url(f"http://127.0.0.1:{port}/")
        if os.environ.get("DETEKTOR_SELFTEST"):
            selftest(window, port, server.log)
    except Exception:
        server.log("BLAD STARTU: " + traceback.format_exc())
        window.load_html(error_html(traceback.format_exc()))


def selftest(window, port, log):
    """DETEKTOR_SELFTEST=1: zapytanie /api/analyze przez serwer, wynik do logu, potem zamkniecie okna jak uzytkownik."""
    import json
    import time
    import urllib.request
    time.sleep(3)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/analyze", method="POST",
                                 data=json.dumps({"text": "This is a short test essay about school. " * 20}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        log("SELFTEST " + json.dumps(json.loads(r.read())["combined"]))
    log(f"SELFTEST port={port} zamykam okno")
    window.destroy()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # torch i transformers importujemy w glownym watku, ZANIM ruszy petla okna (.NET/WinForms): import ich DLL
    # z watku pobocznego przy dzialajacym oknie zakleszczal spakowany EXE. W tle zostaje tylko wczytanie modeli.
    import server
    server.INDEX_PATH = APP_DIR / "index.html"
    server.LOG_PATH = LOG_PATH
    server.log(f"START (okno), ROOT={ROOT}")
    window = webview.create_window(TITLE, html=LOADING_HTML, width=1180, height=860, min_size=(820, 600),
                                   background_color="#f6f4ef")
    webview.start(start_server, (window, server), gui="edgechromium", private_mode=True)
    # okno zamkniete: konczymy natychmiast caly proces (watek serwera, watki torcha), bez czekania na nic
    os._exit(0)


if __name__ == "__main__":
    main()
