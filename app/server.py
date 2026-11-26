import json
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
INDEX_PATH = APP_DIR / "index.html"
LOG_PATH = APP_DIR / "server.log"
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
MODELS = None
LOAD_ERROR = None

# Serwer zyje tylko tak dlugo, jak otwarte jest okno aplikacji: strona trzyma
# polaczenie /api/events, a po jego zerwaniu (i krotkiej przerwie na odswiezenie)
# proces sam sie konczy, zeby nie blokowal plikow modeli.
CLOSE_GRACE_S = 5
NO_WINDOW_LIMIT_S = 180
STATE_LOCK = threading.Lock()
OPEN_CLIENTS = 0
EVER_CONNECTED = False
LAST_DISCONNECT = time.monotonic()
STARTED_AT = time.monotonic()
SHUTDOWN = threading.Event()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def port_free(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0
    finally:
        sock.close()


def detector_running():
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=2) as res:
            return json.loads(res.read().decode("utf-8")).get("ok") is True
    except Exception:
        return False


def show_error(msg):
    log(msg)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "Detektor AI", 0x10)
        except Exception:
            pass


def find_app_browser():
    candidates = [shutil.which("msedge"), shutil.which("chrome")]
    for env in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData"):
        base = os.environ.get(env)
        if base:
            candidates.append(os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"))
            candidates.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def open_window():
    # osobne okno aplikacji (Edge/Chrome --app), a jak ich nie ma - zwykla karta
    browser = find_app_browser()
    if browser:
        try:
            subprocess.Popen([browser, f"--app={URL}", "--window-size=1150,900"])
            return
        except OSError:
            pass
    webbrowser.open(URL)


def request_shutdown(reason):
    if SHUTDOWN.is_set():
        return
    SHUTDOWN.set()
    log(f"STOP ({reason})")


def watchdog():
    while not SHUTDOWN.wait(1):
        with STATE_LOCK:
            clients, ever, last = OPEN_CLIENTS, EVER_CONNECTED, LAST_DISCONNECT
        now = time.monotonic()
        if ever and clients == 0 and now - last > CLOSE_GRACE_S:
            request_shutdown("okno aplikacji zamkniete")
        elif not ever and now - STARTED_AT > NO_WINDOW_LIMIT_S:
            request_shutdown("okno aplikacji nie polaczylo sie")


def load_models_background():
    global MODELS, LOAD_ERROR
    try:
        log("Ladowanie modeli...")
        from helpers import load_models
        MODELS = load_models()
        log("Modele zaladowane")
    except Exception:
        LOAD_ERROR = traceback.format_exc()
        log("LOAD ERROR: " + LOAD_ERROR)


class DetectorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log("%s - %s" % (self.address_string(), fmt % args))

    def send_data(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_data(code, body, "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self.send_data(204, b"", "text/plain; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_data(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self.send_json(200, {
                "ok": True,
                "ready": MODELS is not None,
                "error": LOAD_ERROR.strip().splitlines()[-1] if LOAD_ERROR else None,
                "ml_loaded": bool(MODELS and MODELS["ml_model"] is not None),
                "stat_loaded": bool(MODELS and MODELS["stat_model"] is not None),
            })
        elif path == "/api/events":
            self.hold_window_connection()
        else:
            self.send_data(404, b"Not found", "text/plain; charset=utf-8")

    def hold_window_connection(self):
        global OPEN_CLIENTS, EVER_CONNECTED, LAST_DISCONNECT
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with STATE_LOCK:
            OPEN_CLIENTS += 1
            EVER_CONNECTED = True
        try:
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.flush()
            sock = self.connection
            while not SHUTDOWN.is_set():
                readable, _, _ = select.select([sock], [], [], 2)
                if readable and not sock.recv(1024):
                    break  # przegladarka zamknela polaczenie
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except OSError:
            pass
        finally:
            with STATE_LOCK:
                OPEN_CLIENTS -= 1
                LAST_DISCONNECT = time.monotonic()
            self.close_connection = True

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/shutdown":
            self.send_json(200, {"ok": True})
            request_shutdown("przycisk Zamknij")
            return
        if path != "/api/analyze":
            self.send_data(404, b"Not found", "text/plain; charset=utf-8")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            text = str(body.get("text", "")).strip()
            if not text:
                self.send_json(400, {"error": "Wklej tekst do analizy."})
                return
            if LOAD_ERROR:
                self.send_json(500, {"error": "Modele nie wczytaly sie. Szczegoly w app/server.log."})
                return
            if MODELS is None:
                self.send_json(503, {"error": "Modele jeszcze sie laduja. Sprobuj za chwile."})
                return
            from helpers import analyze_text
            self.send_json(200, analyze_text(text, MODELS))
        except Exception as exc:
            log("ANALYZE ERROR: " + traceback.format_exc())
            self.send_json(500, {"error": f"Blad analizy: {exc}"})


def main():
    log("START")
    if not port_free(HOST, PORT):
        if detector_running():
            log("Detektor juz dziala - otwieram okno")
            open_window()
        else:
            show_error(
                f"Port {PORT} jest zajety przez inny program.\n"
                "Zamknij go i uruchom DetektorAI.exe ponownie."
            )
        return

    server = ThreadingHTTPServer((HOST, PORT), DetectorHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"Listening on {URL}")
    threading.Thread(target=load_models_background, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    open_window()

    SHUTDOWN.wait()
    time.sleep(0.3)  # zeby odpowiedz na /api/shutdown zdazyla wyjsc
    server.shutdown()
    server.server_close()
    log("Serwer zatrzymany")
    os._exit(0)  # nie czekamy na watki torcha


if __name__ == "__main__":
    try:
        main()
    except Exception:
        show_error("Nie udalo sie uruchomic serwera:\n" + traceback.format_exc()[-900:])
        os._exit(1)
