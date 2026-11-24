import json
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from helpers import analyze_text, load_models


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
INDEX_PATH = APP_DIR / "index.html"
LOG_PATH = APP_DIR / "server.log"
HOST = "127.0.0.1"
PORT = 8000
MODELS = None


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


def show_error(msg):
    log(msg)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "Detektor AI", 0x10)
        except Exception:
            pass


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

    def do_OPTIONS(self):
        self.send_data(204, b"", "text/plain; charset=utf-8")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_data(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/health":
            payload = json.dumps({
                "ok": True,
                "ready": MODELS is not None,
                "ml_loaded": bool(MODELS and MODELS["ml_model"] is not None),
                "stat_loaded": bool(MODELS and MODELS["stat_model"] is not None),
            }).encode("utf-8")
            self.send_data(200, payload, "application/json; charset=utf-8")
        else:
            self.send_data(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/analyze":
            self.send_data(404, b"Not found", "text/plain; charset=utf-8")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            text = str(body.get("text", "")).strip()
            if not text:
                payload = json.dumps({"error": "Wklej tekst do analizy."}).encode("utf-8")
                self.send_data(400, payload, "application/json; charset=utf-8")
                return
            if MODELS is None:
                payload = json.dumps({"error": "Modele jeszcze sie laduja. Sprobuj za chwile."}).encode("utf-8")
                self.send_data(503, payload, "application/json; charset=utf-8")
                return
            result = analyze_text(text, MODELS)
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_data(200, payload, "application/json; charset=utf-8")
        except Exception as exc:
            log("ANALYZE ERROR: " + traceback.format_exc())
            payload = json.dumps({"error": f"Blad analizy: {exc}"}).encode("utf-8")
            self.send_data(500, payload, "application/json; charset=utf-8")


def open_browser():
    time.sleep(0.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    global MODELS
    log("START")
    if not port_free(HOST, PORT):
        webbrowser.open(f"http://{HOST}:{PORT}")
        show_error(
            f"Port {PORT} jest juz zajety.\n"
            "Otwieram istniejacą aplikacje w przegladarce.\n"
            "Jesli nie dziala, zamknij stary proces Detektora i uruchom ponownie."
        )
        return

    log("Ladowanie modeli...")
    MODELS = load_models()
    log("Modele zaladowane")
    threading.Thread(target=open_browser, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), DetectorHandler)
    log(f"Listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        show_error("Nie udalo sie uruchomic serwera:\n" + traceback.format_exc()[-900:])
