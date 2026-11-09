from __future__ import annotations

import json
import os
import ctypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


TRAIN_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = TRAIN_DIR / "runs"
STATIC_DIR = Path(__file__).resolve().parent
PORT = 8765


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def process_alive(pid: object) -> bool:
    if not isinstance(pid, int):
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def build_status() -> dict:
    jobs = []
    if RUNS_DIR.is_dir():
        for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
            progress = read_json(run_dir / "progress.json")
            config = read_json(run_dir / "config.json")
            metrics = read_json(run_dir / "metrics.json")
            updated = float(progress.get("updated_at", 0))
            status = progress.get("status", "queued")
            pid_alive = process_alive(config.get("pid"))
            if status in {"preparing", "training"} and not pid_alive and time.time() - updated > 180:
                status = "stalled"
            elif status in {"preparing", "training"} and pid_alive:
                status = "training"
            jobs.append({
                "id": run_dir.name,
                "title": run_dir.name,
                "status": status,
                "progress": progress.get("progress", 0),
                "epoch": progress.get("epoch", 0),
                "max_epochs": progress.get("max_epochs", config.get("epochs", 0)),
                "eta_seconds": progress.get("eta_seconds"),
                "metrics": {**progress.get("metrics", {}), **metrics},
                "config": config,
                "updated_at": updated,
            })
    active = [job for job in jobs if job["status"] == "training"]
    return {
        "server_time": time.time(),
        "done": sum(job["status"] == "done" for job in jobs),
        "training": len(active),
        "queued": sum(job["status"] == "queued" for job in jobs),
        "total": len(jobs),
        "jobs": jobs,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            payload = json.dumps(build_status()).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
        elif path == "/api/history":
            query = parse_qs(urlparse(self.path).query)
            run = (query.get("run") or [""])[0]
            runs_root = RUNS_DIR.resolve()
            run_dir = (RUNS_DIR / run).resolve()
            if not run or run_dir.parent != runs_root or not run_dir.is_dir():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            records: list[dict] = []
            history_path = run_dir / "history.jsonl"
            if history_path.is_file():
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            payload = json.dumps({"run": run, "records": records}).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")


if __name__ == "__main__":
    print(f"dashboard: http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
