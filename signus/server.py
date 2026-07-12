"""Stdlib-only web server: static frontend + POST /api/analyze (raw body,
filename + optional fs/fmt/dtype overrides in the query string — no multipart)."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .pipeline import analyze, survey_web
from .sigio import Meta, decode, parse_name

_WEB = Path(__file__).resolve().parent.parent / "web"
_MIME = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
         ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml"}
_MAX_BODY = 256 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _meta(self, q: dict) -> Meta:
        """Build Meta from query params with filename fallback; raise on unknown."""
        name = q.get("name", [""])[0]
        m = parse_name(name)
        meta = Meta(float(q["fs"][0]) if "fs" in q else m.fs,
                    q.get("fmt", [m.fmt])[0], q.get("dtype", [m.dtype])[0],
                    q.get("endian", [m.endian])[0],
                    q.get("bitrev", ["1" if m.bitrev else "0"])[0] == "1")
        if not meta.ok():
            raise ValueError(f"샘플레이트/포맷을 알 수 없습니다: {name}")
        return meta

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        url = urlparse(self.path)
        if url.path not in ("/api/analyze", "/api/survey"):
            self._json(404, {"error": "not found"})
            return
        q = parse_qs(url.query)
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            self._json(411, {"error": "Content-Length 필요"})
            return
        if n > _MAX_BODY:
            self._json(413, {"error": "파일이 너무 큽니다 (최대 256MB)"})
            return
        try:
            meta = self._meta(q)
            burst = int(q["burst"][0]) if "burst" in q else None
            x = decode(self.rfile.read(n), meta)
            if x.size < 256:
                raise ValueError("신호가 너무 짧습니다")
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        try:
            out = survey_web(x, meta) if url.path == "/api/survey" \
                else analyze(x, meta, burst=burst).to_json()
            self._json(200, out)
        except Exception as exc:  # surface DSP failures to the UI
            self._json(500, {"error": f"분석 실패: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        rel = urlparse(self.path).path.lstrip("/") or "index.html"
        target = (_WEB / rel).resolve()
        if not (target.is_file() and target.is_relative_to(_WEB.resolve())):
            self._json(404, {"error": "not found"})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"signus UI → http://{host}:{srv.server_address[1]}")
    srv.serve_forever()
