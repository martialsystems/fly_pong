#!/usr/bin/env python3
"""Phone-width 390x844 and desktop ~1280 check for the Pong page.

Uses CDP Emulation.setDeviceMetricsOverride on a page target plus a unique
Chrome user-data-dir. Chrome --window-size is not a CSS viewport on this host.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
PHONE = (390, 844)
DESKTOP = (1280, 800)


def find_chrome() -> Optional[str]:
    env = os.environ.get("VIEWPORT_SANITY_CHROME", "").strip()
    if env and Path(env).is_file():
        return env
    cands = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for c in cands:
        if Path(c).is_file():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser"):
        hit = shutil.which(name)
        if hit:
            return hit
    return None


def _serve(root: Path) -> Tuple[ThreadingHTTPServer, int]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt, *args):
            return

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


class Cdp:
    def __init__(self, ws_url: str):
        import urllib.parse

        u = urllib.parse.urlparse(ws_url)
        self._sock = socket.create_connection((u.hostname, u.port), timeout=10)
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        path = u.path + (("?" + u.query) if u.query else "")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {u.hostname}:{u.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._sock.sendall(req.encode("ascii"))
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            hdr += self._sock.recv(4096)
        if b"101" not in hdr.split(b"\r\n", 1)[0]:
            raise RuntimeError("CDP websocket handshake failed")
        self._buf = b""
        self._next = 1

    def _send(self, payload: bytes) -> None:
        header = bytearray([0x81])
        n = len(payload)
        mask = b"\x01\x02\x03\x04"
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(header + masked)

    def _recv_frame(self) -> bytes:
        while True:
            if len(self._buf) < 2:
                self._buf += self._sock.recv(4096)
                continue
            b0, b1 = self._buf[0], self._buf[1]
            masked = b1 & 0x80
            ln = b1 & 0x7F
            idx = 2
            if ln == 126:
                if len(self._buf) < 4:
                    self._buf += self._sock.recv(4096)
                    continue
                ln = struct.unpack("!H", self._buf[2:4])[0]
                idx = 4
            elif ln == 127:
                if len(self._buf) < 10:
                    self._buf += self._sock.recv(4096)
                    continue
                ln = struct.unpack("!Q", self._buf[2:10])[0]
                idx = 10
            mlen = 4 if masked else 0
            need = idx + mlen + ln
            while len(self._buf) < need:
                chunk = self._sock.recv(4096)
                if not chunk:
                    raise RuntimeError("CDP socket closed mid-frame")
                self._buf += chunk
            mask = self._buf[idx : idx + mlen] if masked else b""
            data = self._buf[idx + mlen : need]
            self._buf = self._buf[need:]
            if masked:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            opcode = b0 & 0x0F
            if opcode == 0x8:
                raise RuntimeError("CDP websocket closed")
            if opcode == 0x1 or opcode == 0x2:
                return data

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mid = self._next
        self._next += 1
        msg = {"id": mid, "method": method}
        if params:
            msg["params"] = params
        self._send(json.dumps(msg, separators=(",", ":")).encode("utf-8"))
        deadline = time.time() + 20
        while time.time() < deadline:
            raw = self._recv_frame()
            try:
                data = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if data.get("id") == mid:
                if "error" in data:
                    raise RuntimeError("{0}: {1}".format(method, data["error"]))
                return data.get("result") or {}
        raise RuntimeError("CDP timeout waiting for {0}".format(method))

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def _wait_devtools(port: int, timeout: float = 12.0) -> str:
    list_url = "http://127.0.0.1:{0}/json/list".format(port)
    new_url = "http://127.0.0.1:{0}/json/new?about:blank".format(port)
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(list_url, timeout=1) as resp:
                pages = json.loads(resp.read().decode("utf-8"))
            if isinstance(pages, list):
                for tab in pages:
                    if not isinstance(tab, dict):
                        continue
                    if tab.get("type") in ("page", "webview") and tab.get("webSocketDebuggerUrl"):
                        return str(tab["webSocketDebuggerUrl"]).replace("localhost", "127.0.0.1")
            req = urllib.request.Request(new_url, method="PUT")
            with urllib.request.urlopen(req, timeout=2) as resp:
                created = json.loads(resp.read().decode("utf-8"))
            ws = created.get("webSocketDebuggerUrl")
            if ws:
                return str(ws).replace("localhost", "127.0.0.1")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last = str(exc)
        time.sleep(0.15)
    raise RuntimeError("Chrome page target not up on {0}: {1}".format(port, last))


def _eval(cdp: Cdp, expression: str) -> Any:
    out = cdp.call(
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    if out.get("exceptionDetails"):
        raise RuntimeError("page JS error: {0}".format(out["exceptionDetails"]))
    return (out.get("result") or {}).get("value")


MEASURE_JS = r"""
(async () => {
  const canvas = document.getElementById("court");
  const slider = document.getElementById("skill");
  const score = document.getElementById("score");
  const start = document.getElementById("start");
  if (!canvas) return { error: "missing #court" };
  if (!slider) return { error: "missing #skill" };
  if (!score) return { error: "missing #score" };
  if (!start) return { error: "missing #start" };
  start.click();
  slider.value = "0";
  slider.dispatchEvent(new Event("input", { bubbles: true }));
  slider.value = "1";
  slider.dispatchEvent(new Event("input", { bubbles: true }));
  canvas.dispatchEvent(new MouseEvent("mousemove", { clientX: 40, clientY: 80, bubbles: true }));
  await new Promise((r) => setTimeout(r, 800));
  const cr = canvas.getBoundingClientRect();
  const sr = slider.getBoundingClientRect();
  const sc = score.getBoundingClientRect();
  const st = start.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const inside = (r) => r.width > 0 && r.height > 0 && r.left >= -8 && r.right <= vw + 8;
  const status = (document.getElementById("ai-status") || {}).textContent || "";
  return {
    vw, vh,
    canvas: { w: cr.width, h: cr.height, left: cr.left, top: cr.top },
    sliderIn: inside(sr) && sr.top < vh && sr.bottom > 0,
    scoreIn: inside(sc) && sc.top < vh && sc.bottom > 0,
    startIn: inside(st) && st.top < vh && st.bottom > 0,
    canvasVisible: cr.width > 100 && cr.height > 80 && cr.top < vh,
    startLabel: start.textContent,
    status: status,
    score: score.textContent,
    skill: slider.value
  };
})()
"""


def measure(chrome: str, viewports: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    httpd, http_port = _serve(PUBLIC)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    cdp_port = probe.getsockname()[1]
    probe.close()
    tmp = tempfile.mkdtemp(prefix="flypong-chrome-")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--disable-extensions",
        "--remote-debugging-port={0}".format(cdp_port),
        "--remote-allow-origins=*",
        "--user-data-dir={0}".format(tmp),
        "about:blank",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: List[Dict[str, Any]] = []
    try:
        ws = _wait_devtools(cdp_port)
        cdp = Cdp(ws)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        url = "http://127.0.0.1:{0}/index.html".format(http_port)
        for w, h in viewports:
            metrics = {
                "width": w,
                "height": h,
                "deviceScaleFactor": 1,
                "mobile": w <= 500,
            }
            cdp.call("Emulation.setDeviceMetricsOverride", metrics)
            cdp.call("Page.navigate", {"url": url})
            deadline = time.time() + 12
            while time.time() < deadline:
                try:
                    ready = _eval(cdp, "document.readyState")
                except RuntimeError:
                    ready = ""
                if ready == "complete":
                    break
                time.sleep(0.1)
            cdp.call("Emulation.setDeviceMetricsOverride", metrics)
            time.sleep(0.2)
            data = _eval(cdp, MEASURE_JS)
            data["requested"] = {"w": w, "h": h}
            results.append(data)
        cdp.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        httpd.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def main() -> None:
    chrome = find_chrome()
    if chrome is None:
        print("SKIP: no Chrome binary for viewport_sanity")
        raise SystemExit(0)
    rows = measure(chrome, [PHONE, DESKTOP])
    failed = False
    for row in rows:
        req = row.get("requested") or {}
        print(json.dumps(row, indent=2))
        if row.get("error"):
            print("FAIL:", row["error"])
            failed = True
            continue
        if int(row.get("vw") or 0) != int(req.get("w") or -1):
            print("FAIL: innerWidth {0} != {1}".format(row.get("vw"), req.get("w")))
            failed = True
        if not row.get("canvasVisible"):
            print("FAIL: canvas not visible at", req)
            failed = True
        if not row.get("sliderIn"):
            print("FAIL: skill slider outside viewport at", req)
            failed = True
        if not row.get("scoreIn"):
            print("FAIL: score outside viewport at", req)
            failed = True
        if not row.get("startIn"):
            print("FAIL: start control outside viewport at", req)
            failed = True
        if row.get("startLabel") not in ("Playing", "Play again"):
            print("FAIL: start click did not enter play at", req, row.get("startLabel"))
            failed = True
        if abs(float(row.get("skill") or -1) - 1.0) > 1e-6:
            print("FAIL: skill slider did not accept 1 at", req)
            failed = True
        status = str(row.get("status") or "")
        if status.startswith("AI:"):
            print("FAIL: status still says AI at", req, status)
            failed = True
        if "PPO" not in status and "lag-chase" not in status:
            print("FAIL: status is not PPO or lag-chase at", req, status)
            failed = True
    if failed:
        raise SystemExit(2)
    print("viewport_sanity pass")


if __name__ == "__main__":
    main()
