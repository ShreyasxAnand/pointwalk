#!/usr/bin/env python3
"""The viewer's web server. It sits between the web page and the drawing program.

    browser page  <--HTTP-->  this server  <--pipe-->  build/viewer (GPU)

The page asks for "frame" many times a second and says how far the camera should
move. This server passes that on to the drawing program as one line of text,
reads the picture it sends back, shrinks it to a JPEG, and returns it to the page.

Start it with:
    python3 viewer/server.py --renderer build/viewer --scene build/scene.bin --port 8000
"""
import argparse
import io
import os
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    from PIL import Image          # used to make JPEGs (Colab has it)
except ImportError:                # without it, pictures are sent as BMP: bigger, but still work
    Image = None

HERE = os.path.dirname(os.path.abspath(__file__))

# The most the camera may move in ONE picture, whatever the page asks for.
MAX_STEP_M = 2.0      # metres
MAX_TURN_RAD = 0.5    # radians


class Renderer:
    """Owns the drawing program and talks to it, one picture at a time."""

    def __init__(self, program, scene):
        self.lock = threading.Lock()   # two page requests must never talk to it at once
        self.proc = subprocess.Popen([program, scene], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        hello = self.proc.stdout.readline().decode("ascii", "replace").split()
        if len(hello) != 4 or hello[0] != "POINTWALK":
            raise RuntimeError(f"the drawing program did not start properly (it said: {hello!r})")
        self.width, self.height, self.dots = int(hello[1]), int(hello[2]), int(hello[3])

    def _read_exactly(self, count):
        data = self.proc.stdout.read(count)
        if data is None or len(data) != count:
            raise RuntimeError("the drawing program stopped")
        return data

    def frame(self, command):
        """Send one line, get back (list of 7 numbers as text, raw r g b bytes)."""
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("the drawing program stopped")
            self.proc.stdin.write((command + "\n").encode("ascii"))
            self.proc.stdin.flush()
            header = self.proc.stdout.readline().decode("ascii", "replace").split()
            if len(header) != 8 or header[0] != "FRAME":
                raise RuntimeError(f"unexpected reply from the drawing program: {header!r}")
            pixels = self._read_exactly(self.width * self.height * 3)
            return header[1:], pixels

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def encode_picture(pixels, width, height, quality):
    """Raw r g b bytes -> (file bytes, content type)."""
    if Image is not None:
        out = io.BytesIO()
        Image.frombytes("RGB", (width, height), pixels).save(out, "JPEG", quality=quality)
        return out.getvalue(), "image/jpeg"
    # BMP fallback: rows are stored bottom to top, colours as b g r, each row padded to 4 bytes.
    row_bytes = width * 3
    padding = b"\x00" * ((4 - row_bytes % 4) % 4)
    rows = []
    for y in range(height - 1, -1, -1):
        row = bytearray(pixels[y * row_bytes:(y + 1) * row_bytes])
        row[0::3], row[2::3] = row[2::3], row[0::3]
        rows.append(bytes(row) + padding)
    body = b"".join(rows)
    header = struct.pack("<2sIHHIIiiHHIIiiII", b"BM", 54 + len(body), 0, 0, 54,
                         40, width, height, 1, 24, 0, len(body), 2835, 2835, 0, 0)
    return header + body, "image/bmp"


def clamp(text, limit):
    """Turn a number from the page into a float no bigger than limit either way."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return 0.0
    if value != value:               # not-a-number
        return 0.0
    return max(-limit, min(limit, value))


def make_handler(renderer, quality):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):    # stay quiet: one line per picture would flood the log
            pass

        def send_bytes(self, status, content_type, body, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            name = url.path.rstrip("/").split("/")[-1]   # works under any prefix a proxy adds
            query = parse_qs(url.query)

            def number(key, limit):
                return clamp(query.get(key, ["0"])[0], limit)

            try:
                if name in ("", "index.html"):
                    with open(os.path.join(HERE, "index.html"), "rb") as f:
                        self.send_bytes(200, "text/html; charset=utf-8", f.read())
                elif name == "ping":
                    self.send_bytes(200, "text/plain", b"ok")
                elif name == "frame":
                    if query.get("reset", ["0"])[0] == "1":
                        command = "reset"
                    else:
                        command = "%f %f %f %f %f" % (
                            number("f", MAX_STEP_M), number("r", MAX_STEP_M), number("u", MAX_STEP_M),
                            number("t", MAX_TURN_RAD), number("l", MAX_TURN_RAD))
                    info, pixels = renderer.frame(command)
                    body, content_type = encode_picture(pixels, renderer.width, renderer.height, quality)
                    self.send_bytes(200, content_type, body, {
                        "X-Pose": " ".join(info[0:5]),       # x y z yaw pitch
                        "X-Kernel-Ms": info[5],
                        "X-Frame-Ms": info[6],
                        "X-Dots": str(renderer.dots),
                    })
                else:
                    self.send_bytes(404, "text/plain", b"not found")
            except RuntimeError as problem:
                self.send_bytes(503, "text/plain", str(problem).encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError):
                pass                                        # the page went away mid-reply

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Web server for the pointwalk viewer")
    parser.add_argument("--renderer", required=True, help="path to the drawing program (build/viewer)")
    parser.add_argument("--scene", required=True, help="path to the scene file (build/scene.bin)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality, 1 to 95")
    args = parser.parse_args()

    renderer = Renderer(args.renderer, args.scene)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(renderer, args.quality))
    server.daemon_threads = True
    print(f"viewer server: {renderer.dots} dots, {renderer.width}x{renderer.height}, "
          f"http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
