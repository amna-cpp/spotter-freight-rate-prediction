"""A small JSON endpoint for one load at a time."""

import json
from http.server import BaseHTTPRequestHandler

from model.inference import estimate


class handler(BaseHTTPRequestHandler):
    def respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/predict":
            return self.respond(404, {"error": "Not found."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                return self.respond(413, {"error": "Send one load at a time."})
            payload = json.loads(self.rfile.read(size))
            result = estimate(payload)
        except (ValueError, TypeError, json.JSONDecodeError):
            return self.respond(400, {"error": "Check the load inputs and try again."})
        return self.respond(200, result)

    def do_GET(self):
        return self.respond(405, {"error": "Use POST to estimate a load."})
