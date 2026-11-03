"""Preview the static UI and prediction endpoint locally."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from api.predict import handler


class Preview(SimpleHTTPRequestHandler, handler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, directory=str(Path(__file__).parent / "public"), **kwargs
        )


if __name__ == "__main__":
    print("Rate desk: http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Preview).serve_forever()
