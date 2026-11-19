"""Serve the local listening test with HTTP byte ranges for reliable audio seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        # Avoid retaining media responses from the earlier non-seekable server.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        self.remaining = None
        range_header = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not range_header or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        start, end = 0, size - 1
        valid = match is not None and size > 0
        if valid:
            first, last = match.groups()
            valid = bool(first or last)
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                length = int(last)
                valid = length > 0
                start = max(0, size - length)
            valid = valid and 0 <= start <= end < size
        if not valid:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        remaining = self.remaining
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path, help="Serve an explicit local evaluation directory")
    parser.add_argument("--report", choices=["baseline", "ppo", "band", "adaptive", "cinematic", "notes", "notes-more", "supervised", "partners", "offline", "repair", "joint-rl"], default="baseline")
    args = parser.parse_args()
    folder = {"ppo": "ppo_pilot_v1", "baseline": "evaluation_v1", "band": "band_joint_v1", "adaptive":"adaptive_band_v2", "cinematic":"cinematic_trio_v1", "notes":"note_trio_v1", "notes-more":"note_trio_v2", "supervised":"supervised_trio_v1", "partners":"partner_response_v1", "offline":"offline_ensemble_v1", "repair":"ensemble_repair_v1", "joint-rl":"joint_ensemble_rl_v1"}[args.report]
    root = Path(__file__).resolve().parent / "outputs" / folder
    if args.directory is not None:
        root=args.directory.resolve()
        if not root.is_dir():parser.error('Evaluation directory does not exist')
    handler = partial(RangeRequestHandler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Listening test: http://127.0.0.1:{args.port}/listen.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
