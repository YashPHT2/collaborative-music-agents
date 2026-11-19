"""Regression checks for audio seeking over HTTP (no browser required)."""
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from serve_evaluation import RangeRequestHandler


class QuietHandler(RangeRequestHandler):
    def log_message(self, *args):
        pass


class AudioRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.content = bytes(range(256)) * 4
        (Path(cls.folder.name) / "sample.mp3").write_bytes(cls.content)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=cls.folder.name))
        cls.worker = Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()
        cls.folder.cleanup()

    def request(self, byte_range=None, method="GET"):
        connection = HTTPConnection(*self.server.server_address, timeout=5)
        connection.request(method, "/sample.mp3?playback=2", headers={"Range": byte_range} if byte_range else {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_full_response_advertises_seeking(self):
        status, headers, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(body, self.content)

    def test_partial_response_has_exact_bytes_and_headers(self):
        status, headers, body = self.request("bytes=120-199")
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], "bytes 120-199/1024")
        self.assertEqual(headers["Content-Length"], "80")
        self.assertEqual(body, self.content[120:200])

    def test_suffix_and_open_ended_ranges(self):
        self.assertEqual(self.request("bytes=-50")[2], self.content[-50:])
        self.assertEqual(self.request("bytes=900-")[2], self.content[900:])
        self.assertEqual(self.request("bytes=1000-9999")[2], self.content[1000:])

    def test_invalid_ranges_do_not_return_wrong_audio(self):
        for header in ["bytes=9999-", "bytes=300-200", "bytes=-0", "bytes=-", "bytes=0-2,5-9"]:
            with self.subTest(header=header):
                status, headers, body = self.request(header)
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], "bytes */1024")
                self.assertEqual(body, b"")

    def test_head_returns_range_metadata_without_audio_body(self):
        status, headers, body = self.request("bytes=100-199", method="HEAD")
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Length"], "100")
        self.assertEqual(body, b"")


if __name__ == "__main__":
    unittest.main()
