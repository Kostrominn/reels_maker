"""A local server for the watch page with HTTP Range support, so videos can be
seeked (python -m http.server cannot). Listens on 127.0.0.1 only."""
from __future__ import annotations

import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        m = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start, end = m.groups()
        if start:
            start, end = int(start), min(int(end) if end else size - 1, size - 1)
        else:                                    # suffix range: the last N bytes
            start, end = max(0, size - int(end)), size - 1
        if start >= size or start > end:
            self.send_error(416, 'Requested Range Not Satisfiable')
            return None
        f = open(path, 'rb')
        f.seek(start)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, '_left', None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(1 << 20, left))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break                            # the browser drops ranges it no longer needs
            left -= len(chunk)

    def end_headers(self):
        if not self.headers.get('Range'):
            self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()


def serve(root, port=8765):
    print(f'http://127.0.0.1:{port}/СМОТРЕТЬ.html  ({root})')
    ThreadingHTTPServer(('127.0.0.1', port), partial(RangeHandler, directory=str(root))).serve_forever()
