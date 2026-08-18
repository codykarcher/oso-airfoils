"""
server.py -- static file server for the live dashboard, plus one endpoint.

``python -m http.server`` would do for the files, but the family buttons need to send
a selection back, so this adds:

    GET /select?family=<key|empty>    ->  thickness-matched member of that family
    GET /select?airfoil=<stem|empty>  ->  one named airfoil from the whole library
    GET /savepath?path=<dir>          ->  relocate this run's saved snapshots

Either writes selection.json; they are mutually exclusive (one reference at a time).

The plotter watches selection.json and re-renders the current generation's polar with
that family's geometry as the reference.
"""

import json
import os
import pathlib
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = pathlib.Path(__file__).parent.resolve()

# Per-run state. Several dashboards may be live at once (a GA run and a gradient
# run, or two thicknesses), each on its own port with its own directory, so
# state.json / frames / selection.json must NOT be shared. ROOT is set by
# serve(); it defaults to HERE so the original single-run usage is unchanged.
ROOT = HERE
SELECTION = HERE / 'selection.json'
SAVEPATH = HERE / 'savepath.json'


def _set_root(root):
    global ROOT, SELECTION, SAVEPATH
    ROOT = pathlib.Path(root).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    SELECTION = ROOT / 'selection.json'
    SAVEPATH = ROOT / 'savepath.json'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def list_directory(self, path):
        """Serve the dashboard instead of a directory listing.

        SimpleHTTPRequestHandler resolves a directory request by looking for
        index.html INSIDE that directory, and that lookup does not go through
        translate_path -- so with a per-run ROOT holding only state.json and
        frames/, a request for '/' fell through to a file listing. It returned
        HTTP 200, which is why a status-code check did not catch it.
        """
        idx = HERE / 'index.html'
        if idx.is_file():
            body = idx.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            import io
            return io.BytesIO(body)
        return super().list_directory(path)

    def translate_path(self, path):
        """Serve run artefacts from ROOT, static assets from the package dir.

        A run directory holds only what the run produces (state.json, frames/).
        index.html and any sibling assets live with the sources, so anything not
        found under ROOT falls back to HERE instead of 404ing -- that way a run
        directory needs no copied or symlinked boilerplate.
        """
        p = pathlib.Path(super().translate_path(path))
        if not p.exists() and ROOT != HERE:
            try:
                alt = HERE / p.relative_to(ROOT)
                if alt.exists():
                    return str(alt)
            except ValueError:
                pass
        return str(p)

    def log_message(self, *a):
        pass                                # keep the console for GA output

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/select':
            q = parse_qs(u.query)
            fam = (q.get('family', [''])[0] or '').strip()
            afl = (q.get('airfoil', [''])[0] or '').strip()
            sel = {'family': fam or None, 'airfoil': afl or None}
            tmp = str(SELECTION) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(sel, f)
            os.replace(tmp, SELECTION)
            body = json.dumps({'ok': True, **sel}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == '/savepath':
            # Only RECORD the request. The GA process owns the files and the writer,
            # so it performs the move and reports the outcome in state.json.
            want = (parse_qs(u.query).get('path', [''])[0] or '').strip()
            tmp = str(SAVEPATH) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'path': want or None}, f)
            os.replace(tmp, SAVEPATH)
            body = json.dumps({'ok': True, 'path': want or None}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()


def serve(port=8777, root=None, background=False):
    """Start the dashboard server. Returns the server object.

    `background=True` runs it on a daemon thread so a run driver can host its own
    dashboard in-process -- one command instead of two, and the port travels with
    the run rather than being a separate thing to remember.
    """
    import threading
    _set_root(root or HERE)
    for f in (SELECTION, SAVEPATH):
        if f.exists():
            f.unlink()
    srv = ThreadingHTTPServer(('127.0.0.1', int(port)), Handler)
    print(f'serving {ROOT} at http://localhost:{port}', flush=True)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        srv.serve_forever()
    return srv


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='live dashboard server')
    ap.add_argument('port', nargs='?', type=int, default=8777)
    ap.add_argument('--root', default=None,
                    help='directory holding this run\'s state.json and frames/ '
                         '(default: the package dir). Give each concurrent run '
                         'its own root and its own port so they do not overwrite '
                         'each other.')
    a = ap.parse_args()
    serve(a.port, a.root)
