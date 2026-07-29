"""
ref_library.py — reference-airfoil metadata, loaded OFF the startup path.

`families.survey()` and `families.library()` together take ~95 s on this machine
(measured: 31.5 s + 63.4 s, for 11 families and 87 airfoils). Building them
inline made that a hard startup delay -- and in live_gradient, where they sat in
the meta dict above the server start, the dashboard could not even be served
until the scan finished.

They are only needed for the reference-overlay buttons, which nobody clicks in
the first minute of a run, so they load on a daemon thread and are patched into
the SAME meta dict the Plotter already holds. `write_state` serialises whatever
meta contains at the time, so the buttons simply appear once ready.
"""
import threading


def populate_async(meta, tau, on_done=None):
    """Fill meta['families'] and meta['library'] in the background."""
    def work():
        try:
            from oso_airfoils.live import families
            meta['families'] = families.survey(tau)
            meta['library'] = families.library()
            if on_done:
                on_done(len(meta['families']), len(meta['library']))
        except Exception as e:                      # never take the run down for this
            print(f'  [ref] reference library unavailable: '
                  f'{type(e).__name__}: {e}', flush=True)
    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t
