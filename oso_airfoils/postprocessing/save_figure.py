"""oso_airfoils.postprocessing.save_figure
==========================================
Save a matplotlib figure to multiple formats with an automatic dark-mode copy.

Ported from ``customPlot.py`` that lived alongside earlier publication notebooks.
"""
from __future__ import annotations

import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D


def _is_black(color) -> bool:
    """Return True when *color* resolves to black (or near-black)."""
    if color in ('black', 'k', '#000000'):
        return True
    try:
        arr = np.asarray(plt.matplotlib.colors.to_rgba(color))
        return bool(np.all(arr[:3] < 0.05))
    except Exception:
        return False


def _apply_dark_legend(leg: Legend, legend_dict: dict) -> None:
    """Update one legend for dark-mode rendering."""
    leg.get_frame().set_facecolor(legend_dict['facecolor'])
    leg.get_frame().set_edgecolor(legend_dict['edgecolor'])
    for text in leg.get_texts():
        text.set_color(legend_dict['labelcolor'])
    # Whiten handle lines/markers that are currently black.
    for handle in leg.legend_handles:
        if isinstance(handle, Line2D):
            if _is_black(handle.get_color()):
                handle.set_color('white')
            mfc = handle.get_markerfacecolor()
            if mfc != 'none' and _is_black(mfc):
                handle.set_markerfacecolor('white')
            mec = handle.get_markeredgecolor()
            if mec != 'none' and _is_black(mec):
                handle.set_markeredgecolor('white')


def save_figure(fig, fname: str, *args, **kwargs) -> None:
    """Save *fig* to *fname* (plus ``.svg``, ``.pdf``, and ``.pgf``), and a dark-mode copy.

    Four light-mode files are always written:
      ``<stem>.svg``, ``<stem>.pdf``, ``<stem>.pgf``, ``<stem>.<ext>``

    Three dark-mode files are also written:
      ``<stem>_dark.svg``, ``<stem>_dark.pdf``, ``<stem>_dark.<ext>``

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    fname : str
        Output path with **exactly one** extension, e.g. ``'plots/out.png'``.
    *args, **kwargs
        Forwarded to ``fig.savefig``.  The special kwarg ``legend_dictionary``
        is consumed here and not forwarded — it customises the legend style
        used in the dark-mode copy.
    """
    legend_dict = kwargs.pop('legend_dictionary', None)
    if legend_dict is None:
        legend_dict = {'facecolor': 'black', 'labelcolor': 'white', 'edgecolor': 'white'}
    else:
        legend_dict.setdefault('facecolor', 'black')
        legend_dict.setdefault('labelcolor', 'white')
        legend_dict.setdefault('edgecolor', 'white')

    assert fname.count('.') == 1, (
        'fname must contain exactly one period to separate stem and extension, '
        f'e.g. "out.png".  Got: {fname!r}'
    )
    stem, ext = fname.rsplit('.', 1)

    # Light-mode saves
    fig.savefig(f'{stem}.svg', *args, **kwargs)
    fig.savefig(f'{stem}.pdf', *args, **kwargs)
    fig.savefig(f'{stem}.pgf', *args, **kwargs)
    fig.savefig(f'{stem}.{ext}', *args, **kwargs)

    # Dark-mode copy — build kwargs that force a black background
    # regardless of whether transparent=True was passed for the light saves.
    dark_kwargs = dict(kwargs)
    dark_kwargs['facecolor'] = 'black'
    dark_kwargs['transparent'] = False

    # Pre-deepcopy: apply dark mode to figure-level legends on the ORIGINAL
    # figure so the styling is captured by deepcopy.  copy.deepcopy does not
    # reliably preserve fig.legends, so this guarantees they arrive dark in
    # fig2.  We save the original colours and restore them afterwards.
    _fig_leg_states: list[dict] = []
    for _leg in fig.legends:
        _s: dict = {
            'fc':    tuple(float(x) for x in _leg.get_frame().get_facecolor()),
            'ec':    tuple(float(x) for x in _leg.get_frame().get_edgecolor()),
            'texts': [_t.get_color() for _t in _leg.get_texts()],
        }
        _fig_leg_states.append(_s)
        _apply_dark_legend(_leg, legend_dict)

    fig2 = copy.deepcopy(fig)

    # Restore the original figure's legend colours so the light-mode object
    # is left unchanged after this function returns.
    for _leg, _s in zip(fig.legends, _fig_leg_states):
        _leg.get_frame().set_facecolor(_s['fc'])
        _leg.get_frame().set_edgecolor(_s['ec'])
        for _t, _c in zip(_leg.get_texts(), _s['texts']):
            _t.set_color(_c)
    fig2.set_edgecolor('white')
    fig2.set_facecolor('black')

    for ax in fig2.axes:
        ax.set_facecolor('black')

        if ax.xaxis.label.get_color() == 'black':
            ax.xaxis.label.set_color('white')
        if ax.yaxis.label.get_color() == 'black':
            ax.yaxis.label.set_color('white')

        # Match original: only force white ticks when they are currently black
        if 'color' in ax.xaxis.get_tick_params():
            if ax.xaxis.get_tick_params()['color'] == 'black':
                ax.tick_params(axis='x', which='both', colors='white')
        else:
            ax.tick_params(axis='x', which='both', colors='white')

        if 'color' in ax.yaxis.get_tick_params():
            if ax.yaxis.get_tick_params()['color'] == 'black':
                ax.tick_params(axis='y', which='both', colors='white')
        else:
            ax.tick_params(axis='y', which='both', colors='white')

        for spine in ax.spines.values():
            ec = spine.get_edgecolor()
            if ec[0] == 0.0 and ec[1] == 0.0 and ec[2] == 0.0:
                spine.set_edgecolor('white')

        ax.title.set_color('white')

        # Apply dark styling to ALL legends on this axis — including those
        # added via add_artist() which are not returned by ax.get_legend().
        for child in ax.get_children():
            if isinstance(child, Legend):
                _apply_dark_legend(child, legend_dict)

    # Apply dark styling to figure-level legends (created via fig.legend()).
    for leg in fig2.legends:
        _apply_dark_legend(leg, legend_dict)

    fig2.savefig(f'{stem}_dark.svg', *args, **dark_kwargs)
    fig2.savefig(f'{stem}_dark.pdf', *args, **dark_kwargs)
    fig2.savefig(f'{stem}_dark.{ext}', *args, **dark_kwargs)
    plt.close(fig2)
