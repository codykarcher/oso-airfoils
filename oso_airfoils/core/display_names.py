"""oso_airfoils.core.display_names
==================================
Utilities for converting raw airfoil stem names into human-readable display
labels used in plot legends and figure titles.
"""
from __future__ import annotations

import re


def pretty_name(stem: str) -> str:
    """Convert a raw airfoil stem name to a human-readable display label.

    Rules applied in order
    ----------------------
    * Underscores are replaced with hyphens.
    * ``du-`` prefix (case-insensitive) → ``DU-``.
    * ``-w<digits>-`` substring (e.g. ``-w-``, ``-w2-``) → ``-W<digits>-``.
    * ``riso-a-``, ``riso-b-``, ``riso-p-`` prefix → ``Risø-A-``, etc.

    Names that don't match any rule are returned unchanged.

    Examples
    --------
    >>> pretty_name('du_93-w-210')
    'DU-93-W-210'
    >>> pretty_name('du_91-w2-250')
    'DU-91-W2-250'
    >>> pretty_name('riso-a-21')
    'Risø-A-21'
    >>> pretty_name('riso-b-35')
    'Risø-B-35'
    >>> pretty_name('FFA-W3-211')
    'FFA-W3-211'
    >>> pretty_name('OSO-2025-WT2-T21')
    'OSO-2025-WT2-T21'
    """
    n = stem.replace('_', '-')
    n = re.sub(r'(?i)^du-', 'DU-', n)
    n = re.sub(r'(?i)-w(\d*)-', lambda m: f'-W{m.group(1)}-', n)
    n = re.sub(r'(?i)^riso-([abp])-', lambda m: f'Risø-{m.group(1).upper()}-', n)
    return n
