import os
import pathlib
from metafoil.core.kulfan import Kulfan


def get_geometry_from_dir(name, datfiles_dir, family_name='', strict=False,
                          kulfan_order=None):
    """
    Load a Kulfan airfoil object by searching for a .dat file matching `name`
    inside `datfiles_dir`.

    Matching is case-insensitive and treats underscores as hyphens. By default
    the search first tries the full filename stem, then the stem with the
    leading family-prefix word removed. Pass ``strict=True`` to skip the
    stripped-prefix fallback (required when searching across all families to
    avoid ambiguous short names matching in the wrong family).

    Parameters
    ----------
    name : str or list of str
        Airfoil name(s) to look up. If a list is given, returns a dict
        ``{name: Kulfan}``. Names not found return ``None`` with a warning.
    datfiles_dir : path-like
        Directory containing .dat files.
    family_name : str, optional
        Human-readable family label used in the error message.
    strict : bool, optional
        If True, only match on the full filename stem (no stripped prefix).
    kulfan_order : int, optional
        If given, refit the loaded airfoil to this Kulfan polynomial order
        before returning.
    """
    if isinstance(name, (list, tuple)):
        import warnings
        result = {}
        for n in name:
            try:
                result[n] = get_geometry_from_dir(n, datfiles_dir, family_name,
                                                  strict=strict, kulfan_order=kulfan_order)
            except ValueError:
                warnings.warn('Airfoil "{}" could not be found, returning None.'.format(n))
                result[n] = None
        return result

    datfiles_dir = pathlib.Path(datfiles_dir)
    fls = sorted([f for f in os.listdir(datfiles_dir) if f.endswith('.dat')])
    fnames = [f.split('.')[0].lower().replace('_', '-') for f in fls]
    fnames_stripped = ['-'.join(f.split('-')[1:]) for f in fnames]
    search_name = name.lower().replace('_', '-')

    if search_name in fnames:
        idx = fnames.index(search_name)
    elif not strict and search_name in fnames_stripped:
        idx = fnames_stripped.index(search_name)
    else:
        label = family_name + ' ' if family_name else ''
        raise ValueError(
            'Airfoil not found, valid {}airfoils are: {}'.format(
                label, [f.upper() for f in fnames]
            )
        )

    afl = Kulfan()
    afl.readFile(datfiles_dir / fls[idx])
    if kulfan_order is not None:
        afl.changeOrder(kulfan_order)
    return afl


def _tau_float(afl):
    """Return tau as a plain float regardless of whether it is a Pint quantity."""
    tau = afl.tau
    if hasattr(tau, 'magnitude'):
        tau = tau.to('').magnitude
    return float(tau)


def get_all_geometry_from_dir(datfiles_dir):
    """
    Load every .dat file in *datfiles_dir* and return an ordered dict
    ``{stem: Kulfan}`` sorted first by thickness (tau) then alphabetically
    by filename stem.
    """
    datfiles_dir = pathlib.Path(datfiles_dir)
    fls = sorted([f for f in os.listdir(datfiles_dir) if f.endswith('.dat')])
    entries = []
    for fl in fls:
        stem = fl.split('.')[0]
        afl = Kulfan()
        afl.readFile(datfiles_dir / fl)
        entries.append((_tau_float(afl), stem.lower(), stem, afl))
    entries.sort(key=lambda e: (e[0], e[1]))
    return {e[2]: e[3] for e in entries}


def sort_airfoil_dict(airfoil_dict):
    """Re-sort an existing ``{name: Kulfan}`` dict by thickness then name."""
    entries = [(_tau_float(afl), name.lower(), name, afl)
               for name, afl in airfoil_dict.items()]
    entries.sort(key=lambda e: (e[0], e[1]))
    return {e[2]: e[3] for e in entries}


def plot_family_from_dir(datfiles_dir, title='Airfoil Family', keys=None, save_path=None):
    """
    Plot all (or a filtered subset of) .dat files in `datfiles_dir` on a
    single axes with equal aspect ratio.

    Parameters
    ----------
    datfiles_dir : path-like
        Directory containing .dat files.
    title : str
        Plot title.
    keys : list of str, optional
        If provided, only plot airfoils whose filename stem (upper-cased)
        appears in this list.
    save_path : path-like, optional
        If provided, save the figure to this path at 300 dpi and close it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    datfiles_dir = pathlib.Path(datfiles_dir)
    fls = sorted([f for f in os.listdir(datfiles_dir) if f.endswith('.dat')])
    if keys is not None:
        norm_keys = [k.lower() for k in keys]
        fls = [f for f in fls if f.split('.')[0].lower() in norm_keys]
    fig, ax = plt.subplots(figsize=(12, 6))
    for fl in fls:
        afl = Kulfan()
        afl.readFile(datfiles_dir / fl)
        label = fl.split('.')[0].upper()
        ax.plot(afl.xcoordinates, afl.ycoordinates, label=label)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize='small')
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return fig

