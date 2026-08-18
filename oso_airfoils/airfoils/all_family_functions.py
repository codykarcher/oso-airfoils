import warnings
import importlib
import pathlib
from oso_airfoils.core.colors import default_color_cycle as dcc
from oso_airfoils.core.airfoil_family import get_geometry_from_dir, get_all_geometry_from_dir, sort_airfoil_dict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=dcc)

_here = pathlib.Path(__file__).parent.resolve()


def _discover_family_modules():
    """
    Auto-discover family modules by scanning subdirectories for
    family_functions.py. Skips any directory without that file.
    """
    modules = []
    for d in sorted(_here.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith('.') or d.name.startswith('_'):
            continue
        if not (d / 'family_functions.py').exists():
            continue
        try:
            mod = importlib.import_module(
                'oso_airfoils.airfoils.{}.family_functions'.format(d.name)
            )
            modules.append(mod)
        except ImportError:
            pass
    return modules


_FAMILY_MODULES = _discover_family_modules()


def _try_all_families(single_name):
    """Try each discovered family using strict (full-name-only) matching."""
    for mod in _FAMILY_MODULES:
        if not hasattr(mod, '_datfiles') or not hasattr(mod, '_FAMILY'):
            continue
        try:
            return get_geometry_from_dir(single_name, mod._datfiles, mod._FAMILY, strict=True)
        except (ValueError, FileNotFoundError):
            pass
    raise ValueError('Airfoil "{}" not found in any registered family.'.format(single_name))


def get_geometry(name=None):
    if name is None:
        merged = {}
        for mod in _FAMILY_MODULES:
            if hasattr(mod, '_datfiles'):
                merged.update(get_all_geometry_from_dir(mod._datfiles))
        return sort_airfoil_dict(merged)
    if isinstance(name, (list, tuple)):
        result = {}
        for n in name:
            try:
                result[n] = _try_all_families(n)
            except ValueError:
                warnings.warn('Airfoil "{}" could not be found in any family, returning None.'.format(n))
                result[n] = None
        return result
    return _try_all_families(name)


def get_coordinates(name=None):
    result = get_geometry(name)
    if isinstance(result, dict):
        return {n: (afl.xcoordinates, afl.ycoordinates) if afl is not None else None
                for n, afl in result.items()}
    return result.xcoordinates, result.ycoordinates


def plot_geometry(name):
    _max = len(dcc)
    if isinstance(name, str):
        names = [name]
    else:
        names = list(name)
    if len(names) > _max:
        raise ValueError(
            'The plot_geometry function received {} airfoils but the default color cycle only '
            'has {} entries. This function is not intended to process large numbers '
            'of airfoils — retrieve data with get_geometry() and build a custom '
            'plot instead.'.format(len(names), _max)
        )
    airfoils = get_geometry(names)
    fig, ax = plt.subplots(figsize=(12, 6))
    for n, afl in airfoils.items():
        if afl is None:
            continue
        ax.plot(afl.xcoordinates, afl.ycoordinates, label=n.upper())
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize='small')
    fig.tight_layout()
    return fig


def get_performance_data():
    raise NotImplementedError('TBD')


def ingest_performance_data():
    raise NotImplementedError('TBD')


if __name__ == "__main__":
    # Single lookup
    afl = get_geometry('ffa-w3-211')
    print(afl)

    # Multi lookup
    afls = get_geometry(['s809', 'du_93-w-210', 'riso-a-18', 'nonexistent'])
    for name, afl in afls.items():
        print(name, ':', afl)

    # All lookup
    all_afls = get_geometry()
    for name, afl in all_afls.items():
        print(name, ':', afl)


    # Multi lookup
    afls = plot_geometry(['s809', 'du_93-w-210', 'riso-a-18', 'nonexistent'])
    afls.savefig(_here / 'Example_Plot.png', dpi=300, bbox_inches='tight')
