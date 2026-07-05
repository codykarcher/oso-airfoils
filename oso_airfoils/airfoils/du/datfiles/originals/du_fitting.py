import pathlib

import matplotlib.pyplot as plt

from oso_airfoils.core.process_airfoils import fit_from_image

path_to_here = pathlib.Path(__file__).parent.resolve()

png_dir          = path_to_here / 'png_files'
verification_dir = path_to_here / 'verification_plots'
originals_dir    = path_to_here
datfiles_dir     = path_to_here.parent   # oso_airfoils/airfoils/du/datfiles/

# (png filename, zeroCut, target_tau)
# zeroCut=0.8 is required for DU 93-W-210 to correctly split upper/lower
# near the leading edge of the scan.
# target_tau: last three digits of name / 1000 for individual DU profiles;
# None for the generic family overlays (no named thickness to enforce).
file_configs = [
    ('afl-du-family1.png', 0.15, None),
    ('afl-du-family2.png', 0.15, None),
    ('afl-du-family3.png', 0.15, None),
    ('afl-du-family4.png', 0.15, None),
    ('du_91-w2-250.png',   0.15, 0.250),
    ('du_93-w-210.png',    0.80, 0.210),
    ('du_96-w-180.png',    0.15, 0.180),
    ('du_97-w-300.png',    0.15, 0.300),
]

for filename, zero_cut, target_tau in file_configs:
    stem = filename[:-4]   # strip .png
    print(f'Processing {stem} ...')

    afl, fig = fit_from_image(png_dir / filename, zeroCut=zero_cut, fit_order=8,
                              target_tau=target_tau)

    # Verification plot
    plot_path = verification_dir / (stem + '.png')
    fig.savefig(str(plot_path), dpi=300)
    plt.close(fig)
    print(f'  verification plot → {plot_path}')

    # Dat file: filenames starting with 'du' are promoted to the datfiles
    # folder; family/reference airfoils stay in originals.
    if stem.startswith('du'):
        dat_path = datfiles_dir / (stem + '.dat')
    else:
        dat_path = originals_dir / (stem + '.dat')

    afl.write2file(str(dat_path))
    print(f'  dat file          → {dat_path}')

