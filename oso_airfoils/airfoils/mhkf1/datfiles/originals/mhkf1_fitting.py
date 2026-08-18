import pathlib

import matplotlib.pyplot as plt

from oso_airfoils.core.process_airfoils import fit_from_image

path_to_here = pathlib.Path(__file__).parent.resolve()

png_dir          = path_to_here / 'png_files'
verification_dir = path_to_here / 'verification_plots'
datfiles_dir     = path_to_here.parent   # oso_airfoils/airfoils/mhkf1/datfiles/

# (png filename, zeroCut, target_tau)
file_configs = [
    ('mhkf1-180.png', 0.15, 0.180),
    ('mhkf1-240.png', 0.15, 0.240),
    ('mhkf1-400.png', 0.15, 0.400),
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

    dat_path = datfiles_dir / (stem + '.dat')
    afl.write2file(str(dat_path))
    print(f'  dat file          → {dat_path}')

