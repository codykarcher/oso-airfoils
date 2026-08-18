import pathlib

import matplotlib.pyplot as plt

from oso_airfoils.core.process_airfoils import fit_from_image

path_to_here = pathlib.Path(__file__).parent.resolve()

png_dir          = path_to_here / 'png_files'
verification_dir = path_to_here / 'verification_plots'
datfiles_dir     = path_to_here.parent   # oso_airfoils/airfoils/riso_a/datfiles/

# (png filename, zeroCut, target_tau)
file_configs = [
    ('riso-a-12.png', 0.15, 0.120),
    ('riso-a-15.png', 0.15, 0.150),
    ('riso-a-18.png', 0.15, 0.180),
    ('riso-a-21.png', 0.15, 0.210),
    ('riso-a-24.png', 0.15, 0.240),
    ('riso-a-27.png', 0.15, 0.270),
    ('riso-a-30.png', 0.15, 0.300),
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

