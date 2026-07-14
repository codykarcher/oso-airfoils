import copy
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from metafoil.core.kulfan import Kulfan


def fit_from_image(filepath, zeroCut=0.15, fit_order=8):
    """
    Process a single PNG image that uses a green dot as the chord-line origin
    marker.  Extracts airfoil outline coordinates from pixel data and fits
    Kulfan (CST) coefficients using ``Kulfan.fit2coordinates``.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the PNG image file.
    zeroCut : float, optional
        Normalised x/chord fraction below which upper/lower surface assignment
        is determined by the sign of y rather than the pixel-row midpoint.
        Default is 0.15.  Use 0.8 for DU 93-W-210.
    fit_order : int, optional
        Order of the Kulfan (CST) polynomial.  Default is 8.

    Returns
    -------
    afl : Kulfan
        Fitted Kulfan airfoil object.
    fig : matplotlib.figure.Figure
        Verification figure showing the image, extracted points and the fit.
    """
    img = mpimg.imread(str(filepath))

    Nrows = len(img)
    Ncols = len(img[0])

    zeroRow    = 0
    zeroColumn = 0
    maxColumn  = 0
    minColumn  = Ncols

    min_y_arr = []
    max_y_arr = []

    for j in range(Ncols):
        min_y = Nrows
        max_y = 0
        for i in range(Nrows):
            quad = img[i][j]
            if quad[0] <= 0.1 and quad[1] >= 0.9 and quad[2] <= 0.1:
                zeroRow    = i
                zeroColumn = j
            if quad[0] != 1 or quad[1] != 1 or quad[2] != 1:
                minColumn = min(minColumn, j)
                maxColumn = max(maxColumn, j)
                min_y = min(min_y, i)
                max_y = max(max_y, i)
        min_y_arr.append(min_y)
        max_y_arr.append(max_y)

    # Interpolate min_y_arr / max_y_arr over any column that does not show
    # BOTH surfaces clearly.  "Clearly" means the vertical gap between the
    # topmost and bottommost dark pixel exceeds a minimum thickness threshold.
    # This handles two problematic column types:
    #   (a) Empty gap columns between dashes  → max_y=0, min_y=Nrows → gap<0
    #   (b) Columns with only one solid-line surface → gap ≈ line thickness (3–8 px)
    # Both are excluded from the valid set and filled by linear interpolation
    # using the surrounding columns where both surfaces are visible.
    col_idx  = np.arange(Ncols)
    min_y_np = np.array(min_y_arr, dtype=float)
    max_y_np = np.array(max_y_arr, dtype=float)
    gap      = max_y_np - min_y_np
    # Threshold: larger than a drawn line (~3–8 px) but smaller than the
    # airfoil thickness near mid-chord.  3 % of image height works well.
    thickness_threshold = max(10, int(0.03 * Nrows))
    valid = gap > thickness_threshold
    if valid.sum() > 1:
        min_y_filled = np.where(valid, min_y_np,
                                np.interp(col_idx, col_idx[valid], min_y_np[valid]))
        max_y_filled = np.where(valid, max_y_np,
                                np.interp(col_idx, col_idx[valid], max_y_np[valid]))
    else:
        min_y_filled = min_y_np
        max_y_filled = max_y_np

    upperCoordinates = []
    lowerCoordinates = []
    sorter = 'u'

    for j in range(minColumn, maxColumn):
        for i in range(Nrows):
            quad = img[i][j]
            x =      (j - zeroColumn) / (maxColumn - zeroColumn)
            y = -1.0 * (i - zeroRow)  / (maxColumn - zeroColumn)

            if x <= zeroCut:
                sorter = 'u' if y >= 0 else 'l'
            else:
                sorter = 'u' if i <= (min_y_filled[j] + max_y_filled[j]) / 2.0 else 'l'

            cutoff = 0.8
            if quad[0] <= cutoff and quad[1] <= cutoff and quad[2] <= cutoff and x >= 0:
                if sorter == 'u':
                    upperCoordinates.append([x, y])
                else:
                    lowerCoordinates.append([x, y])

    lc = np.array(lowerCoordinates)
    uc = np.array(upperCoordinates)

    # The LE column (j == zeroColumn) produces many pixels all at exactly x=0.
    # argmin inside fit2coordinates.fallback grabs the first one, which sits
    # inside the upper-surface block rather than at the true upper/lower seam,
    # causing the wrong split.  Fix: exclude the noisy LE pixels (within 2 px
    # of the green-dot column) from the point clouds, then insert one clean
    # (0, 0) point at the seam.  fit2coordinates finds it immediately via the
    # primary try-path and splits correctly.
    le_gap = 2.0 / (maxColumn - zeroColumn)   # 2-pixel exclusion radius
    uc_f = uc[uc[:, 0] > le_gap]
    lc_f = lc[lc[:, 0] > le_gap]

    uc_sorted = uc_f[np.argsort(-uc_f[:, 0])]   # descending x: TE → LE
    lc_sorted = lc_f[np.argsort( lc_f[:, 0])]   # ascending  x: LE → TE

    # Inject the exact origin at the seam so fit2coordinates never falls back.
    combined_psi  = np.concatenate([uc_sorted[:, 0], [0.0], lc_sorted[:, 0]])
    combined_zeta = np.concatenate([uc_sorted[:, 1], [0.0], lc_sorted[:, 1]])

    afl = Kulfan()
    afl.fit2coordinates(copy.deepcopy(combined_psi), copy.deepcopy(combined_zeta),
                        fit_order=fit_order)

    # --- verification figure ---
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(img, aspect='equal', alpha=0.2)
    plt.plot([zeroColumn, maxColumn], [zeroRow, zeroRow], 'b')
    plt.plot(min_y_filled, 'g', lw=1.0, label='min_y (filled)')
    plt.plot(max_y_filled, 'r', lw=1.0, label='max_y (filled)')
    plt.plot(lc[:, 0] * (maxColumn - zeroColumn) + zeroColumn,
             -lc[:, 1] * (maxColumn - zeroColumn) + zeroRow, 'c.')
    plt.plot(uc[:, 0] * (maxColumn - zeroColumn) + zeroColumn,
             -uc[:, 1] * (maxColumn - zeroColumn) + zeroRow, 'm.')
    plt.plot(afl.xcoordinates * (maxColumn - zeroColumn) + zeroColumn,
             -afl.ycoordinates * (maxColumn - zeroColumn) + zeroRow, 'k')

    return afl, fig


if __name__ == '__main__':
    wd = 'png_files'
    files = sorted(
        f for f in os.listdir(wd)
        if os.path.isfile(os.path.join(wd, f))
        and f[0] not in ('.', '_')
        and f.endswith('.png')
        and 'mhkf1' not in f
    )

    for fl in files:
        stem = fl[:-4]
        zero_cut = 0.8 if stem == 'du_93-w-210' else 0.15
        afl, fig = fit_from_image(os.path.join(wd, fl), zeroCut=zero_cut)
        fig.savefig(os.path.join('verification_plots', stem + '.png'), dpi=300)
        plt.close(fig)
        afl.write2file(os.path.join('fitted_datfiles', stem + '.dat'))


