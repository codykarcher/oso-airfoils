import copy
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from metafoil.core.kulfan import Kulfan


def fit_from_image(filepath, zeroCut=0.15, fit_order=8, target_tau=None):
    """
    Process a single PNG image without requiring a green dot origin marker.
    Detects the leading and trailing edges computationally, applies a rotation
    correction so the chord is horizontal, then fits Kulfan (CST) coefficients
    using ``Kulfan.fit2coordinates``.  The trailing-edge gap is split evenly
    about y=0 (``TE_shift`` forced to 0).

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the PNG image file.
    zeroCut : float, optional
        Normalised x/chord fraction below which upper/lower surface assignment
        is based on sign of y rather than pixel-row midpoint.  Default 0.15.
    fit_order : int, optional
        Order of the Kulfan (CST) polynomial.  Default 8.
    target_tau : float or None, optional
        If given, the fitted airfoil is scaled to this thickness/chord ratio
        using ``Kulfan.scaleThickness``.  For DU airfoils the value is the
        last three digits of the name divided by 1000 (e.g. ``du_93-w-210``
        → ``0.21``).  Default is ``None`` (no scaling applied).

    Returns
    -------
    afl : Kulfan
        Fitted Kulfan airfoil object (chord-aligned, TE gap evenly split).
    fig : matplotlib.figure.Figure
        Verification figure showing image, extracted points, and fit.
    """
    img = mpimg.imread(str(filepath))

    Nrows = len(img)
    Ncols = len(img[0])

    maxColumn = 0
    minColumn = Ncols

    min_y_arr = []
    max_y_arr = []

    for j in range(Ncols):
        min_y, max_y = Nrows, 0
        for i in range(Nrows):
            q = img[i][j]
            if q[0] != 1 or q[1] != 1 or q[2] != 1:
                minColumn = min(minColumn, j)
                maxColumn = max(maxColumn, j)
                min_y = min(min_y, i)
                max_y = max(max_y, i)
        min_y_arr.append(min_y)
        max_y_arr.append(max_y)

    min_y_filled = np.array(min_y_arr, dtype=float)
    max_y_filled = np.array(max_y_arr, dtype=float)

    # Identify the origin:
    # 1. minColumn is the leftmost column containing any non-white pixel.
    # 2. Find all non-white rows at minColumn; take their vertical midpoint as
    #    zeroRow.  This handles a vertical leading-edge line (e.g. DU 93-W-210)
    #    where several pixel rows are lit at the same leftmost column.
    # 3. Scan rightward along zeroRow_int from minColumn until the row turns
    #    white; average [minColumn, leading_edge_end] as zeroColumn.
    le_rows = [i for i in range(Nrows)
               if img[i][minColumn][0] != 1
               or img[i][minColumn][1] != 1
               or img[i][minColumn][2] != 1]
    zeroRow     = (min(le_rows) + max(le_rows)) / 2.0
    zeroRow_int = int(round(zeroRow))
    leading_edge_end = minColumn
    for j in range(minColumn, maxColumn):
        q = img[zeroRow_int][j]
        if q[0] == 1 and q[1] == 1 and q[2] == 1:
            leading_edge_end = j
            break
    zeroColumn = (minColumn + leading_edge_end) / 2.0

    # Extract pixel coordinates (stop at valid_max to exclude single-surface
    # extension columns beyond the actual TE gap).
    upperCoordinates = []
    lowerCoordinates = []
    sorter = 'u'

    for j in range(minColumn, maxColumn):
        for i in range(Nrows):
            q = img[i][j]
            x =      (j - zeroColumn) / (maxColumn - zeroColumn)
            y = -1.0 * (i - zeroRow)  / (maxColumn - zeroColumn)

            if x <= zeroCut:
                sorter = 'u' if y >= 0 else 'l'
            else:
                sorter = 'u' if i <= (min_y_filled[j] + max_y_filled[j]) / 2.0 else 'l'

            cutoff = 0.8
            if q[0] <= cutoff and q[1] <= cutoff and q[2] <= cutoff and x >= 0:
                (upperCoordinates if sorter == 'u' else lowerCoordinates).append([x, y])

    lc = np.array(lowerCoordinates)
    uc = np.array(upperCoordinates)

    # --- rotation correction ---
    # Compute the angle the chord makes with horizontal from the TE midpoint,
    # then rotate ALL pixel coordinates to level the chord before fitting.
    # This bakes the correction into the fit rather than applying it only to
    # the verification plot (which was the bug in the original script).
    #
    # Use the last column where *both* surfaces are present (gap > threshold).
    # At maxColumn the outline can taper to a single-pixel tail (min_y==max_y),
    # which corrupts te_mid_y and hence the chord angle.
    _te_threshold = 3
    effective_te_col = maxColumn
    for _j in range(maxColumn, minColumn, -1):
        if max_y_filled[_j] - min_y_filled[_j] > _te_threshold:
            effective_te_col = _j
            break
    te_mid_y = (min_y_filled[effective_te_col] + max_y_filled[effective_te_col]) / 2.0
    y_te     = -1.0 * (te_mid_y - zeroRow) / (maxColumn - zeroColumn)
    angle    = np.arctan2(y_te, 1.0)   # chord tilt angle
    ca, sa   = np.cos(angle), np.sin(angle)

    def _rotate(pts):
        # Rotation by -angle about origin: levels the chord
        # R(-angle) = [[ca, sa], [-sa, ca]]
        x_r =  pts[:, 0] * ca + pts[:, 1] * sa
        y_r = -pts[:, 0] * sa + pts[:, 1] * ca
        return np.column_stack([x_r, y_r])

    uc_rot = _rotate(uc)
    lc_rot = _rotate(lc)

    # Filter noisy LE pixels and inject one clean (0, 0) at the seam so that
    # fit2coordinates finds the origin directly (same fix as green-dot version).
    le_gap  = 2.0 / (maxColumn - zeroColumn)
    uc_f    = uc_rot[uc_rot[:, 0] > le_gap]
    lc_f    = lc_rot[lc_rot[:, 0] > le_gap]

    uc_sorted = uc_f[np.argsort(-uc_f[:, 0])]   # descending x: TE → LE
    lc_sorted = lc_f[np.argsort( lc_f[:, 0])]   # ascending  x: LE → TE

    # Trim both surfaces to their shared TE x: whichever surface terminates
    # earlier (shorter x_max) sets the limit for both.  This handles images
    # where one surface line extends further right as a single-surface tail.
    if len(lc_sorted) > 0 and len(uc_sorted) > 0:
        x_te = min(uc_sorted[0, 0], lc_sorted[-1, 0])  # uc descending, lc ascending
        uc_sorted = uc_sorted[uc_sorted[:, 0] <= x_te + le_gap]
        lc_sorted = lc_sorted[lc_sorted[:, 0] <= x_te + le_gap]

    combined_psi  = np.concatenate([uc_sorted[:, 0], [0.0], lc_sorted[:, 0]])
    combined_zeta = np.concatenate([uc_sorted[:, 1], [0.0], lc_sorted[:, 1]])

    afl = Kulfan()
    afl.fit2coordinates(copy.deepcopy(combined_psi), copy.deepcopy(combined_zeta),
                        fit_order=fit_order)

    # Force TE gap to be evenly split about y=0 (no trailing-edge offset)
    afl.constants.TE_shift = 0.0

    # Optional thickness scaling — pure y-scale that preserves shape exactly.
    # scaleThickness() leaves TE_gap unscaled (changes the TE wedge shape), so
    # we multiply coefficients AND TE_gap by the same factor instead.
    if target_tau is not None:
        cf = target_tau / float(afl.tau)
        afl.upperCoefficients   = afl.upperCoefficients * cf
        afl.lowerCoefficients   = afl.lowerCoefficients * cf
        afl.constants.TE_gap    = afl.constants.TE_gap * cf
        afl.constants.TE_shift  = afl.constants.TE_shift * cf   # stays 0

    # --- verification figure ---
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(img, aspect='equal', alpha=0.2)
    plt.plot([zeroColumn, maxColumn], [zeroRow, zeroRow], 'b')
    plt.plot([minColumn, minColumn], [0, Nrows], 'g--', lw=0.8)
    plt.plot(min_y_filled, 'g')
    plt.plot(max_y_filled, 'r')
    plt.plot(lc[:, 0] * (maxColumn - zeroColumn) + zeroColumn,
             -lc[:, 1] * (maxColumn - zeroColumn) + zeroRow, 'c.')
    plt.plot(uc[:, 0] * (maxColumn - zeroColumn) + zeroColumn,
             -uc[:, 1] * (maxColumn - zeroColumn) + zeroRow, 'm.')
    # Inverse-rotate fit coordinates back to unrotated pixel-normalised space
    # before mapping to pixel space.  _rotate uses R(-angle), so the inverse
    # is R(+angle) = [[ca, -sa], [sa, ca]].
    fit_xy = np.column_stack([afl.xcoordinates, afl.ycoordinates])
    fit_x_unrot = fit_xy[:, 0] * ca - fit_xy[:, 1] * sa
    fit_y_unrot = fit_xy[:, 0] * sa + fit_xy[:, 1] * ca
    plt.plot(fit_x_unrot * (maxColumn - zeroColumn) + zeroColumn,
             -fit_y_unrot * (maxColumn - zeroColumn) + zeroRow, 'k')

    return afl, fig


if __name__ == '__main__':
    wd = 'png_files'
    files = sorted(
        f for f in os.listdir(wd)
        if os.path.isfile(os.path.join(wd, f))
        and f[0] not in ('.', '_')
        and f.endswith('.png')
        and 'mhkf1' in f      # original script processed mhkf1 hydrofoils
    )

    for fl in files:
        stem = fl[:-4]
        afl, fig = fit_from_image(os.path.join(wd, fl))
        fig.savefig(os.path.join('verification_plots', stem + '.png'), dpi=300)
        plt.close(fig)
        afl.write2file(os.path.join('fitted_datfiles', stem + '.dat'))
