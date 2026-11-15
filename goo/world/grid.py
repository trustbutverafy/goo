"""Cells and faces of a rectangle of the panel, flattened in raster order (docs/DYNAMICS.md §6).

Faces are numbered: every x face (between columns) in raster order, then every y face (between rows) in
raster order. Restricted to a sub-rectangle this order keeps relative order, which is what makes a window's
systems the panel's systems. Positions are absolute panel coordinates."""
import numpy as np


class Grid:
    def __init__(self, H, W, dx, i0=0, j0=0):
        self.H, self.W, self.N, self.dx = H, W, H * W, float(dx)
        idx = np.arange(self.N).reshape(H, W)
        xL, xR = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        yL, yR = idx[:-1, :].ravel(), idx[1:, :].ravel()
        self.nx = xL.size
        self.fL = np.concatenate([xL, yL])
        self.fR = np.concatenate([xR, yR])
        self.nF = self.fL.size
        self.isx = np.zeros(self.nF, bool); self.isx[:self.nx] = True
        self.X = np.broadcast_to(((j0 + np.arange(W)) + 0.5) * self.dx, (H, W)).ravel().copy()
        self.Y = np.broadcast_to((((i0 + np.arange(H)) + 0.5) * self.dx)[:, None], (H, W)).ravel().copy()

    def D(self, a):
        """Face difference a[R] - a[L] of a cell array [..., N]."""
        return a[..., self.fR] - a[..., self.fL]

    def div(self, flux):
        """Outflow of a face flux [..., nF] (positive from L to R) per cell [..., N]: + at L, - at R."""
        flux = np.asarray(flux)
        if flux.ndim == 1:
            return np.bincount(self.fL, flux, self.N) - np.bincount(self.fR, flux, self.N)
        out = np.empty(flux.shape[:-1] + (self.N,))
        for k in np.ndindex(flux.shape[:-1]):
            out[k] = np.bincount(self.fL, flux[k], self.N) - np.bincount(self.fR, flux[k], self.N)
        return out

    def cellsum(self, facevals):
        """Sum of a face quantity over the faces of each cell."""
        return np.bincount(self.fL, facevals, self.N) + np.bincount(self.fR, facevals, self.N)


def panel_faces(ax, ay, y0, y1, x0, x1):
    """A window's face array from the panel's x-face [H, W-1] and y-face [H-1, W] arrays, in Grid order."""
    return np.concatenate([ax[y0:y1, x0:x1 - 1].ravel(), ay[y0:y1 - 1, x0:x1].ravel()])


def capillary_pressure(grid, gam_f, s):
    """(1/dx^2) sum_f gamma_f (s_c - s_nbr): the derivative of F_cap per volume, J/m^3."""
    d = grid.D(s) * gam_f                      # gamma (s_R - s_L)
    return (np.bincount(grid.fR, d, grid.N) - np.bincount(grid.fL, d, grid.N)) / grid.dx ** 2
