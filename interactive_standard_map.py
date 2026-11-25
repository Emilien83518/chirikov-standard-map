import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.colors import LogNorm
from numba import njit

mpl.rcParams['toolbar'] = 'None'

N_ITER = 500
K_INIT = 0.971
GRID_RES = 1024
CHUNK_SIZE = 10_000_000
FOUR_PI = 4 * np.pi
TWO_PI = 2 * np.pi


@njit(cache=True)
def compute_orbit_points(theta0, p0, K, N):
    """Return arrays of (theta, p) for N iterations. Use for small N."""
    thetas = np.empty(N)
    ps = np.empty(N)
    theta, p = theta0, p0
    for i in range(N):
        p = p + K * np.sin(theta)
        theta = (theta + p) % FOUR_PI
        p = ((p + np.pi) % TWO_PI) - np.pi
        thetas[i] = theta
        ps[i] = p
    return thetas, ps


@njit(cache=True)
def compute_orbit_density(theta0, p0, K, N, grid, nx, ny):
    """Iterate N steps, accumulate hits into grid, return final (theta, p)."""
    theta, p = theta0, p0
    inv_x = nx / FOUR_PI
    inv_y = ny / TWO_PI
    for i in range(N):
        p = p + K * np.sin(theta)
        theta = (theta + p) % FOUR_PI
        p = ((p + np.pi) % TWO_PI) - np.pi
        ix = int(theta * inv_x)
        iy = int((p + np.pi) * inv_y)
        if ix >= nx:
            ix = nx - 1
        if iy >= ny:
            iy = ny - 1
        grid[iy, ix] += 1
    return theta, p


DENSITY_THRESHOLD = 100_000


class InteractiveMap:
    def __init__(self):
        self.orbits = []
        self.orbit_artists = []
        self.density_artist = None
        self.color_idx = 0
        self.K = K_INIT
        self.N = N_ITER

        self.fig = plt.figure(figsize=(10, 7))
        self.fig.canvas.manager.set_window_title('Chirikov Standard Map')

        self.ax = self.fig.add_axes([0.08, 0.22, 0.88, 0.72])
        self._setup_axes()

        ax_k = self.fig.add_axes([0.12, 0.10, 0.15, 0.04])
        self.tb_k = TextBox(ax_k, 'K  ', initial=str(self.K))
        self.tb_k.on_submit(self._on_k_submit)

        ax_n = self.fig.add_axes([0.48, 0.10, 0.15, 0.04])
        self.tb_n = TextBox(ax_n, 'Iterations  ', initial=str(self.N))
        self.tb_n.on_submit(self._on_n_submit)

        ax_clear = self.fig.add_axes([0.78, 0.10, 0.10, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self._on_clear)

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Warm up numba JIT on first import
        _g = np.zeros((4, 4), dtype=np.int64)
        compute_orbit_density(0.0, 0.0, 0.5, 10, _g, 4, 4)
        compute_orbit_points(0.0, 0.0, 0.5, 10)

    def _setup_axes(self):
        self.ax.set_xlim(0, FOUR_PI)
        self.ax.set_ylim(-np.pi, np.pi)
        self.ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, FOUR_PI])
        self.ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$'])
        self.ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        self.ax.set_xlabel(r'$\theta$', fontsize=13)
        self.ax.set_ylabel(r'$p$', fontsize=13)
        self._update_title()
        self.ax.grid(True, alpha=0.15)

    def _update_title(self):
        if self.N >= 1_000_000_000:
            n_str = f'{self.N / 1e9:.1f}B'
        elif self.N >= 1_000_000:
            n_str = f'{self.N / 1e6:.1f}M'
        elif self.N >= 1_000:
            n_str = f'{self.N / 1e3:.0f}K'
        else:
            n_str = str(self.N)
        self.ax.set_title(
            f'Chirikov Standard Map   $K = {self.K:.3f}$   '
            f'({n_str} iter, {len(self.orbits)} orbits)',
            fontsize=13,
        )

    def _use_density(self):
        total = self.N * max(len(self.orbits), 1)
        return total >= DENSITY_THRESHOLD

    def _add_orbit(self, theta0, p0):
        self.orbits.append((theta0, p0))
        self._redraw_all()

    def _redraw_all(self):
        for artist in self.orbit_artists:
            artist.remove()
        self.orbit_artists.clear()
        if self.density_artist is not None:
            self.density_artist.remove()
            self.density_artist = None

        if not self.orbits:
            self._update_title()
            self.fig.canvas.draw_idle()
            return

        if self._use_density():
            self._draw_density()
        else:
            self._draw_scatter()

        self._update_title()
        self.fig.canvas.draw_idle()

    def _draw_scatter(self):
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf',
        ]
        for i, (theta0, p0) in enumerate(self.orbits):
            thetas, ps = compute_orbit_points(theta0, p0, self.K, self.N)
            c = colors[i % len(colors)]
            line, = self.ax.plot(thetas, ps, 'o', color=c, ms=0.4, alpha=0.8)
            self.orbit_artists.append(line)

    def _draw_density(self):
        nx, ny = GRID_RES, GRID_RES
        grid = np.zeros((ny, nx), dtype=np.int64)

        for theta0, p0 in self.orbits:
            remaining = self.N
            theta, p = theta0, p0
            while remaining > 0:
                chunk = min(remaining, CHUNK_SIZE)
                theta, p = compute_orbit_density(
                    theta, p, self.K, chunk, grid, nx, ny,
                )
                remaining -= chunk

        grid_float = grid.astype(np.float64)
        grid_float[grid_float == 0] = np.nan

        self.density_artist = self.ax.imshow(
            grid_float,
            extent=[0, FOUR_PI, -np.pi, np.pi],
            origin='lower',
            aspect='auto',
            cmap='inferno',
            norm=LogNorm(vmin=1, vmax=grid.max() or 1),
            interpolation='bilinear',
        )

    def _on_k_submit(self, text):
        try:
            self.K = float(text)
            self._redraw_all()
        except ValueError:
            pass

    def _on_n_submit(self, text):
        try:
            val = text.strip().lower()
            multiplier = 1
            if val.endswith('b'):
                multiplier = 1_000_000_000
                val = val[:-1]
            elif val.endswith('m'):
                multiplier = 1_000_000
                val = val[:-1]
            elif val.endswith('k'):
                multiplier = 1_000
                val = val[:-1]
            self.N = int(float(val) * multiplier)
            self._redraw_all()
        except ValueError:
            pass

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self._add_orbit(event.xdata, event.ydata)

    def _on_clear(self, event):
        for artist in self.orbit_artists:
            artist.remove()
        self.orbit_artists.clear()
        if self.density_artist is not None:
            self.density_artist.remove()
            self.density_artist = None
        self.orbits.clear()
        self.color_idx = 0
        self._update_title()
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    app = InteractiveMap()
    plt.show()
