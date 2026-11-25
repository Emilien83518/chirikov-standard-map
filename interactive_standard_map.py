import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from numba import njit, prange

mpl.rcParams['toolbar'] = 'None'

N_ITER = 500
K_INIT = 0.971
FOUR_PI = 4 * np.pi
TWO_PI = 2 * np.pi
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]


@njit(parallel=True, cache=True)
def compute_all_orbits(theta0s, p0s, K, N):
    """Compute N iterations for all orbits in parallel.

    Parameters
    ----------
    theta0s, p0s : 1-D arrays of length n_orbits
    K : float
    N : int

    Returns
    -------
    all_thetas : (n_orbits, N) array
    all_ps     : (n_orbits, N) array
    """
    n_orbits = theta0s.shape[0]
    all_thetas = np.empty((n_orbits, N))
    all_ps = np.empty((n_orbits, N))
    for j in prange(n_orbits):
        theta = theta0s[j]
        p = p0s[j]
        for i in range(N):
            p = p + K * np.sin(theta)
            theta = (theta + p) % FOUR_PI
            p = ((p + np.pi) % TWO_PI) - np.pi
            all_thetas[j, i] = theta
            all_ps[j, i] = p
    return all_thetas, all_ps


class InteractiveMap:
    def __init__(self):
        self.theta0s = []
        self.p0s = []
        self.orbit_artists = []
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

        # Warm up JIT
        _t = np.array([0.0])
        _p = np.array([0.0])
        compute_all_orbits(_t, _p, 0.5, 10)

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
        n = self.N
        if n >= 1_000_000_000:
            n_str = f'{n / 1e9:.1f}B'
        elif n >= 1_000_000:
            n_str = f'{n / 1e6:.1f}M'
        elif n >= 1_000:
            n_str = f'{n / 1e3:.0f}K'
        else:
            n_str = str(n)
        self.ax.set_title(
            f'Chirikov Standard Map   $K = {self.K:.3f}$   '
            f'({n_str} iter, {len(self.theta0s)} orbits)',
            fontsize=13,
        )

    def _redraw_all(self):
        for artist in self.orbit_artists:
            artist.remove()
        self.orbit_artists.clear()

        if not self.theta0s:
            self._update_title()
            self.fig.canvas.draw_idle()
            return

        t0_arr = np.array(self.theta0s)
        p0_arr = np.array(self.p0s)
        all_thetas, all_ps = compute_all_orbits(t0_arr, p0_arr, self.K, self.N)

        for i in range(len(self.theta0s)):
            c = COLORS[i % len(COLORS)]
            line, = self.ax.plot(
                all_thetas[i], all_ps[i], 'o', color=c, ms=0.4, alpha=0.8,
            )
            self.orbit_artists.append(line)

        self._update_title()
        self.fig.canvas.draw_idle()

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
        self.theta0s.append(event.xdata)
        self.p0s.append(event.ydata)
        self._redraw_all()

    def _on_clear(self, event):
        for artist in self.orbit_artists:
            artist.remove()
        self.orbit_artists.clear()
        self.theta0s.clear()
        self.p0s.clear()
        self._update_title()
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    app = InteractiveMap()
    plt.show()
