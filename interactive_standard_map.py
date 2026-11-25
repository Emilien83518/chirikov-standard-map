import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

mpl.rcParams['toolbar'] = 'None'

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
]

N_ITER = 500
K_INIT = 0.971


def compute_orbit(theta0, p0, K, N):
    thetas = np.empty(N)
    ps = np.empty(N)
    theta, p = theta0, p0
    for i in range(N):
        p = p + K * np.sin(theta)
        theta = (theta + p) % (2 * np.pi)
        p = ((p + np.pi) % (2 * np.pi)) - np.pi
        thetas[i] = theta
        ps[i] = p
    return thetas, ps


class InteractiveMap:
    def __init__(self):
        self.orbits = []
        self.orbit_artists = []
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

    def _setup_axes(self):
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-np.pi, np.pi)
        self.ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        self.ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        self.ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        self.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        self.ax.set_xlabel(r'$\theta$', fontsize=13)
        self.ax.set_ylabel(r'$p$', fontsize=13)
        self.ax.set_title(f'Chirikov Standard Map   $K = {self.K:.3f}$', fontsize=14)
        self.ax.grid(True, alpha=0.15)

    def _next_color(self):
        c = COLORS[self.color_idx % len(COLORS)]
        self.color_idx += 1
        return c

    def _add_orbit(self, theta0, p0):
        color = self._next_color()
        self.orbits.append((theta0, p0, color))
        self._draw_orbit(theta0, p0, color)
        self.fig.canvas.draw_idle()

    def _draw_orbit(self, theta0, p0, color):
        thetas, ps = compute_orbit(theta0, p0, self.K, self.N)
        line, = self.ax.plot(thetas, ps, 'o', color=color, ms=0.4, alpha=0.8)
        self.orbit_artists.append(line)

    def _redraw_all(self):
        for artist in self.orbit_artists:
            artist.remove()
        self.orbit_artists.clear()
        for theta0, p0, color in self.orbits:
            self._draw_orbit(theta0, p0, color)
        self.ax.set_title(f'Chirikov Standard Map   $K = {self.K:.3f}$', fontsize=14)
        self.fig.canvas.draw_idle()

    def _on_k_submit(self, text):
        try:
            self.K = float(text)
            self._redraw_all()
        except ValueError:
            pass

    def _on_n_submit(self, text):
        try:
            self.N = int(float(text))
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
        self.orbits.clear()
        self.color_idx = 0
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    app = InteractiveMap()
    plt.show()
