import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_spectrogram(sound_analysis, sides='onesided', Fs=44100, NFFT=2048, noverlap=1024):
    fig, ax = plt.subplots()
    Pxx, freqs, bins, im = ax.specgram(sound_analysis.sound.data, sides=sides,
                                       Fs=Fs, NFFT=NFFT, noverlap=noverlap)
    plt.show()


def plot_wave(sound_analysis):
    plt.plot(sound_analysis.sound.data)


def plot_env(ax, sa, color):
    ax.plot(sa.env, color=color)


def plot_peaks(ax, sa, color):
    for p in sa.peaks:
        ax.axvline(p, color=color)


def plot_bpf(ax, sa, color):
    for bps in sa.bpf_list:
        b = bps.breakpoints
        line = plt.Line2D(b[:, 0], b[:, 1], color=color)
        ax.add_line(line)


class MultibandPlot:
    def __init__(self, msa):
        self.msa = msa
        self.num_bands = len(msa.bands)

    def audio(self):
        from IPython.display import Audio
        return Audio(self.msa.sound.filepath)

    def plot(self, types=None, sharey=False, figsize=(8, 8), to_fig=None):
        assert self.num_bands > 2
        if types is None:
            types = ['env', 'peaks', 'bpf']

        num_rows = 0
        if len([t for t in types if t in ['env', 'peaks', 'bpf']]) > 0:
            num_rows += self.num_bands
        if 'memb_fun' in types:
            num_rows += 1
        assert num_rows > 0

        fig, axes = plt.subplots(num_rows, ncols=1, sharex=True, sharey=sharey,
                                 figsize=figsize, squeeze=False)
        axes = axes[:, 0]  # ensure we always have a 1D array of axes

        for ax in axes:
            y_max = self.msa.max_env
            self._plot_memb_funs(fig, ax, 'gray', 'gray', y_max)
        if 'env' in types:
            self._plot_env(fig, axes, 'black')
        if 'peaks' in types:
            self._plot_peaks(fig, axes, 'red')
        if 'bpf' in types:
            self._plot_bpf(fig, axes, 'orange')

        fig.subplots_adjust(hspace=0)
        for ax in axes:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ymargin(0)
            ax.set_xticks([0, self.msa.bands[0].len])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: str(float(x) / 44100)))
        plt.setp(axes[-1].get_yticklabels(), visible=True)
        plt.setp(axes[-1].get_xticklabels(), visible=True)
        axes[0].set_yticks([0, round(axes[0].get_ylim()[1] - axes[0].get_ylim()[1] * 0.05, 2)])
        plt.xlabel('Time (seconds)')
        plt.plot()

        if to_fig is not None:
            fig.savefig(to_fig, bbox_inches='tight')

    def _plot_env(self, fig, axes, color):
        for i, ax in enumerate(axes):
            plot_env(ax, self.msa.bands[i], color=color)

    def _plot_peaks(self, fig, axes, color):
        for i, ax in enumerate(axes):
            for p in self.msa.bands[i].peaks:
                ax.axvline(p, color=color)

    def _plot_bpf(self, fig, axes, color):
        for i, ax in enumerate(axes):
            for bps in self.msa.bands[i].bpf_list:
                b = bps.breakpoints
                line = plt.Line2D(b[:, 0], b[:, 1], color=color)
                ax.add_line(line)

    def _plot_memb_funs(self, fig, ax, color_beg, color_end, y_max=1):
        # FIXME this function calculates only two-class membership function ad-hoc
        x_max = self.msa.bands[0].len - 1
        ax.add_line(plt.Line2D([0, x_max], [y_max, 0], color=color_beg,
                               linestyle=':', linewidth=0.5))
        ax.add_line(plt.Line2D([0, x_max], [0, y_max], color=color_end,
                               linestyle='--', linewidth=0.5))

    def _plot_fuzzy(self, fig, ax, color_beg, color_end, color_peak):
        self._plot_memb_funs(fig, ax, color_beg, color_end)
        for b in self.msa.bands:
            for p in b.peaks:
                ax.axvline(p, color=color_peak)
