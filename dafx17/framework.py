import os
from io import StringIO

import librosa
import numpy as np
import pandas as pd

from dsp import get_envelope, crossover, find_nonsilence, get_split_freqs, find_peaks, pairwise


def _lin_solve(xy1, xy2):
    a = np.array([[xy1[0], 1], [xy2[0], 1]])
    b = np.array([xy1[1], xy2[1]])
    return np.linalg.solve(a, b)


class BPF:
    def __init__(self, x, y, bounds_behavior='clip_edges'):
        assert len(x) == len(y)
        assert bounds_behavior in {'clip_edges', 'zero_edges'}

        self.bounds_behavior = bounds_behavior
        self.breakpoints = np.array([x, y]).T
        self.poly1d_list = [np.poly1d(_lin_solve(xy1, xy2))
                            for xy1, xy2 in pairwise(self.breakpoints)]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        r = StringIO()
        r.write('BPF: ')
        for bp in self.breakpoints:
            r.write('[{:.2f}, {:.2f}]'.format(bp[0], bp[1]))
            r.write(', ')
        return r.getvalue()

    def _clip_edges(self, x):
        if x < self.breakpoints[0][0]:
            return self.breakpoints[0][1]
        elif x > self.breakpoints[-1][0]:
            return self.breakpoints[-1][1]
        else:
            raise ValueError

    def eval_x(self, x):
        if not (self.breakpoints[0][0] <= x <= self.breakpoints[-1][0]):
            if self.bounds_behavior == 'zero_edges':
                return 0
            elif self.bounds_behavior == 'clip_edges':
                return self._clip_edges(x)
            else:
                raise ValueError('unknown bounds behavior')
        for i, (xy1, xy2) in enumerate(pairwise(self.breakpoints)):
            if xy1[0] <= x <= xy2[0]:
                return np.polyval(self.poly1d_list[i], x)
        raise ValueError

    def to_array(self):
        # TODO test this method with non-integer x break-point coordinates
        start_x = int(self.breakpoints[0][0])
        length = int(self.breakpoints[-1][0] - self.breakpoints[0][0] + 1)
        arr = np.zeros(length)
        for xy1, xy2 in pairwise(self.breakpoints):
            x1, x2 = int(xy1[0]) - start_x, int(xy2[0]) - start_x
            y1, y2 = xy1[1], xy2[1]
            arr[x1:x2 + 1] = np.linspace(y1, y2, x2 - x1 + 1)
        return arr


class SettingsDSP:
    nonsilence_thresh = 0.001  # -60dB
    nonsilence_merge_thresh = 22050  # 500 ms
    nonsilence_discard_thresh = 221  # 5ms
    peak_thresh = 0.8
    peak_merge_thresh = 441
    bands_split = get_split_freqs(8)


class MonoSound:
    def __init__(self, filepath=None, data=None):
        assert filepath is not None or data is not None
        self._filepath = filepath
        self._data = data

    @staticmethod
    def from_file(filepath):
        assert os.path.isfile(filepath)
        return MonoSound(filepath, None)

    @staticmethod
    def from_data(data):
        return MonoSound(None, data)

    @property
    def filepath(self):
        return self._filepath

    @property
    def data(self):
        if self._data is None:
            self._data = librosa.load(self.filepath, sr=44100, mono=True)[0]
        return self._data


class SoundAnalysis(SettingsDSP):
    def __init__(self, sound):
        self._sound = sound
        self._sound_env = None
        self._max = None
        self._max_env = None
        self._segments = None
        self._peaks = None
        self._bpf_list = None

    def __len__(self):
        return self.len()

    @property
    def len(self):
        return len(self._sound.data)

    @property
    def sound(self):
        return self._sound

    @property
    def env(self):
        if self._sound_env is None:
            self._sound_env = get_envelope(self.sound.data)
        return self._sound_env

    @property
    def max(self):
        if self._max is None:
            self._max = np.max(self.sound.data)
        return self._max

    @property
    def max_env(self):
        if self._max_env is None:
            self._max_env = np.max(self.env)
        return self._max_env

    @property
    def segments(self):
        if self._segments is None:
            self._segments = find_nonsilence(self.env, self.nonsilence_thresh,
                                             self.nonsilence_merge_thresh,
                                             self.nonsilence_discard_thresh)
        return self._segments

    @property
    def peaks(self):
        if self._peaks is None:
            peaks = []
            for s in self.segments:
                senv = self.env[s[0]:s[1]]
                psegm = find_peaks(senv, self.peak_thresh, self.peak_merge_thresh, 'AHD')
                peaks.extend([p + s[0] for p in psegm])
            self._peaks = peaks
        return self._peaks

    @property
    def bpf_list(self):
        if self._bpf_list is None:
            bpfs = []
            peaks_np = np.array(self.peaks)
            for s in self.segments:
                peaks_in_segm = peaks_np[(peaks_np > s[0]) & (peaks_np < s[1])]
                px = np.sort(np.append(s, peaks_in_segm))
                py = self.env[px]
                bpfs.append(BPF(px, py))
            self._bpf_list = bpfs
        return self._bpf_list

    def render_bpf(self):
        d = np.zeros(len(self.env))
        for b in self.bpf_list:
            x1, x2 = int(b.breakpoints[0, 0]), int(b.breakpoints[-1, 0])
            d[x1:x2 + 1] = b.to_array()
        return d


class MultibandAnalysis(SettingsDSP):
    def __init__(self, sound):
        self._sound = sound
        self._analysis_bands = None
        self._max = None
        self._max_env = None

    @property
    def sound(self):
        return self._sound

    @property
    def bands(self):
        if self._analysis_bands is None:
            bands_sdata = crossover(self.sound.data, self.bands_split)
            self._analysis_bands = [SoundAnalysis(MonoSound.from_data(b)) for b in bands_sdata]
            for b in self._analysis_bands:
                b.nonsilence_merge_thresh = 3308  # 75ms FIXME make sr dependent
        return self._analysis_bands

    @property
    def len(self):
        return self.bands[0].len

    @property
    def max(self):
        if self._max is None:
            self._max = sorted([b.max for b in self.bands])[-1]
        return self._max

    @property
    def max_env(self):
        if self._max_env is None:
            self._max_env = sorted([b.max_env for b in self.bands])[-1]
        return self._max_env


class Movement:
    def __init__(self, mbsa):
        self.mbsa = mbsa
        self.bpfs = [
            BPF([0., 1.], [1., 0.]),
            BPF([0., 1.], [0., 1.]),
        ]
        self._report_data = None

    def _check_membership(self, x):
        assert 0. <= x <= 1.
        return np.array(
            [bpf.eval_x(x) for bpf in self.bpfs]
        )

    @property
    def report_data(self):
        if self._report_data is None:
            report = np.zeros((len(self.mbsa.bands), len(self.bpfs)))
            for i, band in enumerate(self.mbsa.bands):
                p_vals = band.env[band.peaks]
                p_idx_norm = [float(p) / band.len for p in band.peaks]
                memberships = [self._check_membership(p) for p in p_idx_norm]
                for m, v in zip(memberships, p_vals):
                    report[i] += m * v
            self._report_data = report
        return self._report_data

    @property
    def cumul_peak_vals(self):
        return np.sum(self.report_data, axis=0)

    @property
    def centroids(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            band_enum = np.arange(len(self.report_data), dtype=np.float32)
            centroids = np.divide(
                np.sum((band_enum * self.report_data.T).T, axis=0),
                self.cumul_peak_vals
            )
            return centroids * (self.cumul_peak_vals > 0)

    @property
    def report(self):
        return pd.DataFrame({
            'Band Centroid': self.centroids.tolist(),
            'Cumulative Peak Values': self.cumul_peak_vals.tolist()
        }, index=['Beginning', 'End']).T


###


if __name__ == '__main__':
    def get_split_freqs_glitch_fix(num_bands):  # FIXME glitch in low-frequency bands
        if num_bands > 12:
            return get_split_freqs(num_bands)[2:]
        else:
            return get_split_freqs(num_bands)


    from sklearn.metrics import mean_squared_error as ms_err
    import json

    fn = 'data/test_noises/1818-2828--6060-5656--up.wav'
    rows = []
    num_bands_list = np.linspace(2, 20, 10, dtype=int)
    for num_bands in num_bands_list:
        msa = MultibandAnalysis(MonoSound.from_file(fn))
        msa.bands_split = get_split_freqs_glitch_fix(num_bands)  # FIXME
        m = Movement(msa)
        row = [fn, int(num_bands)]
        row += list(m.report.loc['Band Centroid', :])
        row += list(m.report.loc['Cumulative Peak Values', :])
        row += [json.dumps([ms_err(b.env, b.render_bpf()) for b in msa.bands])]
        row += [json.dumps(m.report_data.tolist())]
        rows.append(row)
    print('lal')
