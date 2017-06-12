from glob import glob

import numpy as np
from numpy import pi, polymul
from scipy import signal as sig
from scipy.signal import bilinear

from util import pairwise


def erbs2freq(erb: float) -> float:
    return 676170.4 / (47.06538 - np.power(np.e, 0.08950404 * erb)) - 14678.49


def get_split_freqs(num_bands):
    assert num_bands <= 37
    width = 37. / num_bands
    split_log = [i * width for i in range(1, num_bands)]
    split_hz = [erbs2freq(b) for b in split_log]
    return split_hz


def crossover(audio, split_freqs, order=4, sr=44100):
    assert len(split_freqs) > 0
    nyq = sr / 2
    bands = []

    # first band is lowpass
    lp_b, lp_a = sig.butter(order, split_freqs[0] / nyq)
    bands.append(sig.lfilter(lp_b, lp_a, audio))

    # then series of bandpass filters
    if len(split_freqs) > 1:
        for low, high in pairwise(split_freqs):
            b, a = sig.butter(order, [low / nyq, high / nyq], 'bandpass')
            bands.append(sig.lfilter(b, a, audio))

    # last band is highpass
    hp_b, hp_a = sig.butter(order, split_freqs[-1] / nyq, 'highpass')
    bands.append(sig.lfilter(hp_b, hp_a, audio))

    return bands


def A_weighting(fs):
    # see https://gist.github.com/endolith/148112
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = polymul([1, 4 * pi * f4, (2 * pi * f4) ** 2],
                   [1, 4 * pi * f1, (2 * pi * f1) ** 2])
    DENs = polymul(polymul(DENs, [1, 2 * pi * f3]),
                   [1, 2 * pi * f2])

    return bilinear(NUMs, DENs, fs)


def calc_rms(ts, w_size=1024):
    p = np.power(ts, 2)
    w = np.ones(w_size) / w_size
    c = np.convolve(p, w, 'valid')
    return np.sqrt(c)


aw_b, aw_a = A_weighting(44100)


def get_envelope(x):
    return calc_rms(x, 882)  # 20 ms


def split_segments(bool_array, merge_thresh, discard_thresh=None):
    diff = np.diff(bool_array.astype(np.int))
    ones = np.where(diff == 1)[0]
    mones = np.where(diff == -1)[0]

    if len(ones) == 0:
        ones = np.array([0])
    if len(mones) == 0:
        mones = np.array([len(diff) - 1])
    if ones[0] > mones[0]:
        ones = np.append(0, ones)
    if ones[-1] > mones[-1]:
        mones = np.append(mones, len(diff) - 1)

    assert len(ones) == len(mones)

    segments = np.array([ones, mones]).T
    if discard_thresh is not None:
        s = (segments[:, 1] - segments[:, 0]) > discard_thresh
        segments = segments[s]

    if len(segments) == 0:
        return []

    merged = [segments[0]]
    for a, b in pairwise(segments):
        if b[0] - a[1] < merge_thresh:
            merged[-1][1] = b[1]
        else:
            merged.append(b.tolist())

    return merged


def find_peaks(segm_vol_curve, peak_thresh, gate_thresh, peak_prune_strategy=None):
    assert peak_prune_strategy in {'AD', 'AHD', None}

    thresh = np.max(segm_vol_curve) * peak_thresh
    segm = split_segments(segm_vol_curve > thresh, gate_thresh)
    # peak is the maximum value in each range
    peaks = [int(s[0] + np.argmax(segm_vol_curve[s[0]:s[1] + 1])) for s in segm]

    if len(peaks) <= 1 or peak_prune_strategy is None:
        return peaks

    if peak_prune_strategy == 'AD':
        raise NotImplementedError('AD prune strategy is not implemented yet')
    elif peak_prune_strategy == 'AHD':
        peaks = [peaks[0], peaks[-1]]
    else:
        raise NotImplementedError('Unknown peak prune strategy: ' + str(peak_prune_strategy))

    return peaks


def find_nonsilence(vol_curve, vol_thresh, time_thresh, discard_thresh):
    # TODO filter low-level noise, etc
    return split_segments(vol_curve > vol_thresh, time_thresh, discard_thresh)


######################################################################


def create_bpf_repr(peaks_x, audio_chunk):
    bpf = np.zeros_like(audio_chunk)
    # bpf starts at the beginning of the chunk and ends at the end
    p = np.append(np.append(0, peaks_x), len(audio_chunk) - 1)
    for aa, bb in pairwise(p):
        a, b = aa - p[0], bb - p[0]
        ab = np.linspace(audio_chunk[a], audio_chunk[b - 1], b - a)
        bpf[a:b] = ab
    return bpf


######################################################################

# if __name__ == "__main__":
#     wavs = []
#     for f in glob('audio/*.wav'):
#         load = ess.standard.MonoLoader(filename=f, sampleRate=44100)
#         wavs.append(load())
#
#     env = get_envelope(wavs[2])
#     se = find_nonsilence(env)[0]
#     senv = env[se[0]:se[1]]
#     p = find_peaks(senv)
#     bp = np.sort(np.append(se, p))
#     bpf = create_bpf_repr(bp, senv)
