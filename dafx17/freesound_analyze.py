import yaml

from dsp import *

if __name__ == '__main__':
    audio_path = 'audio/freesound/'
    analysis_yaml = audio_path + 'analysis.yaml'

    with open(analysis_yaml, 'r') as f:
        analysis = yaml.load(f)

    if analysis is None:
        analysis = {}
    if 'band_split_hz' not in analysis:
        analysis['band_split_hz'] = get_bark_split_freqs(28)[::2]

    try:
        for i, a in enumerate(glob(audio_path + '*')):
            try:
                loader = std.MonoLoader(filename=a, sampleRate=44100)
            except RuntimeError:
                print('non-audio file: ' + a)
                continue
            print('{}\t{}'.format(i, a.split('/')[-1]))

            sid = a.split('.')[0].replace(audio_path, '')

            if sid in analysis:
                continue

            cross_bands = crossover(loader(), analysis['band_split_hz'])
            bands_analysis = []

            for band in cross_bands:
                segm_data = []
                env = get_envelope(band)
                for segm in find_nonsilence(env):
                    try:
                        peaks = find_peaks(band[segm[0]:segm[1]])
                    except ValueError:
                        peaks = []
                    segm_data.append({
                        'bounds': [int(e) for e in segm],
                        'peaks': [int(e) for e in peaks]
                    })
                bands_analysis.append({'segments': segm_data})

            analysis[sid] = {'bands': bands_analysis}
    finally:
        with open(audio_path + 'analysis.yaml', 'w') as f:
            f.write(yaml.dump(analysis))
