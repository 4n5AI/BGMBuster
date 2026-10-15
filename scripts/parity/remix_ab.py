"""Objective comparison of the two remix methods on mixes with known stems (brief §4.5).

Given a speech recording, a music track and a sound-effects track (44.1 kHz WAV), builds mixes at a few
music levels, separates them with the upstream PyTorch MRX, and scores the "music removed" output of
  method A: speech_est + sfx_est + g * music_est
  method B: mix - (1 - g) * music_est
against the true speech + sfx (+ g * music) with SI-SDR, for g in {0, 0.25}. Also reports how much music
remains (music residual level) and how much speech/sfx is lost.

usage: python3 remix_ab.py <checkpoint.pth> <speech.wav> <music.wav> <sfx.wav> [seconds=30]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "upstream"))
import parity  # noqa: E402

SR = 44100


def si_sdr(est, ref):
    est = est.astype(np.float64) - est.mean()
    ref = ref.astype(np.float64) - ref.mean()
    scale = np.dot(est.ravel(), ref.ravel()) / (np.dot(ref.ravel(), ref.ravel()) + 1e-12)
    target = scale * ref
    noise = est - target
    return 10 * np.log10(np.sum(target ** 2) / (np.sum(noise ** 2) + 1e-12) + 1e-12)


def rms_db(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def load(path, secs):
    x = parity.read_wav(path)
    n = int(secs * SR)
    if x.shape[1] < n:
        reps = int(np.ceil(n / x.shape[1]))
        x = np.tile(x, (1, reps))
    x = x[:, :n]
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    return x[:2]


def main():
    import torch
    from mrx import MRX

    ckpt, speech_p, music_p, sfx_p = sys.argv[1:5]
    secs = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
    model = MRX().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    speech, music, sfx = (load(p, secs) for p in (speech_p, music_p, sfx_p))
    # level the stems: speech at -20 dB, sfx at -26 dB, music relative to speech
    speech = speech * 10 ** ((-20 - rms_db(speech)) / 20)
    sfx = sfx * 10 ** ((-26 - rms_db(sfx)) / 20)
    print(f"{'music/speech':>13s} {'g':>5s} {'method':>7s} {'SI-SDR':>8s} {'music left':>11s} {'speech+sfx kept':>16s}")
    for music_rel in (-12.0, -6.0, 0.0):
        m = music * 10 ** ((-20 + music_rel - rms_db(music)) / 20)
        mix = (speech + sfx + m).astype(np.float32)
        y, gain, _ = parity.upstream_separate(model, mix.copy())
        est_music, est_speech, est_sfx = y[0], y[1], y[2]
        for g in (0.0, 0.25):
            target = speech + sfx + g * m
            a = est_speech + est_sfx + g * est_music
            b = mix - (1 - g) * est_music
            for name, out in (("A", a), ("B", b)):
                # music residual: project the error onto the music stem
                err = out - target
                music_left = rms_db(err) - rms_db(m * (1 - g) if g < 1 else m)
                kept = si_sdr(out - g * m, speech + sfx)
                print(f"{music_rel:+12.0f}dB {g:5.2f} {name:>7s} {si_sdr(out, target):8.2f} {music_left:10.1f}dB {kept:15.2f}")


if __name__ == "__main__":
    main()
