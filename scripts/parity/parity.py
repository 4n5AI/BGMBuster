"""Parity check: upstream PyTorch MRX vs the NumPy port vs the exported ONNX graphs.

Runs the same signal through
  (1) upstream mrx.py `MRX.forward` + separate.py's -27 LUFS gain (pyloudnorm)    -> reference
  (2) mrx_numpy.py                                                                -> NumPy port
  (3) NumPy STFT -> ONNX masks (onnxruntime, CPU) -> NumPy iSTFT, for each ONNX variant
and reports the SNR of each source against the reference (brief §4.3 asks for >= 60 dB).

usage: python3 parity.py <checkpoint.pth> <onnx_dir> [wav file ...]
Without wav files a 10 s synthetic stereo signal is used.
"""
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "upstream"))

import torch  # noqa: E402
import pyloudnorm  # noqa: E402
import onnxruntime as ort  # noqa: E402
from mrx import MRX  # noqa: E402  (upstream, MIT)
import mrx_numpy as ref  # noqa: E402

SR = 44100


def snr_db(est, tgt):
    return 10 * np.log10(np.sum(tgt.astype(np.float64) ** 2) / (np.sum((est.astype(np.float64) - tgt) ** 2) + 1e-30) + 1e-30)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getframerate() == SR, "expects 16-bit 44.1 kHz WAV"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels()).T
    return (data.astype(np.float32) / 32768.0)


def synthetic(secs=10.0, seed=0):
    rng = np.random.default_rng(seed)
    L = int(secs * SR)
    t = np.arange(L) / SR
    music = 0.2 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 440 * t))
    speech = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * (1 + 0.5 * np.sin(2 * np.pi * 1200 * t))
    sfx = 0.1 * rng.standard_normal(L) * (np.sin(2 * np.pi * 1.5 * t) > 0.8)
    noise = 0.01 * rng.standard_normal(L)
    mix = (music + speech + sfx + noise).astype(np.float32)
    return np.stack([mix, np.roll(mix, 300) * 0.8]).astype(np.float32)


def upstream_separate(model, x):
    """separate.separate_soundtrack with consistency 'pass' (default), without torchaudio."""
    meter = pyloudnorm.Meter(SR)
    loudness = meter.integrated_loudness(x.T.astype(np.float64))
    gain = 10 ** ((-27.0 - loudness) / 20.0)
    with torch.no_grad():
        out = model(torch.from_numpy(x * gain).float()) / gain
    return out.numpy(), gain, loudness


def onnx_separate(sess, x, gain):
    xg = (x * gain).astype(np.float32)
    feeds, specs = {}, []
    for n_fft in ref.WINDOWS:
        spec = ref.stft(xg, n_fft)  # [C, F, T] complex64
        specs.append(spec)
        feeds[f"mag_{n_fft}"] = np.abs(spec).astype(np.float32)
    t0 = time.time()
    masks = sess.run(None, feeds)
    dt = time.time() - t0
    C, L = x.shape
    out = np.zeros((3, C, L), dtype=np.float32)
    for r, (spec, m) in enumerate(zip(specs, masks)):  # m [C, 3, F, T]
        for s in range(3):
            out[s] += ref.istft(m[:, s] * spec, ref.HOP, L)
    return out / gain, dt


def main():
    ckpt, onnx_dir = sys.argv[1], sys.argv[2]
    files = sys.argv[3:]
    signals = [(os.path.basename(f), read_wav(f)) for f in files] or [("synthetic-10s", synthetic())]
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = MRX().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    p = ref.load_state_dict(ckpt)
    variants = [v for v in ("fp32", "fp16", "int8") if os.path.exists(os.path.join(onnx_dir, f"mrx.{v}.onnx"))]
    sessions = {v: ort.InferenceSession(os.path.join(onnx_dir, f"mrx.{v}.onnx"), providers=["CPUExecutionProvider"]) for v in variants}
    print(f"{'signal':16s} {'implementation':22s} " + " ".join(f"{n:>9s}" for n in ref.SOURCE_NAMES) + "   time")
    for name, x in signals:
        t0 = time.time()
        y_ref, gain, lufs = upstream_separate(model, x)
        t_ref = time.time() - t0
        print(f"{name:16s} {'upstream torch':22s} " + " ".join(f"{'ref':>9s}" for _ in range(3)) + f"   {t_ref:5.1f}s  ({lufs:.1f} LUFS, gain {gain:.3f}, {x.shape[1] / SR:.1f}s x {x.shape[0]}ch)")
        t0 = time.time()
        y_np, _, _ = ref.mrx_forward(x * gain, p, log=lambda *_: None)
        y_np /= gain
        print(f"{'':16s} {'numpy port':22s} " + " ".join(f"{snr_db(y_np[s], y_ref[s]):8.1f}d" for s in range(3)) + f"   {time.time() - t0:5.1f}s")
        for v, sess in sessions.items():
            y_onnx, dt = onnx_separate(sess, x, gain)
            print(f"{'':16s} {'onnx ' + v + ' (ort cpu)':22s} " + " ".join(f"{snr_db(y_onnx[s], y_ref[s]):8.1f}d" for s in range(3)) + f"   {dt:5.1f}s (model run only)")
        # chunked vs whole: 20 s chunks with 2 s overlap is examined in M1; here only report the ref sum consistency
        print(f"{'':16s} {'sum(sources) vs mix':22s} {snr_db(y_ref.sum(0), x):8.1f}d (upstream, consistency=pass)")


if __name__ == "__main__":
    main()
