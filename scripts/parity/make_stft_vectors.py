"""Write STFT test vectors (NumPy / torch.stft conventions) for checking the TypeScript STFT.

usage: python3 make_stft_vectors.py <out_dir> [seconds=1.0]
Writes, per n_fft in (1024, 2048, 8192): x.f32 (signal), spec_<n>.f32 (interleaved re/im, frame-major [T, F]),
and meta.json. Everything is little-endian float32.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mrx_numpy as ref  # noqa: E402

out = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
os.makedirs(out, exist_ok=True)
sr = 44100
L = int(secs * sr)
rng = np.random.default_rng(42)
t = np.arange(L) / sr
x = (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 3001.0 * t + 0.3) + 0.05 * rng.standard_normal(L)).astype(np.float32)
x.tofile(os.path.join(out, "x.f32"))
meta = {"sampleRate": sr, "length": L, "hop": ref.HOP, "resolutions": {}}
for n_fft in ref.WINDOWS:
    spec = ref.stft(x[None, :], n_fft)[0]  # [F, T]
    T = spec.shape[1]
    inter = np.empty((T, spec.shape[0], 2), dtype=np.float32)
    inter[:, :, 0] = spec.real.T
    inter[:, :, 1] = spec.imag.T
    inter.tofile(os.path.join(out, f"spec_{n_fft}.f32"))
    meta["resolutions"][str(n_fft)] = {"bins": int(spec.shape[0]), "frames": int(T)}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print("wrote", out, meta)
