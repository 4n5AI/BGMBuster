"""End-to-end parity of the browser pipeline (TypeScript STFT -> ONNX on onnxruntime-web -> TypeScript iSTFT,
chunked) against upstream PyTorch.

  python3 browser_parity.py dump <checkpoint.pth> <dir> [seconds=10]
      writes <dir>/input_L.f32, input_R.f32 and the upstream reference stems ref_<source>_<ch>.f32
  python3 browser_parity.py compare <dir>
      reads browser_<source>_<ch>.f32 (written by scripts/bench/browser-stems.cjs) and prints SNR per source
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "upstream"))
import parity  # noqa: E402

SOURCES = ["music", "speech", "sfx"]


def dump(ckpt, out, secs):
    import torch
    from mrx import MRX

    os.makedirs(out, exist_ok=True)
    model = MRX().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    x = parity.synthetic(secs, seed=5)
    y, gain, lufs = parity.upstream_separate(model, x)
    for c, name in enumerate("LR"):
        x[c].astype("<f4").tofile(os.path.join(out, f"input_{name}.f32"))
        for s, src in enumerate(SOURCES):
            y[s, c].astype("<f4").tofile(os.path.join(out, f"ref_{src}_{name}.f32"))
    print(f"wrote {out}: {secs}s stereo, {lufs:.2f} LUFS, gain {gain:.4f}")


def compare(d):
    print(f"{'source':8s} {'L':>8s} {'R':>8s}  (SNR of browser pipeline vs upstream PyTorch, dB)")
    for src in SOURCES:
        vals = []
        for name in "LR":
            ref = np.fromfile(os.path.join(d, f"ref_{src}_{name}.f32"), dtype="<f4")
            got = np.fromfile(os.path.join(d, f"browser_{src}_{name}.f32"), dtype="<f4")
            assert ref.shape == got.shape, (ref.shape, got.shape)
            vals.append(parity.snr_db(got, ref))
        print(f"{src:8s} {vals[0]:8.1f} {vals[1]:8.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 10.0)
    else:
        compare(sys.argv[2])
