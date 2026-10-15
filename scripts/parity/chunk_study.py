"""How much does chunked processing deviate from whole-file processing (the upstream behaviour)?

Runs the upstream PyTorch MRX on a 60 s synthetic stereo signal as a whole, then in overlapping chunks joined with
the same cross-fade as src/separation/chunking.ts, and reports the SNR of the chunked result against the whole-file
result per source for several chunk lengths, overlaps and cross-fade modes.

usage: python3 chunk_study.py <checkpoint.pth> [seconds=60]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "upstream"))
import torch  # noqa: E402
from mrx import MRX  # noqa: E402
import parity  # noqa: E402

SR = 44100


def plan_chunks(length, chunk, overlap):
    """Same plan as src/separation/chunking.ts: fewest equal chunks of at most `chunk` with >= `overlap`."""
    if length <= chunk:
        return [(0, length)]
    n = int(np.ceil((length - overlap) / (chunk - overlap)))
    ln = min(chunk, int(np.ceil((length + (n - 1) * overlap) / n)))
    stride = ln - overlap
    ranges = []
    for i in range(n):
        start = length - ln if i == n - 1 else min(i * stride, length - ln)
        ranges.append((start, start + ln))
    return ranges


def ramp(i, n, mode):
    s = np.sin((np.pi / 2) * ((i + 0.5) / n))
    return s * s if mode == "equal-gain" else s


def assemble(length, ranges, chunks, mode):
    acc = np.zeros(length)
    wsum = np.zeros(length)
    for k, ((start, end), data) in enumerate(zip(ranges, chunks)):
        n = end - start
        w = np.ones(n)
        if k > 0:
            r = max(0, min(n, ranges[k - 1][1] - start))
            if r:
                w[:r] *= ramp(np.arange(r), r, mode)
        if k < len(ranges) - 1:
            r = max(0, min(n, end - ranges[k + 1][0]))
            if r:
                w[n - r:] *= ramp(np.arange(r)[::-1], r, mode)
        acc[start:end] += w * data
        wsum[start:end] += w
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), 0)


def main():
    ckpt = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    torch.set_num_threads(os.cpu_count() or 1)
    model = MRX().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    x = parity.synthetic(secs, seed=3)
    # richer content: slow level changes so chunks see different loudness
    env = (0.6 + 0.4 * np.sin(2 * np.pi * np.arange(x.shape[1]) / SR / 7.0)).astype(np.float32)
    x = x * env[None, :]
    t0 = time.time()
    y_whole, gain, lufs = parity.upstream_separate(model, x)
    print(f"whole file: {secs:.0f} s x 2ch in {time.time() - t0:.1f}s, {lufs:.1f} LUFS")
    print(f"{'chunk':>6s} {'overlap':>8s} {'crossfade':>12s}   music   speech     sfx   (SNR of chunked vs whole, dB)")
    length = x.shape[1]
    for chunk_s in (10, 20):
        for ov_s in (1, 2, 3):
            ranges = plan_chunks(length, chunk_s * SR, ov_s * SR)
            outs = []  # per chunk: [3, C, n]
            with torch.no_grad():
                for start, end in ranges:
                    seg = torch.from_numpy(x[:, start:end] * gain).float()
                    outs.append(model(seg).numpy() / gain)
            for mode in ("equal-gain", "equal-power"):
                snrs = []
                for s in range(3):
                    chans = [assemble(length, ranges, [o[s, c] for o in outs], mode) for c in range(x.shape[0])]
                    y = np.stack(chans)
                    snrs.append(parity.snr_db(y, y_whole[s]))
                print(f"{chunk_s:5d}s {ov_s:7d}s {mode:>12s}   " + "  ".join(f"{v:6.1f}" for v in snrs) + f"   ({len(ranges)} chunks)")


if __name__ == "__main__":
    main()
