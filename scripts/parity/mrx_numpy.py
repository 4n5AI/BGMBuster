"""NumPy port of MERL's MRX forward pass for parity testing.

Ported from merlresearch/cocktail-fork-separation (mrx.py, separate.py), Copyright (C) 2023 MERL, MIT License.
See THIRD_PARTY_NOTICES.md.

Mirrors torch semantics exactly:
  - torch.stft(n_fft, hop=256, hann_window(periodic), normalized=True, center=True, pad_mode="reflect")
  - BatchNorm1d in eval mode (eps=1e-5)
  - nn.LSTM gate order (i, f, g, o), bidirectional, 3 layers, batch_first
  - torch.istft(normalized=True, center=True, length=signal_length)
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_sd import load_state_dict  # noqa: E402

WINDOWS = [1024, 2048, 8192]
HOP = 256
SOURCE_NAMES = ["music", "speech", "sfx"]  # dnr_dataset.SOURCE_NAMES


def hann_periodic(n):
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2 * np.pi * k / n)).astype(np.float32)


def stft(x, n_fft, hop=HOP):
    """x: [..., L] float32 -> complex64 [..., F, T] (torch.stft semantics, normalized=True, center=True, reflect)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    pad = n_fft // 2
    x2 = np.pad(x2, ((0, 0), (pad, pad)), mode="reflect")
    L = x2.shape[-1]
    n_frames = 1 + (L - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]  # [T, n_fft]
    frames = x2[:, idx]  # [B, T, n_fft]
    win = hann_periodic(n_fft)
    spec = np.fft.rfft(frames * win, axis=-1) / np.sqrt(n_fft)  # normalized=True
    spec = spec.transpose(0, 2, 1).astype(np.complex64)  # [B, F, T]
    return spec.reshape(*lead, spec.shape[-2], spec.shape[-1])


def istft(spec, hop, length):
    """spec: complex [..., F, T] -> float32 [..., length] (torch.istft semantics, normalized=True, center=True)."""
    lead = spec.shape[:-2]
    F, T = spec.shape[-2], spec.shape[-1]
    n_fft = 2 * F - 2
    s = spec.reshape(-1, F, T).transpose(0, 2, 1)  # [B, T, F]
    frames = np.fft.irfft(s * np.sqrt(n_fft), n=n_fft, axis=-1)  # undo normalization
    win = hann_periodic(n_fft).astype(np.float64)
    frames = frames * win
    out_len = n_fft + hop * (T - 1)
    y = np.zeros((frames.shape[0], out_len), dtype=np.float64)
    env = np.zeros(out_len, dtype=np.float64)
    for t in range(T):
        y[:, t * hop:t * hop + n_fft] += frames[:, t]
        env[t * hop:t * hop + n_fft] += win ** 2
    pad = n_fft // 2
    y = y[:, pad:pad + length]
    env = env[pad:pad + length]
    assert np.all(env > 1e-11), "NOLA violated"
    y = y / env
    return y.astype(np.float32).reshape(*lead, length)


def bn(x, p, prefix, eps=1e-5):
    w, b = p[prefix + ".weight"], p[prefix + ".bias"]
    m, v = p[prefix + ".running_mean"], p[prefix + ".running_var"]
    return (x - m) / np.sqrt(v + eps) * w + b


def encoder(spec, p, i):
    """spec: complex [N, F, T] -> [N, T, 512]"""
    N, F, T = spec.shape
    enc = np.abs(spec).transpose(0, 2, 1).reshape(-1, F).astype(np.float32)
    enc = bn(enc, p, f"_encoders.{i}.layer.0")
    enc = enc @ p[f"_encoders.{i}.layer.1.weight"].T
    enc = bn(enc, p, f"_encoders.{i}.layer.2")
    return np.tanh(enc.reshape(N, T, -1))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_direction(x, w_ih, w_hh, b_ih, b_hh, reverse):
    """x: [N, T, D_in] -> [N, T, H]"""
    N, T, _ = x.shape
    H = w_hh.shape[1]
    xw = x @ w_ih.T + (b_ih + b_hh)  # [N, T, 4H]
    h = np.zeros((N, H), dtype=np.float32)
    c = np.zeros((N, H), dtype=np.float32)
    out = np.zeros((N, T, H), dtype=np.float32)
    order = range(T - 1, -1, -1) if reverse else range(T)
    whT = w_hh.T
    for t in order:
        g = xw[:, t] + h @ whT
        i, f, gg, o = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(gg)
        h = sigmoid(o) * np.tanh(c)
        out[:, t] = h
    return out


def bilstm(x, p, branch, n_layers=3):
    for l in range(n_layers):
        pre = f"_cross_net.lstm.{branch}."
        fwd = lstm_direction(x, p[pre + f"weight_ih_l{l}"], p[pre + f"weight_hh_l{l}"],
                             p[pre + f"bias_ih_l{l}"], p[pre + f"bias_hh_l{l}"], False)
        bwd = lstm_direction(x, p[pre + f"weight_ih_l{l}_reverse"], p[pre + f"weight_hh_l{l}_reverse"],
                             p[pre + f"bias_ih_l{l}_reverse"], p[pre + f"bias_hh_l{l}_reverse"], True)
        x = np.concatenate([fwd, bwd], axis=-1)
    return x


def decoder(x, p, s, r):
    """x: [N, T, 1024] -> mask [N, T, F]"""
    N, T, D = x.shape
    pre = f"_decoders.{s}.{r}.block."
    y = x.reshape(-1, D) @ p[pre + "0.weight"].T
    y = np.maximum(bn(y, p, pre + "1"), 0)
    y = y @ p[pre + "3.weight"].T
    y = np.maximum(bn(y, p, pre + "4"), 0)
    return y.reshape(N, T, -1)


def mrx_forward(x, p, log=print):
    """x: float32 [C, L] (channels processed independently) -> [3, C, L] in SOURCE_NAMES order."""
    C, L = x.shape
    specs, encs = [], []
    for i, n_fft in enumerate(WINDOWS):
        spec = stft(x, n_fft)  # [C, F, T]
        specs.append(spec)
        encs.append(encoder(spec, p, i))
    cross_1 = np.mean(np.stack(encs), axis=0)  # [C, T, 512]
    t0 = time.time()
    lstm_outs = [bilstm(cross_1, p, k) for k in range(3)]
    log(f"  3x BiLSTM over T={cross_1.shape[1]} frames: {time.time() - t0:.1f}s (numpy)")
    cross_2 = np.mean(np.stack(lstm_outs), axis=0)
    feat = np.concatenate([cross_1, cross_2], axis=-1)  # [C, T, 1024]
    out = np.zeros((3, C, L), dtype=np.float32)
    masks = {}
    for s in range(3):
        for r, spec in enumerate(specs):
            m = decoder(feat, p, s, r).transpose(0, 2, 1)  # [C, F, T]
            masks[(s, r)] = m
            out[s] += istft(m * spec, HOP, L)
    return out, masks, specs


def bs1770_integrated_lufs(x, sr=44100):
    """ITU-R BS.1770-4 integrated loudness with absolute (-70) and relative (-10) gating. x: [C, L]."""
    from numpy import pi
    # K-weighting: pre-filter (high shelf) + RLB high-pass, coefficients per BS.1770 for 48k; redesign for sr via bilinear approach (pyloudnorm style)
    def shelf(sr):
        f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
        K = np.tan(pi * f0 / sr); Vh = 10 ** (G / 20.0); Vb = Vh ** 0.4996667741545416
        a0 = 1.0 + K / Q + K * K
        b = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
        a = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
        return np.array(b), np.array(a)
    def highpass(sr):
        f0, Q = 38.13547087602444, 0.5003270373238773
        K = np.tan(pi * f0 / sr)
        a = [1.0, 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K), (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)]
        b = [1.0, -2.0, 1.0]
        return np.array(b), np.array(a)
    def lfilter(b, a, x):
        y = np.zeros_like(x, dtype=np.float64)
        x = x.astype(np.float64)
        for n in range(len(x)):
            acc = b[0] * x[n]
            if n >= 1: acc += b[1] * x[n - 1] - a[1] * y[n - 1]
            if n >= 2: acc += b[2] * x[n - 2] - a[2] * y[n - 2]
            y[n] = acc
        return y
    C, L = x.shape
    G = [1.0, 1.0, 1.0, 1.41, 1.41][:C]
    bs, as_ = shelf(sr); bh, ah = highpass(sr)
    T_g = 0.4; overlap = 0.75
    step = int(round(T_g * sr * (1 - overlap))); blk = int(round(T_g * sr))
    n_blocks = max(0, (L - blk) // step + 1)
    z = np.zeros((C, n_blocks))
    for c in range(C):
        y = lfilter(bh, ah, lfilter(bs, as_, x[c]))
        for j in range(n_blocks):
            seg = y[j * step:j * step + blk]
            z[c, j] = np.mean(seg ** 2)
    l = -0.691 + 10 * np.log10(np.sum([G[c] * z[c] for c in range(C)], axis=0) + 1e-30)
    gate_abs = l > -70.0
    if not np.any(gate_abs):
        return -np.inf
    z_avg = np.array([np.mean(z[c][gate_abs]) for c in range(C)])
    rel = -0.691 + 10 * np.log10(np.sum([G[c] * z_avg[c] for c in range(C)]) + 1e-30) - 10.0
    gate = gate_abs & (l > rel)
    if not np.any(gate):
        return -np.inf
    z_avg = np.array([np.mean(z[c][gate]) for c in range(C)])
    return -0.691 + 10 * np.log10(np.sum([G[c] * z_avg[c] for c in range(C)]) + 1e-30)


def snr_db(est, ref):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((est - ref) ** 2) + 1e-20) + 1e-20)


if __name__ == "__main__":
    ckpt = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    p = load_state_dict(ckpt)
    sr = 44100
    L = int(secs * sr)
    rng = np.random.default_rng(0)
    t = np.arange(L) / sr
    # synthetic "mixture": tone chord (music-like) + noise bursts (sfx-like) + AM-modulated harmonic (speech-like)
    music = 0.2 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 440 * t))
    speech = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * (1 + 0.5 * np.sin(2 * np.pi * 1200 * t))
    sfx = 0.1 * rng.standard_normal(L) * (np.sin(2 * np.pi * 1.5 * t) > 0.8)
    mix = (music + speech + sfx).astype(np.float32)
    x = np.stack([mix, 0.8 * mix]).astype(np.float32)  # stereo
    # STFT/iSTFT round trip check
    for n_fft in WINDOWS:
        rt = istft(stft(x, n_fft), HOP, L)
        print(f"STFT->iSTFT round trip n_fft={n_fft}: SNR {snr_db(rt, x):.1f} dB")
    lufs = bs1770_integrated_lufs(x, sr)
    gain = 10 ** ((-27.0 - lufs) / 20.0)
    print(f"integrated loudness {lufs:.2f} LUFS -> gain to -27 LUFS = {gain:.4f}")
    t0 = time.time()
    out, masks, specs = mrx_forward(x * gain, p)
    out /= gain
    print(f"forward done in {time.time() - t0:.1f}s; output shape {out.shape}, finite={np.isfinite(out).all()}")
    for s, name in enumerate(SOURCE_NAMES):
        print(f"  {name:7s} rms={np.sqrt(np.mean(out[s] ** 2)):.4f}")
    print(f"  mixture rms={np.sqrt(np.mean(x ** 2)):.4f}; SNR(sum of 3 sources vs mixture) = {snr_db(out.sum(0), x):.1f} dB")
    if len(sys.argv) > 3:
        np.savez_compressed(sys.argv[3], x=x, out=out, gain=gain)
