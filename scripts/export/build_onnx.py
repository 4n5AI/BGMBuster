"""Build the MRX ONNX graph (the network between the STFTs and the iSTFTs) from MERL's checkpoint.

The graph takes magnitude spectrograms and returns one real mask per source and resolution:

    inputs  : mag_1024 [B, 513, T], mag_2048 [B, 1025, T], mag_8192 [B, 4097, T]   (float32)
    outputs : mask_1024, mask_2048, mask_8192  [B, 3, F, T]                          (float32)
    sources : index 0 = music, 1 = speech, 2 = sfx  (dnr_dataset.SOURCE_NAMES)

STFT / iSTFT stay outside (JS in the browser).  BatchNorm layers are folded into the neighbouring
linear layers; the per-frequency input BatchNorm of each encoder is kept as an elementwise Mul/Add so
that int8 weight quantisation is not hurt by the large per-bin variance differences.

Weight storage variants (same graph, same float32 arithmetic at run time):
    fp32 : float32 initializers                      (~122 MB)
    fp16 : float16 initializers + Cast               (~61 MB)
    int8 : int8 initializers + Cast + Mul(scale)     (~31 MB), symmetric, one scale per output channel
           (per gate row and direction for the LSTM matrices); onnxruntime folds Cast/Mul at load.

The MRX architecture and weights are Copyright (C) 2023 Mitsubishi Electric Research Laboratories (MERL),
MIT License (https://github.com/merlresearch/cocktail-fork-separation).

usage: python3 build_onnx.py <checkpoint.pth> <out_dir> [fp32|fp16|int8 ...]
"""
import hashlib
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "parity"))
from torch_sd import load_state_dict  # noqa: E402

WINDOWS = [1024, 2048, 8192]
N_BINS = [w // 2 + 1 for w in WINDOWS]
HIDDEN = 512
LSTM_HIDDEN = 256
N_LAYERS = 3
N_SOURCES = 3
SOURCE_NAMES = ["music", "speech", "sfx"]
BN_EPS = 1e-5
INT8_MIN_SNR_DB = float(os.environ.get("INT8_MIN_SNR_DB", "40"))  # int8 tensors below this weight SNR are kept in float16 instead
OPSET = 17
IR_VERSION = 9


def bn_affine(p, prefix, eps=BN_EPS):
    """BatchNorm1d (eval) as y = x * s + t."""
    s = p[prefix + ".weight"] / np.sqrt(p[prefix + ".running_var"] + eps)
    t = p[prefix + ".bias"] - p[prefix + ".running_mean"] * s
    return s.astype(np.float32), t.astype(np.float32)


class Builder:
    def __init__(self, storage):
        assert storage in ("fp32", "fp16", "int8")
        self.storage = storage
        self.nodes = []
        self.inits = []
        self.n = 0
        self.stats = {"weights": 0, "quant_snr_db": [], "kept_fp16": []}

    def const(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def weight(self, name, arr, channel_axis):
        """Store a weight matrix in the chosen format; returns the float32 tensor name usable in the graph."""
        arr = np.ascontiguousarray(arr.astype(np.float32))
        self.stats["weights"] += 1
        if self.storage == "fp32":
            return self.const(name, arr)
        if self.storage == "fp16":
            self.const(name + "_h", arr.astype(np.float16))
            self.nodes.append(helper.make_node("Cast", [name + "_h"], [name], to=TensorProto.FLOAT, name="cast_" + name))
            return name
        # int8, symmetric, one scale per channel.  Try the requested axis and the other matrix axis and keep
        # the one with the smaller error; tensors that still fall under INT8_MIN_SNR_DB are stored as float16.
        best = None
        axes = [channel_axis] + [a for a in range(arr.ndim) if a != channel_axis and arr.shape[a] > 2]
        for axis in axes:
            amax = np.max(np.abs(arr), axis=tuple(i for i in range(arr.ndim) if i != axis), keepdims=True)
            scale = np.maximum(amax / 127.0, 1e-12).astype(np.float32)
            q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
            err = np.sum((q.astype(np.float32) * scale - arr) ** 2)
            snr = 10 * np.log10(np.sum(arr ** 2) / (err + 1e-30))
            if best is None or snr > best[0]:
                best = (snr, axis, q, scale)
        snr, axis, q, scale = best
        if snr < INT8_MIN_SNR_DB:
            self.stats["kept_fp16"].append((name, snr))
            self.const(name + "_h", arr.astype(np.float16))
            self.nodes.append(helper.make_node("Cast", [name + "_h"], [name], to=TensorProto.FLOAT, name="cast_" + name))
            return name
        self.stats["quant_snr_db"].append((name, snr, axis))
        self.const(name + "_q", q)
        self.const(name + "_s", scale)
        self.nodes.append(helper.make_node("Cast", [name + "_q"], [name + "_f"], to=TensorProto.FLOAT, name="cast_" + name))
        self.nodes.append(helper.make_node("Mul", [name + "_f", name + "_s"], [name], name="scale_" + name))
        return name

    def node(self, op, inputs, outputs=None, **attrs):
        self.n += 1
        if outputs is None:
            outputs = [f"{op.lower()}_{self.n}"]
        self.nodes.append(helper.make_node(op, inputs, outputs, name=f"{op}_{self.n}", **attrs))
        return outputs[0]


def build(p, storage):
    b = Builder(storage)
    inputs, outputs = [], []
    # ---------------- encoders ----------------
    encs = []
    for r, (n_fft, F) in enumerate(zip(WINDOWS, N_BINS)):
        name = f"mag_{n_fft}"
        inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["B", F, "T"]))
        x = b.node("Transpose", [name], [f"magT_{n_fft}"], perm=[0, 2, 1])  # [B, T, F]
        s0, t0 = bn_affine(p, f"_encoders.{r}.layer.0")
        x = b.node("Mul", [x, b.const(f"enc{r}_bn0_s", s0)])
        x = b.node("Add", [x, b.const(f"enc{r}_bn0_t", t0)])
        w1 = p[f"_encoders.{r}.layer.1.weight"]  # [512, F]
        s2, t2 = bn_affine(p, f"_encoders.{r}.layer.2")
        w_eff = (w1.T * s2[None, :]).astype(np.float32)  # [F, 512], BN2 folded per output column
        x = b.node("MatMul", [x, b.weight(f"enc{r}_w", w_eff, channel_axis=1)])
        x = b.node("Add", [x, b.const(f"enc{r}_b", t2)])
        encs.append(b.node("Tanh", [x], [f"enc_{n_fft}"]))
    s = b.node("Add", [encs[0], encs[1]])
    s = b.node("Add", [s, encs[2]])
    cross1 = b.node("Mul", [s, b.const("third", np.float32(1.0 / 3.0))], ["cross_1"])  # [B, T, 512]
    cross1_tb = b.node("Transpose", [cross1], ["cross_1_tb"], perm=[1, 0, 2])  # [T, B, 512]
    # ---------------- cross net: 3 branches x 3-layer BiLSTM ----------------
    lstm_outs = []
    for k in range(3):
        x = cross1_tb
        for l in range(N_LAYERS):
            pre = f"_cross_net.lstm.{k}."
            ws, rs, bs = [], [], []
            for suffix in ("", "_reverse"):
                w_ih = p[pre + f"weight_ih_l{l}{suffix}"]  # [4H, D] rows i,f,g,o
                w_hh = p[pre + f"weight_hh_l{l}{suffix}"]  # [4H, H]
                b_ih = p[pre + f"bias_ih_l{l}{suffix}"]
                b_hh = p[pre + f"bias_hh_l{l}{suffix}"]
                H = LSTM_HIDDEN
                order = [0, 3, 1, 2]  # torch (i, f, g, o) -> onnx (i, o, f, c)
                reorder = lambda m: np.concatenate([m[g * H:(g + 1) * H] for g in order], axis=0)
                ws.append(reorder(w_ih))
                rs.append(reorder(w_hh))
                bs.append(np.concatenate([reorder(b_ih), reorder(b_hh)], axis=0))
            W = np.stack(ws)  # [2, 4H, D]
            R = np.stack(rs)  # [2, 4H, H]
            Bv = np.stack(bs).astype(np.float32)  # [2, 8H]
            y = b.node("LSTM", [x, b.weight(f"lstm{k}_l{l}_W", W, channel_axis=1), b.weight(f"lstm{k}_l{l}_R", R, channel_axis=1),
                                b.const(f"lstm{k}_l{l}_B", Bv)],
                       [f"lstm{k}_l{l}_Y", "", ""], direction="bidirectional", hidden_size=LSTM_HIDDEN)  # Y [T, 2, B, H]
            y = b.node("Transpose", [y], perm=[0, 2, 1, 3])  # [T, B, 2, H]
            x = b.node("Reshape", [y, b.const(f"lstm{k}_l{l}_shape", np.array([0, 0, 2 * LSTM_HIDDEN], dtype=np.int64))],
                       [f"lstm{k}_l{l}_out"])  # [T, B, 512]
        lstm_outs.append(x)
    s = b.node("Add", [lstm_outs[0], lstm_outs[1]])
    s = b.node("Add", [s, lstm_outs[2]])
    cross2 = b.node("Mul", [s, "third"], ["cross_2_tb"])  # [T, B, 512]
    feat = b.node("Concat", [cross1_tb, cross2], ["feat_tb"], axis=-1)  # [T, B, 1024]
    # ---------------- decoders ----------------
    for r, (n_fft, F) in enumerate(zip(WINDOWS, N_BINS)):
        per_source = []
        for sidx in range(N_SOURCES):
            pre = f"_decoders.{sidx}.{r}.block."
            wa = p[pre + "0.weight"]  # [512, 1024]
            sa, ta = bn_affine(p, pre + "1")
            wa_eff = (wa.T * sa[None, :]).astype(np.float32)  # [1024, 512]
            wb = p[pre + "3.weight"]  # [F, 512]
            sb, tb = bn_affine(p, pre + "4")
            wb_eff = (wb.T * sb[None, :]).astype(np.float32)  # [512, F]
            x = b.node("MatMul", [feat, b.weight(f"dec{sidx}_{r}_wa", wa_eff, channel_axis=1)])
            x = b.node("Add", [x, b.const(f"dec{sidx}_{r}_ba", ta)])
            x = b.node("Relu", [x])
            x = b.node("MatMul", [x, b.weight(f"dec{sidx}_{r}_wb", wb_eff, channel_axis=1)])
            x = b.node("Add", [x, b.const(f"dec{sidx}_{r}_bb", tb)])
            x = b.node("Relu", [x])  # [T, B, F]
            x = b.node("Transpose", [x], perm=[1, 2, 0])  # [B, F, T]
            per_source.append(b.node("Unsqueeze", [x, b.const(f"axes1_{sidx}_{r}", np.array([1], dtype=np.int64))]))  # [B, 1, F, T]
        out = f"mask_{n_fft}"
        b.node("Concat", per_source, [out], axis=1)  # [B, 3, F, T]
        outputs.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, ["B", N_SOURCES, F, "T"]))
    graph = helper.make_graph(b.nodes, "mrx", inputs, outputs, initializer=b.inits)
    model = helper.make_model(graph, producer_name="bgm-buster build_onnx.py", opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model, b.stats


def main():
    ckpt, out_dir = sys.argv[1], sys.argv[2]
    variants = sys.argv[3:] or ["fp32", "fp16", "int8"]
    os.makedirs(out_dir, exist_ok=True)
    ckpt_sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    p = load_state_dict(ckpt)
    for storage in variants:
        model, stats = build(p, storage)
        meta = {
            "model": "MRX (Multi-Resolution CrossNet), Petermann et al., ICASSP 2022",
            "source": "https://github.com/merlresearch/cocktail-fork-separation checkpoints/default_mrx_pre_trained_weights.pth",
            "checkpoint_sha256": ckpt_sha,
            "license": "MIT, Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)",
            "sources": ",".join(SOURCE_NAMES),
            "stft": "hann(periodic) windows 1024/2048/8192, hop 256, normalized (1/sqrt(n_fft)), center, reflect pad, 44100 Hz",
            "input_lufs": "-27",
            "weights": storage,
            "builder": "bgm-buster scripts/export/build_onnx.py",
        }
        for k, v in meta.items():
            model.metadata_props.add(key=k, value=v)
        onnx.checker.check_model(model)
        suffix = os.environ.get("ONNX_SUFFIX", "")
        path = os.path.join(out_dir, f"mrx.{storage}{suffix}.onnx")
        onnx.save(model, path)
        size = os.path.getsize(path)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"{path}: {size:,d} bytes ({size / 1e6:.1f} MB), {stats['weights']} weight tensors, sha256 {sha}")
        if stats["quant_snr_db"]:
            worst = min(stats["quant_snr_db"], key=lambda t: t[1])
            mean = float(np.mean([t[1] for t in stats["quant_snr_db"]]))
            print(f"  int8: {len(stats['quant_snr_db'])} tensors, weight SNR mean {mean:.1f} dB, worst {worst[0]} {worst[1]:.1f} dB (axis {worst[2]})")
            for name, snr, axis in stats["quant_snr_db"]:
                print(f"    {name:16s} {snr:5.1f} dB  axis {axis}")
        if stats["kept_fp16"]:
            print("  kept in float16 (int8 SNR < %.0f dB): " % INT8_MIN_SNR_DB + ", ".join(f"{n} ({s:.1f} dB)" for n, s in stats["kept_fp16"]))


if __name__ == "__main__":
    main()
