"""Compares an exported video with its source: the video stream must be byte-for-byte the same packets
(codec, size, frame count, bitrate), and the audio must line up at the start and the end.

usage: python3 media_check.py <source media> <exported media>
Requires ffprobe / ffmpeg on PATH.
"""
import json
import subprocess
import sys

import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=index,codec_type,codec_name,width,height,nb_read_packets,bit_rate,avg_frame_rate,sample_rate,channels,duration,start_time:format=duration,bit_rate",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def video_hash(path):
    """MD5 of the raw video packets (codec copy), independent of the container."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "md5", "-"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out


def decode_audio(path, sr=48000):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4")


def click_envelope(x, sr):
    """Envelope of the > 1.2 kHz content (the fixtures carry a 1760 Hz click burst once per second), so the
    cross-correlation is not fooled by the periodic 440 / 660 Hz tones."""
    n = len(x)
    spec = np.fft.rfft(x.astype(np.float64))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec[freqs < 1200] = 0
    hp = np.fft.irfft(spec, n)
    env = np.abs(hp)
    k = max(1, int(sr * 0.002))
    return np.convolve(env, np.ones(k) / k, mode="same")


def offset_samples(a, b, sr):
    """Lag (b relative to a, in samples) maximising the cross-correlation of the click envelopes."""
    ea, eb = click_envelope(a, sr), click_envelope(b, sr)
    ea -= ea.mean(); eb -= eb.mean()
    n = 1
    while n < 2 * len(ea):
        n *= 2
    corr = np.fft.irfft(np.fft.rfft(ea, n) * np.conj(np.fft.rfft(eb, n)), n)
    lag = int(np.argmax(corr))
    if lag > n // 2:
        lag -= n
    return -lag


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ps, pd = probe(src), probe(dst)
    vs = next((s for s in ps["streams"] if s["codec_type"] == "video"), None)
    vd = next((s for s in pd["streams"] if s["codec_type"] == "video"), None)
    ok = True
    if vs and vd:
        same = {k: (vs.get(k), vd.get(k)) for k in ("codec_name", "width", "height", "nb_read_packets", "avg_frame_rate")}
        hs, hd = video_hash(src), video_hash(dst)
        print("video:", ", ".join(f"{k} {a}->{b}" for k, (a, b) in same.items()), f"| packets md5 {hs == hd and 'IDENTICAL' or 'DIFFERENT'}")
        print(f"       bit_rate {vs.get('bit_rate')} -> {vd.get('bit_rate')}")
        ok &= hs == hd and all(a == b for a, b in same.values())
    elif vs or vd:
        print("video: missing on one side"); ok = False
    a_s = next((s for s in ps["streams"] if s["codec_type"] == "audio"), None)
    a_d = next((s for s in pd["streams"] if s["codec_type"] == "audio"), None)
    print(f"audio: {a_s and a_s['codec_name']} {a_s and a_s.get('sample_rate')} Hz {a_s and a_s.get('channels')} ch  ->  "
          f"{a_d and a_d['codec_name']} {a_d and a_d.get('sample_rate')} Hz {a_d and a_d.get('channels')} ch, bit_rate {a_d and a_d.get('bit_rate')}")
    print(f"duration: {ps['format'].get('duration')} -> {pd['format'].get('duration')} s")
    xa, xb = decode_audio(src), decode_audio(dst)
    sr = 48000
    n = min(len(xa), len(xb))
    w = min(n, sr * 2)
    head = offset_samples(xa[:w], xb[:w], sr)
    tail = offset_samples(xa[n - w:n], xb[n - w:n], sr)
    print(f"audio sync: start offset {head / sr * 1000:+.2f} ms, end offset {tail / sr * 1000:+.2f} ms (length {len(xa)} vs {len(xb)} samples)")
    ok &= abs(head) <= sr * 0.002 and abs(tail) <= sr * 0.002
    print("OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
