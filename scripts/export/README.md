# scripts/export — ONNX の書き出し

MERL のチェックポイントから、STFT と iSTFT の間のネットワークだけを ONNX にする。

```bash
/path/to/venv/bin/python scripts/export/build_onnx.py scripts/parity/checkpoints/default_mrx_pre_trained_weights.pth out/ fp32 fp16 int8
INT8_MIN_SNR_DB=0 ONNX_SUFFIX=-all /path/to/venv/bin/python scripts/export/build_onnx.py <ckpt> out/ int8   # 全テンソル int8
```

| 変種 | 格納 | サイズ |
|---|---|---|
| fp32 | float32 | 122.0 MB |
| fp16 | float16 ＋ Cast | 61.1 MB |
| int8 | int8 ＋ Cast ＋ Mul（出力チャンネルごとの対称スケール）。重み SNR が `INT8_MIN_SNR_DB`（既定 40 dB）未満のテンソルは float16 | 30.9〜42.4 MB |

入力 `mag_1024` / `mag_2048` / `mag_8192`（[B, F, T]）、出力 `mask_1024` / `mask_2048` / `mask_8192`（[B, 3, F, T]、音源は music / speech / sfx の順）。検証は `scripts/parity/parity.py`。
