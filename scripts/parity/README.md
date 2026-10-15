# scripts/parity — 上流 Python との一致検証

MERL の MRX（[merlresearch/cocktail-fork-separation](https://github.com/merlresearch/cocktail-fork-separation)、MIT）と、ブラウザ実装の出力が一致することを確かめるためのスクリプト。

| ファイル | 内容 | 依存 |
|---|---|---|
| `torch_sd.py` | PyTorch の zip 形式 `.pth` を、torch なしで読み込む（標準ライブラリ＋NumPy） | numpy |
| `mrx_numpy.py` | `mrx.py` / `separate.py` の NumPy 移植（STFT・BatchNorm・BLSTM・マスク・iSTFT・BS.1770）。試験ベクトルの生成と自己検査に使う | numpy |

## チェックポイントの取得

GitHub の Git LFS から取得する（122,317,463 bytes）。リポジトリにはコミットしない（`.gitignore` 済み）。

```bash
mkdir -p scripts/parity/checkpoints
python3 - <<'EOF'
import requests, pathlib
url = "https://media.githubusercontent.com/media/merlresearch/cocktail-fork-separation/main/checkpoints/default_mrx_pre_trained_weights.pth"
pathlib.Path("scripts/parity/checkpoints/default_mrx_pre_trained_weights.pth").write_bytes(requests.get(url, timeout=600).content)
EOF
sha256sum scripts/parity/checkpoints/default_mrx_pre_trained_weights.pth
# 期待値: 9cecc071c9e58afda34d1a98609a1540f61320e718bafecc4c9c1c5e3ed20a0c
```

## 使い方

```bash
cd scripts/parity
python3 torch_sd.py checkpoints/default_mrx_pre_trained_weights.pth      # パラメータ一覧と総数
python3 mrx_numpy.py checkpoints/default_mrx_pre_trained_weights.pth 2.0 ref_2s.npz   # 2 秒の合成信号で forward
```

出力の例（2026-10-15、2 秒ステレオの合成信号）：

```
STFT->iSTFT round trip n_fft=1024: SNR 155.3 dB
STFT->iSTFT round trip n_fft=2048: SNR 158.2 dB
STFT->iSTFT round trip n_fft=8192: SNR 169.1 dB
integrated loudness -11.68 LUFS -> gain to -27 LUFS = 0.1714
forward done in 1.9s; output shape (3, 2, 88200), finite=True
SNR(sum of 3 sources vs mixture) = 27.3 dB
```

## 上流との一致（予定）

torch を導入したうえで、同じ入力を上流 `MRX`（PyTorch）・`mrx_numpy.py`・書き出した ONNX（onnxruntime）に通し、音源ごとの SNR を記録する。目安はブリーフ §4.3 の 60 dB 以上。
