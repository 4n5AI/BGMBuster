# M0 中間報告（調査と検証）— 2026-10-15

ブリーフ §12 の形式。M0 の「上流との一致（SNR 60dB 以上）」と「1チャンクの処理時間」は、Python / npm パッケージの導入承認待ちのため未了。

## 実施したこと

- 上流 MERL 実装（`mrx.py` / `separate.py` / `dnr_dataset.py` / `consistency.py`）を取得し、STFT 仕様・音源の並び・前処理を確認。学習済みチェックポイント（122MB）を取得し、NumPy だけで読み込んで構造を確認。
- Mediabunny 1.61.4 / @mediabunny/aac-encoder 1.61.4 / onnxruntime-web 1.30.0 / Vite 8.3.4 の最新版と API 名を、GitHub のソースと npm レジストリで確認。
- MRX の NumPy 参照実装（`scripts/parity/mrx_numpy.py`）を書き、合成信号で形状と数値の健全性を確認。

## 確認できた事実（出典付き）

### MRX（上流実装）

| 項目 | 事実 | 出典 |
|---|---|---|
| STFT | 窓長 1024 / 2048 / 8192、ホップ 256、Hann 窓（`torch.hann_window` 既定の周期窓）、**`normalized=True`（1/√n_fft でスケール）**、`center=True`、`pad_mode="reflect"` | [mrx.py L95–112](https://github.com/merlresearch/cocktail-fork-separation/blob/main/mrx.py) |
| iSTFT | `normalized=True`、`center=True`、`length=入力長`。解像度ごとに iSTFT してから **加算** | mrx.py L83–84, L115–134 |
| マスク | 各音源×各解像度の decoder が出す **ReLU マスク（上限なし）** を複素スペクトログラムに掛ける | mrx.py L178–205 |
| 音源の並び | `SOURCE_NAMES = ["music", "speech", "sfx"]`。出力テンソルの音源次元はこの順 | [dnr_dataset.py L13](https://github.com/merlresearch/cocktail-fork-separation/blob/main/dnr_dataset.py), separate.py `_mrx_output_to_dict` |
| サンプリング周波数 | **44,100 Hz**（`SAMPLE_RATE = 44100`。`separate_soundtrack_file` は違う fs を 44.1k にリサンプル） | dnr_dataset.py L15, separate.py |
| 前処理 | 入力を **−27 LUFS**（pyloudnorm, ITU-R BS.1770 統合ラウドネス）に合わせるゲインを掛け、出力を同じゲインで戻す。これは上流 `separate.py` の既定（audiojs 独自ではない） | separate.py `_compute_gain`, `separate_soundtrack(input_lufs=-27.0)` |
| チャンネル | 各チャンネルを独立に処理（`reshape(-1, n_samples)`）。ステレオ＝モノ 2 本のバッチと同じ。モデルはステレオ前提ではない | mrx.py L96–98, L143–145 |
| チャンク | **上流にチャンク処理はない**（ファイル全体を一度に BLSTM に通す）。20 秒分割はブラウザ側の近似で、継ぎ目の誤差は別途計測が必要 | separate.py 全体 |
| 残差 | `consistency_mode` 既定は `"pass"`（残差を足し戻さない） | separate.py, consistency.py |
| 構造 | 隠れ 512、BLSTM 3 層（各方向 256）× 3 分岐、encoder は BatchNorm→Linear→BatchNorm→tanh | mrx.py |
| パラメータ数 | **30,566,472**（fp32 で 122.3MB、fp16 で 61.1MB、int8 で約 31MB） | `scripts/parity/torch_sd.py` の出力 |
| チェックポイント | `checkpoints/default_mrx_pre_trained_weights.pth`、122,317,463 bytes、SHA-256 `9cecc071c9e58afda34d1a98609a1540f61320e718bafecc4c9c1c5e3ed20a0c`。SNR 損失で学習し出力レベルが混合と一致する（方式 B の再ミックスに向く） | [README](https://github.com/merlresearch/cocktail-fork-separation#list-of-included-pre-trained-models)、Git LFS（media.githubusercontent.com） |
| ライセンス | コードと重みは MIT（Copyright (c) 2023 MERL）。引用必須：Petermann et al., ICASSP 2022 | LICENSE.md, README |
| 学習データのライセンス | 上流 README には記載なし（DnR v2 の Zenodo ページは本環境から到達不可。**未確認**） | — |

### ライブラリ

| 項目 | 事実 | 出典 |
|---|---|---|
| Mediabunny | 1.61.4（2026-10-12）、MPL-2.0。`Input` / `BlobSource` / `ALL_FORMATS` / `getPrimaryAudioTrack` / `getAudioTracks` / `AudioBufferSink.buffers()` / `Conversion.init({ composable: true, audio: { discard: true }, copy: { mode: 'forced' } })` / `isValid` / `discardedTracks` / `utilizedTracks` / `output.addAudioTrack(new AudioBufferSource({ codec, quality }))` / `output.start()` / `conversion.execute()` / `audioSource.add(AudioBuffer)` / `output.finalize()` が実在 | [converting-media-files.md](https://github.com/Vanilagy/mediabunny/blob/main/docs/guide/converting-media-files.md)「Composable conversions」, media-sources.md, media-sinks.md |
| 映像の無劣化コピー | `copy: { mode: 'forced' }` で「コピーできないトラックは破棄」になる（勝手に再エンコードしない）。`utilizedTracks` に映像トラックが入っているかで可否を判定できる | converting-media-files.md「Copying media data」 |
| `trim.start` | 0 以外を指定すると映像・音声とも再エンコードになる → 使わない | 同上「Trimming」 |
| 出力コンテナ | `Mp4OutputFormat` / `MovOutputFormat` / `WebMOutputFormat` / `WavOutputFormat`。`BufferTarget` は 100MB 未満向け、大きい動画は `StreamTarget` ＋ File System Access API | output-formats.md, writing-media-files.md |
| `canEncodeAudio('aac')` と `registerAacEncoder()` | 実在。AAC は WebCodecs でエンコードできないブラウザがある | aac-encoder.md |
| @mediabunny/aac-encoder のライセンス | パッケージは MPL-2.0。中身は **FFmpeg（libavcodec）の AAC エンコーダーの WASM ビルド**。FFmpeg は `--enable-gpl` なしのビルドでは **LGPL v2.1 以降**（彼らの configure に `--enable-gpl` はない）。パッケージの LICENSE ファイルは MPL-2.0 本文のみで FFmpeg の表記はない | [packages/aac-encoder/README.md](https://github.com/Vanilagy/mediabunny/blob/main/packages/aac-encoder/README.md), [FFmpeg LICENSE.md](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md) |
| onnxruntime-web | 1.30.0（2026-09-18）、MIT。入口 `onnxruntime-web/webgpu`（WASM EP も同梱）、`executionProviders: ['webgpu', 'wasm']`。WASM のマルチスレッドは `crossOriginIsolated` のときだけ有効 | [ep-webgpu.md](https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/tutorials/web/ep-webgpu.md), env-flags-and-session-options.md, npm `exports` |
| WebGPU と LSTM | **v1.30.0 の `ort.webgpu*` ビルドは JSEP ではなくネイティブ WebGPU EP**（`DISABLE_JSEP: 'true'`, `DISABLE_WEBGPU: 'false'`）。この EP は `LSTM`（opset 7–13, 14+）と `GRU` のカーネルを登録しており、WGSL で 1 時間ステップ＝1 ディスパッチのセル実装。旧 JSEP の演算子表（`webgpu-operators.md`）に LSTM がないのは JSEP の話 | [v1.30.0 js/web/script/build.ts](https://github.com/microsoft/onnxruntime/blob/v1.30.0/js/web/script/build.ts), [v1.30.0 webgpu_execution_provider.cc](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc), rnn/lstm.cc |
| Vite / TypeScript / Vitest | vite 8.3.4（MIT）、typescript 6.0.3（Apache-2.0。7.0.2 は 2026-10-14 公開で新しすぎるため見送り推奨）、vitest 5.0.4（MIT） | npm レジストリ |

### 既存ツール（NAMARI）の構成

- NAMARI はバンドラなし（`build.py` で `src/*.js` を 1 枚の `index.html` に埋め込み、GitHub Pages は「Deploy from a branch」）、CSP は meta タグ、`VERSION` ファイル、`.nojekyll`、`THIRD_PARTY_NOTICES.md` は日本語で「同梱しているコード／流用・改変したコード／外部サービス」の 3 節＋ライセンス全文、LICENSE は「Copyright (c) 2026 4n5-Studio」。
- 本ツールは onnxruntime-web・Mediabunny（ESM）・Worker・WASM を束ねる必要があるため、ブリーフどおり Vite + TypeScript を採用し、NAMARI からは **表記の慣例**（LICENSE 文、THIRD_PARTY_NOTICES の体裁、README の章立て、VERSION、`.nojekyll`、CSP meta）を引き継ぐ方針。

## 確認できなかったこと

| 項目 | 状況 |
|---|---|
| `audiojs/mrx`（Hugging Face）のモデルカード・ONNX | **本環境のネットワーク方針で huggingface.co が遮断**（CONNECT 403）。入出力名・int8・48000 の記述は未確認。代わりに MERL のチェックポイントから自前で ONNX を書き出す方針（ブリーフ §4.3 の代替手順） |
| 上流 Python との一致（SNR） | torch 未導入のため未実施。NumPy 参照実装は動作し、STFT→iSTFT 往復 155〜169 dB、2 秒ステレオで出力は有限、3 音源の和と混合の SNR 25〜27 dB（SNR 損失モデルとして妥当な範囲）。これは「移植が自己矛盾していない」ことの傍証で、上流との一致の証明ではない |
| WebGPU の 1 チャンク処理時間 | 本コンテナのヘッドレス Chromium 141 には WebGPU アダプタがない（SwiftShader 指定でも `navigator.gpu` なし）。**利用者の実機で計測する必要がある** |
| WASM の 1 チャンク処理時間 | onnxruntime-web 未導入のため未計測。概算：30.5M MAC/フレーム × 172 フレーム/秒 ≈ 5.3 GMAC/音声 1 秒/1ch。単一スレッド WASM SIMD（5〜15 GMAC/s と仮定）ではステレオで **実時間の 0.7〜2 倍**。10 分の動画で 7〜20 分の見込み（要実測） |
| onnxruntime-web の WASM ビルドに contrib 演算子（DynamicQuantizeLSTM 等）が含まれるか | 未確認。int8 は「重みのみ量子化＋DequantizeLinear」方式なら標準演算子だけで済む |
| DnR v2 の学習データのライセンス | zenodo.org 遮断のため未確認（ブリーフ §4.3 の記述を README に転記する前提） |
| 到達不可だったホスト | huggingface.co、hf-mirror.com、api.github.com、onnxruntime.ai、zenodo.org、download.pytorch.org、cdn.jsdelivr.net。到達可：raw.githubusercontent.com、media.githubusercontent.com、registry.npmjs.org、pypi.org |

## 計測結果

| 計測 | 値 |
|---|---|
| チェックポイント | 122,317,463 bytes、30,566,472 パラメータ |
| ONNX の見込みサイズ | fp32 122MB（GitHub 100MB 超）／fp16 61MB／int8（重みのみ）約 31MB |
| NumPy STFT→iSTFT 往復 SNR | 1024: 155.3 dB、2048: 158.2 dB、8192: 169.1 dB |
| NumPy 参照 forward（2 秒ステレオ） | 1.9 秒（計測目的ではない） |
| 3 音源の和 vs 混合（合成信号） | 25.5〜27.3 dB |
| WebGPU／WASM の処理時間 | 未計測 |

## 決める必要があること

1. **パッケージ導入の承認**（必須。下記一覧）。
2. **ネットワーク許可**：環境設定の Allowed domains に `download.pytorch.org` を足せば CPU 版 torch（約 200MB）で済む。足さない場合は PyPI の CUDA 同梱版（約 555MB ＋ nvidia 依存 2GB 強）になる。任意で `huggingface.co` を足せば audiojs 版との突き合わせもできる。
3. **配布するモデルの形式**：int8（重みのみ、約 31MB、推奨。fp32 との SNR を測って 35 dB 以上なら採用）／fp16（約 61MB）／fp32 を 2 ファイルに分割（122MB）。
4. **リポジトリ名と URL**：ブリーフは `bgm-buster`、実際は `4n5AI/BGMBuster`（Pages は `https://4n5ai.github.io/BGMBuster/`）。推奨：リポジトリ名は変えず、npm の `name` は `bgm-buster`、Vite の `base` は Actions の `GITHUB_REPOSITORY` から自動で決める（改名しても壊れない）。
5. **NAMARI との整合**：上記のとおり「Vite + TS を採用し、表記の慣例だけ引き継ぐ」でよいか。

## 次にやること（承認後）

1. torch で上流 `MRX` を実行し、同じ入力で NumPy 参照と ONNX（onnxruntime）を突き合わせて音源ごとの SNR を表にする。ONNX は出力名を `music` / `speech` / `sfx` と明示して書き出す（`scripts/export/`）。
2. fp16 / int8（重みのみ）を作り、fp32 との SNR とサイズを表にする。
3. onnxruntime-web（Node, WASM 単一スレッド）で 20 秒チャンクの処理時間を計測。利用者の実機で WebGPU を測るための小さな計測ページを用意する。
4. JS の STFT / iSTFT を実装し、NumPy の試験ベクトルで一致を確認（M1 の入口）。
