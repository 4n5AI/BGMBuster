# BGM排除くん（BGM Buster）

動画（または音声）から **BGM（音楽）だけを取り除き、セリフと効果音を残す** ブラウザ完結型ツールです。

**公開URL:** https://4n5ai.github.io/BGMBuster/ ／ English: [README.en.md](README.en.md)

1. **ファイルを選ぶ**（MP4・MOV・WebM・WAV・MP3・M4A）
2. **BGMを除去**（AIモデルは初回だけダウンロード。約 31 MB）
3. **確認して保存**（「元の音／BGM除去後／BGMのみ」を映像と同期して試聴し、BGM と効果音の音量を 0〜100% で調整。動画は映像を再エンコードせずに保存）

- APIキー・ログイン・会員登録なし。ページを開いてすぐ使えます。
- **ファイルは端末の外へ送信しません。** 通信はページ本体と AI モデルの初回ダウンロードだけです。
- 映像は再エンコードしません（画質を落としません）。差し替えるのは音声だけです。

## しくみ

| 段階 | 使うもの |
|---|---|
| 動画・音声の読み込みと書き出し | [Mediabunny](https://mediabunny.dev/)（WebCodecs）。映像はパケットをそのままコピーし、音声だけ新しくエンコード |
| 音の分離 | [MRX](https://github.com/merlresearch/cocktail-fork-separation)（MERL「The Cocktail Fork Problem」ICASSP 2022）の学習済みモデルを [onnxruntime-web](https://onnxruntime.ai/) で実行。WebGPU が使える環境では GPU、なければ WebAssembly（CPU） |
| 再ミックス | セリフ＋効果音に、スライダーの分だけ BGM を足し戻す |

処理はすべて Web Worker の中で行い、キャンセルできます。モデルは Cache Storage に保存し、2 回目以降はダウンロードしません（SHA-256 で照合します）。

## 対応形式と上限

| 項目 | 内容 |
|---|---|
| 入力 | MP4・MOV（H.264 / HEVC など WebCodecs が読める映像、AAC / Opus / PCM 音声）、WebM（VP8 / VP9 / AV1 + Opus / Vorbis）、WAV、MP3、M4A |
| 出力 | MP4・MOV → MP4（映像コピー、音声 AAC 128 kbps 以上）／ WebM → WebM（映像コピー、音声 Opus）／ 音声入力 → WAV（16 bit、元のサンプリング周波数） |
| 個別保存 | セリフ／効果音／BGM をそれぞれ WAV で保存 |
| 長さ | 目安 10 分まで（メモリの都合。長いものは短く書き出してから） |
| チャンネル | ステレオまで。3ch 以上はステレオにまとめて処理 |
| 音声トラック | 複数ある場合は 1 本目だけを処理 |

## 動作環境

| ブラウザ | 状況 |
|---|---|
| Chrome・Edge（Windows / macOS、最新版） | 対応。WebGPU があれば GPU で処理 |
| Chrome（Linux） | WebAssembly で動作（AAC は同梱のエンコーダーを使用） |
| Safari・Firefox | 未確認（WebCodecs の対応状況によっては読み込めない形式があります） |
| スマートフォン | メモリが足りず失敗することがあります。短いファイルでお試しください |

処理時間の目安（WebAssembly、1 スレッド、2.1 GHz のサーバー用 CPU での実測）：音声 1 秒・1 チャンネルあたり約 0.73 秒。ステレオの 1 分の動画で約 1.5 分です。デスクトップの CPU や WebGPU ではこれより速くなります。

## 精度の限界

- 歌の入った BGM は、歌声がセリフ側に残ることがあります。
- チャイムや楽器音のような、音楽に近い効果音は BGM 側に分類されることがあります。
- 逆に、低い持続音（ドローン）や環境音のような BGM は「効果音」側に分類され、BGM の音量を 0 にしても残ることがあります。その場合は「効果音の音量」も下げてください（セリフだけが残ります）。
- 叫び声や鳴き声、声に似た上昇音など「声のような音」はセリフ側に分類され、どのスライダーでも残ります。どの音がどこに入ったかは「ステムを保存」で 3 本を聴き比べると分かります。
- BGM を完全には消さず、スライダーで小さくする使い方もできます。

## プライバシー

ファイルの内容は、この端末の中だけで処理されます。サーバーはありません。アクセス解析や広告もありません。

## モデルと学習データ

- モデル：MRX（Multi-Resolution CrossNet）。Mitsubishi Electric Research Laboratories（MERL）が MIT License で公開している学習済み重み（`default_mrx_pre_trained_weights.pth`）を、このリポジトリの `scripts/export/build_onnx.py` で ONNX に書き出し、重みを int8 で格納しています（計算は float32）。上流の PyTorch 実装との一致を確認しています（`docs/reports/m0.md`）。
- 学習データ：Divide and Remaster（DnR）v2。音楽と効果音は Free Music Archive と FSD50K 由来で、素材ごとにライセンスが異なり、**一部は非商用ライセンスの素材を含みます**。重み自体は MIT で公開されています。
- 論文：D. Petermann, G. Wichern, Z.-Q. Wang, J. Le Roux, "The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks," ICASSP 2022. https://arxiv.org/abs/2110.09958

## 利用上の注意

権利を持っている素材、または使用を許可された素材に使ってください。

## 既知の制限

- MP3 入力では、先頭に約 23 ms の無音が付きます（デコーダーがエンコーダー遅延の情報を使わないため。音声のみの入力なので同期には影響しません）。
- 出力の音声は、エンコーダーのフレーム境界まで数十 ms 長くなることがあります。
- 動画を保存するとき、音声エンコーダーの遅延（ブラウザ内蔵の AAC で 2112 サンプル、同梱のエンコーダーで 1024 サンプルなど）をその場で実測して補正します。実測できない環境では 1024 サンプルとみなします。使われた値は「技術情報」の「書き出し」行に出ます。
- WebGPU での処理時間は環境により大きく異なります。`bench.html` で実機の値を測れます。

## 開発

```bash
npm ci
npm run dev        # http://localhost:5173/
npm run typecheck  # tsc（アプリと Worker）
npm test           # vitest（STFT 往復、チャンク結合、WAV ヘッダー、ラウドネス、辞書のキー）
npm run build      # dist/
```

検証用スクリプト（Python は `scripts/parity/README.md`、試験メディアは `scripts/fixtures/make_fixtures.sh`）：

| スクリプト | 内容 |
|---|---|
| `scripts/export/build_onnx.py` | MERL のチェックポイントから ONNX（fp32 / fp16 / int8）を書き出す |
| `scripts/parity/parity.py` | 上流 PyTorch・NumPy 移植・ONNX の一致（SNR） |
| `scripts/parity/chunk_study.py` | チャンク処理と全体処理の差 |
| `scripts/parity/remix_ab.py` | 再ミックス方式 A / B の比較（既知のステムから） |
| `scripts/bench/ort-web-node.mjs` | onnxruntime-web（WASM）の処理時間とメモリ |
| `scripts/bench/app-e2e.cjs` | ヘッドレス Chromium での通し試験 |
| `scripts/bench/media-roundtrip.cjs` ＋ `scripts/parity/media_check.py` | 映像が再エンコードされていないこと、音ズレがないことの確認 |

## 公開手順（GitHub Pages）

GitHub Actions（`.github/workflows/deploy.yml`）で、`main` への push と手動実行のときに `dist/` を GitHub Pages に配置します。
リポジトリの **Settings → Pages** で Source を **GitHub Actions** にしてください。公開先は `https://<ユーザー名>.github.io/<リポジトリ名>/` です（ビルド時に `BASE_PATH` をリポジトリ名から決めます）。

## ファイル構成

```
index.html / bench.html     ← 本体と計測ページ（Vite の入口）
public/models/mrx-v1.int8.onnx ← 分離モデル（約 31 MB）
src/main.ts                 ← 起動
src/config.ts               ← モデルのパス・SHA-256・チャンク長・上限
src/app/                    ← 状態、パイプライン、画面
src/ui/                     ← DOM ヘルパー、アイコン
src/i18n/                   ← 日本語／英語の辞書と切替
src/audio/                  ← FFT、STFT、ラウドネス、再ミックス、WAV、リサンプル、試聴
src/separation/             ← モデルの抽象化、MRX、チャンク、Worker、モデルのキャッシュ
src/media/                  ← 読み込みとデコード
src/export/                 ← 映像コピー＋音声差し替え、保存
scripts/                    ← ONNX 書き出し、一致検証、計測、試験メディア
tests/                      ← 自動テスト
docs/reports/               ← マイルストーンごとの報告
```

## 開発フェーズ

- [x] M0：調査と検証（上流との一致 115 dB 以上、WASM の計測）
- [x] M1：分離エンジン（Worker、チャンク、キャンセル、再ミックス A / B）
- [x] M2：動画の入出力（映像コピーの確認、音ズレ 0 ms）
- [x] M3：UI（3 ステップ、試聴の同期、スライダー、保存、エラー表示）
- [x] M4：多言語（日本語／英語）
- [ ] M5：公開と実機計測（公開済み。AAC 入力は実機の出力で確認し、音ズレ 24.7 ms を修正。残り：WebGPU の実測、再ミックス方式の聴き比べ）

## License

MIT License — Copyright (c) 2026 4n5-Studio

同梱・流用した第三者のコードとモデルは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を見てください。
