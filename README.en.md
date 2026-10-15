# BGM Buster (BGM排除くん)

Removes **only the background music** from a video (or an audio file) and keeps the dialogue and the sound effects. Everything runs in your browser.

**Live:** https://4n5ai.github.io/BGMBuster/ ／ 日本語: [README.md](README.md)

1. **Choose a file** (MP4, MOV, WebM, WAV, MP3 or M4A)
2. **Remove the music** (the AI model, about 31 MB, is downloaded the first time only)
3. **Review and save**: listen to the original, the result or the music alone in sync with the picture, set the music and sound-effects levels from 0 to 100 %, and save the video without re-encoding the picture

- No API key, no account, no sign-up.
- **Your file never leaves your device.** The only network traffic is the page itself and the first download of the model.
- The picture is never re-encoded; only the audio track is replaced.

## How it works

| Stage | What is used |
|---|---|
| Reading and writing media | [Mediabunny](https://mediabunny.dev/) (WebCodecs). Video packets are copied as they are; only the audio is encoded again |
| Separation | The pretrained [MRX](https://github.com/merlresearch/cocktail-fork-separation) model (MERL, "The Cocktail Fork Problem", ICASSP 2022) running on [onnxruntime-web](https://onnxruntime.ai/): WebGPU when available, WebAssembly (CPU) otherwise |
| Remix | Dialogue + effects, plus as much of the music as the slider says |

Processing runs in a Web Worker and can be cancelled. The model is kept in Cache Storage (verified by SHA-256), so it is downloaded once.

## Formats and limits

| | |
|---|---|
| Input | MP4, MOV (any video WebCodecs can read, e.g. H.264 / HEVC; AAC, Opus or PCM audio), WebM (VP8 / VP9 / AV1 with Opus / Vorbis), WAV, MP3, M4A |
| Output | MP4 / MOV → MP4 (video copied, AAC at 128 kbps or more) · WebM → WebM (video copied, Opus) · audio input → WAV (16-bit, original sample rate) |
| Separate stems | Dialogue, sound effects and music as individual WAV files |
| Length | About 10 minutes (memory); trim longer files first |
| Channels | Up to stereo; more channels are mixed down to stereo |
| Audio tracks | Only the first track is processed |

## Browser support

| Browser | Status |
|---|---|
| Chrome, Edge (Windows / macOS, latest) | Supported; GPU via WebGPU when available |
| Chrome (Linux) | Works on WebAssembly (AAC via the bundled encoder) |
| Safari, Firefox | Not verified; some formats may fail to load depending on WebCodecs support |
| Phones | May run out of memory; try short files |

Speed (measured on WebAssembly, single thread, a 2.1 GHz server CPU): about 0.73 s per second of audio per channel, so roughly 1.5 minutes for a one-minute stereo video. Desktop CPUs and WebGPU are faster.

## Limits of the separation

- Songs with vocals: some of the singing can stay with the dialogue.
- Sound effects that resemble music (chimes, instrument-like sounds) may end up with the music.
- Conversely, drone-like or ambience-like music can be classified as sound effects and remain at music level 0. Lower the sound effects volume as well in that case (only the dialogue stays).
- You can leave some music in with the slider instead of removing it completely.

## Privacy

Files are processed on your device only. There is no server, no analytics and no advertising.

## Model and training data

- Model: MRX (Multi-Resolution CrossNet). The pretrained weights published under the MIT License by Mitsubishi Electric Research Laboratories (`default_mrx_pre_trained_weights.pth`) are exported to ONNX by `scripts/export/build_onnx.py` in this repository, with int8 weights (float32 arithmetic). The export matches the upstream PyTorch implementation (see `docs/reports/m0.md`).
- Training data: Divide and Remaster (DnR) v2. Its music and sound effects come from the Free Music Archive and FSD50K, each clip under its own licence, **some of them non-commercial**. The weights themselves are published under MIT.
- Paper: D. Petermann, G. Wichern, Z.-Q. Wang, J. Le Roux, "The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks," ICASSP 2022. https://arxiv.org/abs/2110.09958

## Please note

Use this only on material you own or have permission to edit.

## Known limitations

- MP3 input gains about 23 ms of silence at the start (the decoder ignores the encoder-delay information; audio-only input, so nothing goes out of sync).
- The output audio can be a few tens of milliseconds longer than the input, up to the encoder's frame boundary.
- WebGPU timings vary a lot between devices; `bench.html` measures your own.

## Development

```bash
npm ci
npm run dev        # http://localhost:5173/
npm run typecheck  # tsc (app and worker)
npm test           # vitest (STFT round trip, chunk joins, WAV header, loudness, dictionary keys)
npm run build      # dist/
```

Verification scripts (Python: see `scripts/parity/README.md`; test media: `scripts/fixtures/make_fixtures.sh`):

| Script | Purpose |
|---|---|
| `scripts/export/build_onnx.py` | Export ONNX (fp32 / fp16 / int8) from MERL's checkpoint |
| `scripts/parity/parity.py` | SNR of the NumPy port and the ONNX graphs against upstream PyTorch |
| `scripts/parity/chunk_study.py` | Chunked vs whole-file processing |
| `scripts/parity/remix_ab.py` | Remix method A vs B on known stems |
| `scripts/bench/ort-web-node.mjs` | onnxruntime-web (WASM) time and memory |
| `scripts/bench/app-e2e.cjs` | End-to-end run in headless Chromium |
| `scripts/bench/media-roundtrip.cjs` + `scripts/parity/media_check.py` | Video packets unchanged, audio in sync |

## Deployment (GitHub Pages)

GitHub Actions (`.github/workflows/deploy.yml`) publishes `dist/` to GitHub Pages on pushes to `main` and on manual runs. Set **Settings → Pages → Source** to **GitHub Actions**. The site lives at `https://<user>.github.io/<repository>/` (the build derives `BASE_PATH` from the repository name).

## License

MIT License — Copyright (c) 2026 4n5-Studio

Third-party code and the model are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
