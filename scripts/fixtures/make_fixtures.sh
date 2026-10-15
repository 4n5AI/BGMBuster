#!/usr/bin/env bash
# Synthetic test media for the input/output round trip (no third-party material). Needs ffmpeg with
# libx264, libvpx-vp9, libopus, aac. usage: make_fixtures.sh <out_dir> [seconds=6]
set -euo pipefail
out="$1"; secs="${2:-6}"
mkdir -p "$out"
# audio: two tones with a click every second so that sync offsets are easy to measure
atone="aevalsrc=exprs='0.3*sin(2*PI*440*t)+0.2*sin(2*PI*1760*t)*lt(mod(t\,1)\,0.02)|0.3*sin(2*PI*660*t)+0.2*sin(2*PI*1760*t)*lt(mod(t\,1)\,0.02)':s=48000:d=$secs"
vsrc="testsrc2=size=1280x720:rate=30:duration=$secs"
vsrc_v="testsrc2=size=720x1280:rate=30:duration=$secs"
common=(-hide_banner -loglevel error -y)
ffmpeg "${common[@]}" -f lavfi -i "$vsrc" -f lavfi -i "$atone" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a libopus -b:a 96k -shortest "$out/h264_opus.mp4"
ffmpeg "${common[@]}" -f lavfi -i "$vsrc" -f lavfi -i "$atone" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac -b:a 128k -shortest "$out/h264_aac.mp4"
ffmpeg "${common[@]}" -f lavfi -i "$vsrc" -f lavfi -i "$atone" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a pcm_s16le -shortest "$out/h264_pcm.mov"
ffmpeg "${common[@]}" -f lavfi -i "$vsrc_v" -f lavfi -i "$atone" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a libopus -b:a 96k -shortest "$out/vertical_h264_opus.mp4"
ffmpeg "${common[@]}" -f lavfi -i "$vsrc" -f lavfi -i "$atone" -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -b:v 1M -c:a libopus -b:a 96k -shortest "$out/vp9_opus.webm"
ffmpeg "${common[@]}" -f lavfi -i "$vsrc" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -an "$out/noaudio.mp4"
ffmpeg "${common[@]}" -f lavfi -i "$vsrc" -f lavfi -i "$atone" -f lavfi -i "$atone" -map 0:v -map 1:a -map 2:a -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a libopus -b:a 96k -shortest "$out/two_audio_tracks.mp4"
ffmpeg "${common[@]}" -f lavfi -i "$atone" -c:a pcm_s16le -ar 44100 "$out/audio_44k.wav"
ffmpeg "${common[@]}" -f lavfi -i "$atone" -c:a libmp3lame -b:a 160k "$out/audio.mp3" 2>/dev/null || ffmpeg "${common[@]}" -f lavfi -i "$atone" -c:a mp3 -b:a 160k "$out/audio.mp3"
ffmpeg "${common[@]}" -f lavfi -i "$atone" -c:a aac -b:a 128k "$out/audio.m4a"
ls -la "$out"
