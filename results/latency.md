# Latency

Measured 2026-11-06 on one NVIDIA GB10 (DGX Spark, unified memory) with nothing else running: the patched llama.cpp server built with CUDA, the ternary Qwen3.8-27B build and the step-37,500 audio projector, all layers on the GPU, 4,096-token context, the "repeat the following text" prompt, greedy decoding. Every request used a different LibriSpeech clip, so the server's prompt cache never held the audio. Medians of 6 clips per length, in milliseconds after the request arrives:

| Clip length | First text token | Server encode + prompt | Full reply (tokens) |
|---|---|---|---|
| ~2 s | 379 | 343 | 523 (2.5) |
| 3 s | 488 | 347 | 826 (9.5) |
| 6 s | 525 | 357 | 1,124 (17) |
| 10 s | 553 | 374 | 1,592 (30.5) |

The Whisper encoder always runs on a fixed 30-second window, so encode + prompt time barely grows with clip length. Generation runs at about 35 ms per token. Not measured: end-of-speech detection (the pause before a non-streaming model can start), text-to-speech, other GPUs, and any server load. Part of the first-token time is HTTP overhead and CPU audio preprocessing. Reproduce with `scripts/latency.py`. A first version of the benchmark that reused each clip's audio hit the prompt cache and understated latency; the numbers above are from the corrected run.

