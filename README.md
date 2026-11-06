# qwen3.8-27b-ultravox-0.6

An [Ultravox](https://github.com/fixie-ai/ultravox) audio adapter (the projector between a Whisper encoder and the LLM) trained against **Qwen3.8-27B (bf16)**, plus the tooling to serve it through a llama.cpp fork on a ternarized build of the same model. English speech in, text out; the LLM and the Whisper encoder are frozen and only the 52.4M-parameter projector is trained.

Weights and the converted audio projector GGUF are on Hugging Face: https://huggingface.co/ianblenke/qwen3.8-27b-ultravox-0.6 This repository holds the training recipe, the patches, the conversion and evaluation scripts, and the measurements.

## Results

Word error rate of a "repeat the following text" request, greedy decoding, first 100 validation clips of each LibriSpeech subset (clean: 2,038 words; other: 1,820 words). Text is lowercased and reduced to letters, digits and apostrophes; replies are scored raw, so a model that does not stop is penalized. One sample per cell: differences of a few tenths of a point are within noise.

| Checkpoint | Serving path | clean | other |
|---|---|---|---|
| step 24,000 | bf16 Qwen3.8-27B (transformers) | 3.78% | 9.18% |
| step 24,000 | ternary build, llama.cpp + trim patch | 4.07% | 8.13% |
| step 37,500 (final) | bf16 Qwen3.8-27B (transformers) | 3.68% | 8.24% |
| step 37,500 (final) | ternary build, llama.cpp + trim patch | 3.97% | 8.35% |

Without `patches/llama.cpp/0001-ultravox-trim-audio-padding.patch` the llama.cpp path did not stop after the sentence (see below); on three clips at step 24,000 WER was 1.20, all run-on. No run in the table produced a run-on (bf16: no reply hit the token cap; llama.cpp: no reply longer than 1.5x the reference plus 10 words).

Training: 2 x DGX Spark (128 GB unified memory each), data parallel, global batch 8, 37,500 steps (~300k samples) of LibriSpeech clean + other continuation and Common Voice 17 English, lr 2e-3 cosine, 500 warmup steps. Mean loss per ~1,850-step window fell 0.427 -> 0.189 -> 0.137 -> ... -> 0.097 (`results/loss.csv`); most of the gain came in the first 8,000 steps. The run was paused once at step 24,000 and resumed from that checkpoint (optimizer and scheduler state restored; data order reshuffled with a new seed).

## A finding worth knowing: padded audio makes the model run on

The Ultravox reference code trims the audio embeddings to the real clip length (a 2.6 s clip gives 17 audio tokens). llama.cpp's Whisper path always encodes a full 30 s window (about 188 audio positions, mostly silence). The projector was trained only on trimmed sequences, and with the padding present the model repeats the sentence instead of stopping. This reproduces in bf16 transformers by padding the clip with silence to 30 s, so it is an input-length mismatch, not a quantization or weights problem. The llama.cpp patch trims the embeddings to the same length rule as the reference processor (`ceil(ceil(mel_frames / 2) / stack_factor)`). Limitation: the patch assumes a single audio item per request.

## Layout
- `patches/transformers/` Transformers 5 / Qwen3.5-architecture compatibility patch against Ultravox `v0.6-1-g69ddc63` (`git apply`).
- `patches/llama.cpp/` the trim patch, against the PrismML llama.cpp fork at `56f21fb`.
- `docker/` image recipes: `Dockerfile.base` (NVIDIA PyTorch 26.07 container + transformers 4.57.3) and `Dockerfile.tf581` (adds transformers 5.8.1).
- `configs/` training configs, including the resume config. `scripts/` multi-node launcher, resume launchers, memory watchdog (set `MASTER_ADDR`, `NIC`, `WORKDIR`).
- `scripts/assemble_ultravox_hf.py`, `merge_mmproj.py`: checkpoint -> converter-ready HF dir -> merged GGUF projector.
- `scripts/bf16_infer.py`, `bf16_infer_pad.py`, `bf16_wer.py`, `wer_eval.py`: inference and WER evaluation (transformers and llama.cpp server).
- `results/` WER table and loss curve.

## Conversion
1. `scripts/assemble_ultravox_hf.py CKPT WHISPER_SNAPSHOT OUT` builds a converter-ready HF directory (projector from the checkpoint + the stock Whisper encoder; fixes the stale `hidden_size`).
2. `convert_hf_to_gguf.py --mmproj OUT --outtype f16` (PrismML llama.cpp fork) writes the audio mmproj (projector type `ultravox`).
3. To keep vision as well, `scripts/merge_mmproj.py VISION.gguf AUDIO.gguf OUT.gguf` combines it with an existing vision mmproj (per-modality projector-type keys; tensors copied unchanged). The merged file used here contains a vision tower we did not train and is not redistributed.

## Limits
English read speech only (LibriSpeech, Common Voice); not evaluated on noisy, accented, conversational or overlapping speech. Only 100 clips per subset were scored, and one early bf16 test clip got a refusal reply ("I cannot repeat that text") that has not been investigated. No safety evaluation. The adapter was trained against the base model; its behavior on fine-tunes of Qwen3.8-27B is untested.

## License and attribution
See `ATTRIBUTION.md`. Code and patches: MIT (derived from MIT-licensed Ultravox and llama.cpp). Adapter weights: Apache-2.0 (the base model is Apache-2.0; Whisper is MIT).
