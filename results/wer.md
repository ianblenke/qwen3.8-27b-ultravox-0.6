# WER, first 100 validation clips of each LibriSpeech subset (clean 2038 words, other 1820 words)
| checkpoint | path | clean | other |
|---|---|---|---|
| 24000 | bf16 transformers | 3.78% (77 err) | 9.18% (167 err) |
| 24000 | ternary build, llama.cpp + trim patch | 4.07% (83) | 8.13% (148) |
| 37500 | bf16 transformers | 3.68% (75) | 8.24% (150) |
| 37500 | ternary build, llama.cpp + trim patch | 3.97% (81) | 8.35% (152) |
Greedy decoding, prompt "Repeat the following text, without any explanation:", raw replies scored. See scripts/bf16_wer.py and scripts/wer_eval.py.
