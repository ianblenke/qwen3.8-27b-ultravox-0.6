#!/usr/bin/env python3
"""Assemble a self-contained Ultravox HF directory for llama.cpp's `convert_hf_to_gguf.py --mmproj`.

Ultravox checkpoints store only the trained projector (diff_state_dict). The converter also needs the Whisper
encoder tensors and an `audio_config`. This copies the projector from CKPT, takes the encoder from the stock
Whisper snapshot (or merges an optional LoRA-trained set of tensors if CKPT has audio_tower.* weights), and
writes model.safetensors + config.json into OUT.

usage: assemble_ultravox_hf.py CKPT_DIR WHISPER_SNAPSHOT_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

ckpt, whisper, out = (Path(a) for a in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)

tensors = {}
with safe_open(ckpt / "model.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith("multi_modal_projector."):
            tensors[k] = f.get_tensor(k)
        elif k.startswith("audio_tower.") and "lora" not in k:
            tensors[k] = f.get_tensor(k)  # present only if the encoder itself was trained
n_proj = sum(k.startswith("multi_modal_projector.") for k in tensors)
assert n_proj == 4, f"expected 4 projector tensors, found {n_proj}"

with safe_open(whisper / "model.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith("model.encoder."):
            name = "audio_tower." + k[len("model.encoder."):]
            tensors.setdefault(name, f.get_tensor(k).to(tensors["multi_modal_projector.linear_1.weight"].dtype))

wcfg = json.load(open(whisper / "config.json"))
cfg = json.load(open(ckpt / "config.json"))
audio_keys = ["model_type", "d_model", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim",
              "num_mel_bins", "max_source_positions", "activation_function", "layer_norm_eps"]
cfg["audio_config"] = {k: wcfg[k] for k in audio_keys if k in wcfg}
cfg["audio_config"].setdefault("model_type", "whisper")
cfg["architectures"] = ["UltravoxModel"]
# Ultravox leaves a stale hidden_size in config.json; the converter writes it as the projection dim, so use the
# projector's real output width (the text model's hidden size).
cfg["hidden_size"] = int(tensors["multi_modal_projector.linear_2.weight"].shape[0])
json.dump(cfg, open(out / "config.json", "w"), indent=2)
save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
print(f"wrote {out}: {len(tensors)} tensors, projector={n_proj}, encoder={sum(k.startswith('audio_tower.') for k in tensors)}")
