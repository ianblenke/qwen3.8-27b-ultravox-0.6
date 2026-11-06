#!/usr/bin/env python3
"""Run an Ultravox checkpoint (bf16 Qwen3.8 + trained projector) on a few wav clips and report whether it stops.

usage: bf16_infer.py CKPT_DIR WAV_DIR [MAX_TOKENS]
The model is built directly (UltravoxModel(config), then the projector tensors are loaded from CKPT_DIR/model.safetensors)
because UltravoxModel.from_pretrained nests from_pretrained calls that transformers 5 rejects under its meta-device init.
Prints each clip's reference text (WAV_DIR/valN.txt), the generated text and the generated-token count; a reply whose
token count equals MAX_TOKENS did not stop on its own.
"""
import glob
import sys

import soundfile as sf
import torch
import transformers
from safetensors.torch import load_file

from ultravox import data as datasets
from ultravox.inference import infer
from ultravox.model import ultravox_config, ultravox_model, ultravox_processing

ckpt, wavdir = sys.argv[1:3]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dtype = torch.bfloat16

config = ultravox_config.UltravoxConfig.from_pretrained(ckpt)
model = ultravox_model.UltravoxModel(config).to(dtype)
state = load_file(f"{ckpt}/model.safetensors")
missing, unexpected = model.load_state_dict(state, strict=False)
proj = [k for k in state if k.startswith("multi_modal_projector.")]
print(f"loaded {len(state)} tensors ({len(proj)} projector); unexpected={len(unexpected)}", flush=True)
assert len(proj) == 4 and not unexpected
model.merge_and_unload()
model.to("cuda")

tok = transformers.AutoTokenizer.from_pretrained(ckpt)
tok.padding_side = "left"
tok.pad_token = tok.eos_token
audio_proc = transformers.AutoProcessor.from_pretrained(config.audio_model_id)
proc = ultravox_processing.UltravoxProcessor(
    audio_proc, tokenizer=tok, stack_factor=config.stack_factor, audio_context_size=model.audio_tower_context_length
)
inf = infer.LocalInference(model=model, processor=proc, tokenizer=tok, dtype=dtype, enable_thinking=False)

prompt = "Repeat the following text, without any explanation: <|audio|>"
for wav in sorted(glob.glob(f"{wavdir}/val*.wav")):
    audio, sr = sf.read(wav, dtype="float32")
    ref = open(wav[:-4] + ".txt").read().strip()
    out = inf.infer(datasets.VoiceSample.from_prompt_and_raw(prompt, audio, sr), max_tokens=cap, temperature=0.0)
    print(f"\nCLIP {wav.split('/')[-1]}\n  ref: {ref}\n  hyp: {out.text!r}\n  tokens={out.output_tokens} cap={cap} stopped_on_own={out.output_tokens < cap}", flush=True)
