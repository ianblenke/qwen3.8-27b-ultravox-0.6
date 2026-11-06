#!/usr/bin/env python3
"""(WER variant) Run an Ultravox checkpoint (bf16 Qwen3.8 + trained projector) on a few wav clips and report whether it stops.

usage: bf16_infer.py CKPT_DIR WAV_DIR [MAX_TOKENS]
The model is built directly (UltravoxModel(config), then the projector tensors are loaded from CKPT_DIR/model.safetensors)
because UltravoxModel.from_pretrained nests from_pretrained calls that transformers 5 rejects under its meta-device init.
Prints each clip's reference text (WAV_DIR/valN.txt), the generated text and the generated-token count; a reply whose
token count equals MAX_TOKENS did not stop on its own.
"""
import glob
import json
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import io

import soundfile as sf
import torch
import transformers
from safetensors.torch import load_file

from ultravox import data as datasets
from ultravox.inference import infer
from ultravox.model import ultravox_config, ultravox_model, ultravox_processing

ckpt, hub, subset, n_clips, outp = sys.argv[1:6]
n_clips = int(n_clips)
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


def norm(t):
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()


def edit_distance(r, h):
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1]


files = sorted(glob.glob(f"{hub}/datasets--fixie-ai--librispeech_asr/snapshots/*/{subset}/validation-*.parquet"))
rows = []
for f in files:
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        rows += pf.read_row_group(g, columns=["audio", "text"]).to_pylist()
        if len(rows) >= n_clips:
            break
    if len(rows) >= n_clips:
        break
rows = rows[:n_clips]
errs = words = runon = 0
log = open(outp, "w")
for i, r in enumerate(rows):
    audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    ref = norm(r["text"])
    cap = int(len(ref) * 1.5) + 12
    out = inf.infer(datasets.VoiceSample.from_prompt_and_raw(prompt, audio, sr), max_tokens=cap, temperature=0.0)
    hyp = norm(out.text)
    e = edit_distance(ref, hyp)
    errs += e
    words += len(ref)
    runon += out.output_tokens >= cap
    print(f"{i:3d} wer={e / max(1, len(ref)):.2f} tokens={out.output_tokens}/{cap} | ref: {' '.join(ref)[:60]} | hyp: {' '.join(hyp)[:60]}", flush=True)
    log.write(json.dumps({"i": i, "ref": r["text"], "hyp": out.text, "errors": e, "ref_words": len(ref), "tokens": out.output_tokens, "cap": cap}) + "\n")
print(f"TOTAL clips={len(rows)} words={words} errors={errs} WER={errs / max(1, words):.4f} hit_cap={runon}")
