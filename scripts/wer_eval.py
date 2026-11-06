#!/usr/bin/env python3
"""Word error rate of audio transcription through a llama.cpp server (OpenAI-style /v1/chat/completions, input_audio).

Reads N clips from local LibriSpeech parquet shards (HF hub cache layout), asks the server to repeat each clip,
and scores normalized text (lowercase, letters/digits/apostrophes only). Prints per-clip lines and a total.

usage: wer_eval.py --hub HUB_DIR --subset clean|other --n 100 --url http://127.0.0.1:8099 [--think]
Needs: pyarrow, soundfile (stdlib otherwise). Output is raw: no trimming of repeated text, so a model that does not
stop is penalized; the token cap is 1.5x the reference word count + 12.
"""
import argparse, base64, glob, io, json, re, sys, time, urllib.request

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--hub", required=True)
ap.add_argument("--subset", default="clean")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--url", default="http://127.0.0.1:8099")
ap.add_argument("--think", action="store_true", help="leave thinking on (default: enable_thinking=false)")
ap.add_argument("--prompt", default="Repeat the following text, without any explanation: ")
ap.add_argument("--out", default=None, help="write per-clip JSONL here")
a = ap.parse_args()


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


files = sorted(glob.glob(f"{a.hub}/datasets--fixie-ai--librispeech_asr/snapshots/*/{a.subset}/validation-*.parquet"))
rows = []
for f in files:
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        rows += pf.read_row_group(g, columns=["audio", "text"]).to_pylist()
        if len(rows) >= a.n:
            break
    if len(rows) >= a.n:
        break
rows = rows[: a.n]

errs = words = 0
log = open(a.out, "w") if a.out else None
for i, r in enumerate(rows):
    ref = norm(r["text"])
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": a.prompt},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(r["audio"]["bytes"]).decode(), "format": "wav"}},
        ]}],
        "temperature": 0,
        "max_tokens": int(len(ref) * 1.5) + 12,
    }
    if not a.think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]
    hyp_text = msg.get("content") or ""
    hyp = norm(hyp_text)
    e = edit_distance(ref, hyp)
    errs += e
    words += len(ref)
    print(f"{i:3d} {time.time() - t0:5.1f}s wer={e / max(1, len(ref)):.2f} | ref: {' '.join(ref)[:70]} | hyp: {' '.join(hyp)[:70]}", flush=True)
    if log:
        log.write(json.dumps({"i": i, "ref": r["text"], "hyp": hyp_text, "errors": e, "ref_words": len(ref)}) + "\n")
print(f"TOTAL clips={len(rows)} words={words} errors={errs} WER={errs / max(1, words):.4f}")
