#!/usr/bin/env python3
"""Latency of the audio path through a llama.cpp server: different clips per request (no audio cache hits).
usage: latency.py HUB_DIR URL OUT.json
Per clip: (a) audio, max_tokens=1  (encoder + prefill + 1 token), (b) audio streamed: time to first token and total,
(c) the reference text instead of audio, max_tokens=1 (prefill + 1 token; subtract from (a) for an encoder estimate)."""
import base64, glob, io, json, statistics, sys, time, urllib.request
import pyarrow.parquet as pq, soundfile as sf

hub, url, outp = sys.argv[1:4]
files = sorted(glob.glob(f"{hub}/datasets--fixie-ai--librispeech_asr/snapshots/*/clean/validation-*.parquet"))
rows = []
for f in files:
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g, columns=["audio", "text"]).to_pylist():
            a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            rows.append((len(a) / sr, r))
        if len(rows) >= 400: break
    if len(rows) >= 400: break
P = "Repeat the following text, without any explanation: "
def post(body, stream=False):
    body.update({"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)
def audio_body(r, **kw):
    b = base64.b64encode(r["audio"]["bytes"]).decode()
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": P}, {"type": "input_audio", "input_audio": {"data": b, "format": "wav"}}]}], **kw}
def text_body(r, **kw):
    return {"messages": [{"role": "user", "content": P + r["text"].lower()}], **kw}
used = set()
def pick(target, n):
    c = sorted((abs(d - target), i) for i, (d, _) in enumerate(rows) if i not in used and d > 0.5)[:n]
    used.update(i for _, i in c); return [i for _, i in c]
for i in pick(5, 2): post(audio_body(rows[i][1], max_tokens=8)).read()   # warm-up
res = []
for target in (1, 3, 6, 10):
    for i in pick(target, 6):
        d, r = rows[i]
        t = time.time(); resp = post(audio_body(r, max_tokens=80, stream=True)); first = None; n = 0; text = ""; tm = {}
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            ch = json.loads(line[5:])
            if ch.get("timings"): tm = ch["timings"]
            c = (ch.get("choices") or [{}])[0].get("delta", {}).get("content")
            if c:
                if first is None: first = time.time() - t
                n += 1; text += c
        tot = time.time() - t
        t = time.time(); post(text_body(r, max_tokens=1)).read(); x1 = time.time() - t
        a1 = first
        res.append({"target": target, "dur": round(d, 2), "audio_1tok": a1, "text_1tok": x1, "ttft": first, "total": tot, "tokens": n, "text": text, "prompt_ms": tm.get("prompt_ms"), "prompt_n": tm.get("prompt_n"), "predicted_ms": tm.get("predicted_ms"), "predicted_n": tm.get("predicted_n")})
        print(f"{target:2d}s clip {d:5.2f}s  audio ttft {a1*1000:5.0f} ms  text+1tok {x1*1000:5.0f} ms  server prompt {(tm.get('prompt_ms') or 0):4.0f} ms ({tm.get('prompt_n')} tok)  total {tot*1000:5.0f} ms  tokens {n}", flush=True)
json.dump(res, open(outp, "w"))
print("\nmedians (ms) by target duration:")
for target in (1, 3, 6, 10):
    g = [x for x in res if x["target"] == target]
    m = lambda k: round(statistics.median(x[k] for x in g) * 1000)
    print(f"{target:2d}s: ttft {m('ttft')}, server prompt {m('prompt_ms')}, text+1tok {m('text_1tok')}, total {m('total')}, tokens {statistics.median(x['tokens'] for x in g)}  (n={len(g)})")
