import itertools, sys
from ultravox.data import datasets, types, registry
from ultravox.training import train
args = types.VoiceDatasetArgs(max_audio_duration_secs=16, shuffle=True)
for n in ["librispeech-clean-continuation","librispeech-other-continuation","commonvoice-en"]:
    ds = registry.create_dataset(n, args)
    try: L = len(ds)
    except Exception as e: L = repr(e)
    it = iter(ds); s = next(it); s2 = next(it)
    print("DS", n, "len", L, "| keys", sorted(s.__dict__.keys())[:6] if hasattr(s,"__dict__") else type(s), flush=True)
    print("   text:", str(getattr(s,"messages",s))[:200].replace("\n"," "), flush=True)
