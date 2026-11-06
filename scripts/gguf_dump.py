import sys
from gguf import GGUFReader
for p in sys.argv[1:]:
    r=GGUFReader(p); print("==",p.split("/")[-1],"tensors",len(r.tensors))
    for k,f in r.fields.items():
        if k.startswith("GGUF.") : continue
        v=f.contents() if hasattr(f,"contents") else None
        print("  KV",k,str(v)[:70])
    names=[t.name for t in r.tensors]
    import re
    pre={}
    for n in names:
        q=re.sub(r"\d+","N",n); pre[q]=pre.get(q,0)+1
    print("  TENSOR-PATTERNS",len(pre)); [print("   ",q,c) for q,c in sorted(pre.items())[:60]]
