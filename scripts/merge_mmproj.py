#!/usr/bin/env python3
"""Merge a vision mmproj GGUF and an audio mmproj GGUF into one file with both encoders.

The llama.cpp loader reads `clip.projector_type` first and applies it to every modality, so the merged file
drops that key and sets `clip.vision.projector_type` and `clip.audio.projector_type` instead. Tensor payloads
are copied unchanged (no re-quantization). Tensor names must not clash (vision: v.*, mm.N.*; audio: a.*, mm.a.*).

usage: merge_mmproj.py VISION.gguf AUDIO.gguf OUT.gguf   (needs the fork's gguf-py on PYTHONPATH)
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

vis_path, aud_path, out_path = sys.argv[1:4]
vis, aud = GGUFReader(vis_path), GGUFReader(aud_path)

vnames = {t.name for t in vis.tensors}
clash = vnames & {t.name for t in aud.tensors}
assert not clash, f"tensor name clash: {sorted(clash)[:5]}"

vproj = vis.fields["clip.projector_type"].contents()
aproj = aud.fields["clip.projector_type"].contents()
assert vis.fields["clip.vision.projection_dim"].contents() == aud.fields["clip.audio.projection_dim"].contents(), \
    "vision and audio projection dims differ (both must equal the text model hidden size)"

SKIP = {"clip.projector_type", "general.architecture", "general.name", "general.size_label", "general.file_type",
        "general.quantization_version"}
w = GGUFWriter(out_path, arch="clip")


def copy_kv(reader):
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in SKIP or key in done:
            continue
        done.add(key)
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vtype)


done = set()
copy_kv(vis)
copy_kv(aud)
w.add_string("general.name", "bonsai-vision-audio-mmproj")
w.add_string("clip.vision.projector_type", vproj)
w.add_string("clip.audio.projector_type", aproj)
w.add_bool("clip.has_vision_encoder", True)
w.add_bool("clip.has_audio_encoder", True)

for reader in (vis, aud):
    for t in reader.tensors:
        w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=False)
w.close()
print(f"wrote {out_path}: vision={vproj} audio={aproj} tensors={len(vis.tensors) + len(aud.tensors)}")
