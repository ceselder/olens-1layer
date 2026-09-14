"""Upload shallow-verbalizer checkpoints + loading code to a (private) HF model repo. Token from the Modal secret."""
import os, sys, glob
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ.pop("HF_HUB_OFFLINE", None)   # image sets offline=1 for the 27B load
from huggingface_hub import HfApi
tokn = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not tokn:
    print("NO HF TOKEN in env; keys:", [k for k in os.environ if "HF" in k or "HUG" in k]); sys.exit(1)
api = HfApi(token=tokn); repo = sys.argv[1]; ckpts = sys.argv[2].split(",")
api.create_repo(repo, repo_type="model", exist_ok=True, private=True)
readme = f"""---
license: other
base_model: Qwen/Qwen3.6-27B
tags: [interpretability, oracle-lens, activation-verbalizer]
---
# Shallow verbalizers for Qwen3.6-27B layer-42 activations

Each checkpoint is a small decoder stack that reads ONE activation vector (block-42 output, d=5120) as a soft token and
verbalizes it as text (trained: 12-token on-policy continuation). Frozen pieces (embeddings, final norm, lm_head, rotary)
are taken from Qwen/Qwen3.6-27B at load time; only the blocks, act_proj and embed_scale are stored here.

| folder | layers | training data | held-out CE (common test file) |
|---|---|---|---|
{chr(10).join(f"| `{c}` | see `n_layers` in the state dict | single pass, see report | see report |" for c in ckpts)}

## Load
```python
import sys; sys.path.insert(0, ".")           # av1_model.py + common.py from this repo
import torch
from av1_model import load_av1               # loads Qwen/Qwen3.6-27B (bf16, ~54GB GPU) for the frozen pieces
av = load_av1("av4_12m/final.pt")             # -> OneLayerAV, eval mode
h42 = ...                                     # [B, 5120] float32 block-42 output vectors (raw residual, norm ~89)
toks = av.generate(h42, 12)                   # greedy 12-token readout
logits = av(h42, token_ids)                   # teacher-forced: logits[:, i] predicts token_ids[:, i]
```
Capture h42 from the 27B with `common.Layer42Hook(model).capture()` around a forward pass (block-42 OUTPUT = hidden_states[43]).
Every number behind these checkpoints is in the code repo: https://github.com/ceselder/olens-1layer (data/*.json).
"""
open("/tmp/README.md", "w").write(readme); api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md", repo_id=repo)
for f in ("av1_model.py", "common.py", "ar_span.py", "feed_inverter_example.py", "feed_example.py"):
    api.upload_file(path_or_fileobj=f"/root/src/{f}", path_in_repo=f, repo_id=repo); print("uploaded", f, flush=True)
for c in ckpts:
    if ":" in c:                                   # explicit  src_path:dest_path  (file) or  src_dir/:dest_dir  (folder)
        src, dst = c.split(":", 1)
        if os.path.isdir(src):
            api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=repo); print("uploaded folder", src, "->", dst, flush=True)
        else:
            api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo); print("uploaded", src, "->", dst, flush=True)
        continue
    p = f"/vol/ckpt/{c}/final.pt"; print("uploading", p, os.path.getsize(p) / 1e9, "GB", flush=True)
    api.upload_file(path_or_fileobj=p, path_in_repo=f"{c}/final.pt", repo_id=repo); print("uploaded", c, flush=True)
print("HF_UPLOAD_DONE", repo, flush=True)
