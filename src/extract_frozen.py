"""Save the frozen pieces small readers need (token embeddings, final RMSNorm, lm_head) so they can train without loading the 27B."""
import os, torch
from common import load_base
base = load_base("cuda"); m = base.model
os.makedirs("/vol/frozen", exist_ok=True)
torch.save({"embed": m.embed_tokens.weight.detach().to(torch.bfloat16).cpu(), "norm": m.norm.weight.detach().float().cpu(),
            "eps": float(getattr(m.norm, "variance_epsilon", 1e-6)), "head": base.lm_head.weight.detach().to(torch.bfloat16).cpu(),
            "vocab": base.lm_head.weight.shape[0], "d": base.lm_head.weight.shape[1]}, "/vol/frozen/qwen36_27b_embed_head.pt")
print("FROZEN_SAVED", base.lm_head.weight.shape, flush=True)
