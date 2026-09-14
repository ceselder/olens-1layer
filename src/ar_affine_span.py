"""Affine AR: text span -> vector, via the FROZEN Qwen3.6-27B block-42 states of the span (last token ⊕ mean over the span)
and the closed-form ridge map (W, b) fit to predict the activation that preceded the span. This is the INPUT of the affAR_* readers.

    from ar_affine_span import AffineARSpan
    ar = AffineARSpan("ar_affine_lastmean/affine.pt")     # loads the 27B (bf16, ~54 GB) for the frozen read
    V = ar(["the cat sat on the mat and"])                  # [1, 5120] float32
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D_MODEL, load_base, load_tokenizer, Layer42Hook, ar_read


class AffineARSpan:
    def __init__(self, affine_path, base=None, device="cuda"):
        self.dev = device; self.tok = load_tokenizer(); self.base = base if base is not None else load_base(device)
        self.hook = Layer42Hook(self.base); a = torch.load(affine_path, map_location=device)
        self.W, self.b, self.feat = a["W"].to(device).float(), a["b"].to(device).float(), a["feat"]

    @torch.no_grad()
    def from_ids(self, ids):                       # ids LongTensor [B, T], bare token ids
        h = ar_read(self.base, self.hook, ids, torch.ones_like(ids), None).float()   # [B, T, 5120] frozen block-42 states
        x = torch.cat([h[:, -1], h.mean(1)], -1) if self.feat == "last+mean" else h[:, -1]
        return x @ self.W + self.b

    def __call__(self, spans, n_tok=None):
        out = []
        for s in spans:
            ids = self.tok(s, add_special_tokens=False)["input_ids"]
            if n_tok: ids = ids[:n_tok]
            out.append(self.from_ids(torch.tensor([ids], device=self.dev))[0])
        return torch.stack(out)
