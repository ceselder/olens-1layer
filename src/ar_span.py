"""AR(span): the activation reconstructor's vector for a text span. This is the INPUT the inverter readers expect.
AR = Qwen3.6-27B blocks 0..42 with the ar_mse_v3 LoRA, read at the span's last token, then value_head (5120->5120).
Trained on exactly-12-token spans (on-policy continuations); shorter/longer spans work but are slightly off-distribution.

    from ar_span import ARSpan
    ar = ARSpan("ar_mse_v3")                      # folder with adapter_config.json, adapter_model.safetensors, value_head.pt
    V = ar(["the cat sat on the mat and", ...])   # [B, 5120] float32, norm ~71
"""
import os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D_MODEL, load_base, load_tokenizer, Layer42Hook, ar_read


class ARSpan:
    def __init__(self, ar_dir, base=None, device="cuda"):
        from peft import PeftModel
        self.dev = device; self.tok = load_tokenizer()
        self.base = base if base is not None else load_base(device)
        self.ar = PeftModel.from_pretrained(self.base, ar_dir, adapter_name="ar").eval()
        self.hook = Layer42Hook(self.ar)
        self.value_head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(device, torch.float32)
        self.value_head.load_state_dict(torch.load(os.path.join(ar_dir, "value_head.pt"), map_location=device))

    @torch.no_grad()
    def from_ids(self, ids):                       # ids: LongTensor [B, T] (same T per batch; bare token ids)
        h = ar_read(self.ar, self.hook, ids, torch.ones_like(ids), self.value_head)
        return self.value_head(h[:, -1].float())  # [B, 5120]

    def __call__(self, spans, n_tok=None):
        out = []
        for s in spans:
            ids = self.tok(s, add_special_tokens=False)["input_ids"]
            if n_tok: ids = ids[:n_tok]
            out.append(self.from_ids(torch.tensor([ids], device=self.dev))[0])
        return torch.stack(out)
