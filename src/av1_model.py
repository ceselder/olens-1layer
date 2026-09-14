"""N-layer AV model class + checkpoint loader (shared by eval/conditioning/playground scripts).
Frozen 27B embed/norm/head/rotary + N trainable blocks (the last N full-attention blocks) + act_proj +
embed_scale. Sequence = [act_proj(h42), embed(tokens)], causal, no prompt. Loads both the new
{"blocks": ...} checkpoints and the original single-block {"block": ...} ones."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import D_MODEL, load_base
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class LSTMStack(nn.Module):
    """'Ordinary RNN' reader: N-layer LSTM (hidden = d_model) over [soft token, embeddings] + identity-init output map.
    Runs in fp32 (cuDNN); returns a tensor shaped like a decoder block output so the rest of the model is unchanged."""
    def __init__(self, n_layers, d=D_MODEL):
        super().__init__()
        self.rnn = nn.LSTM(d, d, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(d, d, bias=True); nn.init.eye_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, **kw):
        with torch.autocast("cuda", enabled=False):
            y, _ = self.rnn(x.float())
            return self.out(y)


class OneLayerAV(nn.Module):
    def __init__(self, base, init_layer=63, n_layers=1, src_layers=None, block_type="attn"):
        super().__init__()
        m = base.model
        self.embed, self.norm, self.head, self.rotary = m.embed_tokens, m.norm, base.lm_head, m.rotary_emb
        for mod in (self.embed, self.norm, self.head):
            for p_ in mod.parameters():
                p_.requires_grad_(False)
        if src_layers is None:
            lt = list(m.config.layer_types) if getattr(m.config, "layer_types", None) else ["full_attention"] * len(m.layers)
            attn_idx = [i for i, t in enumerate(lt) if t == "full_attention"]
            src_layers = attn_idx[-n_layers:] if n_layers > 1 else [init_layer if init_layer >= 0 else 63]
        self.src_layers = list(src_layers)
        self.block_type = block_type
        self.blocks = nn.ModuleList([LSTMStack(n_layers)] if block_type == "lstm" else [copy.deepcopy(m.layers[i]) for i in self.src_layers])
        self.act_proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
        nn.init.eye_(self.act_proj.weight); nn.init.zeros_(self.act_proj.bias)
        self.embed_scale = nn.Parameter(torch.tensor(1.0))

    def hidden(self, h42, tok_ids):
        x0 = self.act_proj(h42.float()).to(self.embed.weight.dtype)[:, None, :]
        xt = self.embed(tok_ids) * self.embed_scale.to(self.embed.weight.dtype)
        x = torch.cat([x0, xt], 1)
        pos = torch.arange(x.shape[1], device=x.device)[None]
        pe = self.rotary(x, pos)
        for blk in self.blocks:
            out = blk(x, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def forward(self, h42, tok_ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.head(self.norm(self.hidden(h42, tok_ids)))

    @torch.no_grad()
    def generate(self, h42, n, temperature=0.0):
        toks = torch.zeros((h42.shape[0], 0), dtype=torch.long, device=h42.device)
        for _ in range(n):
            lg = self.forward(h42, toks)[:, -1].float()
            nxt = lg.argmax(-1) if temperature <= 0 else torch.multinomial(F.softmax(lg / temperature, -1), 1)[:, 0]
            toks = torch.cat([toks, nxt[:, None]], 1)
        return toks


def load_av1(ckpt_path, device="cuda"):
    sd = torch.load(ckpt_path, map_location="cpu")
    base = load_base(device)
    if "blocks" in sd:
        model = OneLayerAV(base, sd.get("init_layer", 63), sd.get("n_layers", 1), sd.get("src_layers"), sd.get("block_type", "attn")).to(device)
        model.blocks.load_state_dict({k: v.to(device) for k, v in sd["blocks"].items()})
    else:                                      # original single-block checkpoint
        model = OneLayerAV(base, sd.get("init_layer", 63), 1).to(device)
        model.blocks[0].load_state_dict({k: v.to(device) for k, v in sd["block"].items()})
    del base; torch.cuda.empty_cache()
    model.act_proj.load_state_dict({k: v.to(device) for k, v in sd["act_proj"].items()})
    model.embed_scale.data = sd["embed_scale"].to(device).float()
    return model.eval()


def load_small(ckpt_path, frozen_path, device="cuda"):
    """Load a SmallReader checkpoint (block_type 'small') without the 27B: frozen embed/norm/head come from extract_frozen.py's file."""
    from onelayer_av_small import SmallReader, FrozenRMSNorm   # re-exported below
    sd = torch.load(ckpt_path, map_location="cpu"); fz = torch.load(frozen_path, map_location="cpu")
    emb = nn.Embedding.from_pretrained(fz["embed"].to(device), freeze=True); norm = FrozenRMSNorm(fz["norm"].to(device), fz["eps"])
    head = nn.Linear(fz["d"], fz["vocab"], bias=False); head.weight = nn.Parameter(fz["head"].to(device), requires_grad=False)
    from onelayer_av_small import FullWidthAttnReader, LinearRNNReader
    m = (LinearRNNReader(emb, norm, head) if sd.get("block_type") == "linrnn" else FullWidthAttnReader(emb, norm, head, sd["inner"], sd["n_layers"], mlp_hidden=sd.get("mlp_hidden", 0), mlp_affine=sd.get("mlp_affine", False), n_heads=sd.get("heads") or None) if sd.get("block_type") == "fullattn" else SmallReader(emb, norm, head, sd["d_small"], sd["n_layers"], rnn_type=sd.get("small_rnn", "none"), use_mlp=not sd.get("small_no_mlp", False))).to(device)
    m.load_state_dict({k: v.to(device) for k, v in sd["trainable_sd"].items()}, strict=False)
    return m.eval()
