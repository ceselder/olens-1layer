"""ONE-LAYER-TRANSFORMER activation verbalizer (AV) for Qwen3.6-27B layer-42 activations.

Can a single transformer block read h42 and say what the model is about to say?
  frozen : Qwen3.6-27B embed_tokens, final norm, lm_head (Qwen token space, 248k vocab)
  train  : ONE decoder block (init from the 27B's block 63 = a trainable skip-lens, or random)
           + act_proj Linear(d,d) identity-init on h42 + a learnable embedding scale
  seq    : [act_proj(h42), embed(tok_1..T)]  causal; position i predicts tok_{i+1}
  target : the on-policy rollout (AV / futurelens warm-start), from the shared harvest

    modal run --detach modal_app.py --task train --gpus 2 --nproc 2 --script onelayer_av.py \
      --args "--data '/vol_data/data/harvest_v1/*.parquet' --steps 6000 --batch 64 --out /vol/ckpt/av1_l63 --wandb-name av1_l63"
"""
import argparse, copy, glob, json, os, time
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import D_MODEL, load_base, load_tokenizer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--init-layer", type=int, default=63, help="27B block to init the (first) layer from; -1 = random")
    p.add_argument("--n-layers", type=int, default=1, help="stack the last N full-attention blocks (63,59,55,...)")
    p.add_argument("--n-tok", type=int, default=12, help="rollout tokens used as target")
    p.add_argument("--block-type", default="attn", choices=["attn", "gdn", "lstm", "small", "fullattn", "linrnn"], help="linrnn = full-width LINEAR recurrence h_t = A h_{t-1} + B x_t (52M); " + "fullattn = attention-only at the 5120 residual width with inner dim --inner (no projections, no MLP); " + "small = narrow pre-LN transformer at width --d-small with trainable 5120<->d projections; " + "attn = last N full-attention blocks (63,59,..); gdn = last N Gated-DeltaNet recurrent blocks (62,61,60,..); lstm = plain N-layer LSTM (hidden 5120), random init")
    p.add_argument("--max-files", type=int, default=0, help="cap parquet files per rank (0 = all; 3M rows per rank OOMs RAM)")
    p.add_argument("--ar", default=None, help="AR-INVERTER mode: soft token = AR(span) from this frozen AR adapter dir (not the real h42); target = span")
    p.add_argument("--exclude", default="", help="comma-separated substrings; any data file containing one is dropped (e.g. the common test file)")
    p.add_argument("--stream", action="store_true", help="stream parquet files one buffer at a time (single pass, unlimited data, RAM-light)")
    p.add_argument("--buffer-files", type=int, default=8, help="--stream: files per shuffle buffer (~4k rows each)")
    p.add_argument("--ar-affine", default=None, help="AR-INVERTER with the AFFINE AR: soft token = W·state(span)+b from this affine.pt (frozen 27B block-42 states); target = span")
    p.add_argument("--ar-cache", default=None, help="dir of precomputed AR(span) parquet files (precompute_ar.py): soft token = cached ar_vec; no AR forward")
    p.add_argument("--d-small", type=int, default=1024, help="--block-type small: reader width")
    p.add_argument("--inner", type=int, default=512, help="--block-type fullattn: attention inner dim (heads = inner/64)")
    p.add_argument("--heads", type=int, default=0, help="--block-type fullattn: number of attention heads (0 = inner/64, i.e. head_dim 64)")
    p.add_argument("--mlp-affine", action="store_true", help="--block-type fullattn: make the MLP sublayer a pure affine map (no GELU); --mlp-hidden = its rank (>=5120 -> full-rank Linear)")
    p.add_argument("--mlp-hidden", type=int, default=0, help="--block-type fullattn: add an MLP sublayer with this hidden width (0 = attention only)")
    p.add_argument("--small-no-mlp", action="store_true", help="--block-type small: attention-only blocks (no MLP sublayer)")
    p.add_argument("--small-rnn", default="none", choices=["none", "rnn", "gru", "lstm"], help="--block-type small: replace the transformer blocks by a classic recurrent stack")
    p.add_argument("--frozen", default=None, help="path to extract_frozen.py output: build embed/norm/head from it and skip loading the 27B (small readers only)")
    p.add_argument("--init-ckpt", default=None, help="warm start: load this checkpoint's trainable_sd into the (same-architecture) model before training")
    p.add_argument("--heldout", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="olens-1layer")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


args = parse()
import torch.distributed as dist
RANK = int(os.environ.get("RANK", "0")); WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LRANK = int(os.environ.get("LOCAL_RANK", "0")); is_dist = WORLD > 1; is_main = RANK == 0
if is_dist:
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
dev = f"cuda:{LRANK}"; torch.cuda.set_device(dev); torch.manual_seed(args.seed)
if is_main:
    os.makedirs(args.out, exist_ok=True)
tok = load_tokenizer(); pad_id = tok.eos_token_id


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


class SmallBlock(nn.Module):
    def __init__(self, d, n_heads, use_mlp=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.use_mlp = use_mlp
        if use_mlp:
            self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        T = x.shape[1]; mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.ln1(x); x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x)) if self.use_mlp else x


class FrozenRMSNorm(nn.Module):
    def __init__(self, w, eps):
        super().__init__(); self.weight = nn.Parameter(w, requires_grad=False); self.eps = eps
    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class SmallReader(nn.Module):
    """Narrow reader: act_proj 5120->d, frozen Qwen embeddings -> tok_in 5120->d, N pre-LN causal blocks at width d,
    out_proj d->5120 -> frozen final RMSNorm -> frozen lm_head. Trainable params ~ 3*5120*d + 12*d^2*N."""
    def __init__(self, embed, norm, head, d, n_layers, n_heads=None, max_pos=64, rnn_type="none", use_mlp=True):
        super().__init__()
        self.rnn_type = rnn_type; self.use_mlp = use_mlp
        self.embed, self.norm, self.head = embed, norm, head
        for mod in (self.embed, self.norm, self.head):
            for p_ in mod.parameters(): p_.requires_grad_(False)
        self.d = d; self.src_layers = []; self.block_type = "small"
        self.act_proj = nn.Linear(D_MODEL, d); self.tok_in = nn.Linear(D_MODEL, d); self.pos = nn.Embedding(max_pos, d)
        if rnn_type == "none":
            self.blocks = nn.ModuleList([SmallBlock(d, n_heads or max(1, d // 64), use_mlp) for _ in range(n_layers)])
        else:                                       # classic recurrent reader: vanilla tanh RNN / GRU / LSTM stack
            cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
            self.blocks = nn.ModuleList([cls(d, d, num_layers=n_layers, batch_first=True)])
        self.out_ln = nn.LayerNorm(d); self.out_proj = nn.Linear(d, D_MODEL)
        self.embed_scale = nn.Parameter(torch.tensor(1.0))
    def hidden(self, h42, tok_ids):
        x0 = self.act_proj(h42.float())[:, None, :]
        xt = self.tok_in(self.embed(tok_ids).float()) * self.embed_scale
        x = torch.cat([x0, xt], 1)
        if self.rnn_type == "none":
            x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
            for blk in self.blocks: x = blk(x)
        else:
            with torch.autocast("cuda", enabled=False):
                x, _ = self.blocks[0](x.float())
        return self.out_proj(self.out_ln(x)).to(self.embed.weight.dtype)
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


class FullWidthAttnBlock(nn.Module):
    """Attention-only block living in the 5120-d residual stream with a small inner dim: pre-RMSNorm -> q,k,v (5120->inner)
    -> causal MHA -> o (inner->5120) -> residual. No MLP, no projections: params = 4*5120*inner (+norm)."""
    def __init__(self, d, inner, n_heads=None, mlp_hidden=0, mlp_affine=False):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-6) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
        self.mlp_hidden = mlp_hidden
        if mlp_hidden:                                  # optional small-hidden MLP sublayer: pre-norm -> d->m GELU m->d, zero-init out
            self.norm2 = nn.RMSNorm(d, eps=1e-6) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
            if mlp_affine:                              # AFFINE "MLP": rank-m linear map + bias, no nonlinearity (m = d -> full-rank single Linear)
                self.mlp = nn.Linear(d, d) if mlp_hidden >= d else nn.Sequential(nn.Linear(d, mlp_hidden, bias=False), nn.Linear(mlp_hidden, d))
                nn.init.zeros_((self.mlp if mlp_hidden >= d else self.mlp[1]).weight)
            else:
                self.mlp = nn.Sequential(nn.Linear(d, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d)); nn.init.zeros_(self.mlp[2].weight)
        self.q = nn.Linear(d, inner, bias=False); self.k = nn.Linear(d, inner, bias=False); self.v = nn.Linear(d, inner, bias=False); self.o = nn.Linear(inner, d, bias=False)
        self.h = n_heads or max(1, inner // 64); nn.init.zeros_(self.o.weight)     # zero-init output: block starts as identity
    def forward(self, x):
        B, T, _ = x.shape; h = self.norm(x)
        q, k, v = (f(h).view(B, T, self.h, -1).transpose(1, 2) for f in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
        x = x + self.o(a)
        return x + self.mlp(self.norm2(x)) if self.mlp_hidden else x


class FullWidthAttnReader(nn.Module):
    """[AR vector * s, frozen embeddings] -> N FullWidthAttnBlocks -> frozen final RMSNorm -> frozen lm_head. Trainable: attention only."""
    def __init__(self, embed, norm, head, inner, n_layers, max_pos=64, mlp_hidden=0, mlp_affine=False, n_heads=None):
        super().__init__()
        self.embed, self.norm, self.head = embed, norm, head; self.mlp_hidden = mlp_hidden; self.mlp_affine = mlp_affine; self.n_heads = n_heads
        for mod in (self.embed, self.norm, self.head):
            for p_ in mod.parameters(): p_.requires_grad_(False)
        self.src_layers = []; self.block_type = "fullattn"; self.inner = inner
        self.blocks = nn.ModuleList([FullWidthAttnBlock(D_MODEL, inner, n_heads=n_heads, mlp_hidden=mlp_hidden, mlp_affine=mlp_affine) for _ in range(n_layers)])
        self.pos = nn.Embedding(max_pos, D_MODEL); nn.init.zeros_(self.pos.weight)
        self.act_scale = nn.Parameter(torch.tensor(1.0)); self.embed_scale = nn.Parameter(torch.tensor(1.0))
        self.act_proj = nn.Identity()
    def hidden(self, h42, tok_ids):
        x0 = (h42.float() * self.act_scale)[:, None, :]
        xt = self.embed(tok_ids).float() * self.embed_scale
        x = torch.cat([x0, xt], 1) + self.pos(torch.arange(h42.shape[0] * 0 + xt.shape[1] + 1, device=h42.device))[None]
        for blk in self.blocks: x = blk(x)
        return x.to(self.embed.weight.dtype)
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


class LinearRNNReader(nn.Module):
    """Linear recurrence at full width, no nonlinearity: h_0 = B v, h_t = A h_{t-1} + B e_t; logits_t = frozen head(frozen norm(h_t)).
    The activation's effect on token t is W_U A^t B v in closed form. Trainable: A, B (5120 x 5120 each, identity-init) + 2 scalars."""
    def __init__(self, embed, norm, head, d=D_MODEL):
        super().__init__()
        self.embed, self.norm, self.head = embed, norm, head
        for mod in (self.embed, self.norm, self.head):
            for p_ in mod.parameters(): p_.requires_grad_(False)
        self.src_layers = []; self.block_type = "linrnn"
        self.A = nn.Linear(d, d, bias=False); nn.init.eye_(self.A.weight)
        self.B = nn.Linear(d, d, bias=True); nn.init.eye_(self.B.weight); nn.init.zeros_(self.B.bias)
        self.act_scale = nn.Parameter(torch.tensor(1.0)); self.embed_scale = nn.Parameter(torch.tensor(1.0))
        self.blocks = nn.ModuleList([self.A, self.B]); self.act_proj = nn.Identity()
    def hidden(self, h42, tok_ids):
        with torch.autocast("cuda", enabled=False):
            x = torch.cat([(h42.float() * self.act_scale)[:, None, :], self.embed(tok_ids).float() * self.embed_scale], 1)
            hs, h = [], None
            for t in range(x.shape[1]):
                u = self.B(x[:, t]); h = u if h is None else self.A(h) + u; hs.append(h)
            return torch.stack(hs, 1).to(self.embed.weight.dtype)
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


class OneLayerAV(nn.Module):
    """Frozen embed/norm/head from the 27B + N trainable decoder blocks (default 1) + act_proj."""

    def __init__(self, base, init_layer, n_layers=1, block_type="attn"):
        super().__init__()
        m = base.model
        self.embed = m.embed_tokens; self.norm = m.norm; self.head = base.lm_head; self.rotary = m.rotary_emb
        for mod in (self.embed, self.norm, self.head):
            for p_ in mod.parameters():
                p_.requires_grad_(False)
        lt = list(m.config.layer_types) if getattr(m.config, "layer_types", None) else ["full_attention"] * len(m.layers)
        want = "linear_attention" if block_type == "gdn" else "full_attention"      # gdn = Gated DeltaNet (recurrent) blocks 62,61,60,58,...
        attn_idx = [i for i, t in enumerate(lt) if t == want]
        src = attn_idx[-n_layers:] if (n_layers > 1 or block_type == "gdn") else [init_layer if init_layer >= 0 else 63]
        self.src_layers = [] if block_type == "lstm" else src
        self.block_type = block_type
        self.blocks = nn.ModuleList([LSTMStack(n_layers)] if block_type == "lstm" else [copy.deepcopy(m.layers[i]) for i in src])
        if init_layer < 0 and block_type != "lstm":
            for blk in self.blocks:
                for mod in blk.modules():
                    cls = type(mod).__name__.lower()
                    for name, p_ in mod.named_parameters(recurse=False):
                        if p_.dim() > 1:
                            nn.init.normal_(p_, std=0.02)
                        elif "norm" in cls:
                            nn.init.ones_(p_)
                        else:
                            nn.init.zeros_(p_)
        for blk in self.blocks:
            for p_ in blk.parameters():
                p_.requires_grad_(True)
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


# ---- build: load the 27B once, take the pieces, free the rest ------------------------------------
if args.block_type in ("small", "fullattn", "linrnn"):
    if args.frozen:
        fz = torch.load(args.frozen, map_location="cpu")
        _emb = nn.Embedding.from_pretrained(fz["embed"].to(dev), freeze=True)
        _norm = FrozenRMSNorm(fz["norm"].to(dev), fz["eps"])
        _head = nn.Linear(fz["d"], fz["vocab"], bias=False); _head.weight = nn.Parameter(fz["head"].to(dev), requires_grad=False)
        base = None
    else:
        base = load_base(dev); _emb, _norm, _head = base.model.embed_tokens, base.model.norm, base.lm_head
    model = (LinearRNNReader(_emb, _norm, _head) if args.block_type == "linrnn" else FullWidthAttnReader(_emb, _norm, _head, args.inner, args.n_layers, mlp_hidden=args.mlp_hidden, mlp_affine=args.mlp_affine, n_heads=args.heads or None) if args.block_type == "fullattn" else SmallReader(_emb, _norm, _head, args.d_small, args.n_layers, rnn_type=args.small_rnn, use_mlp=not args.small_no_mlp)).to(dev)
else:
    base = load_base(dev)
    model = OneLayerAV(base, args.init_layer, args.n_layers, block_type=args.block_type).to(dev)   # blocks deep-copied BEFORE any LoRA
AR = ARHOOK = value_head = None
if args.ar:                                        # AR-inverter: frozen AR (LoRA on blocks 0..42 + value head)
    from peft import PeftModel
    from common import Layer42Hook, ar_read
    AR = PeftModel.from_pretrained(base, args.ar, adapter_name="ar").eval()
    for p_ in AR.parameters():
        p_.requires_grad_(False)
    value_head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32)
    value_head.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
    ARHOOK = Layer42Hook(AR)
    if is_main: print(f"[inverter] soft token = AR(span) from {args.ar}; target = span", flush=True)
elif args.ar_affine:                               # AR-inverter with the closed-form AFFINE AR (frozen base, no adapter)
    from common import Layer42Hook, ar_read
    _aff = torch.load(args.ar_affine, map_location=dev)
    AFF_W, AFF_b, AFF_feat = _aff["W"].to(dev).float(), _aff["b"].to(dev).float(), _aff["feat"]
    AR = base.eval(); ARHOOK = Layer42Hook(AR)
    if is_main: print(f"[inverter] soft token = affine AR(span) from {args.ar_affine} (feat={AFF_feat}); target = span", flush=True)
else:
    if base is not None: del base
torch.cuda.empty_cache()
if args.init_ckpt:
    _sd = torch.load(args.init_ckpt, map_location="cpu"); missing = model.load_state_dict({k: v.to(dev) for k, v in _sd["trainable_sd"].items()}, strict=False)
    if is_main: print(f"[init] loaded trainable_sd from {args.init_ckpt} ({len(_sd['trainable_sd'])} tensors)", flush=True)
trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
n_tr = sum(p_.numel() for p_ in trainable)
if is_main:
    print(f"[model] {args.n_layers}-layer AV ({args.block_type}, d={args.d_small if args.block_type == 'small' else D_MODEL}, inner={args.inner}) | src blocks {model.src_layers} | trainable {n_tr/1e6:.1f}M | "
          f"frozen embed+head {sum(p_.numel() for p_ in list(model.embed.parameters())+list(model.head.parameters()))/1e9:.2f}B", flush=True)
for p_ in trainable:
    p_.data = p_.data.float()                       # fp32 master for the trainable block

# ---- data: per-rank file shard, numpy ------------------------------------------------------------
files = sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), []))
files = [f for f in files if not any(x and x in f for x in args.exclude.split(","))]; my_files = files[RANK::WORLD] if is_dist else files
if args.max_files > 0:
    my_files = my_files[: args.max_files]
if args.stream:
    # STREAM: rank 0 keeps its first file as held-out (removed from the stream); row counts from parquet metadata.
    held_file = my_files[0] if is_main else None
    stream_files = my_files[1:] if is_main else list(my_files)
    if args.ar_cache:
        _cf = lambda f: args.ar_cache + "/" + f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
        stream_files = [f for f in stream_files if os.path.exists(_cf(f))]
        assert not is_main or os.path.exists(_cf(held_file)), "held-out file has no AR cache entry"
    counts = [pq.ParquetFile(f).metadata.num_rows for f in stream_files]
    n_local = int(sum(counts))
    if is_main:
        hf_ = (args.ar_cache + "/" + held_file.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")) if args.ar_cache else held_file
        tb = pq.ParquetFile(hf_).read(columns=(["roll_ids", "ar_vec"] if args.ar_cache else (["roll_ids"] if (args.ar or args.ar_affine) else ["h42", "roll_ids"]))); nh = min(tb.num_rows, args.heldout)
        H = tb.column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, D_MODEL).astype(np.float16)[:nh] if args.ar_cache else None if (args.ar or args.ar_affine) else tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, D_MODEL).astype(np.float16)[:nh]
        R = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, 12).astype(np.int32)[:nh]
        held = np.arange(nh)
        print(f"[data] STREAM rank0: {n_local} train rows in {len(stream_files)} files | held {nh} rows from {held_file.split('/')[-1]}", flush=True)
    else:
        H = R = None; held = np.arange(0)
    train_idx = np.arange(n_local)

    def _read(f):
        if args.ar_cache:                                                  # cached AR(span) stands in for h42
            cf = args.ar_cache + "/" + f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
            tb = pq.ParquetFile(cf).read(columns=["roll_ids", "ar_vec"]); n = tb.num_rows
            return (tb.column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float16),
                    tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int32))
        tb = pq.ParquetFile(f).read(columns=(["roll_ids"] if (args.ar or args.ar_affine) else ["h42", "roll_ids"])); n = tb.num_rows
        h = None if (args.ar or args.ar_affine) else tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float16)
        r = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int32)
        return h, r

    def stream_batches(rng):
        """Single pass: files in random order, shuffle buffer of --buffer-files files, leftovers carried forward.
        Files of a buffer are read in parallel threads and the NEXT buffer is prefetched while the current one trains."""
        from concurrent.futures import ThreadPoolExecutor
        order = rng.permutation(len(stream_files)); carry_h, carry_r = [], []
        pool = ThreadPoolExecutor(max_workers=min(16, args.buffer_files))
        chunks = [order[b0:b0 + args.buffer_files] for b0 in range(0, len(order), args.buffer_files)]
        def load(chunk): return list(pool.map(lambda fi: _read(stream_files[fi]), chunk))
        nxt = pool.submit(load, chunks[0]) if chunks else None
        for ci in range(len(chunks)):
            got = nxt.result(); nxt = pool.submit(load, chunks[ci + 1]) if ci + 1 < len(chunks) else None
            hs, rs = list(carry_h), list(carry_r)
            for h, r in got:
                rs.append(r)
                if h is not None: hs.append(h)
            Rb = np.concatenate(rs); Hb = np.concatenate(hs) if hs else None
            p = rng.permutation(len(Rb)); Rb = Rb[p]; Hb = Hb[p] if Hb is not None else None
            nfull = (len(Rb) // args.batch) * args.batch
            for a in range(0, nfull, args.batch):
                yield (None if Hb is None else Hb[a:a + args.batch]), Rb[a:a + args.batch]
            carry_r = [Rb[nfull:]]; carry_h = [] if Hb is None else [Hb[nfull:]]
else:
    H = []; R = []
    for f in my_files:
        tb = pq.ParquetFile(f).read(columns=(["roll_ids"] if (args.ar or args.ar_affine) else ["h42", "roll_ids"])); n = tb.num_rows
        if not (args.ar or args.ar_affine):
            H.append(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float16))
        R.append(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int32))
    R = np.concatenate(R); H = np.concatenate(H) if H else None; n_local = len(R)
    held = np.arange(args.heldout) if is_main else np.arange(0)
    train_idx = np.arange(args.heldout if is_main else 0, n_local)
    if is_main:
        print(f"[data] rank0 local {n_local} (held {len(held)} train {len(train_idx)}) | files/rank {len(my_files)}", flush=True)


def batch(idx):
    if isinstance(idx, tuple):                       # streamed (h_np|None, r_np)
        h_np, r_np = idx
        t = torch.tensor(r_np[:, :args.n_tok].astype(np.int64), device=dev)
    else:
        h_np, r_np = None, None
        t = torch.tensor(R[idx, :args.n_tok].astype(np.int64), device=dev)
    if args.ar:                                     # AR(span): early-exit read at block 42, last token -> value head
        with torch.no_grad():
            hh = ar_read(AR, ARHOOK, t, torch.ones_like(t), value_head)
            h = value_head(hh[:, -1].float())
    elif args.ar_affine:                            # affine AR(span): frozen block-42 state(s) -> W x + b
        with torch.no_grad():
            hh = ar_read(AR, ARHOOK, t, torch.ones_like(t), None).float()
            x = torch.cat([hh[:, -1], hh.mean(1)], -1) if AFF_feat == "last+mean" else hh[:, -1]
            h = x @ AFF_W + AFF_b
    elif h_np is not None:
        h = torch.tensor(h_np.astype(np.float32), device=dev)
    else:
        h = torch.tensor(H[idx].astype(np.float32), device=dev)
    return h, t


def loss_fn(idx):
    h, t = batch(idx)
    logits = model(h, t)                            # [B,T+1,V] (autocast lives inside forward)
    lg = logits[:, :-1].float()                     # predict t[:, 0..T-1] from positions 0..T-1
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1))


optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, s / max(1, args.warmup)) * max(0.05, 1 - s / args.steps))
eff_batch = args.batch * WORLD
wb = None
if is_main and not args.no_wandb and os.environ.get("WANDB_API_KEY"):
    import wandb
    wb = wandb.init(project=args.wandb_project, name=args.wandb_name or f"av1_l{args.init_layer}",
                    config={**vars(args), "eff_batch": eff_batch, "gpus": WORLD, "trainable_M": n_tr / 1e6})


@torch.no_grad()
def evaluate():
    model.eval(); ces = []
    for a in range(0, min(len(held), 1024), args.batch):
        sub = held[a:a + args.batch]
        if len(sub):
            ces.append(loss_fn(sub).item())
    # a few greedy readouts
    h, t = batch(held[:4])
    g = model.generate(h, args.n_tok)
    samples = [(tok.decode(t[i]), tok.decode(g[i])) for i in range(4)]
    model.train()
    return float(np.mean(ces)), samples


if is_main:
    print(f"[train] steps {args.steps} eff_batch {eff_batch} lr {args.lr}", flush=True)
t0 = time.time(); model.train(); rng = np.random.default_rng(args.seed + 1 + RANK)
perm = rng.permutation(train_idx)                       # ONE pass: every row at most once, no re-draws
max_steps = len(perm) // args.batch
if args.stream:
    max_steps = n_local // args.batch - len(stream_files) // args.buffer_files - 1   # minus per-buffer remainder loss
    batches = stream_batches(np.random.default_rng(args.seed + 7 + RANK))
if is_dist:                                             # ALL ranks must run the same number of steps (collectives)
    ms = torch.tensor([max_steps], device=dev); dist.all_reduce(ms, op=dist.ReduceOp.MIN); max_steps = int(ms.item())
    print(f"[data] rank {RANK}: {len(perm)} train rows -> local cap {len(perm)//args.batch}, global cap {max_steps}", flush=True)
if args.steps > max_steps:
    if is_main: print(f"[data] capping steps {args.steps} -> {max_steps} (single pass over {len(perm)} rows/rank)", flush=True)
    args.steps = max_steps
for step in range(args.steps):
    optim.zero_grad(set_to_none=True)
    idx = next(batches) if args.stream else perm[step * args.batch:(step + 1) * args.batch]
    loss = loss_fn(idx); loss.backward()
    if is_dist:
        for p_ in trainable:
            if p_.grad is None:
                p_.grad = torch.zeros_like(p_)
            dist.all_reduce(p_.grad, op=dist.ReduceOp.SUM); p_.grad.div_(WORLD)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); optim.step(); sched.step()
    if is_main and step % 25 == 0:
        print(f"step {step:05d} | ce {loss.item():.4f} | ppl {np.exp(min(loss.item(),20)):.1f} | gn {float(gn):.2f} | {(time.time()-t0)/60:.1f}min", flush=True)
    if is_main:
        log = {"step": step, "ce": loss.item(), "lr": sched.get_last_lr()[0], "grad_norm": float(gn)}
        if step > 0 and step % args.eval_every == 0:
            ev, samples = evaluate(); log["eval_ce"] = ev
            print(f"  [eval {step}] ce {ev:.4f} ppl {np.exp(min(ev,20)):.1f}", flush=True)
            for tr, gen in samples[:2]:
                print(f"    TRUE {tr!r}\n    GEN  {gen!r}", flush=True)
        if wb:
            wb.log(log)
        if step > 0 and step % args.save_every == 0:
            torch.save({"blocks": model.blocks.state_dict(), "act_proj": model.act_proj.state_dict(), "n_layers": args.n_layers, "trainable_sd": {n_: p_.detach().cpu() for n_, p_ in model.named_parameters() if p_.requires_grad}, "d_small": args.d_small, "small_rnn": args.small_rnn, "small_no_mlp": args.small_no_mlp, "inner": args.inner, "mlp_hidden": args.mlp_hidden, "mlp_affine": args.mlp_affine, "heads": args.heads, "ar": args.ar or (args.ar_cache and "/vol_data/ckpt/ar_mse_v3/final"), "ar_affine": args.ar_affine, "ar_cache": args.ar_cache, "block_type": args.block_type,
                        "src_layers": model.src_layers, "embed_scale": model.embed_scale.data, "init_layer": args.init_layer},
                       f"{args.out}/step_{step:06d}.pt")
    if is_dist and step > 0 and step % args.eval_every == 0:
        dist.barrier()
if is_main:
    ev, samples = evaluate()
    torch.save({"blocks": model.blocks.state_dict(), "act_proj": model.act_proj.state_dict(), "n_layers": args.n_layers, "trainable_sd": {n_: p_.detach().cpu() for n_, p_ in model.named_parameters() if p_.requires_grad}, "d_small": args.d_small, "small_rnn": args.small_rnn, "small_no_mlp": args.small_no_mlp, "inner": args.inner, "mlp_hidden": args.mlp_hidden, "mlp_affine": args.mlp_affine, "heads": args.heads, "ar": args.ar or (args.ar_cache and "/vol_data/ckpt/ar_mse_v3/final"), "ar_affine": args.ar_affine, "ar_cache": args.ar_cache, "block_type": args.block_type,
                "src_layers": model.src_layers, "embed_scale": model.embed_scale.data, "init_layer": args.init_layer}, f"{args.out}/final.pt")
    json.dump({"args": vars(args), "final_eval_ce": ev, "trainable_M": n_tr / 1e6,
               "samples": samples, "elapsed_min": (time.time()-t0)/60}, open(f"{args.out}/meta.json", "w"), indent=1)
    print(f"AV1_DONE final_ce {ev:.4f}", flush=True)
if is_dist:
    dist.barrier(); dist.destroy_process_group()
