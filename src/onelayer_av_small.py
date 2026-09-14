import torch, torch.nn as nn, torch.nn.functional as F
from common import D_MODEL


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
