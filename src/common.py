"""Shared pieces for olens-new-arch: model loading, layer-42 hook (capture / patch), tail-coarsened
top-k KL, whitener, LoRA config.

Conventions (must match everywhere: harvest, AR training, olens RL, WorkspaceBench readouts):
  * subject = Qwen/Qwen3.6-27B, bf16; READ_LAYER = 42 means the OUTPUT of decoder block 42
    (= HF hidden_states[43]), RAW residual stream, no norm.
  * contexts are plain token ids, NO extra BOS/sink prepended (workspace-bench captures plainly).
  * "patch" = replace the block-42 output at ONE position (the last context token) with a vector
    norm-matched to the gold activation's norm; everything else untouched.
"""
from __future__ import annotations

import math
import os
import re

import torch
import torch.nn.functional as F

MODEL = os.environ.get("OLENS_MODEL", "Qwen/Qwen3.6-27B")
READ_LAYER = int(os.environ.get("OLENS_READ_LAYER", "42"))
D_MODEL = 5120

# LoRA targets: attention + gated-DeltaNet projections + MLP, all decoder blocks (same regex family
# as the olens-surprisal ARs; excludes mtp/visual towers).
LORA_TARGET_RE = (r"(?!.*(?:^|\.)(?:mtp|visual)\.).*layers\.\d+\."
                  r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)"
                  r"|mlp\.(?:gate_proj|up_proj|down_proj))")


def lora_target_re(max_layer: int | None = None) -> str:
    """Regex for LoRA target modules; max_layer restricts to blocks 0..max_layer (AR read path)."""
    if max_layer is None:
        return LORA_TARGET_RE
    layer_alt = "|".join(str(i) for i in range(max_layer + 1))
    return LORA_TARGET_RE.replace(r"layers\.\d+\.", r"layers\.(?:" + layer_alt + r")\.")


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def load_base(device="cuda", dtype=torch.bfloat16):
    """Full causal LM, weights verified to load (the modlens lesson: a wrong skeleton silently
    trains on random init)."""
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, device_map={"": device}, output_loading_info=True,
        attn_implementation="sdpa")
    miss = [k for k in info.get("missing_keys", []) if "lora" not in k]
    if miss:
        raise SystemExit(f"REFUSING: {len(miss)} weights did not load, e.g. {miss[:4]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def backbone(model):
    """The decoder stack (has .layers, .norm, .embed_tokens), unwrapping PEFT/DDP."""
    m = model.module if hasattr(model, "module") else model
    m = m.get_base_model() if hasattr(m, "get_base_model") else m
    return m.model


def lm_head_weight(model):
    m = model.module if hasattr(model, "module") else model
    m = m.get_base_model() if hasattr(m, "get_base_model") else m
    return m.get_output_embeddings().weight


class StopForward(Exception):
    """Raised by the hook in capture_grad mode to early-exit the forward after block READ_LAYER,
    so the AR read runs only layers 0..READ_LAYER instead of all 64 (grad graph stops there too)."""


class Layer42Hook:
    """One forward hook on decoder block READ_LAYER with several modes.

      mode=None          : pass-through
      mode="capture"     : store the block output (RAW residual, DETACHED) in .captured  [B, T, d]
      mode="capture_grad": store the block output WITH its autograd graph, then raise StopForward
                           (AR read path: only layers 0..READ_LAYER run)
      mode="patch"       : replace output[b, pos[b]] with vec[b] (vec keeps its autograd graph)
    The hook must stay registered for the life of the model (register once)."""

    def __init__(self, model, layer: int = READ_LAYER):
        self.mode = None
        self.captured = None
        self.pos = None
        self.vec = None
        self.calls = 0
        self.layer = layer
        self._handle = backbone(model).layers[layer].register_forward_hook(self)

    def __call__(self, _module, _inputs, output):
        self.calls += 1
        h = output[0] if isinstance(output, tuple) else output
        if self.mode == "capture":
            if self.captured is None:          # prefill only (generate: later calls are 1-token)
                self.captured = h.detach()
            return output
        if self.mode == "capture_grad":
            self.captured = h                  # keep the graph
            raise StopForward
        if self.mode == "patch":
            if h.shape[1] <= 1:                  # decode step under KV cache: already patched
                return output
            h = h.clone()
            b = torch.arange(h.shape[0], device=h.device)
            h[b, self.pos.to(h.device)] = self.vec.to(device=h.device, dtype=h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return output

    def capture(self):
        self.mode, self.captured = "capture", None
        return self

    def capture_grad(self):
        self.mode, self.captured = "capture_grad", None
        return self

    def patch(self, pos: torch.Tensor, vec: torch.Tensor):
        self.mode, self.pos, self.vec = "patch", pos, vec
        return self

    def off(self):
        self.mode, self.pos, self.vec = None, None, None
        return self

    def remove(self):
        self._handle.remove()


def ar_read(model, hook, span_ids, attn, value_head):
    """AR forward: read block-42 hidden states over the bare span (early-exit at layer 42), map
    every position through value_head. Returns hidden [B,T,d] (block-42 output, grad) so callers can
    take the last real token (KL) or all positions (MSE all-pos). value_head is applied by the caller.
    Adapter must be ON (the trainable AR LoRA)."""
    hook.capture_grad()
    try:
        model(input_ids=span_ids, attention_mask=attn, use_cache=False)
    except StopForward:
        pass
    finally:
        h = hook.captured
        hook.off()
    return h  # [B, T, d], requires_grad


def norm_match(vec: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scale each row of vec to the norm of the matching row of ref (grad flows through vec)."""
    return vec * (ref.norm(dim=-1, keepdim=True).clamp_min(eps) / vec.norm(dim=-1, keepdim=True).clamp_min(eps))


def topk_tail_kl(clean_tlp: torch.Tensor, clean_tki: torch.Tensor, patched_logits: torch.Tensor) -> torch.Tensor:
    """Coarsened KL(clean || patched) per row over {top-k ids} ∪ {tail bucket}.
    clean_tlp [N,k] full-vocab-normalized log-probs of the clean top-k ids; clean_tki [N,k] ids;
    patched_logits [N,V] (grad ok). Returns [N]. (Ported from EasyNLA nla/utils/kl.py.)"""
    lse = torch.logsumexp(patched_logits, dim=-1, keepdim=True)
    q_lp = patched_logits.gather(-1, clean_tki) - lse
    p = clean_tlp.exp()
    p_tail = (1.0 - p.sum(-1)).clamp_min(1e-9)
    q_tail = (1.0 - q_lp.exp().sum(-1)).clamp_min(1e-9)
    kl_top = (p * (clean_tlp - q_lp)).sum(-1)
    return kl_top + p_tail * (p_tail.log() - q_tail.log())


def discount_weights(n: int, gamma: float, device) -> torch.Tensor:
    w = gamma ** torch.arange(n, device=device, dtype=torch.float32)
    return w / w.sum()


class Whitener:
    """Full-covariance whitening W = V diag(1/sqrt(max(λ, floor))) Vᵀ with floor = frac·λ_max.
    whiten(x) = W (x - mu). Fitted on RAW activations (fit_whitener.py); stored as fp32."""

    def __init__(self, mu: torch.Tensor, W: torch.Tensor, meta: dict | None = None):
        self.mu, self.W, self.meta = mu, W, meta or {}

    @classmethod
    def load(cls, path: str, device="cuda"):
        d = torch.load(path, map_location=device)
        return cls(d["mu"].to(device, torch.float32), d["W"].to(device, torch.float32), d.get("meta", {}))

    def save(self, path: str):
        torch.save({"mu": self.mu.cpu(), "W": self.W.cpu(), "meta": self.meta}, path)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mu) @ self.W.t()

    def fve(self, pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        """Per-row whitened FVE = 1 - ||W(pred-gold)||² / ||W(gold-mu)||²."""
        pw, gw = self(pred), self(gold)
        return 1.0 - ((pw - gw) ** 2).sum(-1) / (gw ** 2).sum(-1).clamp_min(1e-9)


def pairwise_selfcos(x: torch.Tensor) -> float:
    """Mean off-diagonal cosine across rows (→1 flags collapse to a constant vector)."""
    if x.shape[0] < 2:
        return float("nan")
    xn = F.normalize(x.float(), dim=-1)
    m = xn @ xn.t()
    off = m[~torch.eye(x.shape[0], dtype=torch.bool, device=m.device)]
    return float(off.mean())


# ---------------------------------------------------------------------------------------------
# Layer-1 activation injection (AV / olens): norm-matched ADD at a marker token, prefill only.
#   h'_pos = h_pos + ||h_pos|| * coeff * unit(v)     (matches the activation-oracle convention)
# Marker ㈜ (id 158983) reuses the olens-surprisal convention so the base tokenizer already has it.
# ---------------------------------------------------------------------------------------------
MARKER_ID = 158983
INJECT_LAYER = 1


class InjectL1:
    """Persistent hook on decoder block INJECT_LAYER's output. Call set(vec, input_ids) before each
    forward; it norm-match-adds vec at every MARKER_ID position. Skips decode steps (KV cache:
    the marker was injected at prefill). vec keeps its grad graph if the AV/olens is trained by a
    loss through the injection (it is not here — vec is detached activation), but grad-safe either way."""

    def __init__(self, model, layer: int = INJECT_LAYER, marker_id: int = MARKER_ID, coeff: float = 1.0):
        self.vec = None
        self.ids = None
        self.marker = marker_id
        self.coeff = coeff
        self._handle = backbone(model).layers[layer].register_forward_hook(self)

    def set(self, vec, input_ids):
        self.vec, self.ids = vec, input_ids
        return self

    def off(self):
        self.vec = None
        return self

    def __call__(self, _m, _i, out):
        if self.vec is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:                                  # decode step: marker already injected
            return out
        mask = (self.ids == self.marker)
        if not bool(mask.any()):
            return out
        b_idx, t_idx = mask.nonzero(as_tuple=True)
        base = h[b_idx, t_idx]
        v = F.normalize(self.vec[b_idx].to(device=h.device, dtype=h.dtype), dim=-1)
        h = h.clone()
        h[b_idx, t_idx] = base + base.norm(dim=-1, keepdim=True) * self.coeff * v
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self._handle.remove()


def build_av_prompt(tok, marker_char="㈜"):
    """Chat-templated AV/olens prompt ending in the injected marker; returns the token id list.
    The activation is injected at the marker; the model then emits text."""
    msg = ("You are shown an internal activation vector from a language model, enclosed in <concept> "
           "tags. It encodes what the model is about to generate next.\n\n<concept>" + marker_char +
           "</concept>\n\nWrite the text that most likely follows.")
    s = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids
