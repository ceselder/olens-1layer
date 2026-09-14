"""Pareto front of the AR-inverter readers: held-out CE (AR(span) -> span, own held slice) vs trainable params, one curve
per training-data size. Pulls runs from wandb (both entities), writes data/pareto.json and pareto_params_data.png/pdf."""
import json, os, re
import wandb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
api = wandb.Api(timeout=60); pts = []
for ent in ("celestedeschamphelaere-personal", "octahedral-systems"):
    for r in api.runs(f"{ent}/olens-1layer", order="-created_at"):
        n = r.name
        if not (("arinv" in n and ("cache" in n or n.startswith("small_") or n.startswith("fw_") or n.startswith("linrnn") or (n.startswith("bm_") and "lr3e-4" in n) or n == "av4_5m_arinv_1pass_b"))): continue
        if (n.startswith("small_") or n.startswith("fw_")) and not n.endswith("_v2"): continue
        if n.startswith("bm_") and r.state != "finished": pass
        hist = r.history(keys=["step", "eval_ce"], pandas=False, samples=5000)
        ev = [(int(h["step"]), float(h["eval_ce"])) for h in hist if h.get("eval_ce") is not None]
        if not ev: continue
        m = re.search(r"_(\d+m)_", n); data = m.group(1) if m else ("5m" if "5m" in n else "30m")
        if n.startswith("bm_"): arch_prefix = "batch16k "
        else: arch_prefix = ""
        params = r.config.get("trainable_M"); arch = (("small %s d%s L%s" % (r.config.get("small_rnn") if r.config.get("small_rnn", "none") != "none" else ("attn-only" if r.config.get("small_no_mlp") else "transformer"), r.config.get("d_small"), r.config.get("n_layers")))) if n.startswith("small_") else f"{r.config.get('block_type','attn')} {r.config.get('n_layers')}L (27B-width)"
        if n.startswith("fw_"): arch = f"full-width attn L{r.config.get('n_layers')} inner{r.config.get('inner')}" + (f" {r.config.get('heads')}h" if r.config.get("heads") else "") + ((f" +affine{r.config.get('mlp_hidden')}" if r.config.get("mlp_affine") else f" +MLP{r.config.get('mlp_hidden')}") if r.config.get("mlp_hidden") else "")
        if n.startswith("linrnn"): arch = "full-width LINEAR RNN"
        arch = arch_prefix + arch
        if n.startswith("rand"): arch = "random-init " + arch
        if n.endswith("b256"): arch += ", batch 256"
        pts.append({"run": n, "state": r.state, "arch": arch, "params_M": params, "data": data, "min_eval_ce": min(v for _, v in ev), "last_step": ev[-1][0], "n_eval": len(ev)})
json.dump(pts, open(f"{root}/data/pareto.json", "w"), indent=1)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1]})
colors = {"2m": "tab:blue", "8m": "tab:orange", "30m": "tab:green", "80m": "tab:red", "84m": "tab:purple", "5m": "tab:gray"}
def kind(p):
    a = p["arch"]
    if a.startswith("full-width LINEAR"): return "linrnn"
    if a.startswith("full-width"): return "fw-mlp" if "+MLP" in a else "fw-aff" if "+affine" in a else "fw"
    return "rnn" if a.startswith("small rnn ") else "gru" if a.startswith("small gru ") else "lstm" if a.startswith("small lstm ") else "attn-only" if a.startswith("small attn-only ") else "tr"
for data in sorted({p["data"] for p in pts}, key=lambda s: int(s[:-1])):
    sel = sorted([p for p in pts if p["data"] == data and p["params_M"]], key=lambda p: p["params_M"])
    tr = [p for p in sel if kind(p) == "tr"]
    if tr: ax.plot([p["params_M"] for p in tr], [p["min_eval_ce"] for p in tr], "o-", color=colors.get(data), label=f"{data.replace('m','M')} rows, transformer readers")
    for kd, mk in (("rnn", "s"), ("gru", "^"), ("lstm", "D"), ("attn-only", "x"), ("fw", "P"), ("fw-mlp", "*"), ("fw-aff", "v"), ("linrnn", "h")):
        rr = [p for p in sel if kind(p) == kd]
        if rr: ax.plot([p["params_M"] for p in rr], [p["min_eval_ce"] for p in rr], mk + "--", color=colors.get(data), alpha=.8, label=f"{data.replace('m','M')} rows, " + {"attn-only": "narrow attention-only (no MLP)", "fw": "full-width attention-only (no projections)", "fw-mlp": "full-width attention + small MLP", "fw-aff": "full-width attention + affine map (no nonlinearity)", "linrnn": "full-width linear RNN (h_t = A h + B x)"}.get(kd, f"classic {kd.upper()}"))
    for k, p in enumerate(sel): ax.annotate(p["arch"].replace("small ", "") + ("" if p["state"] == "finished" else " (running)"), (p["params_M"], p["min_eval_ce"]), fontsize=6, xytext=(4, -9 * (k % 3) + 4), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("trainable parameters (M, log scale)"); ax.set_ylabel("held-out CE, span given AR(span) (nats/token)")
ax.set_title("Eval quality vs parameter count: the lowest curve is the current front;\nevery size is still improving with data, so the front keeps moving down", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
# panel 2: CE vs data per architecture (are the small readers data-limited?)
archs = sorted({p["arch"] for p in pts if kind(p) == "tr" and p["arch"].startswith("small")}, key=lambda a: [q["params_M"] for q in pts if q["arch"] == a][0])
for a in archs:
    ser = sorted([p for p in pts if p["arch"] == a], key=lambda p: int(p["data"][:-1]))
    ax2.plot([int(p["data"][:-1]) for p in ser], [p["min_eval_ce"] for p in ser], "o-", label=f"{a.replace('small transformer ', '')} ({ser[0]['params_M']:.0f}M)")
ax2.set_xscale("log"); ax2.set_xlabel("training rows seen once (M, log scale)"); ax2.set_ylabel("held-out CE (nats/token)")
ax2.set_title("Same readers vs data: no size has flattened yet\n(data is unlimited; each doubling of data still buys 0.3-0.5 nats)", fontsize=10)
ax2.grid(alpha=.3, which="both"); ax2.legend(fontsize=7)
for ext in ("png", "pdf"): fig.savefig(f"{root}/pareto_params_data.{ext}", dpi=150, bbox_inches="tight")
print(f"{len(pts)} points -> pareto_params_data.png/pdf");
for p in sorted(pts, key=lambda p: (p["data"], p["params_M"] or 0)): print(f"  {p['data']:4s} {p['arch']:28s} {p['params_M'] and round(p['params_M'],1):>8} M  min CE {p['min_eval_ce']:.3f}  ({p['state']}, step {p['last_step']})")
