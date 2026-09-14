"""Affine AR scaling law: held-out FVE / cosine of the closed-form ridge map (frozen block-42 span states -> preceding
activation) vs number of training rows, per feature set. Reads results from data/affine_scaling.json (collected from
/vol/ckpt/ar_affine_scaling/*/results.json) and writes affine_scaling.png/pdf."""
import json, os, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pts = json.load(open(f"{root}/data/affine_scaling.json"))
fig, ax = plt.subplots(figsize=(8, 5))
for feat, mk in (("last", "^:"), ("dot", "v-."), ("last+mean", "o-"), ("dot+mean", "D--"), ("all12", "s--")):
    ser = sorted([p for p in pts if p["feat"] == feat], key=lambda p: p["n_rows"])
    if ser: ax.plot([p["n_rows"] / 1e6 for p in ser], [p["heldout_fve"] for p in ser], mk, label={"last": "last span token", "dot": "state at an appended '.' (summary token)", "last+mean": "last ⊕ mean state (10k-d)", "dot+mean": "'.' ⊕ mean state (10k-d)", "all12": "all 12 token states (61k-d)"}[feat])
ax.axhline(0.2089, color="k", ls=":", lw=1); ax.text(0.27, 0.212, "LoRA AR (ar_mse_v3), same test rows: 0.21", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("training rows for the ridge fit (M, log scale)"); ax.set_ylabel("held-out FVE of the preceding activation")
ax.set_title("Affine AR scaling law: gains per doubling of data shrink to ~0.002 by 4M rows; reading at an appended \".\" barely beats the last token\n(ridge on frozen block-42 span states → preceding layer-42 activation; all far below the LoRA AR)", fontsize=9.5)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
for ext in ("png", "pdf"): fig.savefig(f"{root}/affine_scaling.{ext}", dpi=150, bbox_inches="tight")
print("saved affine_scaling.png/pdf", len(pts), "points")
