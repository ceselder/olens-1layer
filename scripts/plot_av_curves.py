"""Held-out CE vs step for every shallow-verbalizer run (data from data/eval_curves.json, pulled from wandb)."""
import json, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(f"{root}/data/eval_curves.json"))
LABEL = {"gdn2_30m_1pass_stream": "2 GDN layers, ~30M rows incl. fresh text, single pass", "av2_30m_1pass_stream": "2 layers, ~30M rows incl. fresh text, single pass", "av2_12m_1pass_stream": "2 layers, 12M rows, single pass (streaming)", "gdn4_8m_1pass": "4 GDN layers, 8M rows, single pass", "gdn2_8m_1pass": "2 GDN layers, 8M rows, single pass", "gdn1_8m_1pass": "1 GDN layer, 8M rows, single pass", "av1_8m_1pass": "1 layer, 8M rows, single pass (real h42 in)", "av2_8m_1pass": "2 layers, 8M rows, single pass (real h42 in)", "av4_12m_1pass": "4 layers, 12M rows, single pass (real h42 in)", "av4_8m_1pass": "4 layers, 8M rows, single pass (real h42 in)", "av1_l63": "1 layer, 1.9M rows, ~2 epochs", "av1_rand": "1 layer (random init), 1.9M rows, ~2 epochs",
         "av2_l5963": "2 layers, 1.9M rows, ~2 epochs", "av4_l51-63": "4 layers, 1.9M rows, ~2 epochs",
         "av1_l63_all": "1 layer, 3M rows, 1 epoch", "av4_all": "4 layers, 3M rows, 1 epoch",
         "av4_5m_1pass": "4 layers, 5M rows, single pass (real h42 in)", "av4_5m_1pass_b": "4 layers, 5M rows, single pass, rerun (real h42 in)",
         "av4_5m_arinv_1pass": "4 layers, 5M rows, single pass, AR(span) in [different task]", "av4_5m_arinv_1pass_b": "4 layers, 5M, single pass, AR(span) in, rerun [different task]"}
fig, ax = plt.subplots(figsize=(10, 6))
for name, r in d.items():
    pts = r["eval_ce"]
    if len(pts) < 2 or (r["state"] == "crashed" and name + "_b" in d): continue   # crashed runs superseded by their reruns
    xs = [p[0] * 256 / 1e6 for p in pts]; ys = [p[1] for p in pts]
    ls = "--" if r["config"].get("ar") else (":" if r["config"].get("block_type") == "gdn" else "-")
    ax.plot(xs, ys, ls, marker=".", ms=3, label=LABEL.get(name, name))
ax.plot([], [], "k:", label="reference: 27B + LoRA verbalizer reaches 1.94 (off-scale)")
ax.set_xlabel("training examples seen (millions; effective batch 256)"); ax.set_ylabel("held-out CE on 12-token on-policy rollout (nats)")
ax.set_title("Shallow verbalizer of Qwen3.6-27B layer 42: held-out CE keeps falling with fresh data and depth;\nre-epoched runs bottom out early (dashed = AR-inverter task; dotted = GDN recurrent blocks; held-out slices differ by run family)", fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=7.5); ax.set_ylim(3.4, 6.0)
for ext in ("png", "pdf"): fig.savefig(f"{root}/av_eval_curves.{ext}", dpi=150, bbox_inches="tight")
print("saved av_eval_curves.png/pdf")
