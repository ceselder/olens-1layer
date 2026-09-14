# olens-1layer — shallow verbalizers / oracle-lens readers for Qwen3.6-27B layer 42

Research code (MATS, Sep 2026). Everything runs on Modal (`modal_app.py`); results, checkpoints and the write-up live elsewhere:

- Report: internal Claude Reports site (ask Celeste); every number in it is in `data/*.json` here
- Checkpoints (HF): `ceselder/olens-ar-inverter-qwen3.6-27b` (AR-inverter readers, RL'd readers, adapters, affine ARs, the 27B Karvonen reader),
  `ceselder/olens-1layer-shallow-verbalizer` and `ceselder/olens-2layer-verbalizer-qwen3.6-27b` (real-h42 readers).
- Running log of every decision and number: `notes/DESIGN.md`; Modal app ids: `notes/*_apps.txt`; report data: `data/*.json`.

## Layout
- `modal_app.py` — Modal app (volumes: `olens-1layer` rw; `olens-new-arch`, `maemm-data`, `celeste-modlens-vol` read-only). Tasks: `train` (torchrun a src script), `train-many` (several spawns in one app), `pyrun`.
- `src/onelayer_av.py` — the reader trainer: block types `attn` / `gdn` / `lstm` (copied 27B blocks), `small` (narrow transformer / RNN / GRU / LSTM, `--small-no-mlp`), `fullattn` (full-width attention ± MLP / affine map, `--heads`), `linrnn`; inputs real h42, `--ar` (LoRA AR on the fly), `--ar-affine`, `--ar-cache` (precomputed AR vectors); streaming single-pass loader with prefetch; `--frozen` to skip loading the 27B.
- `src/av1_model.py`, `src/onelayer_av_small.py` — model classes + loaders (`load_av1`, `load_small`).
- `src/harvest.py`, `src/harvest_fresh.py` — (ctx, h42, on-policy 12-token rollout) harvesters (token corpus / fresh FineFineWeb text).
- `src/precompute_ar.py` — cache AR(span) vectors (`lora` / `affine` / `frozen` modes). `src/extract_frozen.py` — frozen embed/norm/head file.
- `src/ar_affine.py` — closed-form ridge ARs on frozen block-42 span states (`last`, `last+mean`, `firstK`, `allK`, `dot`, `dot+mean`, `embed`).
- `src/fit_h42_to_ar.py` — adapter P: real h42 → AR vector (lets AR-only readers read real activations).
- `src/seed_teacher.py` — K=1 nearest-dictionary teacher (paper stage 3) → SFT seed rows.
- `src/rl_inverter.py` — RL of a reader on real activations: MAEMM ScaleRL bundle (CISPO, batch-level advantages, prompt aggregation, fp32 head), rewards `lora` / `affine` (whitened FVE) or `modlens` (the modulation-lens AR objective), policies `small` or `lora27b` (27B + Karvonen injection). Data-parallel.
- `src/eval_olens_rl.py`, `src/eval_olens_27b.py` — paper-style eval (sample K spans → AR → NNLS in whitened space) + 27B-NLL naturalness. `src/eval_modlens_metric.py` — the modulation oracle's own delta metric for all readers. `src/vibes_compare.py` — WorkspaceBench side-by-side readouts.
- `src/train_av27b.py` — the 27B + rsLoRA Karvonen-injection reader on the same cached rows (full-LM baseline).
- `src/conditioning_test_1layer.py`, `src/ar_vs_h42.py`, `src/readout_examples.py` — diagnostics. `src/upload_hf.py` — HF uploads (token from the Modal secret).
- `scripts/` — plots (Pareto front, eval curves, affine scaling). `data/` — every number in the report as JSON.

## Recommendations if you build on this
This was deliberately an experiment in extremely low parameter counts (readers of 5-70M trainable parameters), so several
choices were made to fit the budget rather than to be good defaults:
- Use an MLP width of 4x d_model. The full-width readers here used MLP hidden sizes of 1024-2048 on a 5120-d stream (0.2-0.4x) to stay under 70M; the narrow readers already use 4x.
- Give attention the full inner dimension (inner = d_model) and 16+ heads; the ablation showed a single head costs ~0.8 nats and every increase in heads helped.
- Keep the effective batch around 2048 for small readers: 16k with a single pass hurt badly (too few steps); lr 5e-4 worked for readers, 1e-5 for RL.
- More data always helped (single pass, never repeat rows); nothing here had flattened by 84M rows.
- Whatever vector the reader inverts matters more than its size: affine-AR vectors were 3x easier to invert than LoRA-AR vectors, and real activations need either an adapter or RL.

API keys are read from environment / Modal secrets; none are in this repo.
