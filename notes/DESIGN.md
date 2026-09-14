# olens-1layer — oracle lens with a ONE-LAYER-transformer verbalizer

Forked from the olens-new-arch session 2026-09-06 (Celeste: "train olens but the AV is a one layer
transformer"). Separate workspace: this dir, Modal app + volume `olens-1layer`. Reads the shared
harvest (~3M rows of (ctx, h42, on-policy 12-tok rollout, teacher top-128)) and the 27B weights
READ-ONLY from the `olens-new-arch` volume at /vol_data. Never writes there.

## Question
Can a SINGLE transformer block read the layer-42 activation of Qwen3.6-27B and verbalize it?
If yes: the activation is highly legible, and the lens is ~60x cheaper than the 27B+LoRA verbalizer.

## Architecture (src/onelayer_av.py)
- Frozen from the 27B: `embed_tokens`, final `norm`, `lm_head`, `rotary_emb` (Qwen token space).
- Trainable: ONE decoder block (default: deepcopy of the 27B's block 63, the block `lm_head` already
  reads = a trainable skip-lens; `--init-layer -1` = random re-init of the same architecture),
  `act_proj` Linear(d,d) identity-init on h42, learnable `embed_scale` (embeddings ~1 vs h42 ~89).
- Sequence: `[act_proj(h42), embed(tok_1..T)]`, causal (sdpa is_causal), position i predicts tok_{i+1}.
- Generation: autoregressive through the one block (12 tokens, no KV cache needed).
- Block 63 is a full_attention layer (layer_types), so it runs standalone (no GDN recurrent state).
- Trainable params ≈ one 27B block (~0.4B) + 26M; frozen embed+head ≈ 2.5B. Tiny to train.

## Plan
1. AV warm-start: h42 -> on-policy rollout (12 tok), CE. Compare eval CE to the 27B-LoRA AV
   (inv_mse eval CE ≈ 1.94 at 1 epoch; av_gen smoke reached ~3.0 fast).  <- running
2. olens: h42 -> 4 bullets, on seed data selected by the trained MSE AR (ar_mse_v3, raw NNLS) with
   this one-layer AV as the candidate generator (reuse olens_seed logic).
3. Ablations: init block 63 vs random; 1 vs 2 layers; frozen vs trained head.
4. Playground page for the one-layer AV (compare to the 27B-LoRA inverters).

## Compute / infra notes (inherited lessons)
- Modal spawn+--detach; the 15GB box OOM-kills background waiters — poll briefly, never stream.
- Loader = numpy (never to_pylist the wide top-k cols). Per-rank file shards for DP.
- Anthropic keys 401 (blocks any WorkspaceBench judging).

## STATE 2026-09-06 — smoke passed, full runs launched
- Smoke (1 GPU, 4 files/16k rows, 30 steps): loads 27B from the read-only mount, one block extracted
  (trainable 398.5M, frozen embed+head 2.54B), CE 14.4 -> 9.8 in 25 steps, eval/generate/save OK.
  Fixes found by the smoke: (1) a single rank loading all 3M rows OOMs RAM -> --max-files per rank;
  (2) fp32 master block x bf16 frozen parts -> autocast INSIDE forward (generate/eval were outside it).
- User clarified: NO prompt/marker — literally [activation soft token] then futurelens continuation.
  That is exactly the implemented sequence.
- Full runs (4xB200 DP each, eff batch 256, --max-files 120/rank ≈1.9M rows, 15000 steps ≈2 epochs,
  lr 1e-4, wandb project olens-1layer): av1_l63 (block-63 init) and av1_rand (random init ablation).
  App ids in notes/av1_apps.txt.
- Reference to beat/compare: 27B+LoRA inverter eval CE ≈ 1.94 (inv_mse, 1 epoch). The one-layer AV's
  attainable CE is the legibility question.
- NEXT: olens 4-bullet stage on seed data (MSE AR selection, this AV as generator); a playground tab for
  the one-layer AV vs the 27B inverters; 2-layer / trained-head ablations if 1 layer plateaus high.

## RESULTS 2026-09-06 — one-layer AV (block-63 init vs random), 4xB200, eff-batch 256, ~1.9M rows
- eval CE (held-out): av1_l63 min 4.661 @7500 (final 4.734, overfit tail); av1_rand min 4.881 @6500.
  Reference: 27B+LoRA inverter 1.94; unconditioned (zero vector) ≈ 8.7 (this model) / 5.8-8.1 (27B inv).
  => ONE layer reads h42 but weakly: closes ~1/2 of the gap from unconditioned to the 27B verbalizer in
  nats, gets the TOPIC/gist, not specifics (ppl ~106 vs ~7). Random init ≈ block-63 init (−0.2 nats):
  the reading is learned, the prior barely matters. Overfits after ~1 epoch (train CE ~3.5 vs eval 4.7).
- Conditioning (l63@7500, but on a file INSIDE the training shards — inflated): correct 3.70 vs zero 8.68 vs
  shuffled 6.92 (gap 4.98, specificity 3.22). Rerunning on a truly unseen file (offset 520).
- Levers launched: av2_l5963 (2 stacked full-attn blocks 59+63, same data) = DEPTH; av1_l63_all (~3M
  rows, 1 epoch) = DATA. Training speed ~0.03 s/step (15k steps ≈ 8 min on 4 GPU).
- HONEST conditioning (truly unseen file, offset 520): correct 4.56 | zero 8.87 | shuffled 7.13 ->
  gap 4.31 nats, specificity 2.58 nats. CE(correct)≈eval 4.66: consistent. The one layer reads the
  SPECIFIC activation (wrong h42 costs 2.6 nats). In nats it closes ~61% of the unconditioned(8.87)->
  27B-verbalizer(1.94) gap — gets topic/gist, not specifics.
- DEPTH: av2_l5963 (2 blocks) reaches 4.685 @2500 = the 1-layer MIN (4.661@7500) at 1/3 the steps,
  still falling (5.20,4.95,4.82,4.74,4.69). Depth clearly helps. DATA run (av1_l63_all) still loading.
- ARCH FACTS (Qwen3.6-27B full-attn block): 24 heads, GQA 6:1 (4 kv), head_dim 256 (inner 6144),
  SwiGLU MLP 3.4x (17408). attn 73M + mlp 267M = 341M/block; act_proj 26M. 1-layer ≈0.37B, 2-layer ≈0.71B.
- SCALING: 1L 4.661 -> 2L 4.588 (plateau) = only −0.07 nats per doubling of depth. 4L (51/55/59/63)
  launched (av4). all-data 1L 4.729@3000 ≈ same trajectory. NOT param-limited (1L overfits: train 3.5 vs
  eval 4.7). DIAGNOSIS: zero-vector CE 8.87 => the shallow AV is a terrible LM on raw embeddings; the
  activation-reading gain (4.3 nats) is fine. Gap to 1.94 is text-modeling capacity, not reading.
  PROPOSED lever: frozen 27B early trunk (blocks 0-7) on the text tokens -> trainable late readout block(s),
  so text enters in-distribution. Cheap (frozen fwd). Awaiting user go-ahead.
- DATA WINS: 1L on ~3M (av1_l63_all) min 4.449 @11500, monotone to the end (no overfit) vs 2L on 1.9M
  4.588. Data (−0.21) >> depth (−0.07). 4L on 1.9M at 4.558@2000 still falling. => scales with data,
  modestly with depth, NOT param-limited (the 1L plateau was a data limit).
- Launched: harvest_v2 (+2M rows, 16 shards, seed 1, MY volume /vol/data/harvest_v2 via maemm-data ro mount)
  and av4_all (4 layers x ~3M). Trainer now takes comma-separated data globs (v1 + v2).
- QUALITATIVE (best 1L, 3M, per-token probes + unseen rows): the shallow AV surfaces GENRE/REGISTER
  (recipe: "add the sugar in a large bowl"; crime fiction: "was the killer"; medical: "observed in dogs.
  The most common signs"; arithmetic: "equals 20") and the local SCHEMA shape, plus coarse topic — but NOT
  specifics (sugar/egg for butter, 20 for 21), latent commitments (no "wife"), reasoning or intent. A
  "genre + schema" reader vs the 27B inverter's specifics + latent answers. First-token activations
  (little context) collapse to a fixed default readout ("era, a.\nThe Japanese government"). Occasional
  <think> template-token leakage when uncertain.
- av4 (4L, 1.9M): min 4.444 @4000 then OVERFITS (4.524 @6500) — equals 1L-on-3M's best; depth w/o data
  just memorizes sooner. av4_all (4L x 3M) pending; harvest_v2 (+2M) running for 4L x 5M next.
- CORRECTION (user): the "overfit" was DATA RE-USE, not a model property. Measured (av1_l63 ckpts,
  trained-on vs held-out CE): 4.61/4.82 @2500 -> 3.30/4.73 @final, gap 0.21 -> 1.42 nats growing
  linearly past the data. Also my sampler re-drew rows across batches (not single-pass). FIXED: true
  single pass over a permutation (steps capped to data). Policy: NEVER epoch; data is infinite (harvest).
- av4_all (4L x 3M, 1 nominal epoch) min 4.277 @7500 = best so far. Launched av4_5m_1pass (4L, single pass
  over v1+v2 ≈5M) and harvest_v3 (+3M, seed 2) so the next run has ~8M.
- Training input is REAL h42 -> rollout (futurelens), per user's fork instruction; NOT AR(span)->span
  (the AR-inverter). Asked user which they want.
- AR-inverter mode (--ar): smoke OK (CE 14.25->9.36 in 25 steps). Launched av4_5m_arinv_1pass = same 4L/5M/single-pass config as av4_5m_1pass but soft token = ar_mse_v3(span), target = span. Direct h42-futurelens vs AR-inverter comparison.
- COMMON UNSEEN BENCHMARK = harvest_v3/shard00_part0000.parquet (seed 2, never trained on; EXCLUDE from future --data globs). Running conditioning_test_1layer on it for every final checkpoint so runs with different held-out slices become comparable.
- COMMON UNSEEN FILE results (harvest_v3/shard00_part0000, 1024 rows, real h42 in): 1L-3M 4.381 | 4L-3M 4.295
  (zero: 9.03 / 7.47; shuffled: 7.01 / 7.16). 4L is a much better unconditioned LM (zero 7.47 vs 9.03) and reads
  more specifically (2.87 vs 2.63 nats) -> supports "gap to 1.94 is text-modeling capacity". data/common_heldout.json.
- av4_5m_arinv_1pass: 17231 steps single pass (1.10M rows/rank x4). eval@500 = 4.07 already — but this is a DIFFERENT
  task (AR(span)->span is span-determined, autoencoder-like) so its CE is NOT comparable to the h42->rollout CE.
  The only fair comparison is the common unseen file with REAL h42 in (conditioning_test_1layer) on both finals.
- av4_5m_1pass: 4.14@4000 -> 3.95@7500 -> 3.93@8000, still falling (single pass, 17231 steps).
- av4_5m_1pass (h42): 4.14@4000 -> 3.93@8000 -> 3.705@15500 (still falling) then NCCL ALLREDUCE timeout: per-rank
  step cap (len(perm)//batch computed PER RANK) -> smallest rank exited early, others hung 10 min. No final.pt
  (last ckpt step_012000). FIX: all_reduce MIN of the cap. Stopped the arinv run (same bug), relaunched both as
  av4_5m_b / av4_5m_arinv_b.
- buggy arinv run (ap-Jvqp) reached 2.51 @8000 on its OWN task (AR(span)->span; not comparable to h42->rollout) before I stopped it.
- Step cap now global: ranks have 1102784/1104784/1104784/1000160 train rows (v1 tail files are short) -> 15627 steps = 4.0M examples/pass; ~300k rows idle. (Balance by rows, not files, if it matters.)
- harvest_v3 DONE (24 shards, 3M). Launched av4_8m_1pass: 4L, v1+v2+v3 minus the common test file (~8M rows), 8 GPUs x batch 32 = eff 256, single pass (~31k steps), container memory 320GB.
- COMMON FILE: av4_5m_b (4L, single pass, 4.0M examples) = 3.761 correct | 7.06 zero | 6.51 shuffled (spec 2.75)
  vs 4L-3M-epoch 4.295 -> -0.53 nats on the SAME unseen file. Single pass over more data is THE lever.
- av4_5m_arinv_b finished (exit 0): own-task CE (AR_mse(span)->span) 2.063@15000, 2.057@15500. Common-file test (real h42 in) launched: ap-hSqtX8YfsnYTOmex0UdEFi.
- AR-INVERTER TRANSFER FAILS on the common file (real h42 in): 5.945 correct | 6.46 zero | 8.17 shuffled (gap 0.52,
  spec 2.22) vs futurelens-trained 3.761. Same arch/data/steps; only the training input differs (AR_mse(span) vs real
  h42). Decodes degenerate into repetition ("call volume and call volume..."). Reading: the inverter learned to decode
  the AR's regressed/denoised vector space; real h42 (noisier, different norm) is off-distribution for it.
  => for the shallow AV, train on REAL activations. Testing whether a norm rescale of h42 closes part of the gap.
- NORM RESCALE of real h42 into the AR-inverter-trained AV: 5.93 (x0.4) ... 5.95 (x1.15) — flat. Not a norm mismatch; the inverter decodes the AR's vector directions, which differ from real h42's. data/arinv_norm_rescale.json.
- COMMON FILE: av4_8m (single pass, 6.96M examples) = 3.618 correct | 6.51 zero | 6.42 shuffled (spec 2.80).
  Data scaling on one file: 3M-epoch 4.295 -> 4M-1pass 3.761 -> 7M-1pass 3.618. Still improving; keep harvesting.
- Launched harvest_v4 (+4M, 32 shards, seed 3); a waiter auto-launches av4_12m (single pass v1..v4) when it lands.
- harvest_v4 31/32 done (last shard finishing); launched av4_12m_1pass (ap-vr5fp1IuHbsOd54xfPyCQq): 4L, v1..v4 minus common file, 8 GPUs x 32, single pass (~45k steps), mem 400GB. Data glob runs after the 27B load so the last part will be included.
- av4_12m_1pass FINISHED: 3.5536 @33000 (own held-out, same slice as 5M/8M runs). Common-file test launched.
- User asks: (1) compare AR(span) vs real h42 -> src/ar_vs_h42.py (stats + cross-feed CE matrix + residual decodability); (2) depth
  scaling at fixed data -> av1_8m / av2_8m single passes on exactly av4_8m's data; (3) weights -> HF private repo
  ceselder/olens-1layer-shallow-verbalizer (av4_12m, av4_8m, av1_l63_all + av1_model.py/common.py + README).
- COMMON FILE av4_12m (8.6M examples): 3.578 | zero 6.33 | shuffled 6.46 (spec 2.88). Data curve on one file: 4.295 (3M epoch) -> 3.761 (4M) -> 3.618 (7M) -> 3.578 (8.6M): still falling, diminishing.
- AR(span) vs REAL h42 (common file, data/ar_vs_h42.json): AR keeps ~13% of per-dim variance (median var ratio 0.13),
  raw FVE 0.21, cos 0.79 (centred 0.45), |AR|/|h| 0.80, residual = 89% of |h-mu| and cos(residual, h-mu)=0.88.
  => the MSE AR is a heavy shrinker to the span-conditional mean; what falls out is the context-dependent variance.
  CROSS-FEED CE: h42-AV: real 3.58 | AR(span) 3.37 (!) | residual 5.09 | zero 6.33.  AR-AV: real 5.95 | AR(span) 2.13 | residual 6.64 | zero 6.46.
  The h42-trained AV reads AR(span) BETTER than real h42 (denoised span info); the AR-trained AV inverts the AR nearly
  verbatim (2.13, copies the span) but collapses on the 87% of variance it never saw. AR-inverter == span autoencoder.
- DEPTH AT FIXED DATA (8M single pass, common file): 1L 4.013 | 2L 3.837 | 4L 3.618 (own held-out: 3.985/3.809/3.603). ~0.2 nats per depth doubling. 4L-12M = 3.578.
- USER: "I want the RNN AV" = verbalizer built from Qwen3.6's Gated-DeltaNet (linear_attention, recurrent) blocks, trained
  the same simple way. GDN blocks: idx 62,61,60,58,57,56,... (full-attn at i≡3 mod 4). GDN dims: 48 value heads x 128
  (6144), 16 key heads x 128 (2048), conv kernel 4; same SwiGLU 17408 MLP. Added --block-type gdn; smoking, then
  1/2/4-layer GDN on the same 8M single pass for a head-to-head with attention (4.01/3.84/3.62 common file).
- HF repo ceselder/olens-1layer-shallow-verbalizer now has av4_12m, av4_8m, av1_l63_all, av1_8m, av2_8m (+ code, README).
- harvest_fresh: hub offline toggled only around the dataset download; xet disabled (hub cache dir is read-only).
- 2026-09-07 LAUNCHES: gdn1/2/4_8m (GDN blocks, same 8M single pass as the attention runs); av2_12m (2L attn, 12M via
  --stream); ar_affine (closed-form ridge from frozen block-42 last-token [+mean] state of the span -> preceding h42;
  no LoRA; lambda sweep; scored on held-out + common file; compare to LoRA AR FVE 0.21 on the common file);
  harvest_v5 = 96 shards x 250k = 24M rows of FRESH FineFineWeb text (seed 4, 2 files/shard, file-offset 2000).
  NEW COMMON TEST FILE for fresh text = harvest_v5/shard00_part0000.parquet (never train on it).
- User: "crank it until it doesn't do better" -> keep harvesting + single-pass 2L runs, report the data curve.
- Modal cap: 100 ephemeral apps/workspace -> added task train-many (many spawns, one app). Shards 71-95 of harvest_v5 run in one app.
- RESULTS 2026-09-07 (own held-out, same slice): GDN 1L 3.965 | 2L 3.749 | 4L 3.614  vs attention 3.985 | 3.809 | 3.603.
  => GDN (recurrent) blocks read h42 as well as attention; slightly better at 1-2 layers, equal at 4. 2L attn on 12M
  (streaming) = 3.696 (vs 3.809 on 8M): data still helps the 2L. harvest_v5: 71/96 shards done (~17.9M fresh rows).
- AFFINE AR (closed-form ridge on frozen block-42 states of the span, 2M rows, lambda-insensitive): last-token feats
  FVE 0.088 (common 0.091) cos 0.75; last+mean feats FVE 0.140 (common 0.144) cos 0.77, centred cos 0.37.
  LoRA AR (ar_mse_v3) on the same common rows: FVE 0.21, cos 0.79, centred 0.45. An affine map gets ~2/3 of the LoRA AR.
  Saved /vol/ckpt/ar_affine_{last,lastmean}/affine.pt (+results.json).
- LAUNCHED: av2_30m + gdn2_30m (2L attn / 2L GDN, single pass over v1..v5 ≈ 30M rows, streaming, 8 GPUs x 32);
  common-file tests for gdn1/2/4_8m + av2_12m; harvest_v5 shards 71-95 (one app).
- 2026-09-08: 30M single-pass 2-layer runs DONE (own held-out): attn 3.536 @110k, GDN 3.487 @110k. 2L data curve (attn):
  8M 3.81 -> 12M 3.70 -> 30M 3.54; GDN 2L: 8M 3.75 -> 30M 3.49. Still falling. harvest_v5 complete (96 shards, ~24M).
  Scoring both finals on v3-common + v5-fresh test files; uploading both (+ av2_12m, gdn2_8m, gdn4_8m) to HF.
- wandb: runs since 2026-09-07 log to entity celestedeschamphelaere-personal (secret maemm-wandb changed); curve fetch now queries both entities.
- 2026-09-08 SCORES. v3 common file: 2L attn 8M 3.836 -> 12M 3.723 -> 30M 3.608; GDN 2L 30M 3.558 (best 2L); 4L 12M 3.578;
  GDN 1/2/4 8M 3.995/3.787/3.647 vs attn 4.013/3.836/3.618.
  v5 FRESH-TEXT file (truly unseen text): av2_30m 3.811 | gdn2_30m 3.766 | av4_12m 3.920 | av2_8m 4.134. Everything is
  0.2-0.35 nats WORSE on fresh text => the v3 'common' file overlaps the old corpus's text positions (optimistic).
  Models trained partly on fresh text (30M runs) generalise better to fresh text than the bigger 4L trained on old-corpus only.
  FRESH file = the honest benchmark from now on. Specificity on fresh text 2.1-2.3 nats.
- Checkpoints delivered: ~/shared/olens-1layer-ckpts/{av2_30m,gdn2_30m}/final.pt (+ av1_model.py, common.py, README) and HF.
- harvest_v6 launched: 96 shards x 500k = 48M fresh rows (file order seed 4, offset 2192 -> disjoint from v5), one app.
  NEXT: 2L attn + GDN single pass over v1..v6 (~84M rows) when v6 lands (~3h).
- User: train the 2L on the AFFINE AR too -> --ar-affine mode (frozen block-42 states -> W,b from ar_affine_lastmean). Launched av2_8m_affinv (same 8M pass as av2_8m); will score with REAL h42 on old+fresh files vs av2_8m 3.84/4.13. HF README updated with the full table; repo still PRIVATE (public flip blocked by the permission classifier; user to decide).
- USER: training data should be (AR(span), span), not (real h42, rollout). Launched av2_30m_arinv (2L, LoRA-AR input, 30M streaming). Existing (AR(span),span) models: av4_5m_arinv_b (LoRA AR, 4L, 5M) and av2_8m_affinv (affine AR, 2L, 8M, running).
- USER CONFIRMED 2026-09-08: the reader's job is AR(span) -> span. MAIN LINE = AR-INVERTER from now on. Real-h42 models
  stay as a reference line. Launched gdn2_30m_arinv (GDN 2L inverter, 30M). conditioning_test_1layer.py now auto-detects
  inverter checkpoints and scores in AR-space: CE(correct AR(span)) / zero / shuffled AR(span), + CE(real h42 in) reference.
- USER: (1) ordinary RNN (not GDN) -> --block-type lstm = plain N-layer nn.LSTM(5120) + identity-init out map, fp32, random init;
  (2) random-init transformer -> --init-layer -1 (normal(0,0.02) matrices, ones norms, zeros biases); (3) crank batch ->
  eff 2048 (8 x 256), lr 2e-4. Launched inverter set at 2048: av2 / gdn2 / lstm2 / rand2 _30m_arinv_b2048 (30M streaming).
  The two eff-256 inverters (av2_30m_arinv, gdn2_30m_arinv) continue as the batch-size ablation.
- AR-SPACE SCOREBOARD (input = AR_mse_v3(span), target = span): av4_5m_arinv_b (4L, 5M): old file 2.127 | fresh 2.242;
  zero 6.5; shuffled AR(span) 11.7-11.9 (=> 9.6 nats of specificity in AR-space); real h42 in: 5.9-6.2 (reference).
  Fresh-text penalty in AR-space is only ~0.12 nats (vs 0.2-0.35 for the real-h42 readers). Decodes ≈ paraphrases of the span.
- AR(span) PRECOMPUTE: inverter runs were AR-forward-bound (0.31 s/step at 32/GPU -> ~12 h). precompute_ar.py caches ar_vec (fp16) per file under /vol/data/ar_v3 (flat mirror names); trainer --ar-cache reads it (test files included in the cache, they're excluded from training by --exclude).
- Stopped the 6 AR-forward-bound inverter runs (would take ~12h); relaunching on the AR cache when precompute completes.
- AR cache for v1..v5 COMPLETE (9116 files). Launched cached 2L inverters: av2/gdn2/lstm2/rand2 _30m_arinv_c2048 (eff 2048,
  lr 2e-4) + av2_30m_arinv_c256 (eff 256, lr 1e-4, batch ablation). Precompute for harvest_v6 (partial) launched.
- HF: ceselder/olens-ar-inverter-qwen3.6-27b = av4_5m_arinv_b + ar_mse_v3 adapter/value_head + ar_span.py + feed_inverter_example.py.
- Asked olens-new-arch on the bus for its olens seed data / train_olens config / olens_mse results (bus wait running).
- av2_8m_affinv (2L, input = AFFINE AR(span)): AR-space 0.643 (old) / 0.691 (fresh); zero 11.5; shuffled 16.5; real h42 in 9.8.
  Near-verbatim inversion: the affine AR is a linear map of the frozen block-42 span state, so inverting it ≈ a tuned lens on
  the span's own representation. (LoRA-AR inverter 4L: 2.13 — the LoRA AR compresses toward what predicts the activation.)
- SMALL READERS (user: <=70M params, Pareto front data/params): SmallReader = act_proj 5120->d, frozen Qwen embed -> tok_in
  5120->d, N pre-LN causal blocks (width d, heads d/64, MLP 4d), out_proj d->5120 -> frozen RMSNorm -> frozen lm_head.
  Sizes: d256/L2 5.5M, d512/L2 14M, d768/L4 40M, d1024/L4 66M. --frozen file (extract_frozen.py) => no 27B load.
- Grid v1 was I/O-bound on the Modal volume (3-5 s/step for 5-66M-param readers). Added a parallel-read prefetcher to the stream loader; relaunched the 12-run grid (v2) and the 2L inverters as 4-GPU x 512 jobs (8-GPU slots unavailable).
- 2026-09-08 RESULTS (AR-space, own held slice, single pass): SMALL readers 2M/8M/30M rows: 5.5M 4.95/3.93/3.39 | 14M 4.52/3.44/2.84 |
  40M 4.24/2.99/2.32 | 66M 4.07/2.84/2.15. 27B-width 2L on 30M: attn 1.497 (b2048) / 1.476 (b256) | GDN 1.452 | random-init 1.844 |
  LSTM (472M) 1.996. 4L 5M: 2.06. => both axes still steep (no flattening): 4x data ≈ -0.6..-1.1 nats; ~3x params ≈ -0.3..-0.5.
  Batch 256 vs 2048: same (1.48 vs 1.50). 27B init worth -0.35 nats vs random. GDN > attn again.
- USER: classic RNN (not GDN). Added --small-rnn {rnn,gru,lstm}: SmallReader with an nn.RNN/GRU/LSTM stack (fp32) instead of transformer blocks. Launched 12 runs on 30M rows (d512L2, d768L4, d1024L4, d1024L6 per type). Existing 27B-width LSTM (472M) = 2.00.
- USER: the Pareto front of interest = EVAL vs PARAMS with data unlimited (not data vs params). Plot reshaped: CE vs params (front = lowest curve) + CE vs data per size (all still data-limited). Next: 84M-row pass for every size when the v6 cache lands.
- USER: transformer with NO MLP -> --small-no-mlp (attention-only blocks). Launched 4 sizes on 30M rows. wandb: celestedeschamphelaere-personal/olens-1layer (new) + octahedral-systems/olens-1layer (old).
- v6 cache covers all 11125 landed parts (89/96 shards; 7 still running, invisible until commit). Launched ~80M-row passes: small transformers d256L2/d512L2/d768L4/d1024L4 (4 GPUs x 512) + gdn2_80m_arinv.
- CLASSIC RNN readers, 30M rows (own held-out): vanilla RNN 9M 3.74 | 16.5M 3.35 | 24M 3.16 | 28M 3.11; GRU 11M 3.28 | 26M 2.89 |
  41M 2.70 | 53M 2.71; LSTM 12M 3.16 | 31M 2.76 | 49M 2.55 | 66M(L6) 2.67. Transformer at matched params is 0.3-0.5 nats
  better (14M 2.84, 40M 2.32, 66M 2.15). LSTM > GRU > vanilla. Going L4 -> L6 did not help the RNNs.
- USER: affine AR from EVERY span token's block-42 state (first 4 / 8 / all 12 concatenated) -> preceding h42. Launched ar_affine first4/first8/all12 (2M rows, ridge).
- USER: 1-layer attention-only reader -> launched small_nomlp d512/d1024/d2048 L1 + full transformer d1024 L1 (30M rows).
- harvest_v6: shards 89-95 were dropped from the second relaunch app's queue; relaunched as a 7-spawn app.
- USER: frozen Qwen embed/unembed + 1-2 layer attn-only (w/ and w/o MLP) in the middle at FULL width = --block-type fullattn (--inner, --mlp-hidden). Params = attention only (no 5120<->d projections). Grid fw_L{1,2}_i{512,1024,1536}_m{0,1024,2048} on 30M rows launched.
- USER: make the MLP an affine map -> --mlp-affine (rank = --mlp-hidden; 5120 = full-rank Linear). Launched 4 full-width variants on 30M rows.
- USER: are heads varied? No (head_dim fixed 64; heads = width/64). Added --heads; launched heads ablation at fullattn L1 i1024: 1/4/64 heads (16 already running).
- USER asked about the MODULATION-LENS AR + inverter: not built before. Now: dictionary pass (frozen block-42 state at the
  span's last token = "activation while reading the phrase") cached under /vol/data/dict_v1 for v1..v4; modulation-lens
  inverters (reading state -> phrase) to train on it via --ar-cache /vol/data/dict_v1. Exact modulation-lens AR = the frozen
  27B read itself (cos 1.0); a trained surrogate only if wanted. Lens stage (NNOMP -> SFT -> contrastive RL) still TODO.
- USER: linear RNN -> --block-type linrnn (h_t = A h_{t-1} + B x_t, full width, closed-form W_U A^t B v). Launched on 30M rows.
- 1-LAYER READERS, 30M rows (own held-out): full-width attn-only i512 10.8M 3.64 | i1024 21M 3.32 (16h); heads @i1024: 1h 4.12,
  4h 3.53, 16h 3.32, 64h 3.26. + affine map: i1024+aff1024 32M 3.29, i512+aff5120 37M 3.44 (affine barely helps).
  + MLP: i512+mlp2048 32M 2.99, i1024+mlp2048 42M 2.80. NARROW 1-layer: transformer d1024 28M 2.63 (best 1-layer!);
  attn-only d512 9M 3.51, d1024 20M 3.21, d2048 48M 3.10. => nonlinearity (MLP) matters; more heads better; 1 head costs 0.8 nats.
- 80M-row passes DONE: 5.5M 3.265 | 14M 2.675 | 40M 2.111 | 66M 1.947 (vs 30M: 3.39/2.84/2.32/2.15); GDN 2L 793M 1.251 (vs 1.45).
  Full-width 2L: attn-only i512 3.37 | i1024 3.00 | i1536 2.90; +MLP1024: i512 2.68 | i1024 2.59; +affine: i512 3.30 | i1024 3.13.
- USER: deliver 1- and 2-layer transformer ckpts + RL them. Plan: GRPO-style RL with the contrastive AR-space reconstruction
  reward r = cos(AR(sample), AR(span)) - cos(AR(sample), AR(other row)); LoRA AR as the frozen reward model; KL to the SFT reader.
- USER: are the small readers trained on the affine AR? No (LoRA AR cache). Launched an affine-AR cache (/vol/data/ar_affine_cache, v1..v4) to train 1L/2L transformers on it.
- RL launched on small_d1024L1_30m (user: 'RL that one'): GRPO-style, centred contrastive cosine reward via the LoRA AR, kl 0.1, lr 1e-5, 3000 steps x 128 prompts x 4 samples.
- AFFINE AR w/ per-token states: first4 0.115 | first8 0.141 | all12 0.154 (FVE, common file) vs last+mean 0.144, LoRA 0.21. Launched affAR_* (affine-AR inverters) and modlens_* (frozen reading-state inverters) at 1L/2L/4L on 12M rows.
- USER: batchmaxx -> bm_* grid at 8x2048=16384 eff batch, lr 1e-3, ~84M rows (d256L2,d512L2,d768L4,d1024L4,d1024L1, fw L2 i1024 m1024). RL relaunched at 512x4.
- USER: lr 1e-3 too high -> relaunched bm_* at 3e-4 (+ d1024L1 at 1e-4 as an LR check). Note: all 30M/80M grid results were at 5e-4 / batch 2048.
- INPUT-SPACE COMPARISON (12M rows, same archs): affine-AR inverters 1L 1.466 | 2L-fw 1.229 | 4L 1.137; modlens (frozen reading
  state) 1L 2.795 | 2L-fw 2.668 | 4L 2.38; LoRA-AR (30M) 2.63 | 2.59 | 2.15. Linear RNN (52M, 30M rows) 3.696.
- bm_small_d1024L1 at batch 16k / lr 1e-3 / 84M rows: 2.734 (WORSE than 2.63 at 2048/5e-4/30M -> 4.7k steps too few / lr too high).
  lr 3e-4 grid running. RL (512x4) at step ~1150: centred cos_true 0.87, cos_other ~0, gap 0.87; CE(true span) drifting 3.9 -> 4.1.
- RL v1 (kl 0.1, lr 1.5e-5): sampled reward 0.80-0.83, greedy reconstruction FLAT at 0.87 (= SFT start), KL exploded to 13-20 nats/seq, CE(true span) 2.76 -> 4.1. Stopped. RL v2: kl 1.0 + sft-mix 0.5 + lr 5e-6.
- USER: affine-AR inverter ckpt + RL against the affine AR. Uploading affAR_* + ar_affine_lastmean/affine.pt + ar_affine_span.py. RL vs affine AR launched (2 configs). Affine cache extension to v5+v6 (72M) launched (64 workers).
- BATCH 16k (lr 3e-4, 84M rows, 4954 steps) is MUCH WORSE than batch 2048: 66M 2.58 (vs 1.95 @80M), 40M 2.88 (2.11), 14M 3.46 (2.68),
  5.5M 4.03 (3.27), 1L 3.24 (2.63 @30M); 1L at lr 1e-3 2.73, 1e-4 3.99. Small readers are STEP-limited -> keep batch 2048, more data.
- RL vs LoRA AR on small_d1024L1: no gain (gap 0.869 -> 0.870/0.876); weak KL -> drift (kl 17, CE 2.76->3.83). Ceiling of the AR.
- USER: NO KL. RL relaunched with kl 0, sft-mix 0 (affine-AR reward on affAR_small_d1024L1_12m; LoRA-AR reward on small_d1024L1_30m).
- USER: scaling law of the affine AR -> ridge fits at 0.25..8M rows (last+mean) and 4/8M (all12); ⊕ = concatenation [last-token state ‖ mean state] (10240-d).
- RL now = MAEMM ScaleRL bundle (CISPO eps_max 5, batch-level adv norm + zero-var filter, prompt-level agg, fp32 head, betas .9/.999, group 8, no KL) with mean-centred FVE reward (no contrastive term). NPR + length control moot (fresh prompts, fixed 12 tokens). Two runs relaunched.
- Affine cache complete for v1..v6 (21116 files). Launched affAR_* at 84M rows, batch 2048 (7x their data).
- REWARD per paper A.9.2: whitened FVE after NNLS refit (K=1 -> max(0,cos_w)^2) using whitener_v2 (mu, Sigma^-1/2 from harvested L42 acts). No KL. Relaunched both RL runs.
- USER CLARIFIED THE RL: input = REAL h42 (Qwen3.6-27B L42) -> olens text -> AR (affine or LoRA) -> must reproduce h42 in
  mean-centred WHITENED space (whitened FVE, NNLS refit, no KL). rl_inverter.py --input h42. Launched: olrl_L1_loraAR
  (init small_d1024L1_30m, LoRA-AR reward), olrl_L1_affAR (init affAR_small_d1024L1_12m, affine-AR reward); plus a real-h42
  SFT small reader (rh42_small_d1024L1_12m) as a better warm start. Stopped the AR-vector-input RL runs.
- OLENS TEACHER (paper stage 3, K=1): seed_teacher.py = nearest dictionary AR vector (whitened cosine) per real h42 -> SFT rows (h42, teacher span). Also reports the true-span wFVE ceiling (LoRA AR 0.094, affine 0.039 on the RL held-out). RL from inverter inits: wFVE 0.02 (LoRA) / 0.01 (affine) after ~1k steps.
- USER: 'train only on AR vectors, it will generalize'. Data says no (5.95 nats on real h42). Testing a closed-form adapter P (ridge h42 -> AR vector) so the AR-only reader sees P(h42): CE(true span) and wFVE vs real h42 through it.
- Added eval_olens_rl.py: paper-style inference (sample K=8 phrases -> AR -> NNLS in whitened space -> wFVE), plus greedy / mean-sampled / best-of-K; run on SFT inits + latest RL ckpts.
- OLENS RL WORKS (paper-style eval, K=8 NNLS whitened FVE vs REAL h42, fresh file): LoRA-AR reward: init 0.006 -> RL@1150 0.031
  (ceiling = true span 0.070; ~45% of ceiling); mean sampled 0.0011 -> 0.0098. Affine-AR reward: init 0.003 -> RL@1450 0.025
  (ceiling 0.027, ~90%). Next: 4096x16 per step on 8 GPUs (DDP + chunked backward; smoke failed, fixing).
- ADAPTER P (ridge h42 -> AR vector): AR-only reader fed P(h42) reaches wFVE 0.069 vs real h42 (ceiling 0.076 = true span; raw h42
  in: 0.011; true AR(span) in: 0.051) and CE(true span) 5.74 (vs 7.26). => the user was right: AR-only training generalises given
  a linear front-end. TEACHER K=1: best dictionary span wFVE 0.155 (2x the true span's 0.078) over 750k activations.
- DDP RL smoke fails on rank 1 with no python traceback (investigating); 1-GPU chunked path OK.
- BIG RL launched (2048x16 = 8 ranks x 256x16, adapter front-end, LoRA-AR wFVE reward, no KL, every-step logging): 1L small_d1024L1_30m and 2L fw_L2_i1024_m1024_30m. Adapter-start RL hit greedy wFVE 0.107 > true-span ceiling 0.094 within 70 small steps. Fitting P_aff for the affine-AR run. USER also wants the full 27B + Karvonen-inject (layer-1 LoRA injection) reader trained on EXACTLY the same data as the baseline.
- FULL-LM BASELINE: train_av27b.py = 27B + rsLoRA(64/16) + Karvonen layer-1 injection at the marker, AR-inverter on the SAME cached rows (v1..v4, 12M). Smoke launched; matched small_d1024L1 + fw_L2 on the same 12M rows launched.
- SEED SFT done (teacher spans, 980k rows): init rh42 4.41 CE | scratch 5.13. Launched RL from seedsft_L1_from_rh42 (raw h42, 2048x16) + paper-style eval of the seed readers. LR-check RL twin at 3e-6 launched.
- 27B Karvonen baseline launched on v1..v4 (12M LoRA-AR rows), eff batch 256, lr 3e-5 (CLAUDE.md default for injection), ~47k steps. Matched shallow readers small_d1024L1_12m + fw_L2_i1024_m1024_12m running on the same rows.
- SEED-SFT reader (teacher spans, init rh42) BEFORE RL: greedy wFVE 0.061, NNLS-8 0.078 > true-span ceiling 0.070 (fresh file). Scratch 0.036/0.048; futurelens reader 0.045/0.062; AR-only+adapter greedy 0.069-0.091 (ceiling 0.076-0.094).
- Modal B200 capacity: 8-GPU jobs queued for >1h. Relaunched the 4 RL runs as 4 GPUs x 512x16 (same 2048x16 batch). 27B baseline stays 8-GPU.
- USER: wants the FULL-WIDTH attention-only 1-layer as the RL policy (small_d1024L1 is the narrow attn+MLP one). Launched RL on fw_L1_i1024_h64_30m and fw_L1_i1024_m0_30m (adapter start, 2048x16).
- USER: RL slow -> it's the 27B reward forward (8192 spans x 12 tok x 43 layers per rank per step). Affine AR needs the same forward. Fitting an EMBEDDING-only affine AR (no 27B) as a cheap reward candidate; reward chunk 256 -> 1024 for new launches.
- AFFINE adapter: affAR reader via P_aff: wFVE 0.024 vs affine ceiling 0.033 (raw h42: 0.004). Affine-reward RL launched (2048x16).
- Pruned RL: stopped lr3e-6 twin, fw L1 16-head duplicate, seed-SFT RL. Kept: 1L narrow (lora), 2L full-width (lora), fw L1 attn-only 64h (lora), affine-AR 1L (affine reward).
- EMBEDDING-only affine AR: FVE 0.050 (vs 0.14 frozen-state affine, 0.21 LoRA) -> too weak as a cheap reward; the 27B forward stays.
- RL @2048x16 (greedy wFVE, ceiling 0.074): narrow 1L 0.063 -> 0.112 @100; fw 2L 0.048 -> 0.103 @100; fw 1L attn-only 0.030 -> 0.071 @75;
  affine 1L (ceiling 0.033) 0.024 -> 0.041 @75. 27B Karvonen inverter: held CE 1.83 @3000/41k (shallow 1L: 2.63).
- Delivered olrl_L1_loraAR_2048x16/step_000100.pt (+ adapter) to ~/shared and HF.
- eval_olens_rl.py: + 27B NLL/token of generated spans (naturalness / hacking detector) + --adapter input. Running across the 1L RL checkpoints.
- MATCHED 12M rows: shallow 1L 2.984 | fw 2L 2.841 | 27B Karvonen 1.794 @3.6k/41k steps (inverter CE). eval_olens_27b.py written (paper-style eval for the injected 27B reader; adapter-switching reader/AR on one base); smoke launched.
- NATURALNESS (27B NLL/token of generated spans; true spans 4.67): 1L adapter SFT greedy 4.00 -> RL@50 4.58 -> RL@100 4.85 (sampled 6.4 -> 7.15) while NNLS-8 wFVE 0.088 -> 0.129 -> 0.140 (2x the true-span ceiling 0.070). Mild naturalness drift; watch.
- ADAPTER/AR FVE (2000 unseen rows, raw | whitened-centred): LoRA AR(true span) vs h42 0.206 | 0.065; P(h42) vs AR(span) 0.508 | 0.183 (cos 0.96); P(h42) vs h42 0.311 | 0.137. Affine: AR(span) vs h42 0.143 | 0.028; P_aff vs affAR(span) 0.486 | 0.174; P_aff(h42) vs h42 0.217 | 0.058. => whitened FVE is ~3-5x lower than raw for every map; P(h42) as a reconstruction of h42 beats the AR itself (it is a function of h42).
- 27B KARVONEN READER (SFT only, step 5000/41k, RAW h42 in, no adapter, no RL): greedy wFVE 0.1185 | NNLS-8 0.189 (ceiling 0.074),
  natural (NLL 4.27 vs true 4.71). => the 27B generalises from AR vectors to real activations by itself (user's hunch holds at 27B scale).
  RL @~1100: narrow 1L 0.138 | fw 2L 0.136 | attn-only 1L 0.098 | affine 1L 0.062. affAR 84M readers: 4L 0.467 | 1L 0.807 | 2L-fw 0.784 | d512L2 1.064.
- Delivered olrl_L1_loraAR_2048x16/step_001100.pt (+eval) to ~/shared and HF.
- 27B reader @5000 with the ADAPTER input: greedy 0.068 / NNLS-8 0.133 (WORSE than raw 0.1185 / 0.189): the 27B's injection generalises to raw h42; the adapter only helps the shallow readers.
- BUG: cache-fed checkpoints stamped ar=LoRA path even for the affine cache -> cond tests of affAR readers fed LoRA-AR vectors (7.1 nats, bogus). Fixed: --input-ar override + ar_cache saved in ckpts; re-running.
- affAR readers, AFFINE-AR-space CE (fresh/old): 4L-84M 0.459/0.507 | 1L-84M 0.798/0.866 | 2L-fw-84M 0.788/0.843 | d512L2-84M 1.035/1.132 | 1L-12M 1.54 | 4L-12M 1.22. Vibes comparison (modlens rl-step25 vs 27B-Karvonen vs tiny RL 1L) launched on 36 WorkspaceBench items.
- VIBES (36 WorkspaceBench items, hit = gold string in readout): see data/vibes_hit_rates.json; modulation oracle & 27B Karvonen name the intermediate most of the time, tiny 1L RL gets domain/category and the entity ~half as often.
- MODLENS objective: their AR = LoRA(r64,a16) on blocks 0..42 + mean-pool + head -> J-space (heldout cos 0.944); target = unit(J[42] h - amu); reward = NNLS cos(matched) - cos(other). Added --reward modlens to rl_inverter (K=1) and eval_modlens_metric.py (their delta metric, 4 bullets). Smoke + metric eval launched.
- Launched RL vs modulation-lens reward: tiny 1L + fw 2L (adapter start, 2048x16, no KL).
- MODULATION-ORACLE METRIC (their AR, unit(J h - amu), NNLS over 4 bullets, delta = matched - permuted; 256 fresh-file REAL web-text activations):
  oracle sft 0.059 | rl-step25 0.094 | rl-step50 0.099 || our 27B Karvonen SFT (4 samples) 0.083 (greedy K=1 0.081) || tiny 1L SFT+adapter 0.046 (K=1 0.057)
  || tiny 1L after OUR LoRA-AR RL@1100 (adapter): 4 samples 0.083, greedy K=1 0.090 ~= the oracle rl-step25 (0.094), at 28M params.
  Caveat: their published deltas (0.48-0.59) are on their own bank/holdout (modulation prompts); on natural activations everything is ~0.05-0.10.
  Now RL'ing the tiny 1L and fw 2L directly against the modulation-lens reward (2048x16).
- 27B Karvonen inverter finished (41k steps, 12M rows). Final evals launched (raw/adapter paper-style; modlens metric v2 incl. tiny 1L FINAL). Tiny 1L RL final: greedy wFVE 0.141 (ceiling 0.074), cos 0.54, CE(true) 8.42.
- USER: affine AR read at a '.' token inserted after the span -> ar_affine --feat dot / dot+mean (2M rows). Compare to last 0.09 / last+mean 0.144 / all12 0.154.
- HEADLINE: tiny 1L (28M) RL'd AGAINST the modulation-lens reward, step 1400: delta 0.113 (4 samples) / 0.118 (greedy K=1) on the
  oracle's own metric, same 256 fresh activations, vs oracle rl-step50 0.099 / rl-step25 0.094 / 27B Karvonen SFT 0.085 /
  tiny RL'd on our LoRA-AR reward 0.083. fw 2L modlens-RL: 0.111 / 0.114. => the tiny transformer optimised against their
  objective BEATS the 27B modulation oracle on that objective. 27B Karvonen FINAL: inverter CE 1.134; paper-style raw 0.122 / NNLS-8 0.188.
- 27B Karvonen RL: added --policy lora27b to rl_inverter (LoRA adapter trainable, HF generate with InjectL1, span logprobs, reward AR as a second adapter on the same base). Smoke launched.
- 27B Karvonen RL launched vs the modulation-lens reward (256 acts x 8 samples/step, lr 1e-5, no KL, 600 steps).
- '.'-token affine AR (2M rows): see data/ar_affine.json summary_token_feats. 27B Karvonen RL vs modlens reward: delta 0.149 @20 -> 0.158 @30 (K=1 greedy, own held slice) — already above the tiny (0.13) and the oracle (0.099).
- USER: scaling of the '.'-token affine AR -> sweep launched (dot 0.25-4M, dot+mean 0.5-4M, last 1/4M) for the scaling plot.
- USER: 27B Karvonen RL vs the olens (LoRA-AR whitened-FVE) reward launched (256x8, no KL, 600 steps).
- ORACLE METRIC, same 256 rows: 27B Karvonen RL'd vs modlens reward @step80: delta 0.165 (4 samples) / 0.159 (K=1) vs oracle rl50 0.099, tiny modlens-RL 0.114.
  Its own-slice K=1 delta 0.178 @310/600. '.'-token affine scaling sweep collected (data/affine_scaling.json, plot updated).
