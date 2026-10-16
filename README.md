# CLOVE — soybean pilot

**C**ontinual **L**earning for Scalable **O**pen-World Agricultural **V**isual **E**valuation, with the optional
**PRISM** memory policy (Provenance-aware Retrieval and Interference-Sensitive Memory).
This package implements `CLOVE_PROTOCOL.md` v1.0 (16 Oct 2026) on top of the existing SAGE resources.
It is *experimental infrastructure*: nothing in this repository is a finding until produced by `clove run` on
the real backend and summarised by `clove analyze`.

**Contract (enforced in code and tests).** The backbone VLM is frozen. There is no fine-tuning, no gradient,
no LoRA, no image demonstrations and no labelled reference images: **k_img = 0** for every stage and method.
The only image in a call is the query photo. Text evidence comes only from SAGE class records admitted at
the current episode. SAGE files are read-only; CLOVE writes only to `paths.artifacts`.

## 1. Notation

| symbol | meaning | where in code |
|:--|:--|:--|
| 𝒞 | admissible soybean classes (test images ≥ `min_test_images` and KB claims ≥ `min_claims`) | `data/splits.eligibility` |
| 𝒟 = {(xᵢ, yᵢ, gᵢ, sᵢ)} | fixed test images, labels, specimen/duplicate group, source | `data/manifest.CanonicalManifest` |
| 𝒦 = {K_c} | immutable registry of source-linked claims | `kb/reader.KBRegistry` |
| 𝒞ₜ, 𝒰ₜ, Δₜ | admitted / not-admitted / admitted-at-t classes; 𝒞ₜ₊₁ = 𝒞ₜ ∪ Δₜ | `kb/inventory.ClassLedger` |
| f_θ | frozen VLM (θ never changes; fingerprint checked for in-process models) | `inference/backend.py` |
| Mₜ | derived, versioned memory (cards of claim ids) | `memory/base.MemoryState` |
| ŷₜ(x) ∈ 𝒞ₜ ∪ {⊥} | prediction; ⊥ = unknown/abstain. `UNRESOLVED` is scored as ⊥ (pre-registered) | `orchestration/policy.py` |
| b_KB | retrieved KB evidence tokens (exact tokenizer count, strict cap) | `budgets.yaml: kb_tokens` |
| b_out | max generation tokens | `max_new_tokens` |
| r | number of candidate classes offered as options | `r` |
| k_img | labelled visual references — **identically 0**, not configurable | `ImageGuard` |

**Three token quantities are kept apart in every record:** `kb_tokens` (b_KB, counted before the call),
`prompt_tokens` (the backend's own count, which includes image tokens) and `output_tokens`. `image_tokens` is
recorded when the backend exposes it and is otherwise `null`. It is never assumed to be zero.

### Forgetting contrasts

* Expansion: F_exp(t) = A(D_old; 𝒞₀, M₀) − A(D_old; 𝒞ₜ, Mₜ). More candidate labels make the task harder, so
  this is **not** memory forgetting.
* Memory history: F_memory(t) = A(D_old; 𝒞ₜ, Mₜ^fresh) − A(D_old; 𝒞ₜ, Mₜ^persistent). Both sides use the same
  inventory, images, retrieval caps and token budgets. Positive means persistent memory is worse.
* Compute: F_tok(j,t) = (T_{j,t} − T_{j,t_j}) / max(1, T_{j,t_j}), where T is end-to-end prompt + output tokens.
  It is reported conditionally (correct at both t_j and t) and unconditionally, together with the prompt,
  output, KB, call and latency deltas separately.
* Class-wise accuracy forgetting F_c = max_{τ∈[t_c,T]} a_{τ,c} − a_{T,c} (mean and worst) and BWT.
* Acquisition: Q_pre(c) = P(⊥ | y=c, t<t_c) and Q_post(c) = P(ŷ=c | y=c, t≥t_c) are two separate endpoints.
  They are never subtracted.

### Denominators

* **Known** = images whose class is admitted at t (`old` = admitted earlier, `new` = admitted now). Rejecting
  a known image counts as an error for open-world accuracy. Macro accuracy is the mean over classes of
  per-class open-world accuracy.
* **Unknown** = `pending_unknown` (admissible, admitted later) plus `permanent_unknown` (images but no
  admissible KB, or held back). "Unknown" is relative to the inventory: the pretrained VLM may well know
  these diseases.
* Unknown-detection metrics treat unknown images as positives, with score s = 1 − max known-option
  probability. They are `NA` when either side is empty, never 0. The final stage records
  `no_eligible_unknowns` when it applies.

## 2. Data actually used (all read-only)

| source | path (from `configs/pilot.yaml`) | role |
|:--|:--|:--|
| WACV frozen manifest | `sage-nova-mirror/wacv2027_nova/data/manifest/soybean.json` | 26 classes, 1,234 test ids, per-class `ref` ids |
| WACV KB | `sage-nova-mirror/wacv2027_kb/soybean/internet.xlsx` | Visual Description per class (no per-sentence URL → `kb_description` claims) |
| WACV provenance | `wacv2027_nova/data/part_index/soybean_sources.json` | verbatim `{url, quote}` → `quote` claims |
| GitHub SAGE registry | `SAGE/disease_registry/outputs/Soybean/final_registry.json` ([tirtho149/SAGE](https://github.com/tirtho149/SAGE)) | per-field value/url/quote, `look_alikes` → hard-pair list |
| image pool | `sage-nova-mirror/dup_audit/scale_images`, `sage-wacv2027/data/images` | content-addressed `<sha1>.jpg` |
| (optional) Prepared_Dataset | `SAGE/Prepared_Dataset/Soybean{,_test}` | alternative folder-layout image source |

To use the registry, clone the GitHub repo next to this one: `git clone https://github.com/tirtho149/SAGE ../SAGE`.
On macOS, two `Bean_Pod_Mottle_{V,v}irus` test files collide on the case-insensitive filesystem; clone on
Linux (Nova) if you use `prepared_dataset`. Without the registry the audit marks it `exists=False` and the
other two KB sources are used.

**Dev split.** SAGE's per-class `ref` images are **never shown to the model**. Under CLOVE they are the
development split, used only to calibrate the unknown threshold and for PRISM's protection prior. A dev image
in the same duplicate group as any test image is dropped.

**Recorded decisions.** Review these, because each one is a scientific choice:
* Aliases (`classes.aliases`): `Soybean_rust → Rust`, `Septoria_brown_spot → Brown_Spot`,
  `Green_stem_disorder → Green_Stem`, `Fusarium → Fusarium_Disease`, `Rhizoctonia_Damping-Off → Rhizoctonia`.
  Phomopsis is **not** merged into Diaporthe.
* Claims that name a not-yet-admitted class are masked until that class is admitted. This prevents leakage,
  and it is also the concrete mechanism by which append-only memory, whose cards are compiled once at
  admission, can lack later source-cited contrasts. Pathogen binomials (e.g. *Cercospora sojina*) are not
  counted as class mentions.
* `sage0` reuses the SAGE k=0 prompt wording (vendored from `sage-wacv2027/wacv/prompts.py`) with two logged
  departures: one added sentence that allows `UNKNOWN`, and masking of KB sentences that name unadmitted
  classes. It sees *all* admitted descriptions (no b_KB cap) as the fixed-resource reference, and its
  tokens are still counted.

### Local state (audit run on 2026-10-16, this laptop)

Only 556 of the 1,234 WACV test images are present locally, so 16 classes meet `min_test_images = 10` and the
derived schedule is 4→8→12→16. On the full Nova pool, 23 KB-backed classes have test images (20 with 50 each, 3 with 21–32), which
derives 8→13→18→23 or better. Herbicide_Injury, Iron_Deficiency_Chlorosis and Potassium_Deficiency have
images but **no KB** in any SAGE source, so they become permanent unknowns (abiotic soybean stresses). Source
OOD and field OOD are **NOT AVAILABLE**: no SAGE file carries verified source/site/year metadata.
**Freeze the manifest on the machine that has the full image pool** (see §5).

## 3. Methods

| tier | `--methods` key | what it isolates |
|:--|:--|:--|
| essential | `names` | pretrained prior: all admitted names, no KB |
| essential | `sage0` | SAGE k=0 with the active KB (fixed-resource reference) |
| essential | `append` | persistent append-only memory: card compiled once at admission |
| essential | `fresh` | Mₜ^fresh: every card rebuilt from the authorised KB at each t |
| essential | `prism` | interference-triggered card refresh + utility-ranked candidates + query-contrastive packing |
| diagnostic | `oracle` | true class forced into the candidates (known images only): retriever vs recogniser |
| secondary | `uniform`, `recency`, `random`, `bm25` | mechanism ablations |
| negative control | `shuffled` | evidence blocks deranged across options (matched length) |
| ablation | `prism_noscan`, `prism_noprotect` | PRISM without interference scan / without protection prior |

All methods share r, b_KB, b_out, the excerpt cap, the instruction cap, the unknown gate and the visual
retriever. PRISM's utility is u(c|x,t) = α·s_visual + β·s_int + γ·p_protect. The coefficients are fixed in
config before testing, and p_protect comes from **dev** errors only.

**Visual candidate retrieval.** `retrieval.visual: siglip` scores the query photo against each admitted
class's *text*, using a frozen dual encoder. It is not a VLM call and uses no reference image. It is the only
thing that makes r < |𝒞ₜ| meaningful. With `visual: none`, every admitted class is a candidate, and PRISM
then collapses to almost exactly `fresh` (seen in the synthetic run). Treat `none` as a debugging setting.

## 4. Inference contract

One VLM call per prediction. A second call happens only if `policy.second_call: true` and the unknown score
falls in the pre-registered band. The output is strict JSON `{"answer": "<option code>"}`; there is no CoT.
The option distribution comes from the top-logprobs at the answer token, and the decoder expands it into the
record `{label_id, decision, unknown_score, evidence_ids}`. Responses are cached only under byte-identical
inputs: image sha256, prompt hash, model/revision/decoding, b_out, method, and memory-state hash.

Backends: `openai_compat` (vLLM server; exact `usage` token counts), `transformers` (in-process; weight
fingerprint checked before and after), `claude_headless` (Claude Code `claude -p`; see below), and `mock` (**tests only**). Real runs refuse `mock`, and anything
the mock produces is stamped `synthetic` and rendered as "SYNTHETIC — NOT RESULTS".

## 5. Commands

```bash
make venv && make test                     # CPU-only suite (unit, leakage, statistics, integration)

python -m clove.cli audit        --config configs/pilot.yaml   # Gate 0 -> artifacts/reports/audit.md
python -m clove.cli manifest     --config configs/pilot.yaml   # freeze manifest + class orders (immutable)
python -m clove.cli test-leakage --config configs/pilot.yaml   # must PASS before any run
python -m clove.cli smoke        --config configs/pilot.yaml --synthetic   # pipeline check, not results
python -m clove.cli smoke        --config configs/pilot.yaml   # Gate 3 on the real frozen VLM
python -m clove.cli estimate     --config configs/pilot.yaml   # GPU-hours from measured smoke latency
python -m clove.cli run          --config configs/pilot.yaml --methods names,sage0,append,fresh,prism
python -m clove.cli analyze      --run-dir artifacts/runs/<RUN_ID>
```

Every command writes `status.json` (`COMPLETE` / `PARTIAL` / `BLOCKED`) and exits nonzero on failure. When
it is blocked, the file contains the exact resume command. `run` refuses to start without a passing
leakage report and an in-budget compute estimate. It writes `preregistration.json`, a hashed copy of the
stats/methods/gate config, before the first prediction, and refuses to continue if that changes. Predictions
are append-only JSONL keyed by `(image, variant, episode, method, order, seed, budget, inference-config)`.
Rerunning the same command resumes, and `compute.max_calls` stops the run cleanly with `PARTIAL`.

Staged expansion: `--orders O1_r0 --limit-images 5` first, then all orders, then the budget sweep
(`budgets.sweep_enabled: true`) and `shifts.enabled: [covariate]` (Gate 5).

### Serving the frozen model on Nova (example)

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct --max-model-len 8192 --limit-mm-per-prompt '{"image":1}' --port 8000
# configs/pilot.yaml: model.base_url = http://<node>:8000 ; set model.revision to the exact commit
```
`--limit-mm-per-prompt image=1` makes the server itself reject any second image.

### Claude via Claude Code headless (`configs/claude.yaml`)

The same data, manifest, orders, budgets and statistics as `pilot.yaml`; only the frozen backbone changes.
It follows SAGE's `open_agentic` convention: one `claude -p` subprocess per image, `stream-json` output, and
a cleaned environment (`CLAUDE*` vars and `ANTHROPIC_API_KEY` stripped, so the logged-in account is used).
Every call also runs with `--no-session-persistence --setting-sources "" --strict-mcp-config
--disable-slash-commands`, and the system prompt is *replaced* rather than appended, so Claude Code's own
long system prompt never enters the token count.

* `image_mode: inline` (default) sends the query photo as a base64 image block in one stream-json user
  message with **all tools disabled** (`--tools ""`). The model has no file access, so k_img = 0 holds by
  construction. Any tool use is treated as a violation.
* `image_mode: read_tool` is SAGE-style: the photo is copied under a neutral name into an empty temp working
  directory, and only `Read` is allowed. The trace is audited, and a Read of any other path becomes a
  `REFERENCE_IMAGE_VIOLATION` error.
* Accounting: `prompt_tokens` = input + cache-creation + cache-read tokens from the `result` event (exact,
  image included), `output_tokens` likewise, and `total_cost_usd` per call. b_KB is counted with the regex
  approximation because Claude's tokenizer is not public (`tokenizer_exact: false` in every record). The
  b_out cap is passed via `CLAUDE_CODE_MAX_OUTPUT_TOKENS` on a best-effort basis; actual output tokens are
  always recorded.
* There are no logprobs, so option scores are hard. Unknown detection is the explicit "0 = none of these"
  answer (`gate.mode: argmax_only`), and AUROC-type metrics reduce to a single operating point. Decoding is
  not seeded, and the response cache keeps the first answer.
* `compute.parallel` runs concurrent subprocesses. `compute.max_usd` is a hard stop: the run ends `PARTIAL`
  with a resume command. `estimate` projects USD and wall-clock from the real smoke calls and blocks if the
  projection exceeds `max_usd`.

```bash
python -m clove.cli smoke    --config configs/claude.yaml      # ~100 real calls; measures $/call and latency
python -m clove.cli estimate --config configs/claude.yaml
python -m clove.cli run      --config configs/claude.yaml --methods names,sage0,append,fresh,prism
```

Tests never call Claude: `tests/fixtures/fake_claude.py` stands in for the CLI.

### Local 2× Titan RTX with Qwen2.5-VL-3B (`configs/qwen3b_titan.yaml`)

```bash
# on the GPU machine (Linux, CUDA), after copying Research/code/ (clove + SAGE trees)
python3 -m venv .venv && .venv/bin/pip install -e '.[gpu,dev]' vllm
bash scripts/serve_qwen_titan.sh                      # vLLM on GPU0:8000 and GPU1:8001
python -m clove.cli audit        --config configs/qwen3b_titan.yaml
python -m clove.cli manifest     --config configs/qwen3b_titan.yaml   # freeze on the FULL image pool
python -m clove.cli test-leakage --config configs/qwen3b_titan.yaml
python -m clove.cli smoke        --config configs/qwen3b_titan.yaml
python -m clove.cli estimate     --config configs/qwen3b_titan.yaml   # GPU-hours from measured latency
python -m clove.cli run          --config configs/qwen3b_titan.yaml --methods names,sage0,append,fresh,prism
```

One server runs per GPU, because a 3B model fits on one card. The client spreads requests across both
(`model.base_urls`) with 16 requests in flight. Server addresses are kept out of the cache key, so the two
replicas share cached responses.

Turing-specific settings: `--dtype half` (no bf16 on Turing) and `max_pixels` = 802,816, about 1,024 visual
tokens per image. Both are recorded in the cache key.

Logprobs are available, so the unknown gate is threshold-calibrated on dev images. SigLIP retrieval runs on
GPU0 in the ~3.5 GB that vLLM leaves free at `GPU_UTIL=0.85`.

Back-of-envelope cost is about 120k calls for the full pool, roughly 4–8 h wall clock (8–16 GPU-hours) and
about $1 of electricity. `estimate` replaces this with measured numbers after `smoke`.

## 6. Statistics (pre-registered in `configs/stats.yaml`)

The primary endpoint is PRISM − append on old-class macro accuracy, averaged over t ≥ 1 at the primary
budget. It is tested with:
* a hierarchical paired bootstrap (orders, then class-stratified specimen groups; 10,000 resamples, 95%
  percentile CI, fixed seed),
* a cluster sign-flip permutation test,
* Holm correction over the primary family, and BH over the exploratory family.

Non-inferiority is claimed only if the lower CI bound clears the pre-registered −2 pp margin. Item-wise
McNemar is reported as well. Non-significance is never reported as equivalence.

## 7. Layout

```
configs/{pilot,budgets,stats}.yaml
src/clove/  contracts.py config.py util.py resources.py cli.py
  data/       audit manifest splits leakage transforms
  kb/         reader (stdlib xlsx) provenance inventory (ledger + AuthorizedKB)
  memory/     base (cards, pack_evidence, snapshot/restore) policy append fresh prism baselines snapshot
  retrieval/  text (TF-IDF) bm25 embedding (SigLIP / precomputed) selector (retrieve_active, scan_interference)
  inference/  backend (vLLM / HF / mock + ImageGuard) prompt decode token_meter
  orchestration/ episode (controller) policy (unknown gate) tools (8 protocol tools) trace (cache, logs)
  evaluation/ metrics continual ood statistics
  reports/    tables figures render
tests/{unit,leakage,statistics,integration}   scripts/00-04   artifacts/ (generated)
```
