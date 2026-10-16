# CLOVE: Token-Efficient Continual Open-World Visual Evaluation

**Soybean-only, zero-reference-image pilot | Version 1.0 | 16 October 2026**  
**Benchmark:** CLOVE — Continual Learning for Scalable Open-World Agricultural Visual Evaluation.  
**Proposed optional method:** PRISM — Provenance-aware Retrieval and Interference-Sensitive Memory.  
**Status:** experimental design and implementation instructions; *not* experimental findings.

> **Non-negotiable contract.** The backbone VLM is frozen. No fine-tuning, gradient updates, LoRA, image demonstrations, or labeled reference images at inference. Image-reference budget is **k_img = 0 at every stage and for every baseline**. Text evidence is retrieved only from existing SAGE class records that are authorized at the current episode. Build before executing; produce actual results only when files and hardware are verified.

## 1. Executive scientific specification

The original SAGE manuscript provides a large source-grounded crop disease image resource, class-specific symptom records, an anatomical index, and a diagnostic evaluation with both k=0 and reference-image comparisons. CLOVE **does not recollect these data**. It changes the *evaluation axis*: from a fixed class universe to an expanding class universe, with a fixed pretrained vision-language model and versioned external memory. The soybean pilot is intended to test (H1) unknown rejection, (H2) knowledge-only new-class acquisition, (H3) retention under changing class inventories, (H4) retrieval interference and memory-history effects, (H5) token/compute forgetting, and (H6) distribution robustness. These hypotheses are not claimed as established findings.

**Data provenance:** SAGE WACV manuscript, Sections 3–6, particularly its source-first disease registry, KB-consistent image filtering, and k=0 evaluation. The manuscript reports approximately 839K images, 335 crops, and 1,251 disease classes globally. The source reports *different soybean class counts in different settings*, so the exact eligible soybean class universe must be reconstructed from files, not guessed. Source-grounded KB claims must retain quote, URL, document ID, and canonical class ID. The manuscript itself acknowledges that filtering images using the same KB later used in evaluation may introduce selection bias.

**Related internal research context:** A separately located October 2026 lab experimental plan called *Learning to Spend Less Without Forgetting* studies a frozen reasoning model, an adaptive token-budget controller, and compute forgetting. CLOVE adapts the **evaluation distinction between accuracy forgetting and compute forgetting**, not that plan's datasets or assumption that a controller must be trained. The exact paper called “FastPaper” has not been verified from the available material; do not attribute methodology to it without locating its definitive manuscript.

### 1.1 What qualifies as novelty, and what does not

Creating a class-incremental split, adding a rejection threshold, running ordinary RAG, or renaming standard memory tools is not by itself a new contribution. The potentially testable contribution is *source-preserving, token-bounded continual memory with interference-sensitive retrieval* under a frozen VLM, evaluated against matched-inventory fresh-state controls. If this does not produce an observable memory-specific failure mode or a measurable gain over stronger baselines, report a negative pilot and do not make a novelty claim. A soybean pilot cannot establish cross-crop or cross-domain generalization.

## 2. Notation and formal task

Let $\mathcal{C}$ denote eligible canonical soybean classes and $\mathcal{D}=\{(x_i,y_i,g_i,s_i)\}$ the fixed evaluation images, labels, specimen/duplicate grouping $g_i$, and image source $s_i$. The immutable, already-published class knowledge registry is $\mathcal{K}=\{K_c:c\in\mathcal{C}\}$; each $K_c$ contains source-linked claims. At episode $t$, the allowed known classes are $\mathcal{C}_t$, unknown classes $\mathcal{U}_t=\mathcal{C}\setminus\mathcal{C}_t$, and acquisition batch $\Delta_t\subset\mathcal{U}_t$, so $\mathcal{C}_{t+1}=\mathcal{C}_t\cup\Delta_t$. The frozen visual-language model is $f_\theta$; $\theta$ never changes. The agent's derived, versioned retrieval state is $M_t$, which may be updated using authorized KB records, but test labels are never written to it.

The classifier returns $\hat y_t(x)\in\mathcal{C}_t\cup\{\bot\}$, where $\bot$ denotes unknown/abstain. `unresolved` may be a separate operational state, but its scored mapping must be specified in advance. The KB text retrieval budget $b_{KB}$ is distinct from max generation tokens $b_{out}$ and class candidate count $r$; the labeled visual-reference budget is identically $k_{img}=0$. The input photograph is allowed: “zero-reference” does **not** mean zero images overall.

An expanded inventory is harder because there are more candidate labels; it does not imply memory forgetting. Define two contrasts for an old-class test set $D_{old}$:

$$F_{exp}(t)=A(D_{old};\mathcal{C}_0,M_0)-A(D_{old};\mathcal{C}_t,M_t)$$

$$F_{memory}(t)=A(D_{old};\mathcal{C}_t,M_t^{fresh})-A(D_{old};\mathcal{C}_t,M_t^{persistent})$$

Here $M_t^{fresh}$ is reconstructed from the **same authorized knowledge** and evaluated with the **same input/generation budget** and same retrieval constraints. The second contrast isolates memory-history effects more directly; it still requires matched prompts, candidate policies, and inference settings. Standard continual learning backward transfer and worst-class forgetting are separate supplemental endpoints.

**Compute forgetting**, only for paired cases whose response remains correct (plus an unconditional complementary analysis), is:

$$F_{tok}(j,t)=\frac{T_{j,t}-T_{j,t_j}}{\max(1,T_{j,t_j})}$$

where $T$ is end-to-end charged token usage at the original acquisition episode $t_j$ and later time $t$. Report retrieval tokens, prompt tokens, generation tokens, repeated calls, and latency separately; do not hide longer prompts inside a smaller `max_new_tokens` number.

## 3. Core soybean-only dataset construction

**Audit first.** Inspect original soybean manifest, canonical labels, class aliases, KB completeness, anatomical metadata, source datasets, filtering flags, and duplicate groups. Confirm exact eligible class count and sample count per class. Freeze file hashes, original paths and all manifests before evaluation. Do not infer disease labels from VLM outputs. Do not use the original KB-filter step again to generate the evaluation labels.

**Preferred 25-class schedule if viable:** $T_0=10$ active classes, then three batches of five: 10→15→20→25. If there are fewer eligible classes, derive a feasible schedule and mark departures; no manufactured classes or invented images. Prefer >=3 independent acquisition episodes. All class orders are fixed and committed before inspecting final results.

**Partition hierarchy:** assign images to original specimens/capture sequences/source groups before test splitting. Exact hashes plus perceptual duplicate review separate development/held-out records. Each class's held-out images persist unchanged through all episodes. A class introduced at $T_2$ should have its *same* fixed images tested as unknown at $T_0,T_1$ and as newly known at $T_2,T_3$. Tests should never become memory updates.

**Orders:** (O1) stratified random; (O2) documented confusable-pair class B introduced after active A; (O3) comparable pairs with reversed/delayed ordering; (O4, optional) pathogen-group ordering. Hard pairs must come from an existing independently sourced pair list; otherwise use KB similarity only as a **proxy** and label it transparently. At least three, ideally five, independent randomized orders for an adequately powered expanded pilot; for very few classes, retain exploratory uncertainty.

**Unknown classes:** classes not yet released constitute *inventory unknown*, not necessarily model-pretraining-unseen biology. Hold back 2–5 external known-to-be-soybean stress classes if independently labeled and licensed; do not claim full end-state unknown testing if there are none. Differentiate `known-but-hard` from `not-in-inventory`, and `unsupported/poor-quality` from `truly new category`.

## 4. IID, OOD and orthogonal shifts

| Condition | Actual intervention | Ground truth and caution |
|:--|:--|:--|
| IID | Fixed soybean images from same source regime; specimen-disjoint | Known/new/unknown labels depend on episode |
| Semantic near-OOD | Withheld soybean disease confusable with active disease | Unknown until admission; define hard pairs independently |
| Semantic far-OOD | Independently labeled dissimilar crop/stress images | May be easy to reject from crop ID alone; report separately |
| Source OOD | Soybean images from an independent collection or acquisition source | Only if source metadata genuinely verifies independence |
| Natural field OOD | New field/site/year/camera, if metadata permits | Do not invent geographic or time splits |
| Synthetic covariate shift | Brightness, contrast, mild blur, compression, resolution, white balance | Pre-register strengths; identify truly label-preserving versus obscuring transforms |
| Information degradation | Severe blur, occlusion, cropping away cues | Analyze as evidence loss, not automatically benign OOD |
| Unknown prevalence shift | 10%, 25%, 50% unknown mix | Keep predictions and calibration threshold fixed; report prevalence |
| Memory / temporal OOD | Changed class inventory and aged external agent state | Separate from image distribution OOD |

For each image keep `origin_image_id`, `variant_id`, `source_dataset`, `group_id`, `episode`, `ground_truth`, `is_known_at_t`, `shift_type`, `severity`, and `rights_status`. Never use converted variants across training/calibration and test partitions as independent specimens. Covariate transforms must use exactly the same original test image and paired method conditions.

## 5. Explicit token-optimized inference contract

**Hard constraints:** $k_{img}=0$; no labeled reference image loading; no montage, retrieved prototype image, image similarity exemplar, or agent image-to-image comparison as a proxy. One incoming query image only. No vision calls for deterministic taxonomy, registry updates, cost accounting, caching, or statistics. One VLM call for a normal prediction and a second call **only when explicitly permitted** by a pre-registered uncertainty policy; baseline comparisons must match caps.

**Default budgets for pilot (to be audited, not assumed optimal):** class-name filter produces at most `r=4` candidate KB records; evidence excerpts each at most 96 tokenizer tokens; total KB evidence <=512 input tokens; general instructions <=180 tokens; maximum new tokens 96; agent calls <=1 main path, <=2 conditional; no visual few-shot tokens whatsoever. Optional budget sweep: 128/256/512/1024 KB tokens, 1/2/4/8 text candidates, 48/96/192 generation tokens. Also evaluate a larger context upper-bound only on an affordable, fixed subset. *Budget sweep measurements are empirical; “under 512” is a target verified with the actual tokenizer.*

Avoid model-generated verbose CoT. Require concise output in strict JSON `{label_id, decision, confidence_optional, evidence_ids}`. If logits available, prefer score-based unknown detection without generating an explanatory paragraph. The model need not expose private reasoning or write long textual traces. Log the externally observable tool trace and KB IDs. Unknown detection and class selection can share the same frozen image encoding when engine supports reuse, otherwise count each actual call.

**Caching:** cache source documents; normalized KB claims; tokenized per-class summaries (by class/version/tokenizer); static text embeddings; eligible index snapshots; deterministic retrieval query outputs; hashes; and model predictions *only when image hash, input token sequence, active-class set, model revision, decoding, and retrieval state are byte-for-byte equivalent*. In particular, never reuse a prediction across different memory methods merely because the query image is the same.

**Token cost:** `$C=\sum_v(p^{in}_v N^{in}_v+p^{out}_v N^{out}_v)$` for hosted models where prices exist, and `GPU_seconds`, tokens, peak VRAM and elapsed time for locally hosted models. Store raw tokenizer counts and report mean, median, 90th percentile, total per run, and correct-resolutions-per-1M-input-tokens. Prompt-length-matched baselines are compulsory: a shorter PRISM input may improve efficiency without proving improved knowledge retention.

**Compute forgetting:** use matched image/episode comparisons to test whether same-correct old tasks require more tokens at later stages; supplement with all-case total token cost so cherry-picking correct-only tasks is impossible. A prompt with a capped 96-token response could still explode in prompt tokens: measure both.

## 6. Research tool suite, algorithm and failure semantics

The full original SAGE KB is immutable and access-controlled by episode. Tools are pure or versioned functions with typed interfaces, bounded outputs, explicit failure codes, and unit tests.

| Tool | Contract | Key risk |
|:--|:--|:--|
| `admit_class(id,episode)` | only authorized new ID → activate existing KB claim IDs | future-class leakage |
| `retrieve_active(image_features,top_r,b)` | only active IDs → bounded cited snippets | wrong-class omission |
| `scan_interference(class_id)` | source-claim overlap and documented contrasts | omissions misread as contradictions |
| `compile_memory(B,policy)` | immutable claims → context-limited view | lossy compression |
| `gate_unknown(scores,threshold)` | calibration-defined known/unknown decision | OOD false accepts |
| `snapshot()` / `restore()` | version derived state; never change authorized class ledger | accidental removal of acquired class |
| `budget_meter()` | tokenizer-exact prompt + generation accounting | illusory savings |
| `audit_episode()` | verify allowed class IDs, citations, hashes, manifest | hidden class contamination |

**PRISM hypothesis.** Allocate limited KB context to maximize *retrieval coverage of likely hard competitor classes* and *source-supported discriminative information* while preserving coverage for old classes. Use existing disease organ and symptom fields as inputs, not invented absence statements. A safe heuristic is:

$$u(c|x,t)=\alpha s_{visual}(x,c)+\beta s_{text}(q_x,K_c)+\gamma p_{protect}(c,t),$$

subject to $\sum_c b_c\le B$, $b_c\ge0$, $c\in\mathcal{C}_t$. Coefficients are **development-calibrated** or fixed before final testing, never tuned on held-out images. $p_{protect}$ can depend on previous memory coverage and class-confusion priors from *development* data, not test errors. Baseline PRISM implementation should use deterministic greedy, source-preserving selection before proposing anything learned. Contrastive claims must be source-backed: field absent in B does not imply medically or biologically absent from B.

The algorithm's performance should be judged against append-only with identical retrieval caps, random subset, recency subset, uniform summarization, BM25, embedding retrieval, and a fresh-state full-authorized-KB oracle. If PRISM cannot outperform these at matched token cost, it remains an exploratory method.

## 7. Executable staged pilot and comparison plan

**Gate 0: corpus feasibility** — at least enough KB-backed soybean classes and held-out images for a meaningful stream; otherwise stop and write an evidence-based feasibility report.

**Gate 1: environment tests** — establish Python environment, model checkpoint/version, image decode, tokenizer, CPU-only manifest runs, GPU capacity, and instrumented single-image inference. No remote upload or network call without configured credentials/permission.

**Gate 2: simulation without VLM** — 3 synthetic dummy records solely for unit testing authorization, unknown labels, admission, rollback, source citations, token budgeting and metrics. Dummy records must never enter evaluation.

**Gate 3: micro-run** — 2–4 eligible classes, 2–5 genuine held-out images per class; compare fixed current KB, append-only, and fresh-state reconstruction. Ensure test results are saved.

**Gate 4: full soybean pilot** — preferred 25-class/4-stage schedule; minimum viable: enough images and stages to study transfer; reserve confirmation-style hypotheses before looking at results. Prioritize 3 primary baseline methods, 3 orders, and 1 robust budget before launching a combinatorial matrix.

**Gate 5: extensions** — add source-OOD if genuine, covariate severity, extra budgets, memory corruption, multiple model sizes and optional autonomous discovery. Run only after primary pipeline is valid.

### 7.1 Baselines: strict priority

| Tier | Method | Distinguishes |
|:--|:--|:--|
| Essential | Class names + unknown gate | pretrained prior / no KB |
| Essential | Original SAGE k=0 with active KB | fixed-resource reference |
| Essential | Append-only RAG at fixed evidence budget | naive continual |
| Essential | Fresh-state rebuilt from authorized KB at every t | separates memory history |
| Essential | PRISM at exactly matched budgets | proposed memory improvement |
| Diagnostic | Oracle in-inventory candidate retrieval (known only) | retriever vs recognizer ceiling |
| Secondary | Uniform compression / recency / BM25 / embeddings | mechanism ablations |
| Negative controls | matched length shuffled KB, unrelated KB, name-only | evidence provenance / prior dependence |

**Ablations:** without interference scan, without protection, without token cap, without organ index, without full provenance field, without unknown gate; evaluate cost and accuracy changes. Keep ablations methodologically meaningful: removing unknown gate changes prediction space and therefore may not be directly comparable for unknown metrics.

### 7.2 Mandatory factorial axes and stage outcome

For each authorized class order and seed, run episode $t$ × evaluation split × method × approved token budget. Store one immutable prediction record per `(image_id, image_variant, stage, method, order_id, seed, inference_config_hash)` including class set fingerprint, retrieval list, input/output token counts, number of VLM calls, raw response location, timeout and error status.

Measure three separate trajectories: **old-class performance**, **newly admitted class performance**, and **not-yet-admitted unknown rejection**. At the final stage with no hidden soybean classes, do not report soybean-internal unknown rejection unless separate valid hidden classes remain. Include an explicit `no_eligible_unknowns` flag in that case.

## 8. Metrics and mathematically correct evaluation

**Known recognition:** accuracy, macro-F1, balanced accuracy and per-class recall on inputs from $\mathcal{C}_t$. A rejection on a known image is incorrect for overall open-world accuracy; selective-classification curves separately measure accepted-case accuracy.

**Unknown detection:** AUROC, AUPR-In, AUPR-Out, FPR@95% known recall (define score orientation explicitly), TNR at fixed known TPR, OSCR, area under risk-coverage curve. Unknown AUROC is undefined when no eligible unknown examples; mark `NA` rather than `0`.

**Acquisition:** for each new class $c$ admitted at $t_c$, evaluate $Q_c^{pre}=P(\hat y=\bot\mid y=c,t<t_c)$ and $Q_c^{post}=P(\hat y=c\mid y=c,t\ge t_c)$ as **different endpoints**, not the subtraction of different definitions. Track retention after additional acquisitions.

**Accuracy forgetting:** class-wise $F_c=\max_{\tau\in[t_c,T]}a_{\tau,c}-a_{T,c}$, with zero-denominator handling and fixed images. Report average and worst-class, plus standard BWT with a clearly defined acquisition stage.

**Memory forgetting:** $F_{memory}(t)$ using persistent vs fresh memory at matched active inventory, retrieval and inference budgets. Report signed effect and paired CI. Separate class-set growth from state history and context truncation effects.

**Compute forgetting:** token growth after class acquisition on paired old-domain images; separate prompt growth, output growth, extra VLM calls and wall-clock cost. For correctness-matched comparison, include counts and full-population aggregate.

**Efficiency:** accuracy vs full input+output tokens, source retrieval bytes, tool calls, end-to-end time, GPU seconds. Define normalized cost per correct answer: `total_tokens / max(1,num_correct)`; do not evaluate solely on cost per attempted image.

**Calibration:** expected calibration error and Brier only if an operationally meaningful probability score is available. Do not interpret free-form 0.95 VLM confidence text as calibrated probability.

## 9. Statistics, experimental dependence, and sample size

**Primary confirmatory endpoint:** difference in matched-inventory old-class macro accuracy (PRISM minus append-only) averaged over predesignated post-acquisition episodes, with one prespecified memory token budget. **Co-primary only if explicitly preregistered:** token cost relative reduction subject to a prechosen noninferiority margin for old-class accuracy. Do not promote whichever metric happens to win.

- Predictions must be paired by image, stage, shift, and class order. Use paired permutation/McNemar for binary per-image outcomes where suitable; sample-dependent metrics should use paired bootstrap.
- When images share capture sequence, source collection, leaf, or specimen, resample at the largest defensible independent unit; optionally stratify by class before resampling.
- Hierarchical bootstrap: sample independent class-order seeds, then class/specimen groups; note that three seeds do not produce strong between-order precision.
- AUROC: use paired bootstrap; DeLong where independent-subject assumptions apply. Confidence intervals can fail at small near-OOD positive counts; disclose counts and instability.
- Use **95% two-sided CIs**, raw effect size, and adjusted p-values. Holm correction across the *predeclared primary comparison family*. Exploratory analyses use BH-FDR with the family labeled exploratory.
- Statistical non-significance **does not establish equivalence**. To claim comparable accuracy with fewer tokens, define a noninferiority margin (for example −2 percentage points, justified with development power analysis) and test it formally; do not silently take a convenient margin after seeing results.
- Report per-class sample counts, class-level variability, successful run fraction, missing predictions, confidence intervals, and available effective independent cluster count. Do not treat repeated class-order reruns as independent biological observations.
- For 25 classes, a small pilot may be underpowered to claim universal improvement; report estimation uncertainty, not guaranteed significance. A prospective power/precision analysis should use development or pilot stage data only; final test results remain confirmatory if pre-registered.
- Target 10,000 paired clustered bootstrap resamples for finalized tables, with deterministic seeds and confidence interval method recorded. Use exact or permutation tests when counts are too small.

Report predetermined failure codes: `RETRIEVAL_MISS`, `KNOWN_FALSE_UNKNOWN`, `UNKNOWN_FALSE_KNOWN`, `CLASS_COMPETITION`, `CONTEXT_OVERFLOW`, `LOSSY_MEMORY`, `CITATION_MISMATCH`, `FUTURE_LEAKAGE`, `OUTPUT_PARSE`, `OOM`, `TIMEOUT`, `INVALID_IMAGE`, `UNRESOLVED`, `UNKNOWN_ROOT_CAUSE`. Attribution of failure mechanism requires logs, not model explanations alone.

## 10. Engineering architecture — small, isolated, auditable modules

Use **src-layout, typed Python, single-responsibility packages, explicit interfaces, immutable source data, small vertical increments, continuous tests**. “SAIZ/Agile style” is interpreted as a disciplined incremental modular architecture; no formal lab-specific SAIZ convention was provided. Do not pretend to know undocumented local standards. Adopt existing SAGE repository conventions where compatible.

```text
clove/
  README.md                     # precise task notation, usage, resource contract
  pyproject.toml                # package + pinned dependency ranges
  Makefile                      # audit, test, smoke, pilot, report
  configs/{pilot.yaml,budgets.yaml,stats.yaml}
  src/clove/
    contracts.py                # dataclasses, explicit enums and schemas
    data/{audit.py,manifest.py,splits.py,leakage.py,transforms.py}
    kb/{reader.py,provenance.py,inventory.py}
    memory/{base.py,append.py,fresh.py,prism.py,snapshot.py}
    retrieval/{bm25.py,embedding.py,selector.py}
    inference/{backend.py,prompt.py,decode.py,token_meter.py}
    orchestration/{episode.py,policy.py,trace.py}
    evaluation/{metrics.py,continual.py,ood.py,statistics.py}
    reports/{figures.py,tables.py,render.py}
    cli.py
  tests/{unit,integration,leakage,statistics}
  scripts/{00_audit.py,01_build_manifest.py,02_smoke.py,03_run_pilot.py,04_analyze.py}
  assets/                        # source-paper references only, not copyrighted replicas
  artifacts/{manifests,orders,predictions,memory,metrics,figures,logs,reports}
```

**Design principles:** Use interfaces `KBReader`, `MemoryPolicy`, `Retriever`, `VLMBackend`, `UnknownGate`, and `TokenMeter`; no module should independently read hidden KBs. The episode controller should issue an `AccessScope` token specifying allowed class IDs and stage; all retrieval calls enforce it. Avoid frameworks requiring expensive multi-agent prompting; the default orchestrator is code-driven and makes a single VLM inference call. CPU-only tests should work without a model checkpoint. All variable external inputs are documented in `README.md` and `configs/`.

**CLI contract:**

```bash
python -m clove.cli audit --config configs/pilot.yaml
python -m clove.cli manifest --config configs/pilot.yaml
python -m clove.cli test-leakage --config configs/pilot.yaml
python -m clove.cli smoke --config configs/pilot.yaml
python -m clove.cli run --config configs/pilot.yaml --methods names,sage0,append,fresh,prism
python -m clove.cli analyze --run-dir artifacts/<RUN_ID>
```

Each command returns nonzero on failure, writes a machine-readable `status.json` (`COMPLETE/PARTIAL/BLOCKED`) and never silently uses fabricated fallback data. Resume idempotently using stable keys; per-image predictions are append-only parquet/JSONL with a deterministic collision policy.

## 11. Minimal code contracts — to be implemented and tested

```python
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

class Decision(str, Enum):
    KNOWN = 'KNOWN'
    UNKNOWN = 'UNKNOWN'
    UNRESOLVED = 'UNRESOLVED'

@dataclass(frozen=True)
class AccessScope:
    episode: int
    allowed_class_ids: frozenset[str]
    inventory_hash: str

@dataclass(frozen=True)
class Claim:
    claim_id: str
    class_id: str
    text: str
    verbatim_quote: str
    source_url: str

@dataclass(frozen=True)
class Prediction:
    image_id: str
    episode: int
    decision: Decision
    label_id: str | None
    evidence_ids: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    vlm_calls: int

class MemoryPolicy(Protocol):
    def update(self, scope: AccessScope, new_classes: tuple[str,...]) -> None: ...
    def select(self, scope: AccessScope, query: str, token_budget: int) -> tuple[Claim,...]: ...

class VLMBackend(Protocol):
    def infer(self, image_path: str, prompt: str, max_new_tokens: int) -> str: ...
```

**Leakage-safe invariant:** For any response, every `claim.class_id` and `prediction.label_id` (unless unknown) must belong to `scope.allowed_class_ids`. A future-class *ground truth* can appear in evaluator-only metadata but never in prompts, index searches, hidden file paths exposed to tools, or debug messages. Test the invariant with a deliberately hidden synthetic class.

**Token-bound evidence packing:** tokenizer counts include all delimiters, identifiers and source fields. If an excerpt cannot fit, drop the lowest-ranked full claim rather than truncating citations unpredictably. Never truncate JSON output schema instructions. Log unfilled capacity; strict cap violations are errors.

```python
def pack_evidence(ranked_claims, scope, tokenizer, max_tokens):
    kept, used = [], 0
    for claim in ranked_claims:
        if claim.class_id not in scope.allowed_class_ids:
            raise PermissionError('future-class KB leak')
        line = f'[{claim.claim_id}] {claim.class_id}: {claim.text}'
        cost = len(tokenizer.encode(line, add_special_tokens=False))
        if used + cost <= max_tokens:
            kept.append((claim, line)); used += cost
    return kept, used
```

This snippet is a **reference implementation for one bounded operation**, not a complete inference engine. The full repository must add accurate chat-template token accounting, image-token usage when exposed by backend, validation and tests.

## 12. Experiment and compute strategy

Do not run 20 × 5 × 4 × all token budgets × 25 classes with a 32B VLM blindly. First identify image count $N$, episodes $E$, methods $M$, orders $O$, budgets $B$, allowed calls $V$, and measured mean per-call time $\bar t$:

$$\widehat{GPUh}=\frac{NEMOBV\bar t}{3600\cdot\eta}$$

where $\eta$ represents effective parallel utilization, *not* a guaranteed speedup; correct for method/episode reuse only after checking exact prompt equivalence. Print an evaluation-count manifest and upper/lower time bounds before any long GPU run.

**Recommended first order:** (1) data audit, (2) unit tests, (3) 12-image smoke, (4) 100–200 distinct-image primary pairing, (5) complete soybean primary endpoint, (6) confirmatory intervals, (7) one OOD axis, (8) extra budgets and secondary ablations. The protocol must adapt to actual available images, not fabricate target counts. Do not start long jobs without a resource/cost ceiling in configuration. Do not kill unrelated GPU tasks. Checkpoint after every batch; fail closed and resume on crash.

**FastPaper-inspired cost experiment (with attribution caveat):** test *compute forgetting* independently of answer forgetting. A fixed-budget frozen model can still consume increasing **prompt** tokens as memory grows. If using an adaptive inference-budget controller, trainable controllers constitute a *separate* experiment and must be clearly flagged as violating the strict no-training PRIMARY setup; recommended primary protocol uses a deterministic, development-calibrated nonlearning policy.

## 13. Publication figures, tables and release readiness

**Core figures:** (F1) task timeline and class-universe growth, (F2) frozen-VLM single-image inference with bounded KB retrieval and persistent memory, (F3) old/new/unknown performance across stages, (F4) fresh-vs-persistent memory controlled contrast, (F5) near-unknown ROC and risk-coverage, (F6) correct-answer rate versus total token cost, (F7) compute forgetting on old cases, (F8) confusable-class error transitions, (F9) one genuine source-OOD axis if supported. All uncertainty bands from actual saved predictions. Do not make decorative estimates look like data.

**Core tables:** dataset eligibility and per-class sample counts; class order manifest; all methods at prespecified budget; paired effect sizes, clustered confidence intervals and adjusted p-values; memory token compliance and per-tool costs; failure analysis and missing/invalid images. Include row counts and number of independent specimen groups.

**Reproducibility package:** immutable manifest and SHA256, source citation and license ledger, configs, checkpoint revision, backend versions, GPU model, exact prompts/templates, all raw prediction records, retrieval traces, unknown score and threshold, statistical scripts, figure source and README reproduction commands. No public release of third-party images without verifying license conditions. Do not mislabel inference model's internal explanation as a faithful causal trace.

**Go/no-go decision:** (a) unknown rejection must be better than trivial baselines on near-OOD, (b) KB acquisition must measurably help identification, (c) persistent-versus-fresh contrast must establish whether a memory-history problem exists, (d) PRISM must be compared at matched token cost, and (e) data leakage audits must pass. If any fail, revise research claim rather than force a positive result.

---

# Appendix A — Two-stage autonomous Claude Code master prompt

Paste the prompt below into Claude Code *after placing it in the SAGE repository*. It is intended to autonomously set up the codebase first and then run the soybean pilot. No instructions can guarantee that a model agent has GPU allocation, network permissions or the required local files; the agent must report specific blockers and never claim work it has not performed.

```text
ROLE: You are a lead continual-learning researcher, expert agricultural computer vision engineer, codebase maintainer, experimental statistician, and reproducibility auditor. Your task is to create a complete CLOVE soybean-only pilot in the EXISTING SAGE project. Treat the attached CLOVE protocol PDF/Markdown and the original SAGE WACV paper as controlling scientific specifications. Do not change the SAGE originals.

MISSION: Deliver a code-driven, aggressively token-optimized, training-free, ZERO-REFERENCE-IMAGE continual open-world agent. The published SAGE class KB already exists; DO NOT crawl the web or regenerate it. k_img=0 for every run. Qwen3-VL-32B frozen if available. No image exemplars, no in-context few-shot examples, no model fine-tuning, no test-time gradient learning. The only incoming image in a prediction is the query image itself. Use concise JSON outputs, deterministic acquisition orchestration, exact tokenizer instrumentation, cached static records and single-call default inference.

STAGE 1 — ENGINEERING BEFORE EXPERIMENTS
1. Inventory repository, soybean classes, image IDs/paths, KB schema/citations, original KB-filter flags, image source metadata, code and GPUs. Write reports/audit.md and environment report with real counts. If manuscript counts and manifests differ, trust verified current files and explain discrepancies.
2. Read existing conventions and create clove/ src-layout modular package. Use typed interfaces, small modules, Pydantic or dataclasses, config-driven boundaries, deterministic hashes, `pyproject.toml`, `Makefile`, README notation and CLI. Minimize unnecessary agent frameworks and VLM API calls. Do not overwrite SAGE code.
3. Implement canonical manifests, specimen-aware splitting, dataset fingerprint, source/duplicate audit, class-order generation, episode AccessScope, immutable original KB, versioned active inventory, fresh and persistent memory states.
4. Implement tools: admit_class, retrieve_active, scan_interference, compile_memory, gate_unknown, snapshot, restore, budget_meter, audit_episode. Make operations truly functional. Reject hidden-class KB requests. Preserve per-claim source URLs and verbatim evidence.
5. Implement inference adapter for locally available frozen backbone, prompt template, structured JSON decoder, calibrated unknown gate and exact input/output token metering. Enforce no reference-image retrieval. A normal evaluation prediction should use at most one VLM call. More calls are opt-in and capped.
6. Add baselines: names-only; SAGE k=0; append-only limited RAG; fresh-state reconstruction; PRISM; BM25/uniform compression if feasible. Keep the same eligible class inventory and budgets across comparisons.
7. Write unit tests for all data and algorithm contracts, statistics formulas, frozen weights, no future-class access, no reference images, token cap, deterministic sequences, seed handling and resumption. CPU-only synthetic unit tests are allowed but synthetic data must never contaminate real benchmarks.
8. Document every symbol, all three notions of token budget, what counts as unknown, data leakage, known/new/unknown evaluation denominators, and every CLI command in README with exact example invocation.
9. DO NOT proceed to full inference until tests pass and a human-readable resource estimate is written. Run smoke tests with actual soybean images.

STAGE 2 — EXECUTE SOYBEAN PILOT AUTONOMOUSLY
10. Build eligible class stream from files: target 10 -> 15 -> 20 -> 25 if at least 25 quality-verified soybean classes exist; otherwise formulate an honest feasible schedule. Keep all held-out images constant across stages; use predesignated class orders random and confusable introduction. Entire future-class KB inaccessible until admitted.
11. Pre-register primary method contrast PRISM versus append-only RAG at identical context/token/tool budgets; primary metric matched-inventory old-class retention. Secondary unknown AUROC/OSCR, new-class identification, memory-specific forgetting, compute forgetting, source/synthetic-covariate OOD, and latency. Never select significance tests after viewing a winning result.
12. Verify hardware/model; run one-image test; profile real token and wall-clock costs; report estimated total GPU hours and stop if above a configurable approved budget. Do not monopolize, terminate, or preempt other users' jobs. Save resumable run plan.
13. Run smallest real end-to-end task first, inspect failures. Then run primary minimal method × episode × image pairs, storing predictions and exact prompts by hashes. Expand sequentially only if baseline integrity and unknown calibration are sound.
14. Run genuine IID, semantic near-OOD, semantic far-OOD and source-OOD only when metadata/data support them. Generate predeclared paired covariate perturbations from originals without leaking across partitions. Clearly mark unsupported OOD modes as NOT AVAILABLE.
15. Repeat exact old images after each knowledge acquisition. Compare A: frozen previous inventory, B: fresh expanded inventory, C: persistent append-only, D: PRISM with identical expanded inventory. Never call A-vs-D accuracy gap purely catastrophic forgetting. Evaluate both memory-history loss and compute-forgetting cost increases.
16. Add token budgets 128/256/512/1024 and output caps 48/96/192 only after stable primary evaluation. Distinguish per-class KB records, retrieved evidence tokens, prompt tokens and generation tokens. All runs k_img=0. Cache only identical inference inputs.
17. Analyze paired per-image records using clustered bootstrap (at least 10k resamples where feasible); paired McNemar or permutation tests; paired AUROC intervals; predeclared Holm correction and exploratory BH. Report macro accuracy, per-class variance, unknown prevalence effects, 95% CI and meaningful effect sizes. Non-significant improvement does not establish equivalence; test noninferiority separately with prechosen margin if claiming similar accuracy.
18. Produce actual plots and machine-readable tables: retention/acquisition/unknown curves, fresh versus persistent, hard-pair confusion, compute forgetting, accuracy-vs-tokens, OOD robustness, provenance and failure logs. Never fabricate results, axes or statistical significance.
19. Create clear reports: dataset audit, algorithm/mathematics, implementation validation, experimental matrices, compute report, actual pilot findings, negative findings, limitations and CVPR readiness. Each report records COMPLETE/PARTIAL/BLOCKED status and paths to evidence.
20. Perform final scientific audit: class-manifest leakage, exclusion of references, frozen weights, no test-label adaptation, no double dipping between KB-based filtering and evaluation, all code reproducible from fresh checkout. Retain all run configs and exact environment versions. Summarize whether CLOVE is scientifically justified and whether PRISM outperforms strong baselines at matched tokens.

AUTONOMY: Do not repeatedly ask for trivial decisions; use justified safe defaults and log them. You may write code, run tests and compute within existing permissions. You MUST NOT invent missing directories, data, accessible GPUs, permissions, source labels, or successful results. On a blocker, produce the exact needed input/permission and a runnable resume command. Do not submit papers, modify external SAGE assets, or publish anything.

FINAL: A clean incremental codebase + README with full notation + tested one-command workflow + completed reproducible soybean pilot to the degree hardware and data permit + empirical reports with 95% CI and token accounting. Start by inspecting the actual SAGE repository, not by writing speculative code.
```

## Appendix B — Reviewer-facing checklist and known scientific hazards

1. **Semantic unknown is inventory-relative:** the pretrained VLM may already know disease names; do not claim truly unseen pretrained classes.
2. **SAGE filtering circularity:** source-based image filtering may preferentially select visually KB-consistent samples. Require a non-KB-filtered independent set if available and highlight this limitation everywhere.
3. **Class count inconsistency:** verify soybean counts by manifest, not manuscript table heading.
4. **No real gradient forgetting:** any deterioration is agent-level behavioral forgetting / retrieval interference.
5. **Token objective can game itself:** a model can refuse everything cheaply; always pair cost with known accuracy and unknown rejection.
6. **Single incoming image only:** source-grounded text ≠ reference visual examples.
7. **Near OOD may be harder than far OOD:** report separately, include hard-pair exposure order.
8. **Memory selection cannot read evaluation labels:** safeguard all internal APIs.
9. **No evidence of source OOD without provenance:** random splits are not source OOD.
10. **Tiny pilot is exploratory:** significant p-values alone do not establish effect generality across soybean, crops, or models.
11. **Claim-level biological validity:** differences between KB entries may indicate missing documentation, not true differentiating traits.
12. **Properly account for image tokens:** if backend does not expose their counts, report a separately explained image-token estimate and instrument actual GPU cost; do not assert zero visual tokens just because visual references are absent.
13. **Controller training is outside the core contract:** any adaptive learned-budget model belongs to optional explicitly separate experiments.
14. **Comparison to strong retrieval baselines:** PRISM benefit must survive fresh-state, same-inventory, matched-cost comparisons.
15. **Report negative findings:** if no memory-history interference appears, say so and simplify method.

## Appendix C — Proposed release gates

| Gate | Required artifact | Pass criterion |
|:--|:--|:--|
| Data | verified soybean inventory and hashes | no duplicate/label leakage and sufficient classes |
| Code | test suite and frozen checkpoint | zero reference-image loads; all policy tests pass |
| Compute | profiler output | measured pilot within approved budget |
| Benchmark | episode manifest | correct active/unknown scope at every stage |
| Method | matched-cost baseline results | observable effect and rigorous uncertainty |
| Statistics | paired predictions and cluster IDs | independently rerunnable tests |
| Paper | verified results + limitations | claims match evidence and scope |

**Bottom line:** The CLOVE pilot must be *a controlled, token-audited study of knowledge-only category acquisition and agent memory history*. Build the minimal system first, run soybean only, and scale only if the measurable mechanism survives comparison to a fresh-state, full-authorized-KB agent. A negative outcome is not a software failure; it is a scientifically legitimate decision point.
