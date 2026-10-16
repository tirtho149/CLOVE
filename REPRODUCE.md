# Reproducing CLOVE (soybean pilot)

This file says how to rebuild every CLOVE result that exists as of 2026-10-16. The method itself is in
`README.md`, and the specification it implements is `docs/CLOVE_PROTOCOL.md` (v1.0, 16 Oct 2026).

**Status.** There are no findings yet. What exists is two Sonnet smoke runs and one partial Sonnet pilot run
on a laptop subset of the data. They check the pipeline and costs. They are not results. The full pilot is
meant to run on Nova (Qwen3-VL via vLLM, full image pool, manifest frozen there).

## 1. Bundle contents

The zip `clove_<commit>_repro.zip` has three top-level folders:

| path | what | source |
|:--|:--|:--|
| `clove/` | the full repository at the named commit: code, configs, tests, this file, the protocol, `requirements-lock.txt` | `git archive` |
| `records/laptop_partial/` | everything the 2026-10-16 laptop runs wrote: audit, frozen subset manifest, class orders, response cache, the three runs with raw responses | `clove/artifacts/laptop_partial/` (git-ignored) |
| `inputs/` | the SAGE text inputs CLOVE reads (WACV manifest, KB xlsx, provenance quotes, and the optional GitHub registry) with SHA-256 sums in `inputs/SHA256SUMS` | `code/sage-nova-mirror/…`, `tirtho149/SAGE` |

**Not included: the images.** The image pool is about 8 GB of content-addressed `<sha1>.jpg` files in
`code/sage-nova-mirror/dup_audit/scale_images`. It was copied from Nova to the laptop over Globus on
2026-10-15. The exact image ids each run used are listed in `records/laptop_partial/manifests/manifest.json`
and in every prediction record (`image_id`).

## 2. Environment

| item | value used |
|:--|:--|
| machine | MacBook, macOS 26.6, arm64 |
| Python | 3.12.14 |
| Python packages | `requirements-lock.txt` (numpy 2.5.4, PyYAML 6.0.3, pillow 12.3.0, pytest 9.1.1) |
| Claude Code CLI | 2.1.301, logged in to the user's account (`strip_api_key: true`, so no API key is used) |
| Claude model | `claude-sonnet-5-5`, `effort: low`, `image_mode: inline`, all tools disabled |

```bash
cd clove
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest          # expected: 57 passed (CPU only, never calls Claude)
```

For the GPU backends (vLLM / Qwen), install `.[gpu]` and `vllm` as described in README section 5.

## 3. Directory layout the configs expect

The paths in `configs/pilot.yaml` are relative to `clove/configs/`:

```
code/
  clove/                                   <- this repo
  sage-nova-mirror/
    wacv2027_nova/data/manifest/soybean.json
    wacv2027_nova/data/part_index/soybean_sources.json
    wacv2027_kb/soybean/internet.xlsx
    dup_audit/scale_images/<sha1>.jpg      <- images (not in the zip)
  sage-wacv2027/data/images/               <- second image dir; absent on the laptop, ignored if missing
  SAGE/disease_registry/outputs/Soybean/final_registry.json   <- optional (git clone tirtho149/SAGE)
```

To rebuild this layout from the zip, copy each file in `inputs/` to the path shown in `inputs/PATHS.txt`.
You can also leave the files where they are and pass a `--override` with absolute paths, for example
`--override '{"paths": {"kb_xlsx": "/abs/inputs/internet.xlsx", ...}}'`.

**Note on the laptop runs:** `SAGE/` was not cloned on the laptop, so the three runs in `records/` ran
**without** the GitHub registry (no registry `quote` claims or `look_alikes`). They used the WACV KB and the
provenance quotes only. To reproduce them exactly, leave the registry out.

## 4. Reproducing the laptop runs (Claude Sonnet)

All three runs used `configs/claude.yaml` with artifacts redirected to `artifacts/laptop_partial`:

```bash
OV='{"paths": {"artifacts": "'$PWD'/artifacts/laptop_partial"}}'

# Gates 0-2: audit, freeze the subset manifest + class orders, leakage tests
for c in audit manifest test-leakage; do
  .venv/bin/python -m clove.cli $c --config configs/claude.yaml --override "$OV"
done

# Gate 3: real smoke (~102 calls)
.venv/bin/python -m clove.cli smoke    --config configs/claude.yaml --override "$OV"
.venv/bin/python -m clove.cli estimate --config configs/claude.yaml --override "$OV"

# Gate 4: one class order, $60 hard stop (this is the run that was stopped)
OV60='{"paths": {"artifacts": "'$PWD'/artifacts/laptop_partial"}, "compute": {"max_usd": 60}}'
.venv/bin/python -m clove.cli run --config configs/claude.yaml --override "$OV60" \
    --orders O1_r0 --run-id sonnet_O1_r0
```

The freeze must give the same fingerprints. `manifest` refuses to overwrite a different freeze.

| fingerprint | value |
|:--|:--|
| manifest | `4a7352212574ea94` |
| KB | `d5ed34ee7872f1ec` |
| episode plan | `4a2d518fa95abf71` (4 stages of 4 classes; orders O1_r0, O1_r1, O1_r2, O2_confusable, O3_reversed) |
| run config hash (`sonnet_O1_r0`) | `241708e929ec8ce2` |

**Replaying without paying.** Copy `records/laptop_partial/` to `clove/artifacts/laptop_partial/`. Then the
commands above hit the response cache (`cache/`) for every call that was already made, and give back the
same answers at no cost. Claude decoding is not seeded, so new calls (for example, the rest of
`sonnet_O1_r0`) will not match a different session's answers exactly. The cache keeps the first answer it
saw.

**Re-analysis only:** `.venv/bin/python -m clove.cli analyze --run-dir artifacts/laptop_partial/runs/<run>`.

## 5. What each recorded run is

| run | what | calls | cost | outcome |
|:--|:--|:--|:--|:--|
| `smoke_c20bf2fc` | first Sonnet smoke, sage0/append/fresh, b_out = 96 for all methods | 102 new | $0.84 | 31/34 sage0 calls failed: the SAGE k=0 prompt asks for reasoning before its JSON, which cannot fit 96 output tokens |
| `smoke_70c1cc45` | rerun after `method_max_new_tokens: {sage0: 512}` (approved by the user) | 34 new + 68 cached | $0.15 | 0 errors |
| `sonnet_O1_r0` | pilot, order O1_r0, methods names → sage0 → append → fresh → prism, $60 cap | 821 predictions (names 522 = episode 0 complete; sage0 299/522) | $22.35 at stop ($14.98 at the last checkpoint, `progress.json`) | **stopped by hand, PARTIAL**: see below |

Why `sonnet_O1_r0` was stopped: the run goes method by method, so the same prompt prefix comes back only
after the prompt cache has expired. Cache reads were 1% of input tokens, so nearly every call paid the
cache-write price, about $0.028 per call against the $0.0023 the smoke-based estimate assumed. After the
stop, `claude.prompt_caching: false` was added to `configs/claude.yaml`. It sets `DISABLE_PROMPT_CACHING` and
is not part of the cache key, so the config hash and cached answers are unchanged. **The partial run was made
with prompt caching on.** With caching off, the projections are about $125 for Sonnet O1_r0 and about $500
for all five orders. Haiku would be much cheaper but needs its own smoke run first.

## 6. Logged deviations and caveats

1. **Laptop subset.** The laptop has 556 of the 1,234 WACV soybean test images, which leaves 16 admissible
   classes. The main manifest (`artifacts/manifests/`) is deliberately **not frozen**. Freeze it on Nova
   with the full pool.
2. **b_out departure.** sage0 uses b_out = 512 and every CLOVE method uses 96 (`configs/claude.yaml`,
   `budgets.method_max_new_tokens`).
3. **b_out is not a hard cap on Claude.** `claude -p` keeps going past `CLAUDE_CODE_MAX_OUTPUT_TOKENS` by
   sending several API requests and adding them up. Actual output tokens are always recorded, and calls over
   the cap are flagged `OUTPUT_CAP_EXCEEDED` (10 in `sonnet_O1_r0`).
4. **No registry** in the laptop runs (section 3).
5. **Claude specifics.** There are no logprobs, so unknown detection is the explicit "0 = none of these"
   option (`gate.mode: argmax_only`) and AUROC-type metrics are a single operating point. b_KB is counted with
   an approximate tokenizer (`tokenizer_exact: false`).
6. **Code version.** The runs were made while the code was being written on 2026-10-16. The commit in this
   bundle comes after all three runs. These changes came after `sonnet_O1_r0` started, and none of
   them touches the Claude prediction path:
   - the `prompt_caching` switch (`claude_headless.py`)
   - round-robin over several vLLM servers (`backend.py`, plus a new unit test)
   - the per-episode call plan and the wall-clock/GPU-hour formula in `estimate` (`plan_calls.py`, `cli.py`)
