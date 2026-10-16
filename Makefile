# CLOVE soybean pilot — one target per protocol gate.
PY      ?= .venv/bin/python
CONFIG  ?= configs/pilot.yaml
METHODS ?= names,sage0,append,fresh,prism
RUN_DIR ?=

.PHONY: venv test audit manifest leakage smoke smoke-synthetic estimate pilot analyze clean-synthetic smoke-claude estimate-claude pilot-claude serve-titan smoke-titan estimate-titan pilot-titan

venv:
	python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,figures]'

test:            ## Gate 1/2: CPU-only unit, leakage, statistics, integration tests
	$(PY) -m pytest

audit:           ## Gate 0: verified inventory -> artifacts/reports/audit.md
	$(PY) -m clove.cli audit --config $(CONFIG)

manifest:        ## freeze manifest + class orders (refuses to overwrite a different freeze)
	$(PY) -m clove.cli manifest --config $(CONFIG)

leakage:
	$(PY) -m clove.cli test-leakage --config $(CONFIG)

smoke-synthetic: ## pipeline check with the mock backend (NOT results)
	$(PY) -m clove.cli smoke --config $(CONFIG) --synthetic

smoke:           ## Gate 3: real micro-run against the frozen VLM
	$(PY) -m clove.cli smoke --config $(CONFIG)

estimate:        ## GPU-hour estimate from measured smoke latency; blocks if over budget
	$(PY) -m clove.cli estimate --config $(CONFIG)

pilot:           ## Gate 4
	$(PY) -m clove.cli run --config $(CONFIG) --methods $(METHODS)

smoke-claude:    ## Gate 3 on Claude via `claude -p` (real calls, billed to the logged-in account)
	$(PY) -m clove.cli smoke --config configs/claude.yaml

estimate-claude:
	$(PY) -m clove.cli estimate --config configs/claude.yaml

pilot-claude:
	$(PY) -m clove.cli run --config configs/claude.yaml --methods $(METHODS)

serve-titan:     ## local 2x Titan RTX: one vLLM (Qwen2.5-VL-3B) per GPU
	bash scripts/serve_qwen_titan.sh

smoke-titan:
	$(PY) -m clove.cli smoke --config configs/qwen3b_titan.yaml

estimate-titan:
	$(PY) -m clove.cli estimate --config configs/qwen3b_titan.yaml

pilot-titan:
	$(PY) -m clove.cli run --config configs/qwen3b_titan.yaml --methods $(METHODS)

analyze:
	$(PY) -m clove.cli analyze --run-dir $(RUN_DIR)

clean-synthetic:
	rm -rf artifacts/synthetic_runs
