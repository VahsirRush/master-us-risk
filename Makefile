.PHONY: help install test lint fmt status clean \
        data engine baselines master ablations factors risk join reports

PY := PYTHONPATH=src python

help:
	@echo "MASTER-US"
	@echo ""
	@echo "  make install     install package + dev dependencies"
	@echo "  make test        run test suite"
	@echo "  make lint        ruff + mypy"
	@echo "  make fmt         ruff format"
	@echo "  make status      show build phase ladder"
	@echo ""
	@echo "Phases (each gated — do not skip):"
	@echo "  make data        0  panel construction + PIT tests"
	@echo "  make engine      1  backtest engine  [gate: 12-1 momentum reproduces]"
	@echo "  make baselines   2  ridge / lgbm / lstm / ungated"
	@echo "  make master      3  MASTER architecture"
	@echo "  make ablations   4  ablation grid + beta sweep + cost breakeven"
	@echo "  make factors     5  Barra descriptors + factor returns"
	@echo "  make risk        6  covariance  [gate: bias statistic in 0.9-1.1]"
	@echo "  make join        7  attribution + neutralized alpha"
	@echo "  make reports     8  research terminal + static export"

install:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -v

lint:
	ruff check src tests
	mypy src

fmt:
	ruff format src tests
	ruff check --fix src tests

status:
	@$(PY) -c "print('phase ladder renderer not implemented (output-layer-spec 3.5)')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache

# ---------------------------------------------------------------- phases

data:
	@echo "phase 0: not implemented — see docs/data-sources-contract.md"

engine:
	@echo "phase 1: not implemented — see docs/implementation-spec.md section 5"

baselines:
	@echo "phase 2: not implemented — see docs/implementation-spec.md section 6"

master:
	@echo "phase 3: not implemented — see docs/implementation-spec.md section 7"

ablations:
	@echo "phase 4: not implemented — see docs/implementation-spec.md section 8"

factors:
	@echo "phase 5: not implemented — see docs/implementation-spec.md section 9"

risk:
	@echo "phase 6: not implemented — see docs/implementation-spec.md section 9.4"

join:
	@echo "phase 7: not implemented — see docs/implementation-spec.md section 10"

reports:
	@echo "phase 8: not implemented — see docs/research-terminal-spec.md"
