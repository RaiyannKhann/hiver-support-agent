# Reproducibility contract (see CLAUDE.md):
#   make eval-cached  -- headline tables from committed artifacts, no API key, no raw CSV
#   make eval-live    -- the same pipeline with real LLM calls (needs ANTHROPIC_API_KEY)
#   make prep         -- one-off artifacts that need the raw CSV (split sidecar, retrieval index)
PY := .venv/bin/python
CACHED := eval/preds_agent.csv eval/judge_scores.csv

.PHONY: eval-cached eval-live prep

eval-cached: eval/golden_meta.csv $(CACHED)
	@mkdir -p data/interim
	$(PY) eval/baseline_trivial.py
	$(PY) eval/baseline_simple.py
	@echo; echo "################ TRIVIAL BASELINE ################"
	$(PY) eval/harness.py --preds data/interim/preds_trivial.csv
	@echo; echo "################ SIMPLE BASELINE ################"
	$(PY) eval/harness.py --preds data/interim/preds_simple.csv
	@echo; echo "################ AGENT ################"
	$(PY) eval/harness.py --preds eval/preds_agent.csv
	$(PY) eval/judge.py
	@echo
	@# agreement needs the hand-filled sheet; report its absence rather than fail the build
	@if [ -f data/golden/judge_human_scores_labeled.csv ]; then $(PY) eval/agreement.py; \
	 else echo "judge agreement: data/golden/judge_human_scores_labeled.csv not present (run make judge-sheet, then score it by hand)"; fi

# Live path: fills whatever is missing from the caches, then prints the same tables.
eval-live: prep
	$(PY) src/agent.py
	$(PY) eval/judge.py
	$(MAKE) eval-cached

# Needs data/raw/twcs/twcs.csv. Both outputs are committed, so a clean clone never runs this.
prep: eval/golden_meta.csv data/interim/retrieval_index.npy
eval/golden_meta.csv:
	$(PY) eval/make_split.py
data/interim/retrieval_index.npy:
	$(PY) src/build_retrieval_index.py

judge-sheet:
	$(PY) eval/make_judge_sheet.py

$(CACHED):
	@echo "missing committed artifact: $@ -- run 'make eval-live' (needs a key)"; exit 1
