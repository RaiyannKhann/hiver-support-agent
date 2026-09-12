# hiver-support-agent

An AI customer-support agent for **British Airways**, built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, and the evaluation that shows how well it works. One inbound tweet in;
intent, escalation decision, and a draft public reply out. No thread history.

The evaluation is the larger half of this repo. The agent is ~150 lines.

## Quickstart (no API key, no dataset, ~5 seconds)

```bash
git clone <this repo> && cd hiver-support-agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make eval-cached
```

Prints the headline table for the two baselines and the agent, the LLM-judge
pass rates, and judge-vs-human agreement — all from committed artifacts.
The Makefile assumes the venv lives at `.venv/`.

## Try the agent on your own message

```bash
export ANTHROPIC_API_KEY=...
.venv/bin/python src/try_agent.py "@British_Airways my bag never arrived at JFK, what do I do?"
.venv/bin/python src/try_agent.py "..." --show-prompt   # also prints the retrieved examples
```

Needs the retrieval index, which is built locally from the raw CSV (see below).
Costs well under a cent per message on `claude-sonnet-5`.

## Headline results

Test split, n=154, British Airways. Raw figures with 95% Wilson intervals;
reweighted figures correct for the stratified golden-set draw.

| | trivial | simple | agent |
|---|---|---|---|
| intent accuracy, raw | 0.065 | 0.558 | **0.695** [0.618, 0.762] |
| intent accuracy, reweighted | 0.079 | 0.488 | **0.675** [0.586, 0.752] |
| escalate-recall at dev-tuned threshold | 1.000 | 1.000 | 0.942 [0.871, 0.975] |
| auto-handle rate at that threshold | 0.000 | 0.000 | **0.292** [0.226, 0.368] |
| auto-handle rate, mixed intents only | 0.000 | 0.000 | 0.176 [0.104, 0.284] |
| safety-flagged escalate-recall | 1.000 | 1.000 | 1.000 (25/25) |

Judge (Haiku 4.5, binary rubric) on the agent's test drafts: all five
non-reference criteria pass together on 0.636 [0.558, 0.708]. Judge-vs-human
agreement, pooled over 275 judgments: **κ = 0.31 [0.13, 0.48]**.

What these numbers do and do not mean — including why the auto-handle rate is
29% and not the 61% the model would choose for itself, and why one judge
criterion is not reportable — is the subject of the report in `reports/`.

## Layout

```
src/                  pipeline scripts, one job each
  profile_brands.py         which brand to build for (writes data/interim/brand_profile.csv)
  cluster_intents.py        intent discovery: KMeans readouts at k=6,8,10,12
  dump_for_reading.py       300 openers for hand-reading
  build_labeling_set.py     the stratified 220-row labelling worksheet (+ sampling weights)
  build_retrieval_index.py  past BA conversations the agent may quote (pre-test-cutoff only)
  agent.py                  the agent: retrieve 5 examples -> Claude -> structured verdict, cached
  try_agent.py              run the agent on one message of your own
eval/                 harness, rubric, baselines
  make_split.py             time-based dev/test split -> golden_meta.csv (committed)
  harness.py                scores any predictions file: Wilson CIs, reweighting, operating curve
  baseline_trivial.py       dev-majority intent, always escalate
  baseline_simple.py        keyword intent + P(escalate | bucket) fitted on dev
  judge.py                  LLM-as-judge, 6 binary criteria, cached
  make_judge_sheet.py       blank sheet for human scoring of the same drafts
  agreement.py              Cohen's kappa, judge vs human, bootstrap intervals
  golden_meta.csv           split + sampling weight per golden id
  labeling_strata.csv       the stratified draw: bucket, population, weight
  preds_agent.csv           the agent's cached predictions  (the reproducibility contract)
  judge_scores.csv          the judge's cached verdicts     (the reproducibility contract)
data/golden/          hand-labelled evaluation set, committed, never machine-written
  hiver_ba_golden_set_corrected.csv   220 rows: intent, safety_flag, auto_or_escalate, reason
  to_label.csv                        the blank worksheet the labels were made on (pristine text)
  judge_human_scores_labeled.csv      50 drafts scored by hand on the rubric
data/raw/             twcs.csv, gitignored — download it yourself (below)
data/interim/         caches, embeddings, indexes — gitignored, regenerated locally
reports/              report, failure analysis, decision log
DECISIONS.md          running log of non-obvious choices
SURPRISES.md          things in the data that broke expectations
```

## The live path

Download the dataset from Kaggle and place it at `data/raw/twcs/twcs.csv`
(~516 MB, 2.8M rows; every script reads it in chunks). Then:

```bash
export ANTHROPIC_API_KEY=...
make prep        # split sidecar + retrieval index, from the raw CSV (~1 min, MiniLM download on first run)
make eval-live   # fills any missing rows in the agent/judge caches, then prints the tables
```

Both caches are complete in this repo, so `make eval-live` makes no calls
unless you delete `eval/preds_agent.csv` or `eval/judge_scores.csv`. Deleting
them and regenerating changes the headline — LLM output is not deterministic,
which is why the caches are committed. A full regeneration is roughly $2.

## Methodology, in brief

- Golden set: 220 BA opening messages, stratified across 9 intents by a keyword
  pre-filter, hand-labelled. Weights are stored; every rate is reported raw and reweighted.
- Split by time, not at random: earliest 30% is dev (threshold selection only), the rest is test.
- Escalation is never reported as accuracy. The threshold that reaches 95% escalate-recall
  on dev is applied unchanged to test, and the auto-handle rate at that point is the number.
- Judge criteria are binary. The judge is a different model family from the agent.
  Agreement with a human is measured on 50 drafts and reported with a bootstrap interval.
- Retrieval examples are cut strictly before the earliest test timestamp and exclude every golden id.
- Runs offline: `make eval-cached` needs no key and no raw data.

## Data attribution

Tweets are from the Kaggle *Customer Support on Twitter* dataset (Thought Vector /
Stuart Axelbrooke), redistributed here only as the 220 hand-labelled rows the
evaluation rests on plus their brand replies. Customer handles were anonymised
to numeric ids by the dataset authors. Check the dataset's license before
reusing the golden set beyond this evaluation.
