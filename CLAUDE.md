# CLAUDE.md

Repo instructions. Read this before writing any code.

## What this project is

A take-home assignment: build an AI customer-support agent for one brand,
using real Twitter support conversations, and prove it works.

The graded deliverables are, in rough order of weight:

1. A golden evaluation set of 150-250 hand-labelled examples
2. An eval harness with automated metrics plus an LLM-as-judge rubric,
   including measured agreement between the judge and a human
3. Results against two baselines (one trivial, one simple)
4. A failure analysis with real examples and hypotheses
5. A mandatory section on what is misleading about the headline number
6. A runnable pipeline that reproduces headline results in under 15 minutes

**Evaluation rigour is the product. The agent is the smaller half of this
assignment.** When a change would improve the agent but cost eval time,
say so and default to protecting the eval work.

The author will be asked to explain and modify this code live in an
interview. Favour obvious code over clever code.

## Hard constraints

- **CLI only.** No web server, no FastAPI, no UI, no Docker.
- **No external vector DB.** No Pinecone, Weaviate, Chroma server.
  Use FAISS or `sklearn.neighbors` over embeddings cached to disk.
- **No fine-tuning, no training loops, no PyTorch.**
- **Must run offline.** Every headline result reproduces from cached
  artifacts with no API key set. Live LLM calls are a separate opt-in path.
- **Subsampling is expected.** Never write code that requires processing
  the full 3M-row CSV. Chunk reads, cache intermediates.
- **No multi-turn conversation state.** Single inbound message in,
  classification + draft reply + escalation decision out. Thread history
  is out of scope by decision, not by accident.

## Repo layout

```
data/raw/          twcs.csv, gitignored, never modified
data/interim/      cached subsamples, embeddings (.npy), gitignored
data/golden/       hand-labelled evaluation set, COMMITTED
src/               pipeline scripts
eval/              harness, rubric, baselines
reports/           report, decision log, failure analysis
DECISIONS.md       running log of non-obvious choices
SURPRISES.md       running log of things in the data that broke expectations
```

## Dataset facts

`data/raw/twcs.csv`, ~2.8M rows, roughly Oct-Dec 2017.

Columns: `tweet_id`, `author_id`, `inbound`, `created_at`, `text`,
`response_tweet_id`, `in_response_to_tweet_id`.

- `inbound` is True for customer messages, False for brand replies
- customer `author_id` values are numeric strings; brands are handles
- do not assume column names beyond this list; check before using

## Code style

- One script, one job. Target under 150 lines. If a file grows past that,
  stop and ask whether it should be split.
- No abstraction layers, base classes, plugin registries, or config
  frameworks. Plain functions and `argparse`.
- Every non-obvious choice gets a one-line comment explaining the
  reasoning, not the mechanics.
- Deterministic by default: seed everything, and print the seed.
- Print progress on anything that takes over ten seconds.

## Methodology rules

Brand: British_Airways.

These encode decisions already made. Do not silently deviate.

- **Split by time, not randomly.** Near-duplicate tweets cluster around
  outage days; a random split leaks twins across train and test and
  inflates retrieval scores.
- **Deduplicate near-identical messages** before any split or scoring.
- **Escalation is asymmetric.** Never report escalation as plain accuracy.
  Report an operating curve: auto-handle rate at a fixed escalate-recall.
- **Judge rubric criteria are binary**, not 1-5 scales.
- **Report intervals, not point estimates.** Any accuracy figure over the
  golden set gets a Wilson confidence interval alongside it.
- **The golden set is stratified**, and sampling weights are stored with it.
  Population estimates must be reweighted, and both numbers reported.

## What you must not do

- **Do not create, edit, extend, or relabel anything in `data/golden/`.**
  Those labels are hand-made by the author and are the ground truth the
  whole submission rests on. Read them, never write them.
- **Do not use an LLM to produce evaluation labels or judge-agreement
  human scores.** Both are human-authored by definition.
- **Do not write the report prose, the failure analysis, or the
  decision log.** You may produce tables, numbers, and extracted examples
  for the author to write from.
- **Do not tune against the golden set.** Any threshold search happens on
  a separate dev slice.
- **Do not add dependencies** without asking. Current allowance: pandas,
  numpy, scikit-learn, sentence-transformers, faiss-cpu, matplotlib,
  anthropic.

## Reproducibility contract

`make eval-cached` must run end-to-end on a clean clone, with no API key,
in under 15 minutes, and print the headline table.

`make eval-live` is the same pipeline with real LLM calls.

Any change that risks the 15-minute budget must be flagged before it lands.

## Working agreement

- Ask before starting anything that spans more than one file.
- After each script, state in one sentence what decision it encodes so it
  can be added to `DECISIONS.md`.
- If a result looks too good, say so and suggest what leak might explain it
  before moving on.