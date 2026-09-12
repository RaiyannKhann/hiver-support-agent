"""Simple baseline: keyword intent rules plus an intent -> escalate lookup fitted on dev.

No model, no embeddings. Intent is the same keyword pre-filter that stratified the
golden set; the escalate score is the dev escalate rate of the predicted bucket,
so the harness threshold sweep has a real curve to walk. Reads dev labels only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # the pre-filter lives in src/
from build_labeling_set import bucket_of  # noqa: E402
from harness import load_golden  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/interim/preds_simple.csv")
    args = ap.parse_args()

    gold = load_golden()
    gold["bucket"] = gold["inbound_text"].map(bucket_of)
    dev = gold[gold["split"] == "dev"]
    base = (dev["auto_or_escalate"] == "escalate").mean()
    esc_rate = dev.groupby("bucket")["auto_or_escalate"].apply(lambda s: (s == "escalate").mean())
    print("dev escalate rate by predicted bucket:")
    for bucket, r in esc_rate.sort_values().items():
        print(f"  {bucket:20} {r:.2f}  (n={int((dev['bucket'] == bucket).sum())})")

    # a bucket unseen on dev falls back to the dev base rate rather than a guess
    score = gold["bucket"].map(esc_rate).fillna(base)
    preds = pd.DataFrame({"id": gold["id"], "intent": gold["bucket"],
                          "auto_or_escalate": np.where(score >= 0.5, "escalate", "auto"),
                          "escalate_score": score.round(4), "draft_reply": ""})
    preds.to_csv(args.out, index=False)
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
