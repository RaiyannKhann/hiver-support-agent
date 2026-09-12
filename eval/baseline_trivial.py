"""Trivial baseline: the dev-majority intent for everything, and always escalate.

The floor every real system must clear. Reads dev labels only, never test.
"""
import argparse

import pandas as pd

from harness import load_golden


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/interim/preds_trivial.csv")
    args = ap.parse_args()

    gold = load_golden()
    dev = gold[gold["split"] == "dev"]
    counts = dev["intent"].value_counts()
    print(f"dev majority intent: {counts.idxmax()} ({counts.max()}/{len(dev)})")
    print(f"dev escalate share: {(dev['auto_or_escalate'] == 'escalate').mean():.3f}"
          " -> always escalate")

    preds = pd.DataFrame({"id": gold["id"], "intent": counts.idxmax(),
                          "auto_or_escalate": "escalate", "escalate_score": 1.0,
                          "draft_reply": ""})
    preds.to_csv(args.out, index=False)
    print(f"wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
