"""Score a predictions file against the golden set and print the headline table.

Reads the hand labels (never writes them), joins the pristine worksheet text and
the committed split/weight sidecar, picks the escalation threshold on dev only,
and reports every test rate raw and reweighted, each with a Wilson interval.

Predictions CSV columns: id, intent, auto_or_escalate, escalate_score, draft_reply
(escalate_score in [0, 1], higher = more likely to need escalation).
"""
import argparse
import math

import numpy as np
import pandas as pd

GOLDEN = "data/golden/hiver_ba_golden_set_corrected.csv"
WORKSHEET = "data/golden/to_label.csv"  # pristine text; the labelled copy was reformatted by an editor
META = "eval/golden_meta.csv"
# Intents whose hand labels split between auto and escalate. Everywhere else the
# decision is fully determined by the intent, so this slice is the real escalation test.
MIXED = {"policy_question", "digital_failure", "service_complaint"}
RECALL_LEVELS = (0.90, 0.95, 0.99)


def load_golden():
    gold = pd.read_csv(GOLDEN, dtype=str).fillna("").drop(columns=["inbound_text", "reply_text"])
    text = pd.read_csv(WORKSHEET, dtype=str).fillna("")[["id", "inbound_text", "reply_text"]]
    meta = pd.read_csv(META, dtype={"id": str})[["id", "split", "weight"]]
    return gold.merge(text, on="id").merge(meta, on="id")


def wilson(k, n, z=1.96):
    if n == 0:
        return math.nan, math.nan
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)  # float noise can dip below 0


def rate(hit, w):
    """Raw rate with a Wilson CI, plus the weight-corrected rate with a Wilson CI on
    Kish's effective n, so the reweighted interval does not pretend to more data."""
    hit, w = np.asarray(hit, dtype=float), np.asarray(w, dtype=float)
    if len(hit) == 0:
        return math.nan, (math.nan, math.nan), math.nan, (math.nan, math.nan)
    weighted = (hit * w).sum() / w.sum()
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return hit.mean(), wilson(hit.sum(), len(hit)), weighted, wilson(weighted * n_eff, n_eff)


def fmt(hit, w):
    raw, (lo, hi), wt, (wlo, whi) = rate(hit, w)
    return f"{raw:6.3f} [{lo:.3f}, {hi:.3f}]     {wt:6.3f} [{wlo:.3f}, {whi:.3f}]"


def threshold_for_recall(scores, is_esc, target):
    """Largest threshold that still escalates >= target of the true escalations.
    Rows scoring >= threshold are escalated; a lower threshold escalates more."""
    for th in sorted(set(scores[is_esc]), reverse=True):
        if (scores[is_esc] >= th).mean() >= target:
            return th
    return scores.min()  # target unreachable: escalate everything


def escalation_lines(sub, esc_pred, label):
    is_esc = sub["is_esc"].values
    recall = fmt(esc_pred[is_esc], sub["weight"].values[is_esc])
    auto = fmt(~esc_pred, sub["weight"].values)
    print(f"  {label:22} n={len(sub):<4} recall {recall}")
    print(f"  {'':22} {'':6} auto   {auto}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--recall", type=float, default=0.95,
                    help="escalate-recall the threshold must reach on dev")
    args = ap.parse_args()

    df = load_golden().merge(pd.read_csv(args.preds, dtype={"id": str}).fillna(""),
                             on="id", how="left", suffixes=("", "_pred"))
    assert df["intent_pred"].notna().all(), "some golden ids have no prediction"
    df["escalate_score"] = df["escalate_score"].astype(float)
    df["is_esc"] = df["auto_or_escalate"] == "escalate"
    dev, test = df[df["split"] == "dev"], df[df["split"] == "test"]

    print(f"predictions: {args.preds}   dev={len(dev)}  test={len(test)}")
    print(f"{'':44}{'raw [95% Wilson]':<24}{'reweighted [95% Wilson, Kish n]'}\n")
    print("== intent accuracy ==")
    print(f"  {'test, all intents':22} n={len(test):<4} acc    "
          f"{fmt(test['intent'] == test['intent_pred'], test['weight'])}")
    for intent, grp in test.groupby("intent"):
        raw, (lo, hi), _, _ = rate(grp["intent"] == grp["intent_pred"], grp["weight"])
        print(f"    {intent:20} n={len(grp):<4} acc    {raw:6.3f} [{lo:.3f}, {hi:.3f}]")

    th = threshold_for_recall(dev["escalate_score"].values, dev["is_esc"].values, args.recall)
    dev_recall = (dev["escalate_score"][dev["is_esc"]] >= th).mean()
    print(f"\n== escalation: threshold {th:.3f} picked on dev for recall >= {args.recall} "
          f"(dev recall {dev_recall:.3f}) ==")
    esc = (test["escalate_score"] >= th).values
    escalation_lines(test, esc, "test, all")
    mixed = test["intent"].isin(MIXED).values
    escalation_lines(test[mixed], esc[mixed], "test, mixed intents")
    flagged = (test["safety_flag"] != "").values
    print(f"  {'test, safety-flagged':22} n={flagged.sum():<4} recall "
          f"{fmt(esc[flagged], test['weight'].values[flagged])}")
    own = (test["auto_or_escalate_pred"] == "escalate").values
    escalation_lines(test, own, "test, model's own call")

    print("\n== operating curve (threshold from dev, rates on test, raw) ==")
    for lvl in RECALL_LEVELS:
        t = threshold_for_recall(dev["escalate_score"].values, dev["is_esc"].values, lvl)
        e = (test["escalate_score"] >= t).values
        print(f"  target recall {lvl:.2f}: threshold {t:.3f}  "
              f"test recall {e[test['is_esc'].values].mean():.3f}  auto-handle {(~e).mean():.3f}")
    print(f"\ndrafts: {(test['draft_reply'] != '').sum()}/{len(test)} test rows have a draft reply")


if __name__ == "__main__":
    main()
