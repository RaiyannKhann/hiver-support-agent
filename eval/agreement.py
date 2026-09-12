"""Judge-vs-human agreement on the rubric, per criterion.

Reads the judge's cached verdicts and the human-filled sheet, joins on id, and
reports raw agreement and Cohen's kappa with a bootstrap interval for every
criterion. Kappa is the honest number: raw agreement is inflated wherever both
sides say yes to almost everything -- and with that few "no"s kappa itself is
noisy, hence the interval.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from judge import CRITERIA

SEED = 42
YES = {"yes", "y", "1", "true", "pass"}
NO = {"no", "n", "0", "false", "fail"}


def normalise(v):
    v = str(v).strip().lower()
    return "yes" if v in YES else "no" if v in NO else None  # blank / n/a / typos drop out


def kappa(h, j):
    """nan when neither rater varies: agreement is then undefined, not perfect."""
    return cohen_kappa_score(h, j) if len(set(h) | set(j)) > 1 else float("nan")


def kappa_ci(h, j, n_boot=1000):
    rng = np.random.default_rng(SEED)
    h, j = np.asarray(h), np.asarray(j)
    boots = [kappa(h[i], j[i]) for i in (rng.integers(0, len(h), len(h)) for _ in range(n_boot))]
    if np.all(np.isnan(boots)):
        return float("nan"), float("nan")
    return np.nanpercentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="eval/judge_scores.csv")
    ap.add_argument("--human", default="data/golden/judge_human_scores_labeled.csv")
    args = ap.parse_args()

    j = pd.read_csv(args.judge, dtype=str, keep_default_na=False)  # "n/a" is a value, not NaN
    h = pd.read_csv(args.human, dtype=str, keep_default_na=False)
    d = h.merge(j, on="id", suffixes=("_h", "_j"))
    print(f"human-scored rows: {len(h)}   also judged: {len(d)}   (bootstrap seed={SEED})\n")
    print(f"{'criterion':26} {'n':>3}  {'agree':>6}  {'kappa':>6} {'[95% boot]':>15}  "
          f"{'judge yes':>9}  {'human yes':>9}")
    pooled_h, pooled_j = [], []
    for name, _ in CRITERIA:
        hv, jv = d[f"{name}_h"].map(normalise), d[f"{name}_j"].map(normalise)
        ok = hv.notna() & jv.notna()
        if ok.sum() == 0:
            print(f"{name:26} {0:>3}  (no scored rows yet)")
            continue
        hv, jv = hv[ok].tolist(), jv[ok].tolist()
        lo, hi = kappa_ci(hv, jv)
        flag = "  <- a rater never varied" if len(set(hv)) == 1 or len(set(jv)) == 1 else ""
        print(f"{name:26} {len(hv):>3}  {np.mean(np.array(hv) == np.array(jv)):6.2f}  "
              f"{kappa(hv, jv):6.2f} [{lo:5.2f}, {hi:5.2f}]  {jv.count('yes') / len(jv):9.2f}  "
              f"{hv.count('yes') / len(hv):9.2f}{flag}")
        pooled_h += hv
        pooled_j += jv
    if pooled_h:
        lo, hi = kappa_ci(pooled_h, pooled_j)
        print(f"\n{'pooled, all criteria':26} {len(pooled_h):>3}  "
              f"{np.mean(np.array(pooled_h) == np.array(pooled_j)):6.2f}  "
              f"{kappa(pooled_h, pooled_j):6.2f} [{lo:5.2f}, {hi:5.2f}]")
        print("\nwhere a rater never varied, kappa is 0 or nan by construction and says nothing "
              "about the judge.")


if __name__ == "__main__":
    main()
