"""Fix the dev/test split of the golden set and bundle it with the sampling weights.

Writes eval/golden_meta.csv (id, created_at, split, bucket, weight) -- the one
committed sidecar the harness joins onto the hand labels. Split is by time, not
at random, per the methodology rules; the earliest slice is dev.
"""
import argparse

import pandas as pd

GOLDEN = "data/golden/hiver_ba_golden_set_corrected.csv"
STRATA = "eval/labeling_strata.csv"
CREATED_FMT = "%a %b %d %H:%M:%S %z %Y"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out", default="eval/golden_meta.csv")
    ap.add_argument("--dev-frac", type=float, default=0.3,
                    help="earliest share of the golden set that becomes dev")
    ap.add_argument("--chunksize", type=int, default=400_000)
    args = ap.parse_args()

    ids = set(pd.read_csv(GOLDEN, dtype=str)["id"])
    found = []
    for i, chunk in enumerate(pd.read_csv(args.csv, usecols=["tweet_id", "created_at"],
                                          dtype=str, chunksize=args.chunksize)):
        found.append(chunk[chunk["tweet_id"].isin(ids)])
        print(f"  chunk {i + 1}: {sum(len(f) for f in found)}/{len(ids)} ids found", flush=True)
    meta = pd.concat(found).rename(columns={"tweet_id": "id"})
    assert len(meta) == len(ids), "some golden ids are missing from the raw CSV"

    meta["ts"] = pd.to_datetime(meta["created_at"], format=CREATED_FMT)
    meta = meta.sort_values(["ts", "id"]).reset_index(drop=True)  # id breaks ts ties
    n_dev = int(len(meta) * args.dev_frac)
    meta["split"] = ["dev"] * n_dev + ["test"] * (len(meta) - n_dev)

    strata = pd.read_csv(STRATA, dtype={"id": str})
    meta = meta.merge(strata[["id", "bucket", "weight"]], on="id", how="left")
    assert meta["weight"].notna().all(), "golden ids missing from the strata file"

    meta[["id", "created_at", "split", "bucket", "weight"]].to_csv(args.out, index=False)
    cut = meta.loc[n_dev - 1, "ts"]
    print(f"\nwrote {args.out}: dev={n_dev} (up to {cut}), test={len(meta) - n_dev}")


if __name__ == "__main__":
    main()
