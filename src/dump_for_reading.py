"""Dump a plain-text sample of BA opening messages for the author to read.

Reading output, not labelled data. Reuses the opener definition and the
near-duplicate key from cluster_intents so this sample and the clustering
sample are drawn from exactly the same population.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from cluster_intents import BRAND, extract_openers, norm_key

SEED = 7  # deliberately not the clustering seed, so reading is not anchored on those 3000
WS_RE = re.compile(r"\s+")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out", default="data/interim/reading_sample.txt", type=Path)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    print(f"seed={SEED}  brand={BRAND}  n={args.n}")
    openers = extract_openers(args.csv, BRAND, args.chunksize)
    openers = openers.assign(key=openers["text"].map(norm_key))
    pool = openers[openers["key"].str.len() > 0].drop_duplicates("key")
    print(f"{len(openers):,} openers -> {len(pool):,} after near-duplicate dedup")

    take = min(args.n, len(pool))
    sample = pool.sample(n=take, random_state=SEED).reset_index(drop=True)
    width = len(str(take))
    with args.out.open("w") as fh:
        fh.write(f"# {take} BA inbound opening messages, seed={SEED}, "
                 f"sampled from {len(pool):,} deduped openers\n")
        fh.write("# for hand-reading only -- not labels, not a golden set\n\n")
        for i, text in enumerate(sample["text"], start=1):
            fh.write(f"{i:>{width}}  {WS_RE.sub(' ', text).strip()}\n")
    print(f"wrote {take} messages to {args.out}")


if __name__ == "__main__":
    main()
