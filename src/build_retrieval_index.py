"""Build the agent's retrieval corpus: past BA conversations it may quote as examples.

Keeps (opener, first BA reply) pairs that predate every test item, drops the
golden ids themselves, dedups, embeds the openers with MiniLM, and caches to
data/interim/retrieval_index.{csv,npy}. The agent looks up nearest neighbours here.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_labeling_set import CREATED_FMT, first_replies  # noqa: E402
from cluster_intents import BRAND, MODEL, clean, extract_openers, norm_key  # noqa: E402

GOLDEN = "data/golden/hiver_ba_golden_set_corrected.csv"
META = "eval/golden_meta.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out-dir", default="data/interim", type=Path)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    meta = pd.read_csv(META, dtype={"id": str})
    meta["ts"] = pd.to_datetime(meta["created_at"], format=CREATED_FMT)
    # strictly before the earliest test item, so no example can be a test item's future
    cutoff = meta.loc[meta["split"] == "test", "ts"].min()
    golden_ids = set(pd.read_csv(GOLDEN, dtype=str)["id"])
    print(f"corpus cutoff: before {cutoff}   excluding {len(golden_ids)} golden ids")

    openers = extract_openers(args.csv, BRAND, args.chunksize)
    openers["ts"] = pd.to_datetime(openers["created_at"], format=CREATED_FMT)
    openers = openers[(openers["ts"] < cutoff) & ~openers["tweet_id"].isin(golden_ids)]
    openers = openers.assign(key=openers["text"].map(norm_key))
    openers = openers[openers["key"].str.len() > 0].drop_duplicates("key")
    print(f"{len(openers):,} deduped pre-cutoff openers outside the golden set")

    replies = first_replies(args.csv, BRAND, set(openers["tweet_id"]), args.chunksize)
    openers["reply_text"] = openers["tweet_id"].map(replies)
    corpus = openers.dropna(subset=["reply_text"]).reset_index(drop=True)
    print(f"{len(corpus):,} of them have a BA reply -> that is the corpus")

    from sentence_transformers import SentenceTransformer

    print(f"embedding with {MODEL}")
    vecs = SentenceTransformer(MODEL).encode([clean(t) for t in corpus["text"]], batch_size=64,
                                             show_progress_bar=True, normalize_embeddings=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = corpus.rename(columns={"tweet_id": "id", "text": "inbound_text"})
    corpus[["id", "created_at", "inbound_text", "reply_text"]].to_csv(
        args.out_dir / "retrieval_index.csv", index=False)
    np.save(args.out_dir / "retrieval_index.npy", vecs)
    print(f"wrote {args.out_dir / 'retrieval_index.csv'} and .npy {vecs.shape}")


if __name__ == "__main__":
    main()
