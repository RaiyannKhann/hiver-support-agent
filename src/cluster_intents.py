"""Discover the intent taxonomy in British Airways' inbound opening messages.

Pulls BA openers out of twcs.csv (chunked), drops near-duplicates, takes a
seeded sample, embeds it once into a cached .npy, then dumps KMeans readouts at
several k so a human can read them and name the intents. Nothing here labels.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BRAND = "British_Airways"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KS = (6, 8, 10, 12)
COLS = ["tweet_id", "created_at", "inbound", "text", "in_response_to_tweet_id"]

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
NONWORD_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")


def clean(text):
    """Mentions and links carry no intent signal -- the @brand handle is constant
    across the whole sample, and t.co links are opaque."""
    return WS_RE.sub(" ", URL_RE.sub(" ", MENTION_RE.sub(" ", text))).strip()


def norm_key(text):
    """Near-duplicate key. Outage days produce hundreds of all-but-identical
    complaints; left in, they would own a cluster by weight of repetition alone."""
    return WS_RE.sub(" ", NONWORD_RE.sub(" ", clean(text).lower())).strip()


def extract_openers(path, brand, chunksize):
    """Inbound, no parent tweet, and @-mentions the brand. Returns id, created_at, text."""
    handle = "@" + brand.lower()
    found = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=COLS, dtype=str,
                                          chunksize=chunksize)):
        chunk = chunk.dropna(subset=["text"])
        opener = (chunk["inbound"].str.lower() == "true") & \
                 chunk["in_response_to_tweet_id"].isna()
        # substring match is safe here: no other handle contains this one
        hit = chunk[opener & chunk["text"].str.lower().str.contains(handle, regex=False)]
        found.append(hit[["tweet_id", "created_at", "text"]])
        print(f"  chunk {i + 1}: {sum(len(f) for f in found):,} openers so far", flush=True)
    return pd.concat(found, ignore_index=True)


def build_sample(frame, n):
    frame = frame.assign(key=frame["text"].map(norm_key))
    frame = frame[frame["key"].str.len() > 0]
    deduped = frame.drop_duplicates("key")  # keeps first occurrence, so file order fixes it
    print(f"{len(frame):,} openers -> {len(deduped):,} after near-duplicate dedup")
    if len(deduped) <= n:
        print(f"only {len(deduped):,} unique openers, using all of them")
        return deduped.reset_index(drop=True)
    return deduped.sample(n=n, random_state=SEED).reset_index(drop=True)


def embed(texts, cache):
    if cache.exists():
        vecs = np.load(cache)
        if len(vecs) == len(texts):
            print(f"loaded cached embeddings {vecs.shape} from {cache}")
            return vecs
        print(f"cache has {len(vecs)} rows but sample has {len(texts)}, re-embedding")
    from sentence_transformers import SentenceTransformer

    print(f"embedding {len(texts):,} messages with {MODEL}")
    print("(first run downloads the model; after that this path is offline)")
    model = SentenceTransformer(MODEL)
    # unit-norm vectors, so KMeans' euclidean distance ranks the same as cosine
    vecs = model.encode([clean(t) for t in texts], batch_size=64,
                        show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, vecs)
    print(f"cached embeddings {vecs.shape} to {cache}")
    return vecs


def write_clusters(k, vecs, display, out_dir, top):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(vecs)
    path = out_dir / f"clusters_k{k}.txt"
    with path.open("w") as fh:
        fh.write(f"k={k}  n={len(display):,}  seed={SEED}  "
                 f"model={MODEL}  inertia={km.inertia_:.1f}\n")
        fh.write("distance in brackets is euclidean to the centroid on unit-norm "
                 "embeddings\n")
        for c in range(k):
            idx = np.flatnonzero(km.labels_ == c)
            dist = np.linalg.norm(vecs[idx] - km.cluster_centers_[c], axis=1)
            fh.write(f"\n## cluster {c}  size={len(idx)}  "
                     f"({len(idx) / len(display):.1%} of sample)\n")
            for pos in np.argsort(dist)[:top]:
                fh.write(f"  [{dist[pos]:.3f}] {display[idx[pos]][:220]}\n")
    sizes = np.bincount(km.labels_, minlength=k)
    print(f"  k={k:>2}: wrote {path}  sizes={sorted(sizes.tolist(), reverse=True)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out-dir", default="data/interim", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--refresh", action="store_true",
                    help="re-extract from the CSV even if the cached sample exists")
    args = ap.parse_args()

    print(f"seed={SEED}  brand={BRAND}  n={args.n}  k={list(KS)}")
    sample_path = args.out_dir / "ba_openers.csv"
    # the sample is cached alongside the vectors so reruns never touch the raw CSV
    if sample_path.exists() and not args.refresh:
        sample = pd.read_csv(sample_path, dtype=str)
        print(f"loaded cached sample of {len(sample):,} from {sample_path}")
    else:
        sample = build_sample(extract_openers(args.csv, BRAND, args.chunksize), args.n)
        sample[["tweet_id", "text"]].to_csv(sample_path, index=False)
        print(f"wrote sample of {len(sample):,} to {sample_path}")

    texts = sample["text"].tolist()
    vecs = embed(texts, args.out_dir / "ba_embeddings.npy")
    display = [WS_RE.sub(" ", t) for t in texts]  # one message per line in the readout
    for k in KS:
        write_clusters(k, vecs, display, args.out_dir, args.top)


if __name__ == "__main__":
    main()
