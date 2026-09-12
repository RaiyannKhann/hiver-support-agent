"""Profile the highest-volume brands in twcs.csv to pick one to build the agent for.

Two chunked passes: pass 1 counts outbound tweets per author to find the top N
brands, pass 2 profiles only those. The file is never loaded whole.
"""
import argparse
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

COLS = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
DTYPES = {"author_id": str, "inbound": str, "text": str}

# A reply that pushes the customer to a private channel, or that is nothing but a
# link, resolves nothing in the channel we can evaluate -- that is what we measure.
DM_RE = re.compile(r"\b(dm|dms|d\.m\.?|direct message|private message|inbox|pm us)\b", re.I)
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@(\w+)")
CUSTOMER_RE = re.compile(r"@(\d+)")  # customer ids are anonymised to digits, brands are handles
SIGNATURE_RE = re.compile(r"\^\s*\w{1,4}\s*$")  # trailing agent initials, e.g. "^RR"
MIN_SUBSTANTIVE_WORDS = 5  # below this a reply is an ack ("Thanks!"), not a support turn


def content_words(text):
    """Words left after stripping mentions, links and the agent signature."""
    return SIGNATURE_RE.sub("", URL_RE.sub("", MENTION_RE.sub("", text))).split()


def is_deflection(text, words):
    return bool(DM_RE.search(text)) or (bool(URL_RE.search(text)) and len(words) <= 3)


def find_top_brands(path, chunksize, top_n):
    """Pass 1: outbound volume per author. Brands are the only outbound authors."""
    counts = Counter()
    for i, chunk in enumerate(pd.read_csv(path, usecols=["author_id", "inbound"],
                                          dtype=DTYPES, chunksize=chunksize)):
        outbound = chunk[chunk["inbound"].str.lower() != "true"]
        counts.update(outbound["author_id"].dropna())
        print(f"  pass 1 chunk {i + 1}: {sum(counts.values()):,} outbound rows", flush=True)
    return [brand for brand, _ in counts.most_common(top_n)]


def profile(path, chunksize, brands):
    """Pass 2: everything else, for the chosen brands only."""
    by_handle = {b.lower(): b for b in brands}
    brand_set = set(brands)
    stats = {b: {"replies": 0, "deflections": 0, "words": []} for b in brands}
    replied_to = defaultdict(set)   # inbound tweet ids the brand answered
    addressed = defaultdict(set)    # inbound tweet ids that @-mention the brand
    turns = defaultdict(Counter)    # brand -> customer -> substantive reply count

    for i, chunk in enumerate(pd.read_csv(path, usecols=COLS, dtype=DTYPES,
                                          chunksize=chunksize)):
        chunk = chunk.dropna(subset=["text", "author_id", "tweet_id"])
        inbound = chunk["inbound"].str.lower() == "true"

        out = chunk[~inbound & chunk["author_id"].isin(brand_set)]
        for author, text in zip(out["author_id"], out["text"]):
            words = content_words(text)
            row = stats[author]
            row["replies"] += 1
            row["words"].append(len(words))
            deflected = is_deflection(text, words)
            row["deflections"] += deflected
            customer = CUSTOMER_RE.search(text)
            if customer:
                # += 0 still registers the customer, so the denominator counts every
                # conversation the brand touched, not just the substantive ones.
                turns[author][customer.group(1)] += int(
                    not deflected and len(words) >= MIN_SUBSTANTIVE_WORDS)

        answered = out.dropna(subset=["in_response_to_tweet_id"])
        for author, parent in zip(answered["author_id"],
                                  answered["in_response_to_tweet_id"].astype("int64")):
            replied_to[author].add(parent)

        inb = chunk[inbound]
        for tweet_id, text in zip(inb["tweet_id"].astype("int64"), inb["text"]):
            for handle in MENTION_RE.findall(text):
                brand = by_handle.get(handle.lower())
                if brand:
                    addressed[brand].add(tweet_id)
        print(f"  pass 2 chunk {i + 1}: {sum(s['replies'] for s in stats.values()):,} "
              f"replies from top brands", flush=True)
    return stats, replied_to, addressed, turns


def summarise(brands, stats, replied_to, addressed, turns):
    rows = []
    for brand in brands:
        row = stats[brand]
        seen, threads = addressed[brand], turns[brand]
        rows.append({
            "brand": brand,
            "outbound_replies": row["replies"],
            "deflection_rate": row["deflections"] / max(row["replies"], 1),
            "median_reply_words": float(np.median(row["words"])) if row["words"] else 0.0,
            "inbound_answered_rate": len(seen & replied_to[brand]) / max(len(seen), 1),
            "multi_turn_rate": sum(v >= 2 for v in threads.values()) / max(len(threads), 1),
            "distinct_customers": len(threads),
            "inbound_addressed": len(seen),
        })
    frame = pd.DataFrame(rows).sort_values("outbound_replies", ascending=False)
    return frame.round({"deflection_rate": 4, "inbound_answered_rate": 4,
                        "multi_turn_rate": 4, "median_reply_words": 1})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out", default="data/interim/brand_profile.csv")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    print(f"reading {args.csv} in chunks of {args.chunksize:,} rows (full scan, no sampling)")
    brands = find_top_brands(args.csv, args.chunksize, args.top)
    print(f"top {len(brands)} brands by outbound volume: {', '.join(brands)}")
    stats, replied_to, addressed, turns = profile(args.csv, args.chunksize, brands)
    table = summarise(brands, stats, replied_to, addressed, turns)
    table.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")
    print(table.to_string(index=False))
    print("\nthread = one (brand, customer) pair; multi_turn_rate is the share of those "
          "\nwhere the brand posted 2+ non-deflecting replies of "
          f"{MIN_SUBSTANTIVE_WORDS}+ content words.")


if __name__ == "__main__":
    main()
