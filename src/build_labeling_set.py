"""Build the blank labelling worksheet for the golden set.

Stratifies BA openers across 9 intents with a crude keyword pre-filter, purely so
every intent has coverage. The pre-filter bucket is NEVER written into the intent
column -- it goes to a sidecar -- because showing a guessed label anchors the
human labeller. All label columns ship blank and stay that way.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from cluster_intents import BRAND, extract_openers, norm_key

SEED = 42
RANDOM_N = 10  # unstratified, so safety-flag cases the keywords cannot see still appear
CREATED_FMT = "%a %b %d %H:%M:%S %z %Y"
REPLY_COLS = ["author_id", "created_at", "text", "in_response_to_tweet_id"]
OUT_COLS = ["id", "brand", "inbound_text", "reply_text", "intent", "safety_flag",
            "auto_or_escalate", "reason", "reference_reply"]
TARGETS = {"policy_question": 30, "booking_and_seats": 30, "digital_failure": 28,
           "other": 28, "praise": 22, "claims_and_cases": 22, "service_complaint": 22,
           "disruption": 18, "mishandled_baggage": 10}

# Ordered: first match wins, most specific first. These are throwaway heuristics for
# coverage only -- no hint list was supplied, so they are read off the k=12 clusters.
PATTERNS = [
    ("mishandled_baggage", r"\b(bag|bags|baggage|luggage|suitcase|pram|stroller|"
                           r"buggy|carousel|lost my \w+|delayed bag)\b"),
    ("disruption", r"\b(delay|delayed|cancel|cancelled|canceled|diverted|divert|"
                   r"missed (my )?connection|rebook|re-book|strike|stranded|on time)\b"),
    ("digital_failure", r"\b(website|web site|app|online|error|site|login|log in|"
                        r"password|crash|glitch|404|won'?t load|not working|down)\b"),
    ("claims_and_cases", r"\b(claim|compensation|comp|refund|reimburse|case (ref|number)|"
                         r"reference number|eu ?261|still waiting|no (reply|response)|chase)\b"),
    ("booking_and_seats", r"\b(seat|seats|seating|book|booking|booked|reserve|"
                          r"reservation|upgrade|check ?in|allocat)\w*\b"),
    ("policy_question", r"\b(can i|could i|am i (allowed|able)|do you (allow|accept)|"
                        r"policy|allowance|rules?|visa|pet|infant|liquids|what happens if|"
                        r"how (do|much|many))\b|\?"),
    ("service_complaint", r"\b(rude|staff|service|disgusting|appalling|awful|worst|"
                          r"shocking|disgrace|never (fly|flying)|unacceptable|complain)\w*\b"),
]
COMPILED = [(name, re.compile(pat, re.I)) for name, pat in PATTERNS]

# Praise is checked last and needs a strong marker with no complaint word in the same
# message: bare "thanks" is British politeness, not sentiment, and sarcasm reads as praise.
PRAISE_RE = re.compile(r"\b(great|excellent|well done|impressed)\b|thank you so much", re.I)
COMPLAINT_RE = re.compile(r"\b(sorry|disappointed|poor|worst|waiting|lost|no response)\b",
                          re.I)


def bucket_of(text):
    for name, rx in COMPILED:
        if rx.search(text):
            return name
    if PRAISE_RE.search(text) and not COMPLAINT_RE.search(text):
        return "praise"
    return "other"


def first_replies(path, brand, ids, chunksize):
    """The brand's earliest direct reply to each opener, by timestamp."""
    hits = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=REPLY_COLS, dtype=str,
                                          chunksize=chunksize)):
        chunk = chunk.dropna(subset=["text", "in_response_to_tweet_id"])
        hits.append(chunk[(chunk["author_id"] == brand)
                          & chunk["in_response_to_tweet_id"].isin(ids)])
        print(f"  reply scan chunk {i + 1}: {sum(len(h) for h in hits):,} matched",
              flush=True)
    replies = pd.concat(hits, ignore_index=True)
    replies["ts"] = pd.to_datetime(replies["created_at"], format=CREATED_FMT)
    replies = replies.sort_values("ts").drop_duplicates("in_response_to_tweet_id")
    return dict(zip(replies["in_response_to_tweet_id"], replies["text"]))


def draw(pool, targets, random_n):
    """Stratified draw, then a small unstratified top-up from what is left."""
    picks, taken = [], set()
    for bucket, target in targets.items():
        avail = pool[(pool["bucket"] == bucket) & ~pool["tweet_id"].isin(taken)]
        got = avail.sample(n=min(target, len(avail)), random_state=SEED)
        picks.append(got.assign(draw="stratified"))
        taken.update(got["tweet_id"])
    rest = pool[~pool["tweet_id"].isin(taken)]
    picks.append(rest.sample(n=min(random_n, len(rest)),
                             random_state=SEED).assign(draw="random"))
    return pd.concat(picks, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/twcs/twcs.csv")
    ap.add_argument("--out", default="data/golden/to_label.csv", type=Path)
    ap.add_argument("--strata-out", default="eval/labeling_strata.csv", type=Path)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing worksheet (DESTROYS hand-entered labels)")
    args = ap.parse_args()

    # data/golden is hand-authored ground truth; never clobber it silently.
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists -- refusing to overwrite it. "
                         f"Pass --force only if you are certain no hand labels are in it.")

    print(f"seed={SEED}  brand={BRAND}  targets={sum(TARGETS.values())}+{RANDOM_N} random")
    openers = extract_openers(args.csv, BRAND, args.chunksize)
    openers = openers.assign(key=openers["text"].map(norm_key))
    pool = openers[openers["key"].str.len() > 0].drop_duplicates("key").copy()
    pool["bucket"] = pool["text"].map(bucket_of)
    print(f"{len(openers):,} openers -> {len(pool):,} after near-duplicate dedup")

    picked = draw(pool, TARGETS, RANDOM_N)
    replies = first_replies(args.csv, BRAND, set(picked["tweet_id"]), args.chunksize)
    # shuffle so the worksheet is not ordered by bucket, which would anchor the labeller
    picked = picked.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out = pd.DataFrame({"id": picked["tweet_id"], "brand": BRAND,
                        "inbound_text": picked["text"],
                        "reply_text": picked["tweet_id"].map(replies).fillna("")})
    for col in OUT_COLS[4:]:
        out[col] = ""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out[OUT_COLS].to_csv(args.out, index=False)

    # sampling weights live beside the worksheet, not in it: the methodology needs them
    # for reweighted population estimates, but the labeller must not see the bucket.
    sizes = pool["bucket"].value_counts()
    drawn = picked["bucket"].value_counts()
    strata = picked[["tweet_id", "bucket", "draw"]].rename(columns={"tweet_id": "id"})
    strata["stratum_population"] = strata["bucket"].map(sizes)
    strata["stratum_sampled"] = strata["bucket"].map(drawn)
    strata["weight"] = strata["stratum_population"] / strata["stratum_sampled"]
    strata.to_csv(args.strata_out, index=False)

    print(f"\nwrote {len(out)} rows to {args.out}  (all label columns blank)")
    print(f"wrote strata + weights to {args.strata_out}\n")
    print(f"{'bucket':20} {'target':>6} {'strat':>6} {'rand':>5} {'pool':>7} {'weight':>8}")
    strat = picked[picked["draw"] == "stratified"]["bucket"].value_counts()
    rand = picked[picked["draw"] == "random"]["bucket"].value_counts()
    for bucket in TARGETS:
        s, r, target = strat.get(bucket, 0), rand.get(bucket, 0), TARGETS[bucket]
        short = "  <-- SHORT" if s < target else ""
        print(f"{bucket:20} {target:>6} {s:>6} {r:>5} {sizes.get(bucket, 0):>7} "
              f"{sizes.get(bucket, 0) / max(s + r, 1):>8.1f}{short}")
    print(f"{'TOTAL':20} {sum(TARGETS.values()):>6} {strat.sum():>6} "
          f"{rand.sum():>5} {len(pool):>7}")
    print(f"\nwith a reply: {(out['reply_text'] != '').sum()}/{len(out)}")


if __name__ == "__main__":
    main()
