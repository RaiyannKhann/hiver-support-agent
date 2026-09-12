"""The support agent: one inbound tweet in, intent + escalation + draft reply out.

Retrieves the k most similar past BA conversations from the cached index as
style examples, asks Claude for a structured verdict, and appends it to the
predictions file. That file is also the cache: ids already in it are skipped, so
the offline path needs no key and a killed run resumes where it stopped.
Reads the worksheet text only -- never the hand labels.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_intents import MODEL as EMBED_MODEL, clean  # noqa: E402

WORKSHEET = "data/golden/to_label.csv"
INDEX = Path("data/interim/retrieval_index")
INTENTS = ["policy_question", "booking_and_seats", "digital_failure", "disruption",
           "mishandled_baggage", "claims_and_cases", "service_complaint", "praise", "other"]
FLAGS = ["legal", "medical", "minor", "pii", "staff_conduct", "third_party_booking", "vulnerable"]
OUT_COLS = ["id", "intent", "auto_or_escalate", "escalate_score", "draft_reply",
            "safety_flags", "reason", "model"]
WS_RE = re.compile(r"\s+")

SYSTEM = """You are the social media support agent for British Airways. You handle one inbound public tweet at a time, with no conversation history, and produce the customer's intent, a decision whether a public reply can handle it or a human with account access must take over, and a draft public reply.

Intents (pick exactly one):
- policy_question: asks what is allowed or how something works (baggage allowance, pets, visas, fare rules, lounges). Usually answerable from public information.
- booking_and_seats: a specific booking, seat selection, upgrade, check-in or ticket change.
- digital_failure: website, app, login or online check-in not working; error messages.
- disruption: a delay, cancellation, diversion, missed connection, or rebooking after one.
- mishandled_baggage: lost, delayed or damaged bags.
- claims_and_cases: an existing complaint, refund, compensation claim or case reference; chasing a response.
- service_complaint: a bad experience with staff, crew, food, cabin or service, not tied to an open case.
- praise: positive feedback with no request.
- other: anything else, including things not about BA support, marketing, jokes, empty mentions.

Escalation policy. Escalate when resolving the message needs anything a public reply cannot do: looking up a booking, claim or account; issuing a refund, rebooking or compensation; investigating a specific incident or member of staff. Always escalate when any safety flag applies. Choose auto when a public reply fully resolves it: general policy answers, generic technical guidance, acknowledging feedback or praise, or messages needing no action.

Safety flags (zero or more): legal (threatens legal action, mentions lawyers or regulators); medical (illness, injury, medication, disability needs); minor (a child is affected); pii (the message already exposes booking references, phone numbers, passport or card details); staff_conduct (accuses a specific or named employee); third_party_booking (booked via an agent or another airline); vulnerable (bereavement, distress, or an elderly or disabled traveller in difficulty).

Draft reply rules. Write the public tweet BA would post: under 280 characters, in the warm, plain British tone of the examples, addressed to the customer. Never invent flight status, policy specifics, compensation amounts or timelines you cannot know from the message. If escalating, say what you need (usually a DM with the booking reference) and promise no outcome. If auto, answer or acknowledge. No hashtags or emoji unless the customer used them. Do not start with an @handle; the platform adds it. Sign off with " ^BA".

escalate_score is your probability, 0 to 1, that a human with account access is needed. Use the whole range, not just 0 and 1."""

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "safety_flags": {"type": "array", "items": {"type": "string", "enum": FLAGS}},
        "reason": {"type": "string", "description": "one short sentence"},
        "escalate_score": {"type": "number"},
        "auto_or_escalate": {"type": "string", "enum": ["auto", "escalate"]},
        "draft_reply": {"type": "string"},
    },
    "required": ["intent", "safety_flags", "reason", "escalate_score", "auto_or_escalate",
                 "draft_reply"],
    "additionalProperties": False,
}


def build_user_message(text, corpus, neighbours):
    lines = ["Similar past conversations (customer tweet -> BA's public reply):", ""]
    for n, i in enumerate(neighbours, 1):
        lines.append(f"{n}. Customer: {WS_RE.sub(' ', corpus['inbound_text'][i])}")
        lines.append(f"   BA: {WS_RE.sub(' ', corpus['reply_text'][i])}")
    return "\n".join(lines + ["", "Now handle this customer tweet:", "", text])


def ask(client, model, user_message):
    resp = client.messages.create(
        model=model, max_tokens=1024, system=SYSTEM,
        messages=[{"role": "user", "content": user_message}],
        # low effort: short classification + a tweet; thinking tokens dominate cost otherwise
        output_config={"effort": "low",
                       "format": {"type": "json_schema", "schema": SCHEMA}})
    if resp.stop_reason == "refusal":  # safest possible row, so the harness still runs
        return {"intent": "other", "safety_flags": [], "reason": "model refused",
                "escalate_score": 1.0, "auto_or_escalate": "escalate", "draft_reply": ""}
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def retriever(k):
    """Loads the cached index once; returns a function that builds the full prompt for a message."""
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors

    corpus = pd.read_csv(INDEX.with_suffix(".csv"), dtype=str)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(np.load(INDEX.with_suffix(".npy")))
    embedder = SentenceTransformer(EMBED_MODEL)

    def prompt_for(text):
        _, idx = nn.kneighbors(embedder.encode([clean(text)], normalize_embeddings=True))
        return build_user_message(text, corpus, idx[0])
    return prompt_for


def client_or_exit(why):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or a profile
    if not (client.api_key or client.auth_token or client.credentials):
        raise SystemExit(f"no API credentials found and {why}; set ANTHROPIC_API_KEY")
    return client


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="eval/preds_agent.csv", help="predictions + cache")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--k", type=int, default=5, help="retrieved examples per query")
    ap.add_argument("--limit", type=int, help="stop after this many API calls (smoke test)")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the assembled prompt for the first pending id and exit")
    args = ap.parse_args()

    todo = pd.read_csv(WORKSHEET, dtype=str)[["id", "inbound_text"]]
    done = (pd.read_csv(args.out, dtype=str) if Path(args.out).exists()
            else pd.DataFrame(columns=OUT_COLS))
    pending = todo[~todo["id"].isin(done["id"])]
    print(f"model={args.model}  k={args.k}  cached={len(done)}  pending={len(pending)}")
    if pending.empty:
        print(f"{args.out} is complete; nothing to call")
        return
    if args.limit:
        pending = pending.head(args.limit)

    prompt_for = retriever(args.k)
    if args.show_prompt:
        print("\n--- system ---\n" + SYSTEM + "\n\n--- user ---\n" + prompt_for(pending.iloc[0]["inbound_text"]))
        return
    client = client_or_exit(f"the cache at {args.out} is incomplete")

    rows = done.to_dict("records")
    for i, (id_, text) in enumerate(zip(pending["id"], pending["inbound_text"]), 1):
        v = ask(client, args.model, prompt_for(text))
        rows.append({"id": id_, "intent": v["intent"], "auto_or_escalate": v["auto_or_escalate"],
                     "escalate_score": min(1.0, max(0.0, float(v["escalate_score"]))),
                     "draft_reply": v["draft_reply"], "safety_flags": ",".join(v["safety_flags"]),
                     "reason": v["reason"], "model": args.model})
        # rewrite after every call: a crash or Ctrl-C loses nothing and the rerun resumes
        pd.DataFrame(rows, columns=OUT_COLS).to_csv(args.out, index=False)
        print(f"  {i:>3}/{len(pending)}  {id_:>8}  {v['intent']:19} {v['auto_or_escalate']:9} "
              f"{v['escalate_score']:.2f}  {v['draft_reply'][:60]}", flush=True)
    print(f"\nwrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
