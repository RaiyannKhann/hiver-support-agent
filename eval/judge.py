"""LLM-as-judge for the agent's draft replies, on binary criteria.

Grades every test-split draft against the rubric below, caches verdicts in
eval/judge_scores.csv (resume-safe, like the agent), and prints pass rates with
Wilson intervals. CRITERIA is the single source of truth: the human scoring sheet
is generated from the same list, so judge and human answer identical questions.
The judge is a different model family from the agent, to avoid self-preference.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from harness import fmt, load_golden

REF = "consistent_with_reference"
CRITERIA = [
    ("addresses_message", "Does the reply respond to the specific thing this customer wrote, "
     "rather than being boilerplate that could answer any tweet?"),
    ("no_fabrication", "Does the reply avoid asserting anything it cannot know from the tweet: "
     "flight status, policy specifics, prices, compensation, timelines, or what the customer "
     "'will' receive?"),
    ("right_next_step", "Does the reply leave the customer knowing what happens next: it answers "
     "the question, or names exactly what to send and where, or, when the tweet needs no action, "
     "acknowledges it so nothing is left hanging?"),
    ("matches_decision", "Given the correct handling shown: for ESCALATE, does it hand off without "
     "promising an outcome; for AUTO, does it resolve publicly without pushing to DM?"),
    ("tone", "Is it polite, warm and professional, with no sarcasm, blame or defensiveness, "
     "and fit to post publicly as British Airways?"),
    (REF, "Only when a reference reply is shown: is the draft's substance consistent with it, "
     "same resolution direction and no contradiction? Answer n/a when there is no reference."),
]
SYSTEM = ("You grade one draft public reply from British Airways' Twitter support account. "
          "Answer every criterion strictly yes or no; a partial pass is a no. Judge only the "
          "draft in front of you and do not rewrite it. The correct handling (AUTO or ESCALATE) "
          "was decided by a human and is given: do not second-guess it, grade whether the draft "
          "acts on it.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": [n for n, _ in CRITERIA],
          "properties": {n: {"type": "object", "additionalProperties": False,
                             "required": ["verdict", "why"],
                             "properties": {"verdict": {"type": "string", "enum":
                                            ["yes", "no"] + (["n/a"] if n == REF else [])},
                                            "why": {"type": "string"}}}
                         for n, _ in CRITERIA}}
COLS = ["id"] + [n for n, _ in CRITERIA] + [f"{n}_why" for n, _ in CRITERIA] + ["model"]


def rubric_text():
    return "\n".join(f"{i}. {n}: {q}" for i, (n, q) in enumerate(CRITERIA, 1))


def test_rows(preds_path):
    """Test-split golden rows joined to the drafts; reference text resolved from the label."""
    g = load_golden().merge(pd.read_csv(preds_path, dtype=str).fillna("")[["id", "draft_reply"]],
                            on="id")
    g["reference"] = g["reply_text"].where(g["reference_reply"] == "see reply_text", "")
    return g[g["split"] == "test"].reset_index(drop=True)


def build_user_message(r):
    ref = (f'Reference reply (what BA actually posted):\n"""{r.reference}"""' if r.reference
           else f"Reference reply: none (answer n/a for {REF})")
    return "\n".join([f'Customer tweet:\n"""{r.inbound_text}"""', "",
                      f"Correct handling (human decision): {r.auto_or_escalate.upper()} -- {r.reason}",
                      "", ref, "", f'Draft reply to grade:\n"""{r.draft_reply}"""', "",
                      "Criteria:", rubric_text()])


def grade(client, model, message):
    resp = client.messages.create(model=model, max_tokens=1024, system=SYSTEM,
                                  messages=[{"role": "user", "content": message}],
                                  output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def summary(rows, scores):
    d = rows.merge(scores, on="id")
    print(f"\n== judge pass rates on test drafts (n={len(d)}, judge={scores['model'].iloc[0]}) ==")
    print(f"{'':44}{'raw [95% Wilson]':<24}reweighted [95% Wilson, Kish n]")
    for n, _ in CRITERIA:
        m = d[n] != "n/a"
        print(f"  {n:26} n={int(m.sum()):<4}     {fmt(d.loc[m, n] == 'yes', d.loc[m, 'weight'])}")
    allc = [n for n, _ in CRITERIA if n != REF]
    print(f"  {'all five non-reference':26} n={len(d):<4}     "
          f"{fmt((d[allc] == 'yes').all(axis=1), d['weight'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="eval/preds_agent.csv")
    ap.add_argument("--out", default="eval/judge_scores.csv", help="verdicts + cache")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, help="stop after this many API calls (smoke test)")
    ap.add_argument("--print-rubric", action="store_true")
    args = ap.parse_args()
    if args.print_rubric:
        print(rubric_text())
        return

    rows = test_rows(args.preds)
    done = (pd.read_csv(args.out, dtype=str, keep_default_na=False)  # "n/a" is a value, not NaN
            if Path(args.out).exists() else pd.DataFrame(columns=COLS))
    pending = rows[~rows["id"].isin(done["id"])]
    print(f"judge={args.model}  cached={len(done)}  pending={len(pending)}")
    if pending.empty:
        summary(rows, done)
        return
    if args.limit:
        pending = pending.head(args.limit)

    import anthropic

    client = anthropic.Anthropic()
    if not (client.api_key or client.auth_token or client.credentials):
        raise SystemExit(f"no API credentials and {args.out} is incomplete; set ANTHROPIC_API_KEY")
    out = done.to_dict("records")
    for i, r in enumerate(pending.itertuples(index=False), 1):
        v = grade(client, args.model, build_user_message(r))
        row = {"id": r.id, "model": args.model}
        row.update({n: v[n]["verdict"] for n, _ in CRITERIA})
        row.update({f"{n}_why": v[n]["why"] for n, _ in CRITERIA})
        out.append(row)
        pd.DataFrame(out, columns=COLS).to_csv(args.out, index=False)  # rewrite per call: resumable
        print(f"  {i:>3}/{len(pending)}  {r.id:>8}  " + " ".join(
            f"{n[:5]}={'-' if v[n]['verdict'] == 'n/a' else v[n]['verdict'][0]}"
            for n, _ in CRITERIA), flush=True)
    if len(out) == len(rows):
        summary(rows, pd.DataFrame(out, columns=COLS))


if __name__ == "__main__":
    main()
