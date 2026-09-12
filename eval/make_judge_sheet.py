"""Write the blank sheet a human fills in to measure judge agreement.

Samples test drafts (half AUTO, half ESCALATE so the reference criterion gets
coverage), shows the scorer exactly what the judge saw and nothing the judge
said, and leaves one blank column per rubric criterion. Also writes the rubric
wording to eval/rubric.txt so it sits next to the sheet while scoring.
"""
import argparse
from pathlib import Path

import pandas as pd

from judge import CRITERIA, rubric_text, test_rows

SEED = 42


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="eval/preds_agent.csv")
    ap.add_argument("--out", default="data/golden/judge_human_scores.csv", type=Path)
    ap.add_argument("--per-class", type=int, default=25, help="rows per AUTO / ESCALATE")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sheet (DESTROYS hand-entered scores)")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists -- refusing to overwrite hand scores without --force")

    rows = test_rows(args.preds)
    picks = [grp.sample(n=min(args.per_class, len(grp)), random_state=SEED)
             for _, grp in rows.groupby("auto_or_escalate")]
    sheet = pd.concat(picks).sample(frac=1.0, random_state=SEED)  # shuffle: no class runs
    sheet = sheet.rename(columns={"auto_or_escalate": "correct_handling",
                                  "reference": "reference_reply_text"})
    sheet = sheet[["id", "inbound_text", "correct_handling", "reason",
                   "reference_reply_text", "draft_reply"]].copy()
    for name, _ in CRITERIA:
        sheet[name] = ""  # fill with yes / no (n/a for the reference criterion only)
    sheet["scorer_notes"] = ""
    sheet.to_csv(args.out, index=False)
    Path("eval/rubric.txt").write_text(rubric_text() + "\n")
    print(f"wrote {len(sheet)} rows to {args.out}  "
          f"({sheet['correct_handling'].value_counts().to_dict()})")
    print("rubric wording written to eval/rubric.txt -- score each criterion yes/no; "
          f"{CRITERIA[-1][0]} is n/a when no reference is shown")


if __name__ == "__main__":
    main()
