"""Try the agent on a message of your own. Prints the verdict; touches no cache.

    python src/try_agent.py "@British_Airways my bag never arrived at JFK, what do I do?"

Shows both the model's own auto/escalate call and what the deployed policy would
do at the harness's dev-tuned threshold, because those differ: the headline
numbers use the threshold, not the model's call.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import ask, client_or_exit, retriever  # noqa: E402

THRESHOLD = 0.15  # the operating point harness.py picked on dev for escalate-recall >= 0.95


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", help="the inbound tweet, in quotes")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--show-prompt", action="store_true",
                    help="also print the retrieved examples the model was shown")
    args = ap.parse_args()

    prompt = retriever(args.k)(args.text)
    if args.show_prompt:
        print(prompt + "\n" + "-" * 72)
    v = ask(client_or_exit("this needs a live call"), args.model, prompt)
    policy = "escalate" if v["escalate_score"] >= args.threshold else "auto"
    print(f"intent          : {v['intent']}")
    print(f"safety_flags    : {', '.join(v['safety_flags']) or 'none'}")
    print(f"model's call    : {v['auto_or_escalate']}   (escalate_score {v['escalate_score']:.2f})")
    print(f"policy @ {args.threshold:.2f}   : {policy}")
    print(f"reason          : {v['reason']}")
    print(f"draft reply     : {v['draft_reply']}   [{len(v['draft_reply'])} chars]")


if __name__ == "__main__":
    main()
