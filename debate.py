#!/usr/bin/env python3
"""
Multi-agent debate: "Research as a Career in the Age of AGI"

Two LLM debaters (Proposition / Opposition) argue the motion from the debate
brief, then an independent judge scores the debate against the brief's
adjudication criteria. Runs on the Groq API (OpenAI-compatible endpoint).

Usage:
    pip install requests
    export GROQ_API_KEY=gsk_...          # or put the key in groq_api_key.txt
    python debate.py                     # default: 2 rebuttal rounds, English
    python debate.py --rounds 3 --lang zh
    python debate.py --prop-model openai/gpt-oss-120b --opp-model openai/gpt-oss-20b
    python debate.py --list-models       # show models available to your key
    python debate.py --dry-run           # print the prompts without calling the API

Output: a Markdown transcript and a JSON log in ./debate_runs/
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Debate brief (from the English tab of the debate doc)
# ---------------------------------------------------------------------------

MOTION = (
    "This House believes that, if AGI arrives in 2047, a first-year undergraduate "
    "who wants research as a career should commit to the scientific path rather "
    "than turn to human-centered work that AI cannot replace."
)

BACKGROUND = """\
In September 2025, Anthropic researcher Jascha Sohl-Dickstein posted the outline of a
talk to young researchers, arguing that AI systems may surpass the best humans at every
intellectual task within a few years. He warned that the worst outcome is to spend two
years making steady, linear progress, only to be overtaken by someone who writes a simple
prompt, and argued that when society is being restructured, the "safe" career choice is
not actually safe. This motion takes his warning seriously but stretches the timeline to
2047 and asks what a student entering university today should do."""

GROUND_RULES = """\
1. Research as a livelihood: the debate concerns research as a profession (pay, advancement,
   supporting a family). Research pursued purely out of curiosity, or by a "gentleman
   scientist" with independent means, is out of scope.
2. AGI in 2047 is a given. Neither side may contest whether or when AGI arrives.
3. Reference timeline: enter college fall 2026 -> bachelor's ~2030 -> PhD ~2035 ->
   tenure-track ~2040 -> tenure decision ~2046, almost exactly when AGI arrives."""

PROP_CASE = """\
Core claim: The two decades before AGI are the last era in which humans set the direction
of science, and quite possibly its most productive one. A student who learns to work
alongside AI will have more leverage over discovery than any generation before.
Key arguments:
1. The window is real and long: 21 years to train deeply, earn a PhD, and lead independent work.
2. Asking the right question (scientific taste) becomes the bottleneck as execution gets cheap.
3. History favors scientists who adopt the new instrument (telescope, computer, sequencer).
4. Flexibility, not abandonment: move between academia, industry labs, startups; pick problems
   that grow more tractable as compute grows.
5. There is no safe harbor: every sector is exposed, so choose the highest upside and purpose.
6. Scientists will shape how AI is used to generate and check knowledge."""

OPP_CASE = """\
Core claim: Academia, and even the arts, may be transformed beyond recognition by AI. The wiser
response is to invest in relational work (nursing and caregiving, teaching and mentoring,
counseling, coaching, hospitality), whose value comes from one human being present for another.
This differs from manual labor and manufacturing, which robots can largely replace. It is a
considered choice and a better way to live, not a retreat.
Key arguments:
1. Research sits on the front line of automation; intellectual work is where AI advances fastest.
2. The academic career pays off too late: stability ~2040, tenure decision as AGI arrives.
3. Human presence will command a premium as everything else becomes cheap (services that depend
   on human time grow relatively more valuable).
4. Relational work builds a life, not just a resume: community, practical wisdom, stability sooner.
5. This is a choice, not a surrender: a good life is measured more by connection than by output."""

POINTS_OF_CLASH = """\
- When will AI acquire scientific taste?
- Can the investment in a research career be recovered?
- Is relational work truly irreplaceable?
- Which path carries less risk?"""

CRITERIA = """\
1. Fidelity to the premise: argues within the 2047 timeline, not "within a few years" or "never".
2. Fidelity to the scope: treats research as a profession (pay, stability, time to payoff),
   not passion or calling.
3. Evidence for irreplaceability: claims about uniquely human value backed by concrete reasons.
4. Engagement with the other side's risks: Proposition must answer the timing mismatch;
   Opposition must answer that relational work will also be disrupted."""

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LANG_NOTE = {
    "en": "Speak in English.",
    "zh": "Speak in Simplified Chinese (简体中文).",
}


def debater_system(side: str, lang: str) -> str:
    own, other = (PROP_CASE, OPP_CASE) if side == "Proposition" else (OPP_CASE, PROP_CASE)
    other_side = "Opposition" if side == "Proposition" else "Proposition"
    return f"""You are the {side} speaker in a formal, Oxford-style debate.

MOTION: {MOTION}

BACKGROUND:
{BACKGROUND}

GROUND RULES (binding on both sides):
{GROUND_RULES}

YOUR CASE ({side}):
{own}

THE {other_side.upper()} CASE YOU MUST DEFEAT:
{other}

POINTS OF CLASH:
{POINTS_OF_CLASH}

HOW YOU WILL BE JUDGED:
{CRITERIA}

Rules of conduct:
- Argue your side wholeheartedly; never concede the motion.
- Engage directly with what your opponent actually said; quote or paraphrase their points before rebutting.
- Use concrete reasoning, examples, and mechanisms rather than slogans. Do not invent statistics or fake citations.
- Stay within the ground rules; if your opponent breaks them, call it out.
- Respect the word limit given for each speech. No headings, write as a spoken speech.
- {LANG_NOTE[lang]}"""


JUDGE_SYSTEM = f"""You are an experienced, impartial debate adjudicator.

MOTION: {MOTION}

GROUND RULES:
{GROUND_RULES}

ADJUDICATION CRITERIA:
{CRITERIA}

Judge only what was argued in the transcript, not your own view of the motion.
Do not reward length or rhetoric alone. Penalize rule violations, unsupported claims,
and dropped arguments."""

SPEECHES = {
    "opening": ("Opening speech", 350,
                "Deliver your opening (constructive) speech. Establish your case and pre-empt the strongest objection."),
    "rebuttal": ("Rebuttal", 300,
                 "Deliver a rebuttal. Attack the strongest point your opponent made in their latest speech, "
                 "defend your case against their attacks, and advance at least one point of clash."),
    "closing": ("Closing speech", 200,
                "Deliver your closing speech. Do not introduce new arguments. Summarize why your side "
                "won the key points of clash."),
}

# ---------------------------------------------------------------------------
# Groq API
# ---------------------------------------------------------------------------


def load_api_key(key_file: str) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    path = Path(key_file)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    sys.exit("No API key found. Set GROQ_API_KEY or create groq_api_key.txt.")


def chat(api_key, model, messages, temperature, max_tokens, retries=6):
    """Call the chat completions endpoint, retrying on rate limits and server errors."""
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  network error ({e}); retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            return text, data.get("usage", {})
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("retry-after", 2 ** (attempt + 1)))
            print(f"  HTTP {r.status_code}; retrying in {wait:.0f}s")
            time.sleep(wait)
            continue
        sys.exit(f"API error {r.status_code}: {r.text[:500]}")
    sys.exit("Gave up after repeated rate-limit / server errors.")


def list_models(api_key):
    r = requests.get(f"{API_BASE}/models",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    r.raise_for_status()
    for m in sorted(r.json()["data"], key=lambda m: m["id"]):
        print(m["id"])


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


def transcript_text(transcript):
    return "\n\n".join(f"[{t['speaker']} - {t['label']}]\n{t['text']}" for t in transcript)


def build_order(rounds):
    order = [("Proposition", "opening"), ("Opposition", "opening")]
    for _ in range(rounds):
        order += [("Proposition", "rebuttal"), ("Opposition", "rebuttal")]
    # Opposition closes first; Proposition, which carries the burden, has the last word.
    order += [("Opposition", "closing"), ("Proposition", "closing")]
    return order


def run(args):
    api_key = None if args.dry_run else load_api_key(args.key_file)
    models = {"Proposition": args.prop_model, "Opposition": args.opp_model}
    transcript, usage_total = [], {"prompt_tokens": 0, "completion_tokens": 0}

    for side, kind in build_order(args.rounds):
        label, words, instruction = SPEECHES[kind]
        history = transcript_text(transcript) if transcript else "(No speeches yet. You speak first.)"
        user_msg = (f"DEBATE SO FAR:\n{history}\n\n"
                    f"YOUR TASK: {instruction}\nWord limit: about {words} words.")
        messages = [{"role": "system", "content": debater_system(side, args.lang)},
                    {"role": "user", "content": user_msg}]

        print(f"\n=== {side} - {label} ({models[side]}) ===")
        if args.dry_run:
            text = f"<{side} {label} would be generated here>"
            if not transcript:
                print(messages[0]["content"][:1500], "\n...")
        else:
            text, usage = chat(api_key, models[side], messages,
                               args.temperature, args.max_tokens)
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)
        print(text)
        transcript.append({"speaker": side, "label": label, "model": models[side], "text": text})

    # --- Judge ---
    judge_user = f"""TRANSCRIPT:
{transcript_text(transcript)}

Evaluate the debate. First write a short adjudication (about 250 words) explaining the
decisive points of clash. Then output ONLY a JSON object in a ```json code block:
{{
  "scores": {{
    "Proposition": {{"premise": 1-10, "scope": 1-10, "evidence": 1-10, "engagement": 1-10}},
    "Opposition":  {{"premise": 1-10, "scope": 1-10, "evidence": 1-10, "engagement": 1-10}}
  }},
  "winner": "Proposition" or "Opposition",
  "reason": "one sentence"
}}
{LANG_NOTE[args.lang]}"""
    print(f"\n=== Judge ({args.judge_model}) ===")
    if args.dry_run:
        verdict_text, verdict = "<judge verdict would be generated here>", None
    else:
        verdict_text, usage = chat(api_key, args.judge_model,
                                   [{"role": "system", "content": JUDGE_SYSTEM},
                                    {"role": "user", "content": judge_user}],
                                   0.2, args.max_tokens)
        for k in usage_total:
            usage_total[k] += usage.get(k, 0)
        verdict = parse_json(verdict_text)
    print(verdict_text)

    save(args, transcript, verdict_text, verdict, usage_total)


def parse_json(text):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.search(r"(\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def save(args, transcript, verdict_text, verdict, usage):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md = [f"# Debate transcript ({stamp})", "", f"**Motion**: {MOTION}", "",
          f"Models: Proposition `{args.prop_model}` · Opposition `{args.opp_model}` · "
          f"Judge `{args.judge_model}` · rebuttal rounds: {args.rounds}", ""]
    for t in transcript:
        md += [f"## {t['speaker']}: {t['label']}", "", t["text"], ""]
    md += ["## Judge's verdict", "", verdict_text, ""]
    md_path = out_dir / f"debate_{stamp}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    log = {"motion": MOTION, "args": {k: v for k, v in vars(args).items() if k != "key_file"},
           "transcript": transcript, "verdict_text": verdict_text,
           "verdict": verdict, "usage": usage}
    (out_dir / f"debate_{stamp}.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved: {md_path} (+ .json)")
    if verdict:
        print(f"Winner: {verdict.get('winner')} - {verdict.get('reason')}")
    if usage["prompt_tokens"]:
        print(f"Tokens used: {usage['prompt_tokens']} in / {usage['completion_tokens']} out")


def main():
    p = argparse.ArgumentParser(description="Run a two-sided multi-agent debate on Groq.")
    p.add_argument("--rounds", type=int, default=2, help="number of rebuttal rounds (default 2)")
    p.add_argument("--lang", choices=["en", "zh"], default="en", help="debate language")
    p.add_argument("--prop-model", default=DEFAULT_MODEL)
    p.add_argument("--opp-model", default=DEFAULT_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.8, help="debater temperature")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="per-call cap; reasoning models spend some tokens thinking")
    p.add_argument("--key-file", default="groq_api_key.txt")
    p.add_argument("--out-dir", default="debate_runs")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.list_models:
        list_models(load_api_key(args.key_file))
        return
    run(args)


if __name__ == "__main__":
    main()
