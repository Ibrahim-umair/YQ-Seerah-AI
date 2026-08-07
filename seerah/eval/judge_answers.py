"""LLM-as-a-judge over the RAG bot's generated answers.

Structured after the course's own offline-evaluation approach (see
cohorts/2024/04-monitoring/offline-rag-evaluation.ipynb in the LLM Zoomcamp
repo): a judge LLM classifies each generated answer as RELEVANT /
PARTLY_RELEVANT / NON_RELEVANT against a reference answer, returns parsable
JSON, and - matching that notebook exactly - is run on a SAMPLE first
(they judged 150, not their whole set), not the full 304 in one pass.

One addition beyond the course's version: a second field, Faithfulness,
judged against the ACTUAL retrieved context this run (not the eval set's
supporting_quotes) - GROUNDED / PARTIALLY_GROUNDED / UNGROUNDED. This catches
hallucination even when an answer happens to sound plausible.

Why Relevance is judged against reference_answer and NEVER against
supporting_quotes' lecture numbers: the lecture series repeats itself, and
supporting_quotes only records the lecture(s) the question-writing agent
happened to find - it is not an exhaustive list of every lecture that
discusses a topic. A generated answer that is correct but cites a different
(uncatalogued) lecture must not be marked wrong. The judge is told this
explicitly and never sees supporting_quotes at all - only the reference
answer's content and the context actually retrieved this run.

Two backends, selected with --agentic:
  - default: seerah.answer.SeerahRAG - one retrieval call, one generation call.
  - --agentic: seerah.agent.SeerahAgent - the model decides how many times to
    search and how to reword the query, up to config.AGENT_MAX_ITERATIONS.
Either way, exactly one judge call follows generation.

Four selection modes:
  --sample FRACTION   stratified by tier, e.g. 0.1 for ~10% (a pilot)
  --full              all 304 - expensive, do this last
  --retest-file PATH  re-judge only the questions that were NOT "RELEVANT" in
                      a prior run's output file. The targeted way to check
                      "did switching to --agentic actually fix the cases the
                      simple pipeline got wrong" without re-paying for the
                      ~250 questions that were already fine.
  --ids A,B,C         judge exactly these question_ids. Combine with --repeat
                      N to run each one N independent times and check
                      self-consistency - this project's own agentic answers
                      have been observed to flip between RELEVANT and
                      PARTLY_RELEVANT on IDENTICAL input with no code change,
                      so a single verdict on a borderline question isn't
                      proof of a reproducible problem. Repeating tells you
                      which questions are consistently bad (a real gap) vs.
                      which just drew an unlucky single sample.

Usage:
    python -m seerah.eval.judge_answers --sample 0.1          # ~30 questions, stratified by tier
    python -m seerah.eval.judge_answers --full                # all 304 - expensive, do this last
    python -m seerah.eval.judge_answers --retest-file data/judge_full.json --agentic
    python -m seerah.eval.judge_answers --ids B3-002,C1-010 --agentic --repeat 3
"""

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from seerah import config
from seerah.agent import SeerahAgent
from seerah.answer import SeerahRAG, format_context
from seerah.eval.run_retrieval import load_questions

console = Console()

JUDGE_INSTRUCTIONS = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system
answering questions about the Seerah (the life of the Prophet Muhammad, peace
be upon him) from Shaykh Dr. Yasir Qadhi's 104-part lecture series.

You will judge TWO separate things. Do not conflate them.

1. Relevance - does the GENERATED ANSWER's content agree with what the
   REFERENCE ANSWER says, in substance? The reference answer is a curated
   summary of the correct answer, not the only acceptable wording, and it is
   NOT an exhaustive list of every lecture that discusses this topic - the
   series repeats itself, and the same event or person is often covered in
   more than one lecture. NEVER penalize the generated answer for citing
   different or additional lectures than the reference answer happens to
   mention, or for adding correct detail the reference answer omits. Judge
   whether its factual claims are correct and consistent with the reference
   answer - not whether its citations match any particular expected lecture.

2. Faithfulness - is the generated answer actually supported by the RETRIEVED
   CONTEXT shown below (what the generator was actually given), or does it
   state things that context does not contain? This checks for hallucination,
   independent of whether the answer happens to be historically correct.

Classify Relevance as exactly one of: "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT".
Classify Faithfulness as exactly one of: "GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED".

Respond with parsable JSON only, no code blocks, no extra text:
{
  "relevance": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT",
  "faithfulness": "GROUNDED" | "PARTIALLY_GROUNDED" | "UNGROUNDED",
  "explanation": "one or two sentences justifying both judgments"
}
""".strip()

JUDGE_PROMPT_TEMPLATE = """
QUESTION: {question}

REFERENCE ANSWER (curated summary, not exhaustive - do not require exact citation match):
{reference_answer}

GENERATED ANSWER (to be judged):
{generated_answer}

RETRIEVED CONTEXT (what the generator was actually given, for the Faithfulness judgment):
{context}
""".strip()

TIERS = ["T1", "T2", "T3"]


def stratified_sample(questions, fraction, seed):
    """Samples `fraction` of each tier independently, not a flat random
    fraction of the whole set - so a small pilot still reflects the tier
    mix (20/38/40) rather than risking an unlucky draw that's all T1."""
    rng = random.Random(seed)
    by_tier = {t: [q for q in questions if q["tier"] == t] for t in TIERS}
    sampled = []
    for tier, group in by_tier.items():
        n = max(1, round(len(group) * fraction))
        sampled.extend(rng.sample(group, min(n, len(group))))
    return sampled


def judge_llm(client, model, prompt):
    response = client.responses.create(
        model=model,
        input=[{"role": "developer", "content": JUDGE_INSTRUCTIONS},
               {"role": "user", "content": prompt}],
    )
    text = response.output_text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{exc}: {text[:200]!r}"


def judge_question(q, bot, judge_model, top_k, agentic):
    if agentic:
        answer, hits, search_log = bot.ask(q["question"])
    else:
        answer, hits = bot.rag(q["question"], top_k=top_k)
        search_log = None
    context = format_context(hits)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=q["question"],
        reference_answer=q["reference_answer"],
        generated_answer=answer,
        context=context,
    )
    verdict, parse_error = judge_llm(bot.llm_client, judge_model, prompt)

    return {
        "question_id": q["question_id"],
        "tier": q["tier"],
        "cross_episode": q["cross_episode"],
        "question": q["question"],
        "generated_answer": answer,
        "num_hits": len(hits),
        "search_log": search_log,
        "relevance": (verdict or {}).get("relevance"),
        "faithfulness": (verdict or {}).get("faithfulness"),
        "explanation": (verdict or {}).get("explanation"),
        "judge_parse_error": parse_error,
    }


def print_summary(results):
    def counts(field, values):
        c = {v: 0 for v in values}
        for r in results:
            if r[field] in c:
                c[r[field]] += 1
        return c

    n = len(results)
    errors = sum(1 for r in results if r["judge_parse_error"])

    console.print(f"\n[bold]Judged {n} questions[/bold]"
                  + (f"  ({errors} judge-output parse failures)" if errors else ""))

    rel_table = Table(title="Relevance (vs reference answer)")
    rel_table.add_column("Verdict")
    rel_table.add_column("n", justify="right")
    rel_table.add_column("%", justify="right")
    for v, c in counts("relevance", ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]).items():
        rel_table.add_row(v, str(c), f"{100 * c / n:.1f}%")
    console.print(rel_table)

    faith_table = Table(title="Faithfulness (vs retrieved context actually used)")
    faith_table.add_column("Verdict")
    faith_table.add_column("n", justify="right")
    faith_table.add_column("%", justify="right")
    for v, c in counts("faithfulness", ["GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED"]).items():
        faith_table.add_row(v, str(c), f"{100 * c / n:.1f}%")
    console.print(faith_table)

    by_tier = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)
    tier_table = Table(title="Relevance by tier")
    tier_table.add_column("Tier")
    tier_table.add_column("n", justify="right")
    tier_table.add_column("RELEVANT", justify="right")
    tier_table.add_column("PARTLY_RELEVANT", justify="right")
    tier_table.add_column("NON_RELEVANT", justify="right")
    for tier in TIERS:
        rows = by_tier.get(tier, [])
        if not rows:
            continue
        c = {"RELEVANT": 0, "PARTLY_RELEVANT": 0, "NON_RELEVANT": 0}
        for r in rows:
            if r["relevance"] in c:
                c[r["relevance"]] += 1
        tier_table.add_row(tier, str(len(rows)), str(c["RELEVANT"]), str(c["PARTLY_RELEVANT"]), str(c["NON_RELEVANT"]))
    console.print(tier_table)


def print_repeat_summary(results, repeat):
    """One row per base question_id (trials share an id suffixed "#1", "#2",
    ... - see main()). Categorizes each as consistently fine, consistently
    bad, or inconsistent, since those need different follow-up: a consistent
    problem is worth fixing, an inconsistent one is a sampling artifact that
    a single-run verdict would have wrongly presented as a fixed defect."""
    abbrev = {"RELEVANT": "R", "PARTLY_RELEVANT": "P", "NON_RELEVANT": "N", None: "?"}
    by_base = {}
    for r in results:
        by_base.setdefault(r["question_id"].split("#")[0], []).append(r)

    table = Table(title=f"Self-consistency across {repeat} independent run(s) per question")
    table.add_column("question_id")
    table.add_column("Trials (in order)")
    table.add_column("RELEVANT", justify="right")
    table.add_column("Read")
    for base_id in sorted(by_base):
        rows = by_base[base_id]
        seq = " ".join(abbrev.get(r["relevance"], "?") for r in rows)
        rel_count = sum(1 for r in rows if r["relevance"] == "RELEVANT")
        n = len(rows)
        if rel_count == n:
            read = "[green]consistently fine - earlier verdict was likely noise[/green]"
        elif rel_count == 0:
            read = "[red]consistently not RELEVANT - a real, reproducible gap[/red]"
        else:
            read = "[yellow]inconsistent - genuinely variable, not a fixed defect[/yellow]"
        table.add_row(base_id, seq, f"{rel_count}/{n}", read)
    console.print(table)


def save_checkpoint(out_path, args, selected, results):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n": len(selected), "seed": args.seed, "judge_model": args.judge_model,
                   "results": results}, f, ensure_ascii=False, indent=2)


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=float, help="fraction of each tier to judge, e.g. 0.1 for ~10%%")
    mode.add_argument("--full", action="store_true", help="judge all 304 questions - expensive, do this last")
    mode.add_argument("--retest-file", help="re-judge only the non-RELEVANT questions from a prior run's output file")
    mode.add_argument("--ids", help="comma-separated question_ids to judge directly")

    parser.add_argument("--repeat", type=int, default=1,
                        help="--ids only: run each question this many independent times, to check self-consistency")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed, for a reproducible pilot")
    parser.add_argument("--top-k", type=int, default=10, help="non-agentic mode only")
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument("--agentic", action="store_true",
                        help="use seerah.agent.SeerahAgent instead of the simple single-shot pipeline")
    parser.add_argument("--max-iterations", type=int, default=config.AGENT_MAX_ITERATIONS, help="--agentic only")
    parser.add_argument("--search-top-k", type=int, default=config.AGENT_SEARCH_TOP_K, help="--agentic only")
    parser.add_argument("--temperature", type=float, default=None,
                        help="--agentic only; 0-2, omit to use the API default (~1.0)")
    parser.add_argument("--out", default=None,
                        help="default: data/judge_pilot.json, data/judge_full.json, or "
                             "data/judge_retest[_agentic].json, depending on mode")
    args = parser.parse_args()

    questions = load_questions()
    if args.full:
        selected = questions
        default_out = config.DATA_DIR / "judge_full.json"
    elif args.retest_file:
        with open(args.retest_file, encoding="utf-8") as f:
            prior = json.load(f)
        weak_ids = {r["question_id"] for r in prior["results"] if r.get("relevance") != "RELEVANT"}
        selected = [q for q in questions if q["question_id"] in weak_ids]
        console.print(f"[bold]Retesting {len(selected)} non-RELEVANT question(s) from {args.retest_file}[/bold]")
        if args.agentic:
            suffix = f"_agentic_t{args.temperature}" if args.temperature is not None else "_agentic"
        else:
            suffix = ""
        default_out = config.DATA_DIR / f"judge_retest{suffix}.json"
    elif args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        base_selected = [q for q in questions if q["question_id"] in wanted]
        missing = wanted - {q["question_id"] for q in base_selected}
        if missing:
            console.print(f"[red]Warning: question_id(s) not found in the eval set: {sorted(missing)}[/red]")
        if args.repeat > 1:
            # Each trial gets a unique question_id ("B3-002#1", "#2", ...) so the
            # existing per-id resume/checkpoint logic works unmodified - a trial
            # is just another "question" as far as that logic is concerned.
            selected = []
            for trial in range(1, args.repeat + 1):
                for q in base_selected:
                    trial_q = dict(q)
                    trial_q["question_id"] = f"{q['question_id']}#{trial}"
                    selected.append(trial_q)
            console.print(f"[bold]{len(base_selected)} question(s) x {args.repeat} independent run(s) "
                          f"= {len(selected)} trials[/bold]")
        else:
            selected = base_selected
        suffix = "_agentic" if args.agentic else ""
        default_out = config.DATA_DIR / f"judge_repeat{suffix}.json"
    else:
        selected = stratified_sample(questions, args.sample, args.seed)
        default_out = config.DATA_DIR / "judge_pilot.json"
    out_path = Path(args.out) if args.out else default_out

    # Resume support: a question only counts as done if it has a real verdict -
    # one that failed to parse gets retried, not silently kept as a gap. This is
    # what makes it safe to re-run this exact command after a kill/crash: only
    # the questions still missing get judged (and paid for) again.
    existing_results = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing_results = json.load(f).get("results", [])
    done_ids = {r["question_id"] for r in existing_results if r.get("relevance")}
    results = [r for r in existing_results if r["question_id"] in done_ids]
    remaining = [q for q in selected if q["question_id"] not in done_ids]

    console.print(f"[bold]Loading indexes and {'agentic ' if args.agentic else ''}RAG bot...[/bold]")
    if args.agentic:
        bot = SeerahAgent(max_iterations=args.max_iterations, search_top_k=args.search_top_k,
                          temperature=args.temperature)
    else:
        bot = SeerahRAG()

    if done_ids:
        console.print(f"[bold]Resuming {out_path}: {len(done_ids)} already judged, "
                      f"{len(remaining)} remaining.[/bold]")
    calls_note = f"up to {args.max_iterations + 1} generate + 1 judge" if args.agentic else "1 generate + 1 judge"
    console.print(f"[bold]Judging {len(remaining)} of {len(selected)} questions ({calls_note} LLM calls each)...[/bold]")

    for i, q in enumerate(remaining, start=1):
        results.append(judge_question(q, bot, args.judge_model, args.top_k, args.agentic))
        if i % 5 == 0 or i == len(remaining):
            console.print(f"  judged {len(results)}/{len(selected)} total")
            save_checkpoint(out_path, args, selected, results)  # checkpoint every 5 - a kill loses at most 5

    save_checkpoint(out_path, args, selected, results)
    if args.ids and args.repeat > 1:
        print_repeat_summary(results, args.repeat)
    else:
        print_summary(results)
    console.print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
