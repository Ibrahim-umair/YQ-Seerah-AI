"""Interactive RAG bot over the full 104-lecture corpus.

By default this runs the AGENTIC pipeline (seerah.agent.SeerahAgent): the LLM
decides for itself whether to search, how many times, and how to reword the
query between attempts - up to a hard cap. Every search it makes is printed
live, with its one-sentence reason ("thinking") and a word-level diff against
the previous query, so a query correction (a misspelled name, a term that
didn't retrieve well) is visible as it happens, not just implied by a better
final answer.

--simple switches to the older single-shot pipeline (seerah.answer.SeerahRAG)
for comparison: one retrieval call, one LLM call, no query correction, no
agentic loop.

Usage:
    python -m seerah.bot
    python -m seerah.bot --max-iterations 5
    python -m seerah.bot --show-context      # also print the chunks the answer was grounded in
    python -m seerah.bot --simple            # old single-shot pipeline, no agent
    python -m seerah.bot --simple --top-k 5
"""

import argparse
import difflib

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from seerah import config
from seerah.agent import SeerahAgent
from seerah.answer import SeerahRAG

console = Console()


def render_query_diff(prev_query, curr_query):
    """Word-level diff of curr_query against prev_query: unchanged words
    plain, dropped words struck through in red, added words bold green -
    so a query correction between search iterations is visible at a glance."""
    prev_words, curr_words = prev_query.split(), curr_query.split()
    matcher = difflib.SequenceMatcher(None, prev_words, curr_words)
    text = Text()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            text.append(" ".join(curr_words[j1:j2]) + " ")
        else:
            if i1 != i2:
                text.append(" ".join(prev_words[i1:i2]) + " ", style="strike red")
            if j1 != j2:
                text.append(" ".join(curr_words[j1:j2]) + " ", style="bold green")
    return text


def make_on_event(console, prev_query_holder):
    def on_event(event):
        if event["type"] == "search":
            console.print(f"\n[bold cyan]Search {event['iteration']}[/bold cyan]"
                          f"  [dim italic]thinking: {event['reason']}[/dim italic]")
            if prev_query_holder[0] is not None:
                console.print("  query: ", render_query_diff(prev_query_holder[0], event["query"]))
            else:
                console.print(f'  query: "{event["query"]}"')
            prev_query_holder[0] = event["query"]
        elif event["type"] == "note":
            console.print(f"[dim]  (agent note: {event['text']})[/dim]")
    return on_event


def print_answer_and_sources(answer, hits, show_context):
    console.print()
    console.print(Panel(Markdown(answer), title="Answer", title_align="left"))

    console.print("\n[dim]Sources:[/dim]")
    for hit in hits:
        console.print(f"  - {hit.citation}, chunk {hit.chunk_index}")

    if show_context:
        console.print("\n[dim]Retrieved context used:[/dim]")
        for i, hit in enumerate(hits, start=1):
            console.print(Panel(hit.text, title=f"#{i}  {hit.citation}  chunk {hit.chunk_index}",
                                title_align="left"))


def main():
    config.use_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--top-k", type=int, default=10, help="--simple mode only")
    parser.add_argument("--max-iterations", type=int, default=config.AGENT_MAX_ITERATIONS,
                        help="agent mode only: max search rounds before a forced answer")
    parser.add_argument("--search-top-k", type=int, default=config.AGENT_SEARCH_TOP_K,
                        help="agent mode only: chunks retrieved per search call")
    parser.add_argument("--show-context", action="store_true",
                        help="also print the retrieved chunks the answer was grounded in")
    parser.add_argument("--simple", action="store_true",
                        help="use the old single-shot pipeline instead of the agent")
    args = parser.parse_args()

    console.print("[bold]Loading indexes...[/bold]")
    if args.simple:
        bot = SeerahRAG()
    else:
        bot = SeerahAgent(max_iterations=args.max_iterations, search_top_k=args.search_top_k)

    console.print(
        f"\n[bold green]Ready[/bold green] ({'simple, non-agentic' if args.simple else 'agentic'} mode). "
        "Ask a question about the Seerah. Type 'exit', 'quit', or submit an empty line to stop.\n"
    )

    while True:
        try:
            query = console.input("[bold cyan]Ask> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break
        if not query or query.lower() in ("exit", "quit"):
            console.print("Bye.")
            break

        if args.simple:
            answer, hits = bot.rag(query, top_k=args.top_k)
        else:
            prev_query_holder = [None]
            answer, hits, _search_log = bot.ask(query, on_event=make_on_event(console, prev_query_holder))

        print_answer_and_sources(answer, hits, args.show_context)
        console.print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()
