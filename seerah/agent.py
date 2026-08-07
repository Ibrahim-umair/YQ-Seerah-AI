"""Agentic RAG: the LLM decides when and how many times to search, instead of
the developer deciding up front. Structured after the DataTalks.Club LLM
Zoomcamp's agentic-loop lessons (01-agentic-rag/lessons/13-function-calling.md
and 14-agentic-loop.md): a `search` tool exposed via the Responses API, and a
message-history loop that keeps calling the model until it stops requesting
tool calls.

Query correction is NOT a separate step here - it isn't one in the course
either. It's the same loop: the model sees its own search results, and if
they're weak, it rewrites the query and searches again. The course's example
is a literal typo ("Olama" -> "Ollama") that self-corrects across two search
calls with no code written to detect typos. On this corpus the equivalent
problem is transliteration: names have no fixed spelling (Badr/Badar/Badur,
Ka'b/Kab/Kaab), so a first search that misses on one spelling can succeed on
a second try with another - the instructions below tell the model this
explicitly, then the loop does the rest.

Every search call also carries a one-sentence `reason` argument - the model's
own explanation for why it's searching, or what it changed since the last
search. That's the "thinking" this module surfaces: small by design (one
sentence), but real, and it's what lets a caller show a bad first query next
to the corrected second one.

Kept entirely separate from seerah.answer.SeerahRAG (the single-shot,
non-agentic pipeline) rather than replacing it, so both can still be run,
judged, and compared - not just so the newer one silently takes over.
"""

import json

from openai import OpenAI

from seerah import config
from seerah.retrieve import Retriever

INSTRUCTIONS = """
You are a research assistant answering questions about the Seerah (the life
of the Prophet Muhammad, peace be upon him), using a `search` tool over
Shaykh Dr. Yasir Qadhi's 104-part Seerah lecture series. You do not have the
lectures memorized - you must search to find evidence, and answer only from
what your searches actually return.

Use the search tool to look things up. Use as many concrete keywords from the
question as possible in your first search.

Names and places in this corpus are transliterated Arabic with no single
fixed spelling (for example "Badr" may also appear as "Badar"; "Ka'b" as
"Kab" or "Kaab"). If a search returns weak or irrelevant results, do not give
up - try again with a different spelling or different keywords before
concluding the corpus doesn't cover it.

If your search results answer the question, synthesize an answer based
*only* on the provided evidence - do not add outside knowledge, and do not
fill gaps with what you already know about Islamic history.
If the results are insufficient or missing key details, reformulate your
query and search again. You may search up to {max_iterations} times total.

Before including any specific factual claim in your answer - a date, a
location, an outcome, a cause, a relationship, what someone said or did, how
something ended - check that your search results state it directly, not that
you are inferring it from related-but-different information. A person being
present somewhere, allied with a group, or active in some period does not by
itself confirm what later happened to them, why something occurred, or how
it concluded. If a specific claim isn't directly stated in what you've
retrieved, search again using terms specific to that exact claim before
including it, or say plainly that your sources don't confirm it. When
double-checking an outcome, search by the person or event's name only, not
the outcome you already found - so a differing account can surface - and if
lectures disagree, trust the more detailed, dedicated one.

Before finishing, check that your answer covers everything the question
specifically asked for - a full outcome, every named event, every person it
asks "who/which" about - not just the first part you found. If your answer
might be missing something from a multi-part question, do one more search
to check - but stop once a search adds nothing new.

Every time you call search, also give a one-sentence `reason` explaining why
you are performing this search - or, on a later call, what you changed since
the previous one and why.

If, after your searches, the evidence still does not answer the question,
say so plainly rather than guessing.
""".strip()

SEARCH_TOOL = {
    "type": "function",
    "name": "search",
    "description": (
        "Search the Seerah lecture corpus (104 lectures, transcribed and "
        "chunked) for passages relevant to a query. Returns the most "
        "relevant excerpts, each with its lecture citation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Rewrite this on later calls if earlier results were weak.",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence: why you are performing this search, "
                                "or what you changed since the last one.",
            },
        },
        "required": ["query", "reason"],
        "additionalProperties": False,
    },
}


def hit_to_dict(hit):
    return {
        "lecture": hit.citation,
        "chunk_index": hit.chunk_index,
        "score": round(hit.score, 4),
        "text": hit.text,
    }


class SeerahAgent:
    """An agentic loop: the model decides how many times to search and how to
    rephrase the query between attempts, capped at max_iterations. On the
    final permitted round, if the model still wants to search, it is denied
    the tool and forced to answer instead - the loop always terminates."""

    def __init__(self, retriever=None, llm_client=None, instructions=INSTRUCTIONS,
                 model=config.ANSWER_MODEL, max_iterations=config.AGENT_MAX_ITERATIONS,
                 search_top_k=config.AGENT_SEARCH_TOP_K, temperature=None):
        self.retriever = retriever or Retriever()
        self.llm_client = llm_client or OpenAI()
        self.instructions = instructions.format(max_iterations=max_iterations)
        self.model = model
        self.max_iterations = max_iterations
        self.search_top_k = search_top_k
        # None = don't send it at all, so the API's own default (~1.0) applies.
        # 0.0 is a real, valid setting - must not be treated as "unset" (a classic
        # falsy-value bug: `if self.temperature` would silently drop temperature=0.0).
        #
        # KNOWN LIMITATION, confirmed live: config.ANSWER_MODEL ("gpt-5.6-luna") REJECTS
        # this parameter outright - "Unsupported parameter: 'temperature' is not
        # supported with this model" (400). It behaves like a reasoning-style model
        # that manages its own internal sampling, with no exposed temperature or seed
        # knob. This parameter is only usable if ANSWER_MODEL is swapped to a model
        # that does support it - see the README/conversation before assuming it works.
        self.temperature = temperature

    def _search(self, query):
        hits, _timings = self.retriever.hybrid_search(query, top_k=self.search_top_k)
        return hits

    def ask(self, question, on_event=None):
        """Runs the agentic loop.

        on_event(dict), if given, is called for every search the model makes
        (type "search": iteration, query, reason) and any incidental message
        text alongside a search (type "note") - see seerah.bot for how these
        are rendered. The final answer is NOT sent through on_event; it comes
        back as this method's return value, same as SeerahRAG.rag().

        Returns (answer, hits, search_log):
          - hits: every chunk retrieved across all iterations, deduplicated
            by (lecture_number, chunk_index) - the union, for citations.
          - search_log: [{iteration, query, reason, num_hits}, ...]
        """

        def emit(event):
            if on_event:
                on_event(event)

        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": question},
        ]

        all_hits = {}
        search_log = []
        iteration = 0
        final_answer = None

        while final_answer is None:
            iteration += 1
            forced = iteration > self.max_iterations
            if forced:
                messages.append({
                    "role": "user",
                    "content": f"You have used the maximum of {self.max_iterations} searches. "
                               f"Answer now with your best answer based on everything found so far.",
                })

            create_kwargs = {"model": self.model, "input": messages,
                             "tools": None if forced else [SEARCH_TOOL]}
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature
            response = self.llm_client.responses.create(**create_kwargs)
            messages.extend(response.output)

            function_calls = [item for item in response.output if item.type == "function_call"]
            note_texts = [item.content[0].text for item in response.output if item.type == "message"]

            for item in function_calls:
                args = json.loads(item.arguments)
                query, reason = args["query"], args.get("reason", "")
                emit({"type": "search", "iteration": iteration, "query": query, "reason": reason})

                hits = self._search(query)
                for hit in hits:
                    all_hits[(hit.lecture_number, hit.chunk_index)] = hit
                search_log.append({"iteration": iteration, "query": query, "reason": reason,
                                   "num_hits": len(hits)})

                messages.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps([hit_to_dict(h) for h in hits], ensure_ascii=False),
                })

            if function_calls:
                # more searching to do - any accompanying text is a note, not the final answer
                for text in note_texts:
                    emit({"type": "note", "iteration": iteration, "text": text})
                continue

            # no function calls this turn -> this IS the final answer
            final_answer = note_texts[0] if note_texts else (
                "I don't have enough information in the lectures to answer that."
            )

        return final_answer, list(all_hits.values()), search_log
