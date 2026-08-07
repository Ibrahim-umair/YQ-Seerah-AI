"""Generation stage: retrieved chunks + a question -> one grounded answer.

Structured the same way as the RAG pattern from the DataTalks.Club LLM
Zoomcamp course (search -> build_prompt -> llm -> rag, a class holding
instructions/prompt_template/model as configurable fields, one plain LLM call
via the Responses API) - adapted to this project's hybrid retriever and
citation format instead of the course's FAQ-document index.

Deliberately NOT agentic: exactly one retrieval call and one LLM call per
question.
  - No query rewriting - `search()` sends the question to hybrid_search()
    exactly as typed.
  - No function calling - `llm()` is a plain Responses API call with a text
    prompt, no tools passed.
  - No agentic loop - the model never decides to retrieve again or take
    another turn. Ask once, retrieve once, answer once.
Those are all real, known limitations (a misspelled name won't be corrected,
a question needing a second lookup won't get one) - left for later, on
purpose, not missed.
"""

from openai import OpenAI

from seerah import config
from seerah.retrieve import Retriever

INSTRUCTIONS = """
You are answering questions about the Seerah (the life of the Prophet
Muhammad, peace be upon him) using excerpts from Shaykh Dr. Yasir Qadhi's
104-part Seerah lecture series.

Use ONLY the excerpts given in the context below - do not use outside
knowledge, and do not fill gaps with what you already know about Islamic
history. If the context does not contain enough to answer the question,
say "I don't have enough information in the lectures to answer that" rather
than guessing.

Every excerpt is labelled with its lecture. Refer to the specific lecture(s)
your answer draws on.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def format_context(hits):
    """Formats a list of Hit into the citation-labelled context block. A
    module-level function (not just SeerahRAG.build_context) so anything
    holding a plain list of Hit - seerah.agent.SeerahAgent's accumulated
    search results, seerah.eval.judge_answers - can format it the same way
    without needing a SeerahRAG instance."""
    lines = []
    for hit in hits:
        lines.append(f"[{hit.citation}, chunk {hit.chunk_index}]")
        lines.append(hit.text)
        lines.append("")
    return "\n".join(lines).strip()


class SeerahRAG:
    """search -> build_prompt -> llm -> rag."""

    def __init__(self, retriever=None, llm_client=None,
                 instructions=INSTRUCTIONS, prompt_template=PROMPT_TEMPLATE,
                 model=config.ANSWER_MODEL):
        self.retriever = retriever or Retriever()
        self.llm_client = llm_client or OpenAI()
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, top_k=10):
        """One hybrid (RRF-fused) retrieval call. No query rewriting."""
        hits, _timings = self.retriever.hybrid_search(query, top_k=top_k)
        return hits

    def build_context(self, hits):
        return format_context(hits)

    def build_prompt(self, query, hits):
        context = self.build_context(hits)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        """One plain Responses API call. No tools, no function calling."""
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.create(model=self.model, input=input_messages)
        return response.output_text

    def rag(self, query, top_k=10):
        """search -> build_prompt -> llm, once each. Returns (answer, hits) -
        hits are returned alongside the answer (unlike the course's rag(),
        which returns just the string) so a caller can display citations."""
        hits = self.search(query, top_k=top_k)
        prompt = self.build_prompt(query, hits)
        answer = self.llm(prompt)
        return answer, hits
