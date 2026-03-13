"""
Q&A Engine — Steps 11, 12
--------------------------
RAG Q&A with confidence filtering and streamed responses.
Uses Groq (LLaMA 3.3 70B) for generation with rate-limit retry.
"""

import time
from groq import Groq, RateLimitError, APIStatusError, APITimeoutError
from prompts import QA_PROMPT, SYSTEM_PROMPT
from embeddings import embed_query, search_faiss

QA_MODEL = "llama-3.3-70b-versatile"
CONFIDENCE_THRESHOLD = 1.2  # L2 distance threshold — lower = more similar
MAX_RETRIES = 5
BASE_WAIT = 10

NO_MATCH_MSG = (
    "⚠️ **This question does not appear to be covered in the document.**\n\n"
    "Try rephrasing your question or asking about a topic discussed in the PDF."
)


def _call_with_retry(func):
    """Retry a Groq API call with exponential backoff on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            return func()
        except (RateLimitError, APIStatusError, APITimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_WAIT * (attempt + 1)
            error_msg = str(e)
            if "Please try again in" in error_msg:
                try:
                    wait_str = error_msg.split("Please try again in ")[1].split("s")[0]
                    wait = float(wait_str) + 1
                except (IndexError, ValueError):
                    pass
            time.sleep(wait)


def answer_question(client: Groq, question: str, chunks: list[dict], faiss_index) -> str:
    """
    Step 11: Answer a question with confidence filtering.

    1. Embed question
    2. Search FAISS → top 5 chunks
    3. Check confidence (L2 distance)
    4. Generate answer with citations
    """
    query_embedding = embed_query(client, question)
    distances, indices = search_faiss(faiss_index, query_embedding, k=5)

    # Confidence filter: reject if best match is too far
    if distances[0] > CONFIDENCE_THRESHOLD:
        return NO_MATCH_MSG, []

    # Build evidence with page citations
    evidence_parts = []
    source_pages = []
    for dist, idx in zip(distances, indices):
        if dist > CONFIDENCE_THRESHOLD:
            break
        chunk = chunks[idx]
        page = chunk["page"]
        source_pages.append(page)
        evidence_parts.append(f"[Page {page}]\n{chunk['text']}")

    evidence = "\n\n---\n\n".join(evidence_parts)
    prompt = QA_PROMPT.format(evidence=evidence, question=question)

    def _call():
        return client.chat.completions.create(
            model=QA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
        )

    response = _call_with_retry(_call)
    answer = response.choices[0].message.content
    return answer, sorted(set(source_pages))


def answer_question_streaming(client: Groq, question: str, chunks: list[dict], faiss_index):
    """
    Step 12: Stream the Q&A answer token by token.

    Yields:
        ("token", text) for each token
        ("sources", page_list) at the end
        ("no_match", message) if question is out of scope
    """
    query_embedding = embed_query(client, question)
    distances, indices = search_faiss(faiss_index, query_embedding, k=5)

    if distances[0] > CONFIDENCE_THRESHOLD:
        yield ("no_match", NO_MATCH_MSG)
        return

    evidence_parts = []
    source_pages = []
    for dist, idx in zip(distances, indices):
        if dist > CONFIDENCE_THRESHOLD:
            break
        chunk = chunks[idx]
        page = chunk["page"]
        source_pages.append(page)
        evidence_parts.append(f"[Page {page}]\n{chunk['text']}")

    evidence = "\n\n---\n\n".join(evidence_parts)
    prompt = QA_PROMPT.format(evidence=evidence, question=question)

    def _call():
        return client.chat.completions.create(
            model=QA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=400,
            stream=True,
        )

    stream = _call_with_retry(_call)

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield ("token", token)

    yield ("sources", sorted(set(source_pages)))
