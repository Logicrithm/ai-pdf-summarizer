"""
Summarization Engine — Steps 7, 8, 9, 10
-----------------------------------------
Section-aware summarization with:
- Dedicated FAISS retrieval per section (Step 7)
- Sequential chunk summarization with Groq (Step 8)
- Structured merge with Groq (Step 9)
- Verification pass (Step 10)
- Two-step limitations inference for papers without explicit limitations
Uses Groq for all LLM calls with rate-limit retry.
"""

import time
from groq import Groq, RateLimitError, APIStatusError, APITimeoutError
from prompts import (
    SYSTEM_PROMPT,
    SECTION_QUERIES,
    SECTION_PROMPTS,
    CONCEPTS_QUERY_2,
    CHUNK_SUMMARY_PROMPT,
    MERGE_PROMPT,
    VERIFICATION_PROMPT,
    LIMITATIONS_INFERENCE_PROMPT,
    CONTRADICTION_CHECK_PROMPT,
)
from embeddings import embed_query, search_faiss

CHUNK_MODEL = "llama-3.3-70b-versatile"
MERGE_MODEL = "llama-3.3-70b-versatile"

MAX_RETRIES = 5
BASE_WAIT = 10  # seconds

# Sections that need more context to avoid truncation
SECTION_OVERRIDES = {
    "limitations": {"k": 4, "max_tokens": 500},
    "applications": {"k": 6, "max_tokens": 500},
    "concepts": {"k": 4, "max_tokens": 500},
}

# Max characters per chunk when building LLM context (prevents 413 errors)
MAX_CHUNK_CHARS = 400

# L2 distance threshold — chunks above this are low relevance
WEAK_MATCH_THRESHOLD = 1.5


def _call_with_retry(func):
    """Retry a Groq API call with exponential backoff on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            return func()
        except (RateLimitError, APIStatusError, APITimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            # Parse wait time from error or use exponential backoff
            wait = BASE_WAIT * (attempt + 1)
            error_msg = str(e)
            if "Please try again in" in error_msg:
                try:
                    wait_str = error_msg.split("Please try again in ")[1].split("s")[0]
                    wait = float(wait_str) + 1
                except (IndexError, ValueError):
                    pass
            time.sleep(wait)


# ─── Step 8: Summarize a single chunk ───
def _summarize_chunk(client: Groq, chunk: dict) -> str:
    """Summarize one chunk using Groq."""
    prompt = CHUNK_SUMMARY_PROMPT.format(
        page=chunk["page"],
        chunk_text=chunk["text"],
    )

    def _call():
        return client.chat.completions.create(
            model=CHUNK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )

    response = _call_with_retry(_call)
    return response.choices[0].message.content


# ─── Step 7: Section-aware retrieval + summarization ───
def _summarize_section(
    client: Groq, section_name: str, query: str,
    chunks: list[dict], faiss_index, k: int = 4,
) -> tuple[str, list[int]]:
    """
    Retrieve top-k chunks relevant to a section query,
    then summarize them.

    For limitations: uses two-step approach.
    Step 1: FAISS search with expanded query.
    Step 2: If results are weak, use LLM inference from broader context.

    Returns:
        Tuple of (section_summary, source_pages).
    """
    # Use overrides for sections that need more context
    overrides = SECTION_OVERRIDES.get(section_name, {})
    actual_k = overrides.get("k", k)
    actual_max_tokens = overrides.get("max_tokens", 300)

    query_embedding = embed_query(client, query)
    distances, indices = search_faiss(faiss_index, query_embedding, k=actual_k)

    retrieved_chunks = [chunks[i] for i in indices if i < len(chunks)]

    # For concepts: run a second query and merge results for broader coverage
    if section_name == "concepts":
        query2_embedding = embed_query(client, CONCEPTS_QUERY_2)
        distances2, indices2 = search_faiss(faiss_index, query2_embedding, k=actual_k)
        for idx in indices2:
            if idx < len(chunks) and chunks[idx] not in retrieved_chunks:
                retrieved_chunks.append(chunks[idx])

    if not retrieved_chunks:
        # For limitations, try inference fallback
        if section_name == "limitations":
            return _infer_limitations(client, chunks)
        return "Not mentioned in document.", []

    source_pages = sorted(set(c["page"] for c in retrieved_chunks))

    # For limitations: check if FAISS results are weak (high L2 distance)
    if section_name == "limitations":
        avg_distance = sum(distances[:len(retrieved_chunks)]) / len(retrieved_chunks)
        if avg_distance > WEAK_MATCH_THRESHOLD:
            # Weak match — fall back to LLM inference from broader context
            return _infer_limitations(client, chunks)

    # Summarize retrieved chunks together (truncate each chunk to fit token limits)
    combined_text = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text'][:MAX_CHUNK_CHARS]}" for c in retrieved_chunks
    )

    # Use section-specific prompt if available, otherwise generic
    if section_name in SECTION_PROMPTS:
        prompt = SECTION_PROMPTS[section_name].format(combined_text=combined_text)
    else:
        prompt = f"Summarize the following text for the '{section_name}' section:\n\n{combined_text}"

    def _call():
        return client.chat.completions.create(
            model=CHUNK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=actual_max_tokens,
        )

    response = _call_with_retry(_call)
    summary = response.choices[0].message.content
    return summary, source_pages


def _infer_limitations(client: Groq, chunks: list[dict]) -> tuple[str, list[int]]:
    """
    Fallback: Ask the LLM to infer implicit limitations from
    the document content (e.g. narrow scope, no evaluation, etc.).
    Uses the last ~8 chunks which typically contain conclusions/future work.
    """
    # Use a mix: first 3 chunks (intro/scope) + last 5 chunks (conclusion/future work)
    selected = chunks[:3] + chunks[-5:]
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for c in selected:
        key = (c["page"], c["text"][:50])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    source_pages = sorted(set(c["page"] for c in unique_chunks))

    chunks_text = "\n\n---\n\n".join(
        f"[Page {c['page']}]\n{c['text'][:500]}" for c in unique_chunks
    )

    prompt = LIMITATIONS_INFERENCE_PROMPT.format(chunks_text=chunks_text)

    def _call():
        return client.chat.completions.create(
            model=CHUNK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )

    response = _call_with_retry(_call)
    summary = response.choices[0].message.content
    return summary, source_pages


def _check_contradictions(client: Groq, section_summaries: dict) -> dict:
    """
    Fix 4: Check if Limitations contradict Findings.
    If contradictions found, remove them from Limitations.
    """
    findings = section_summaries.get("findings", "")
    limitations = section_summaries.get("limitations", "")

    if not findings or not limitations:
        return section_summaries
    if "not mentioned in document" in limitations.lower():
        return section_summaries

    prompt = CONTRADICTION_CHECK_PROMPT.format(
        findings=findings,
        limitations=limitations,
    )

    def _call():
        return client.chat.completions.create(
            model=CHUNK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )

    response = _call_with_retry(_call)
    corrected = response.choices[0].message.content
    section_summaries["limitations"] = corrected
    return section_summaries


# ─── Steps 7+8+9: Full summarization pipeline ───
def summarize_document(
    client: Groq,
    chunks: list[dict],
    faiss_index,
    progress_callback=None,
) -> tuple[str, dict[str, list[int]]]:
    """
    Section-aware summarization pipeline (sequential to respect rate limits):
    1. Run dedicated FAISS search per section (Step 7)
    2. Summarize each section sequentially (Step 8)
    3. Merge into final summary (Step 9)

    Returns:
        Tuple of (final_summary, section_sources).
    """
    section_summaries = {}
    section_sources = {}
    total = len(SECTION_QUERIES)
    completed = 0

    # Step 7+8: Retrieve & summarize per section SEQUENTIALLY (rate limit safe)
    for name, query in SECTION_QUERIES.items():
        summary, pages = _summarize_section(
            client, name, query, chunks, faiss_index, 4
        )
        section_summaries[name] = summary
        section_sources[name] = pages
        completed += 1
        if progress_callback:
            progress_callback(completed, total)

    # Fix 4: Check for contradictions between findings and limitations
    section_summaries = _check_contradictions(client, section_summaries)

    # Step 9: Merge all section summaries
    combined = "\n\n---\n\n".join(
        f"### {name.title()}\n{summary}\n> Source: Pages {', '.join(str(p) for p in section_sources.get(name, []))}"
        for name, summary in section_summaries.items()
    )

    merge_prompt = MERGE_PROMPT.format(section_summaries=combined)

    def _call():
        return client.chat.completions.create(
            model=MERGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": merge_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )

    response = _call_with_retry(_call)
    final_summary = response.choices[0].message.content
    return final_summary, section_sources


# ─── Streaming variant ───
def summarize_document_streaming(
    client: Groq,
    chunks: list[dict],
    faiss_index,
    progress_callback=None,
):
    """
    Same pipeline but streams the final merge step.
    Sections are processed sequentially to avoid rate limits.

    Yields:
        ("section", section_name, completed, total) during retrieval
        ("token", token_text) during final merge streaming
        ("sources", section_sources) at the end
        ("done", final_summary) when complete
    """
    section_summaries = {}
    section_sources = {}
    total = len(SECTION_QUERIES)
    completed = 0

    # Step 7+8: Retrieve & summarize per section SEQUENTIALLY
    for name, query in SECTION_QUERIES.items():
        summary, pages = _summarize_section(
            client, name, query, chunks, faiss_index, 4
        )
        section_summaries[name] = summary
        section_sources[name] = pages
        completed += 1
        if progress_callback:
            progress_callback(completed, total)
        yield ("section", name, completed, total)

    # Fix 4: Check for contradictions between findings and limitations
    section_summaries = _check_contradictions(client, section_summaries)

    # Step 9: Stream the merge
    combined = "\n\n---\n\n".join(
        f"### {name.title()}\n{summary}\n> Source: Pages {', '.join(str(p) for p in section_sources.get(name, []))}"
        for name, summary in section_summaries.items()
    )

    merge_prompt = MERGE_PROMPT.format(section_summaries=combined)

    def _call():
        return client.chat.completions.create(
            model=MERGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": merge_prompt},
            ],
            temperature=0,
            max_tokens=800,
            stream=True,
        )

    stream = _call_with_retry(_call)

    full_summary = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            full_summary += token
            yield ("token", token, 0, 0)

    yield ("sources", section_sources, 0, 0)
    yield ("done", full_summary, 0, 0)


# ─── Step 10: Verification ───
def verify_summary(client: Groq, chunks: list[dict], summary: str) -> str:
    """Run a verification pass against source chunks."""
    source_text = "\n\n---\n\n".join(
        f"[Page {c['page']}] {c['text'][:300]}" for c in chunks[:10]
    )

    prompt = VERIFICATION_PROMPT.format(
        source_chunks=source_text,
        summary=summary,
    )

    def _call():
        return client.chat.completions.create(
            model=MERGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
        )

    response = _call_with_retry(_call)
    return response.choices[0].message.content
