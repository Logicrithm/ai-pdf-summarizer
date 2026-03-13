"""
Prompt Templates — Steps 6, 7, 9, 10, 11
-----------------------------------------
All prompts for summarization, merging, verification, and Q&A.
"""

# ─── Step 6: System Prompt ───
SYSTEM_PROMPT = """You are a precise technical document summarizer.

STRICT RULES:
- Use ONLY information present in the provided text.
- NEVER add external knowledge or assumptions.
- Search carefully through the text for limitations, disadvantages, and challenges before declaring them absent.
- If a section truly does not exist in the text, write: "Not mentioned in document."
- Group related concepts under a parent heading. Maximum 5 top-level concepts.
- Sub-approaches (e.g. instance-transfer, parameter-transfer) must be nested under their parent concept, not listed separately.
- Focus on high-level ideas, not specific algorithm names.

Return output in EXACTLY this structure:

## Overview

## Key Technical Concepts

## Important Findings

## Applications

## Limitations"""


# ─── Step 7: Section Retrieval Queries ───
SECTION_QUERIES = {
    "overview": "introduction overview purpose objective",
    "concepts": "key concepts methods techniques approaches",
    "findings": "results findings outcomes conclusions",
    "applications": "applications use cases examples domains tasks classification localization recognition filtering NLP text image speech sentiment email WiFi",
    "limitations": "limitations drawbacks challenges problems future work scope constraints weaknesses not covered only overview narrow",
}

# Second query for concepts to get broader coverage
CONCEPTS_QUERY_2 = "categories settings types definitions classification framework taxonomy"


# ─── Section-Specific Summarization Prompts ───
SECTION_PROMPTS = {
    "findings": """Summarize the key findings and conclusions from this document section.

Rules:
- Do NOT copy raw numbers or table data
- Explain what the results MEAN, not what they are
- Each finding must be a complete insight in plain English
- If results are mentioned, summarize the overall trend, not individual numbers
- Maximum 5 findings
- Each finding must end with a full stop
- If any two findings express the same idea in different words, keep only the clearer one and remove the duplicate

Text:
{combined_text}

Findings:""",

    "applications": """List ALL applications and use cases mentioned anywhere in this text.

Search carefully for:
- Named tasks (classification, localization, recognition, filtering, etc.)
- Named datasets used as examples
- Named domains (medical, finance, NLP, computer vision, etc.)
- Real world examples mentioned by the authors

Do not stop at 2-3 items. List EVERY application found in the text.
Each item should be a brief description ending with a full stop.

IMPORTANT: Only list real-world tasks and domains (e.g. sentiment analysis, WiFi localization, image classification).
Do NOT list methodology descriptions, transfer learning settings, or internal paper sections as applications.

Text:
{combined_text}

Applications:""",

    "concepts": """Extract ALL major technical concepts, categories, and frameworks from this text.

Group them under parent headings. Include both:
- High level categories (e.g. Transfer Learning Settings)
- Methods within each category (e.g. Inductive, Transductive, Unsupervised)

Maximum 4 parent groups.
Each group can have 3-5 sub-items.
Focus on concepts central to the paper's contribution, not peripheral mentions.

ACCURACY CHECK — Before finalizing, verify each group:
- The parent heading must accurately describe what ALL sub-items have in common
- If sub-items do not belong under a heading, create a more accurate heading
- Example: Do NOT put "concept model types" under "Transfer Learning Settings" — use "Concept Models" instead

Text:
{combined_text}

Key Concepts:""",
}


# ─── Chunk Summarization Prompt ───
CHUNK_SUMMARY_PROMPT = """Summarize the following text excerpt from a technical document.
Use ONLY information present in the text. Do NOT add external knowledge.
Be concise (max 100 words). Include the page number in your summary.

Page: {page}
Text:
{chunk_text}

Summary:"""


# ─── Step 9: Merge Prompt ───
MERGE_PROMPT = """Combine these section summaries into a final structured summary.
Preserve ALL unique topics from every section.
Do NOT discard any section, especially Limitations.
Do NOT add information not present in the summaries.

CRITICAL CITATION RULE:
Each section below has its own "Source: Pages ..." line.
When writing each section, use ONLY the page numbers listed for THAT specific section.
Do NOT copy page numbers from one section to another.
If the Overview source says "Pages 1, 2" and Applications says "Pages 4, 5",
then Overview bullets must cite [Page 1] or [Page 2] ONLY.

Section Summaries:
{section_summaries}

Return output in EXACTLY this structure:

## Overview
(2-3 sentences. Cite pages from the Overview section ONLY.)

## Key Technical Concepts
(Maximum 4 parent groups. Group related sub-items under their parent heading.
Example format:
- **Parent Concept** [Page X]
  - Sub-item 1
  - Sub-item 2
  - Sub-item 3
Do NOT list sub-approaches as separate top-level bullets.
Cite pages from the Concepts section ONLY.)

## Important Findings
(Maximum 5 findings. Each must be a complete insight in plain English.
Do NOT copy raw numbers — explain what results MEAN.
Each bullet ends with [Page X] from Findings section ONLY.)

## Applications
(List ALL applications and use cases found. Do not stop at 2-3 items.
Each bullet ends with [Page X] from Applications section ONLY.
If none: "Not mentioned in document.")

## Limitations
(Bullet list of ANY disadvantages, challenges, or limitations found. Each bullet ends with [Page X] from Limitations section ONLY. ONLY write "Not mentioned in document." if truly none exist.)

Final Summary:"""


# ─── Limitations Inference Prompt ───
LIMITATIONS_INFERENCE_PROMPT = """Based on the following document excerpts, identify any implicit limitations.

Only include REAL limitations such as:
- Methods the authors say do not always work
- Gaps the authors themselves acknowledge
- Scope boundaries explicitly stated in the paper
- Known failure cases mentioned in the text
- Future work suggestions that imply current gaps

Do NOT include:
- General observations about the research field
- Descriptions of what the paper covers
- Opinions not stated by the authors
- Statements about citation counts or references
- Filler content or padding

Automatically REJECT any limitation that:
- Mentions citation counts or "references various studies"
- Says "the field is still evolving"
- Describes what the paper covers rather than what it lacks
- Is an observation about external research, not the paper itself
- Uses speculative language like "may imply", "could suggest", or "might indicate"
- Is inferred by you rather than stated or clearly implied by the authors

Look for:
- Topics the paper admits it does not cover
- Algorithms or methods mentioned but not fully explained
- Statements about scope being narrow (e.g. "brief overview", "survey", "introduction")
- Future work suggestions that imply current gaps
- Lack of empirical evaluation or experiments
- Acknowledged simplifications or assumptions

Document excerpts:
{chunks_text}

If you find implicit limitations, list them as bullet points.
Maximum 5 limitations. Each must be directly supported by the document.
If there are truly NO limitations of any kind, respond with exactly: "Not mentioned in document."

Limitations:"""


# ─── Contradiction Check Prompt ───
CONTRADICTION_CHECK_PROMPT = """Review these two sections for contradictions.

Findings Section:
{findings}

Limitations Section:
{limitations}

If any limitation directly contradicts a finding, remove that limitation from the list.
Return ONLY the corrected Limitations section as a bullet list.
Do not add new content. Do not modify findings.
If no contradictions exist, return the Limitations section unchanged.

Corrected Limitations:"""


# ─── Step 10: Verification Prompt ───
VERIFICATION_PROMPT = """Review this summary against the source chunks provided.
List any claims not supported by the source text.
If all claims are supported, respond: All claims supported.

Source Chunks:
{source_chunks}

Generated Summary:
{summary}

Verification:"""


# ─── Step 11: Q&A Prompt ───
QA_PROMPT = """Answer the question using ONLY the evidence below.
If the provided evidence does not contain enough information to answer the question,
respond with: "This topic is not covered in the document."
Do NOT use external knowledge. Do NOT guess.

Evidence:
{evidence}

Question: {question}

Answer:"""
