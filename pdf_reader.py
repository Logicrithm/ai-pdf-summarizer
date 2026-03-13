"""
PDF Reader — Step 1
-------------------
Extract text page by page with noise filtering.
"""

import fitz  # PyMuPDF
import re

# Lines matching these patterns are noise and should be filtered out
NOISE_PATTERNS = [
    "pip install",
    "import ",
    "Epoch",
    "accuracy:",
    "model.compile",
    ">>>",
]


def _is_noisy_line(line: str) -> bool:
    """Check if a line is noise (code, training logs, etc.)."""
    stripped = line.strip()
    if not stripped:
        return False

    # Check exact noise patterns
    for pattern in NOISE_PATTERNS:
        if pattern in stripped:
            return True

    # Check if line is mostly numbers (more than 5 numbers in a row)
    digit_count = sum(c.isdigit() for c in stripped)
    if digit_count > 5 and digit_count / max(len(stripped), 1) > 0.5:
        return True

    return False


def extract_pages(pdf_file) -> list[dict]:
    """
    Extract text from PDF page by page with noise filtering.

    Returns:
        List of dicts: [{"text": clean_text, "page": page_number}, ...]
    """
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    pages = []

    for i, page in enumerate(doc):
        raw_text = page.get_text()
        lines = raw_text.split("\n")

        # Filter noisy lines
        clean_lines = [line for line in lines if not _is_noisy_line(line)]
        clean_text = "\n".join(clean_lines).strip()

        if clean_text:
            pages.append({"text": clean_text, "page": i + 1})

    doc.close()
    return pages
