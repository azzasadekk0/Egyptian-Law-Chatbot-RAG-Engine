def format_chunk_for_prompt(chunk: dict, index: int) -> str:
    """
    Format a single chunk as a numbered evidence block for the LLM system prompt.

    Example output:
        [المصدر 1]
        القانون: قانون العقوبات المصري (1937)
        المادة: 230
        الصفحات: 45-45
        النص: يُعاقب بالسجن كل من...
    """
    law_name = chunk.get("law_name", "")
    year     = chunk.get("year", "")
    article  = chunk.get("article_number")
    page_s   = chunk.get("page_start", "")
    page_e   = chunk.get("page_end", "")
    text     = chunk.get("text", "").strip()

    lines = [f"[المصدر {index}]"]

    if law_name:
        law_line = f"القانون: {law_name}"
        if year:
            law_line += f" ({year})"
        lines.append(law_line)

    if article:
        lines.append(f"المادة: {article}")

    if page_s:
        page_str = f"{page_s}" if page_s == page_e else f"{page_s}–{page_e}"
        lines.append(f"الصفحات: {page_str}")

    lines.append(f"النص: {text}")
    return "\n".join(lines)


def format_citations_block(chunks: list[dict]) -> str:
    """
    Build a numbered citation block from a list of top chunks.
    Used at the end of the LLM answer so the user can verify sources.

    Example output:
        ━━━ المصادر القانونية المُستشهد بها ━━━
        [1] قانون العقوبات المصري، المادة 230، ص 45
        [2] القانون المدني المصري، المادة 148، ص 32
    """
    if not chunks:
        return ""

    lines = ["━━━ المصادر القانونية المُستشهد بها ━━━"]
    for i, chunk in enumerate(chunks, start=1):
        law_name = chunk.get("law_name", "مصدر غير محدد")
        year     = chunk.get("year", "")
        article  = chunk.get("article_number")
        page_s   = chunk.get("page_start", "")
        page_e   = chunk.get("page_end", "")

        parts = [f"[{i}] {law_name}"]
        if year:
            parts[0] += f" ({year})"
        if article:
            parts.append(f"المادة {article}")
        if page_s:
            page_str = f"ص {page_s}" if page_s == page_e else f"ص {page_s}–{page_e}"
            parts.append(page_str)

        lines.append("، ".join(parts))

    return "\n".join(lines)


def build_evidence_context(chunks: list[dict]) -> str:
    """
    Build the full evidence context string to inject into the LLM prompt.
    Combines all chunk evidence blocks with formatting.
    """
    if not chunks:
        return "لا توجد نصوص قانونية متاحة."

    blocks = [format_chunk_for_prompt(chunk, i) for i, chunk in enumerate(chunks, start=1)]
    return "\n\n".join(blocks)
