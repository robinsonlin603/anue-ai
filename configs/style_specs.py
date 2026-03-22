PHASE2_STYLE_SPEC = """
Follow these rules strictly:

1. Use a formal, objective, and neutral financial news tone.
2. Only use information provided in the source. Do NOT add background, assumptions, opinions, or future projections.
3. Preserve all important numbers, dates, company names, ETF names, and events exactly as in the source.
4. Do NOT derive, calculate, or transform any numbers into new expressions that are not explicitly stated.
5. Output must be in Traditional Chinese.
6. Use plain text only. Do NOT use markdown or bullet points.

Structure requirements (very important):

Paragraph 1: Main event, key numbers, and major beneficiaries
Paragraph 2: Market context, index performance, or overall market growth
Paragraph 3: ETF performance, return comparison, or statistical data
Paragraph 4: Expert opinion, investment usage, or additional supporting information

Important:

1. You MUST cover information from later parts of the source, not only the beginning
2. Each paragraph MUST include at least one concrete element (number, company, ETF, or event)
3. If a type of information does not exist in the source, skip it without inventing content

Output format:

1. One title line
2. 4 paragraphs
3. 320–420 Chinese characters
""".strip()

PHASE3_STYLE_SPEC = """
Follow these rules strictly:

1. Use a formal, objective, and neutral financial news tone.
2. Only use facts from "Source content". Do NOT use facts/names from "Style reference snippets".
3. MANDATORY: Preserve all stock codes exactly like (XXXX-TW) and all numbers/percentages/dates.
4. Output must be in Traditional Chinese, plain text only, no markdown or bullet points.

Structure requirements (Follow strictly):
Paragraph 1: Main event, key numbers, and major beneficiaries (including stock codes)
Paragraph 2: Market context, index performance, or overall market growth
Paragraph 3: Detailed data, comparison, or technical applications
Paragraph 4: Future outlook, expert opinion, or supporting facts from the later part of source

Important:
- Cover information from the entire source, especially the middle and end sections.
- Use the "Style reference snippets" ONLY for learning sentence rhythm and phrasing.
- Total length: 380–520 Chinese characters.
""".strip()
