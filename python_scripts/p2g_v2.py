import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Literal

import pymupdf
import pymupdf4llm
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
BACK_MATTER_RE = re.compile(
    r"^(references|bibliography|acknowledg(?:e)?ments?|declarations?|"
    r"参考文献|声明|作者贡献|利益冲突|致谢|author biographies?|biographies?|"
    r"authors|contributors|about the authors?)$",
    re.IGNORECASE,
)
# pymupdf4llm wraps OCR text read out of a rasterized figure inside HTML
# comments and inline images as <img> tags; highlighted spans come back as
# <mark>…</mark>. None of it is paper prose, so it is stripped before the
# heading tree is built.
PICTURE_TEXT_BLOCK_RE = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?(?:<!--\s*End of picture text\s*-->|$)",
    re.DOTALL | re.IGNORECASE,
)
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MARK_TAG_RE = re.compile(r"</?mark\b[^>]*>", re.IGNORECASE)
EXTRACTION_PROMPT_VERSION = "v2.10"
REDUCE_PROMPT_VERSION = "v2.9"
GROUP_PROMPT_VERSION = "v2.11"
DOCUMENT_PROMPT_VERSION = "v2.18"
GLOBAL_QUESTIONS = [
    "文档的核心研究问题是什么？",
    "文档使用了哪些数据集或材料？",
    "文档采用了哪些主要方法？",
    "比较了哪些模型、方案或对象？",
    "使用了哪些评价指标？",
    "最重要的定量结果是什么？",
    "作者如何解释主要结果？",
    "文档有哪些局限？",
    "未来工作或改进方向是什么？",
    "不同章节之间是否存在数字或结论矛盾？",
]

# Column headers that mean "this table is a config/parameter table", not an
# evaluation-metrics table. `extract_table_facts` labels every numeric row as
# kind="metric", so config rows ("Value=50") would otherwise outrank genuine
# metric definitions for "使用了哪些评价指标？"-style questions.
_GENERIC_METRIC_COLUMNS = {
    "value", "values", "setting", "settings", "parameter", "parameters",
    "description", "amount", "count", "number", "name", "type",
    "base", "tuned", "delta", "average", "avg", "mean", "total", "index",
    "metric",
    "值", "参数", "设置", "描述", "数量", "名称", "类型",
}
_CONFIG_TABLE_MARKERS = ("configuration", "config", "parameters", "设置", "配置")
# Correlation tables legitimately report one metric against several targets
# (social / subjective / economy reward), so identical rows across them are not
# contradictions.
_CORRELATION_TABLE_RE = re.compile(r"correlat|相关", re.IGNORECASE)


class PageMarkdown(BaseModel):
    page: int
    text: str
    metadata: dict = Field(default_factory=dict)


class HeadingNode(BaseModel):
    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    subtree_page_end: int
    parent_id: str | None = None
    text: str = ""
    page_texts: list[PageMarkdown] = Field(default_factory=list, exclude=True)
    children: list["HeadingNode"] = Field(default_factory=list)


class TextChunk(BaseModel):
    index: int
    text: str
    token_estimate: int
    overlap_text: str = ""
    page_start: int | None = None
    page_end: int | None = None
    page_texts: list[PageMarkdown] = Field(default_factory=list, exclude=True)


class Fact(BaseModel):
    fact_id: str = ""
    concept: str
    statement: str
    kind: Literal[
        "definition",
        "formula",
        "method",
        "result",
        "limitation",
        "advantage",
        "disadvantage",
        "example",
        "relation",
        "classification",
        "dataset",
        "metric",
    ]
    source_quote: str
    page: int | None = None
    confidence: float = Field(ge=0, le=1)
    grounded: bool = False

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: str) -> str:
        allowed = {
            "definition",
            "formula",
            "method",
            "result",
            "limitation",
            "advantage",
            "disadvantage",
            "example",
            "relation",
            "classification",
            "dataset",
            "metric",
        }
        aliases = {
            "finding": "result",
            "comparison": "relation",
            "application": "example",
            "challenge": "limitation",
            "purpose": "method",
        }
        normalized = str(value).strip().lower()
        return aliases.get(normalized, normalized if normalized in allowed else "relation")


class ChunkExtraction(BaseModel):
    summary: str
    facts: list[Fact]
    new_concepts: list[str] = Field(default_factory=list)


class TableFact(BaseModel):
    fact_id: str
    section_id: str
    table: str
    # Full caption text, kept once per table (see `extract_table_facts`). It used to
    # be spliced into every row's `statement`, which repeated the same sentence up
    # to 150 times and consumed 55-69% of all table-fact tokens.
    table_caption: str = ""
    row_context: dict[str, str]
    metric: str
    value: float
    display_value: str
    unit: str = ""
    statement: str
    source_quote: str
    page: int
    kind: Literal["metric"] = "metric"
    concept: str
    confidence: float = 1.0
    grounded: bool = True
    bold: bool = False


class DetectedContradiction(BaseModel):
    statement: str
    source_fact_ids: list[str]
    claim_fact: Fact


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


class SectionSynthesis(BaseModel):
    title: str
    summary: str
    core_concepts: list[str]
    key_findings: list[str]
    source_pages: list[int]

    @field_validator("core_concepts", "key_findings", "source_pages", mode="before")
    @classmethod
    def normalize_list_fields(cls, value):
        return _ensure_list(value)


class GroupSynthesis(BaseModel):
    summary: str
    concepts: list[str]
    findings: list[str]
    numeric_claims: list[str]
    limitations: list[str]
    future_work: list[str]
    contradictions: list[str]
    source_fact_ids: list[str] = Field(default_factory=list)
    source_sections: list[str]
    source_pages: list[int]

    @field_validator(
        "concepts",
        "findings",
        "numeric_claims",
        "limitations",
        "future_work",
        "contradictions",
        "source_fact_ids",
        "source_sections",
        "source_pages",
        mode="before",
    )
    @classmethod
    def normalize_list_fields(cls, value):
        return _ensure_list(value)


class GlobalAnswer(BaseModel):
    question: str
    answer: str
    source_fact_ids: list[str] = Field(default_factory=list)
    source_sections: list[str]
    source_pages: list[int]
    confidence: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)

    @field_validator(
        "source_fact_ids", "source_sections", "source_pages", "unsupported_claims", mode="before"
    )
    @classmethod
    def normalize_list_fields(cls, value):
        return _ensure_list(value)


class DocumentOverview(BaseModel):
    title: str
    executive_summary: str
    research_question: str
    methods: list[str]
    datasets: list[str]
    metrics: list[str]
    key_findings: list[str]
    limitations: list[str]
    future_work: list[str]
    contradictions: list[str]
    @field_validator(
        "methods",
        "datasets",
        "metrics",
        "key_findings",
        "limitations",
        "future_work",
        "contradictions",
        mode="before",
    )
    @classmethod
    def normalize_list_fields(cls, value):
        return _ensure_list(value)


class DocumentSynthesis(DocumentOverview):
    global_answers: list[GlobalAnswer] = Field(min_length=10, max_length=10)


def estimate_tokens(text: str) -> int:
    cjk = len(CJK_RE.findall(text))
    ascii_tokens = sum(math.ceil(len(item) / 4) for item in ASCII_TOKEN_RE.findall(text))
    stripped = ASCII_TOKEN_RE.sub("", CJK_RE.sub("", text))
    punctuation = len(re.findall(r"[^\s]", stripped))
    return cjk + ascii_tokens + punctuation


def _prefix_for_budget(text: str, budget: int) -> str:
    if estimate_tokens(text) <= budget:
        return text
    low, high = 1, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    candidate = text[:low]
    minimum = max(1, int(len(candidate) * 0.55))
    boundaries = [candidate.rfind("\n\n"), candidate.rfind("\n")]
    boundaries.extend(candidate.rfind(mark) for mark in "。！？.!?；;")
    boundary = max(boundaries)
    if boundary >= minimum:
        candidate = candidate[: boundary + 1]
    return candidate


def _suffix_for_budget(text: str, budget: int) -> str:
    if not text or budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    low, high = 0, len(text) - 1
    while low < high:
        mid = (low + high) // 2
        if estimate_tokens(text[mid:]) <= budget:
            high = mid
        else:
            low = mid + 1
    suffix = text[low:]
    boundary = min(
        [value for value in (suffix.find("\n"), suffix.find("。"), suffix.find(".")) if value >= 0]
        or [0]
    )
    if 0 < boundary < len(suffix) * 0.45:
        suffix = suffix[boundary + 1 :].lstrip()
    return suffix


def split_text(
    text: str,
    max_tokens: int = 3200,
    overlap_tokens: int = 180,
    page_start: int | None = None,
    page_end: int | None = None,
    page_texts: list[PageMarkdown] | None = None,
) -> list[TextChunk]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("invalid token budget")
    source = text.strip()
    remaining = source
    chunks: list[TextChunk] = []
    previous = ""
    remaining_offset = 0
    while remaining:
        overlap = _suffix_for_budget(previous, overlap_tokens) if previous else ""
        body_budget = max_tokens - estimate_tokens(overlap)
        body = _prefix_for_budget(remaining, body_budget)
        if not body:
            raise ValueError("unable to advance chunk cursor")
        combined = f"{overlap}{body}" if not overlap else f"{overlap}\n{body}"
        if estimate_tokens(combined) > max_tokens:
            body = _prefix_for_budget(body, max(1, body_budget - 1))
            combined = f"{overlap}\n{body}" if overlap else body
        body_start = remaining_offset
        overlap_start = source.rfind(overlap, 0, body_start) if overlap else body_start
        if overlap_start < 0:
            overlap_start = body_start
        chunk_page_texts = _page_texts_for_range(
            source, page_texts or [], overlap_start, body_start + len(body)
        )
        chunk_pages = [item.page for item in chunk_page_texts]
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=combined,
                token_estimate=estimate_tokens(combined),
                overlap_text=overlap,
                page_start=min(chunk_pages) if chunk_pages else page_start,
                page_end=max(chunk_pages) if chunk_pages else page_end,
                page_texts=chunk_page_texts,
            )
        )
        tail = remaining[len(body) :]
        stripped_tail = tail.lstrip()
        remaining_offset += len(body) + len(tail) - len(stripped_tail)
        remaining = stripped_tail
        previous = combined
    return chunks


def _page_texts_for_range(
    source: str, page_texts: list[PageMarkdown], start: int, end: int
) -> list[PageMarkdown]:
    result = []
    cursor = 0
    for item in page_texts:
        fragment = item.text.strip()
        if not fragment:
            continue
        fragment_start = source.find(fragment, cursor)
        if fragment_start < 0:
            continue
        fragment_end = fragment_start + len(fragment)
        cursor = fragment_end
        if fragment_end > start and fragment_start < end:
            result.append(item)
    return result


def _strip_picture_text(text: str) -> str:
    """Remove OCR/picture artifacts that pymupdf4llm emits for rasterized
    figures: the ``<!-- Start of picture text --> …`` HTML comment block (text
    read out of the image), inline ``<img …>`` tags, and ``<mark>`` highlight
    spans (their inner text is kept)."""
    text = PICTURE_TEXT_BLOCK_RE.sub("", text)
    text = IMG_TAG_RE.sub("", text)
    text = MARK_TAG_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def validate_pdf(pdf_path: str | Path) -> int:
    """Open the PDF and return its page count, failing with a usable message.

    A truncated or otherwise damaged file otherwise surfaces as a cryptic
    ``FzErrorArgument: not a dict (null)`` raised deep inside pymupdf4llm's layout
    pass, which says nothing about the real cause. Measured on a real case: a
    27 MB PDF whose upload was cut off (header intact, no ``%%EOF``, no page tree)
    opened as 0 pages and crashed the layout code. Opening the file first turns
    that into an actionable error.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        with pymupdf.open(str(path)) as document:
            pages = document.page_count
    except Exception as error:  # noqa: BLE001 - surface any mupdf failure verbatim
        raise ValueError(
            f"cannot open PDF {path.name!r}: {type(error).__name__}: {error}. "
            "The file may be truncated or damaged; re-download it and retry."
        ) from error
    if pages == 0:
        truncated = b"%%EOF" not in path.read_bytes()[-4096:]
        hint = (
            " It has no %%EOF trailer, so the upload/download was likely cut short."
            if truncated
            else ""
        )
        raise ValueError(
            f"PDF {path.name!r} opened but reports 0 pages, so its page tree is "
            f"unreadable (size {path.stat().st_size} bytes).{hint} "
            "Re-obtain the original file and retry."
        )
    return pages


def extract_pdf_pages(pdf_path: str | Path) -> list[PageMarkdown]:
    path = Path(pdf_path)
    validate_pdf(path)
    raw_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    pages = []
    for index, item in enumerate(raw_pages):
        metadata = item.get("metadata", {})
        page = metadata.get("page_number", metadata.get("page", index + 1))
        text = _strip_picture_text(item.get("text", ""))
        pages.append(PageMarkdown(page=int(page), text=text, metadata=metadata))
    return pages


def _node_id(page: int, level: int, title: str, sequence: int) -> str:
    raw = f"{page}:{level}:{title}:{sequence}".encode("utf-8")
    return f"node-{hashlib.sha1(raw).hexdigest()[:12]}"


def _plain_heading(title: str) -> str:
    title = re.sub(r"<[^>]*>", "", title)
    title = re.sub(r"[*_`]+", "", title)
    return unicodedata.normalize("NFKC", title).strip()


def is_probable_heading(title: str) -> bool:
    cleaned = _plain_heading(title)
    lowered = cleaned.lower()
    if not cleaned or len(cleaned) > 160:
        return False
    if lowered in {"research", "article", "original article", "研究", "论文"}:
        return False
    running_header = re.sub(r"[^a-z\u3400-\u9fff]+", "", lowered)
    if running_header in {"discoverinternetofthings", "发现物联网"}:
        return False
    if re.match(r"^(?:https?://|doi\b|\|)", lowered) or "@" in cleaned:
        return False
    if re.match(r"^(?:table|figure|fig\.?|表|图|公式)[\s�]*\d+", cleaned, re.IGNORECASE):
        return False
    if re.search(r"[。！？!?；;.]\s*$", cleaned):
        return False
    if re.search(r"\b[A-Za-z][A-Za-z0-9_ ]{0,24}\s*=", cleaned):
        return False
    if cleaned.count("·") >= 2 or cleaned.count("|") >= 2:
        return False

    generic_labels = {
        "characters",
        "timeline",
        "notes",
        "output format",
        "output format:",
        "weekly planning",
        "planning & contact prompts",
        "goal setting & reflection",
        "current state",
        "impact statement",
        "system prompts",
        "case studies",
        "case study",
    }
    if re.match(r"^case\s+\d+\b", lowered):
        return False
    if lowered in generic_labels:
        return False
    if lowered.startswith("output format") or lowered.startswith("weekly planning"):
        return False
    if re.match(r"^(?:a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)\.?\d+(?:\.\d+)*", lowered):
        return False
    if re.match(r"^(?:appendix|附录)\s*(?:[a-z]|[一-龥]|\d)", lowered):
        return False
    if re.match(r"^(?:case|案例)\s*(?:[0-9]+|[a-z])", lowered):
        return False
    if len(cleaned.split()) <= 2 and lowered in {"characters", "timeline", "notes"}:
        return False
    return True


def _section_signature(title: str) -> str | None:
    cleaned = _plain_heading(title).lstrip("# ")
    numbered = re.match(r"^(\d{1,2}(?:[.．]\d{1,2})*)\D", cleaned)
    if numbered:
        return f"number:{numbered.group(1).replace('．', '.')}"
    normalized = re.sub(r"[^a-z\u3400-\u9fff]+", "", cleaned.lower())
    aliases = {"abstract": "abstract", "摘要": "abstract"}
    return aliases.get(normalized)


# Unnumbered level-1/2 headings that are document structure, not the document
# title. _section_signature only catches numbered sections and Abstract/摘要, and
# is_back_matter/_is_toc_section do not catch Introduction/References etc., so a
# title candidate must also clear this blacklist.
_FRONT_MATTER_TITLES = {
    "abstract",
    "introduction",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "contents",
    "acknowledgements",
    "acknowledgments",
    "keywords",
    "key words",
    "摘要",
    "引言",
    "绪论",
    "前言",
    "结论",
    "参考文献",
    "目录",
    "关键词",
    "致谢",
}


def _document_key(pdf_path: str | Path) -> str:
    """Stable identity for a source document, used to scope concept memory.

    Hashing the bytes (not the path) means the same paper keeps its memory when
    moved or renamed, while a different paper can never inherit it. `--document-key`
    overrides this when an explicit grouping is wanted.
    """
    path = Path(pdf_path)
    digest = hashlib.sha1()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        # Unreadable source: fall back to the name so scoping still separates
        # documents instead of collapsing everything into one namespace.
        digest.update(str(path).encode("utf-8"))
    return f"doc-{digest.hexdigest()[:16]}"


def _candidate_document_title(root: HeadingNode) -> str | None:
    """Deterministic document-title candidate from the heading tree.

    The paper title usually survives preprocessing as a top-level heading that is
    not a numbered section (``_normalize_levels`` re-bases numbered sections to
    L2 but leaves unnumbered wrappers like the title at their original level, so
    the title may be L1 *or* L2 -- never require a specific level). Take the
    first root child that is not Abstract/numbered/back-matter/TOC/front-matter
    and passes ``is_probable_heading`` (running headers, URLs, captions and
    generic labels are rejected there). Returns None when no clean title exists;
    the overview model then falls back to its own summary.
    """
    for idx, node in enumerate(root.children):
        if node.page_start > 2:  # a real title sits on page 1
            continue
        title = _plain_heading(node.title)
        if not title:
            continue
        if _section_signature(node.title) is not None:
            continue
        if is_back_matter(node.title) or _is_toc_section(node.title):
            continue
        if title.lower() in _FRONT_MATTER_TITLES:
            continue
        if not is_probable_heading(title):
            continue
        return _join_title_lines(root.children, idx, title)
    return None


def _join_title_lines(children: list[HeadingNode], idx: int, title: str) -> str:
    """Re-join a multi-line paper title that preprocessing split into siblings.

    The title block is laid out as separate heading nodes ("Agent AI:\nSurveying
    the Horizons of Multimodal Interaction" becomes two root children), so the
    first plausible line alone is truncated. A trailing colon is the dominant
    title-split signal: while the accumulated title ends with ':' and the next
    sibling is another plausible title line on the same page (passes the same
    filters as the first line), append it. This stops at Abstract/Introduction/
    numbered sections because they fail the filters, and never glues a following
    section onto a colon-less title like "My Paper".
    """
    parts = [title]
    node = children[idx]
    for sibling in children[idx + 1 :]:
        if sibling.page_start != node.page_start or not parts[-1].endswith(":"):
            break
        text = _plain_heading(sibling.title)
        if (
            not text
            or _section_signature(sibling.title) is not None
            or is_back_matter(sibling.title)
            or _is_toc_section(sibling.title)
            or text.lower() in _FRONT_MATTER_TITLES
            or not is_probable_heading(text)
        ):
            break
        parts.append(text)
    return " ".join(parts)


def deduplicate_sections(sections: list[HeadingNode]) -> list[HeadingNode]:
    result: list[HeadingNode] = []
    canonical_by_signature: dict[str, HeadingNode] = {}
    for section in sections:
        signature = _section_signature(section.title)
        canonical = canonical_by_signature.get(signature) if signature else None
        if canonical is None:
            result.append(section)
            if signature:
                canonical_by_signature[signature] = section
            continue
        if section.text.strip() and section.text.strip() not in canonical.text:
            canonical.text = f"{canonical.text}\n{section.text}".strip()
            canonical.page_texts.extend(section.page_texts)
        canonical.page_start = min(canonical.page_start, section.page_start)
        canonical.page_end = max(canonical.page_end, section.page_end)
        canonical.subtree_page_end = max(canonical.subtree_page_end, section.subtree_page_end)
    return result


def build_heading_tree(pages: list[PageMarkdown]) -> HeadingNode:
    first_page = pages[0].page if pages else 1
    root = HeadingNode(
        id="document-root",
        title="Document",
        level=0,
        page_start=first_page,
        page_end=first_page,
        subtree_page_end=first_page,
    )
    stack = [root]
    sequence = 0

    def append_line(node: HeadingNode, page: int, line: str) -> None:
        node.text = f"{node.text}\n{line}".strip()
        if node.page_texts and node.page_texts[-1].page == page:
            node.page_texts[-1].text = f"{node.page_texts[-1].text}\n{line}".strip()
        else:
            node.page_texts.append(PageMarkdown(page=page, text=line.strip()))
        node.page_end = page
        node.subtree_page_end = page

    for page in pages:
        for line in page.text.splitlines():
            match = HEADING_RE.match(line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                if not is_probable_heading(title):
                    current = stack[-1]
                    append_line(current, page.page, line)
                    continue
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                sequence += 1
                parent = stack[-1]
                node = HeadingNode(
                    id=_node_id(page.page, level, title, sequence),
                    title=title,
                    level=level,
                    page_start=page.page,
                    page_end=page.page,
                    subtree_page_end=page.page,
                    parent_id=parent.id,
                )
                parent.children.append(node)
                stack.append(node)
            elif line.strip():
                current = stack[-1]
                append_line(current, page.page, line)
        for node in stack:
            node.subtree_page_end = max(node.subtree_page_end, page.page)

    def propagate(node: HeadingNode) -> int:
        for child in node.children:
            node.subtree_page_end = max(node.subtree_page_end, propagate(child))
        return node.subtree_page_end

    _absorb_noise_nodes(root)
    _merge_low_signal_sections(root)
    _merge_fragment_leaves(root)
    _normalize_levels(root)
    propagate(root)
    return root


# A heading whose body is a figure/table caption, or whose body opens with a
# prompt-transcript role header, or whose bare title is a role word, is not a
# real section: pymupdf4llm promoted a caption/transcript block to a heading.
_FIGURE_CAPTION_LINE_RE = re.compile(
    r"^(?:figure|fig(?:\.|ure)?|table|图|表)\s*\d+", re.IGNORECASE
)
_ROLE_HEADER_LINE_RE = re.compile(
    r"^\*\*(?:prompt|system|user|assistant|human|ai|input|output|response)\s*:?\*\*",
    re.IGNORECASE,
)
_BARE_ROLE_TITLES = {
    "output", "input", "prompt", "response", "system", "user", "assistant",
    "output:", "input:", "prompt:", "response:", "system:", "user:", "assistant:",
}


def _noise_node_reason(child: HeadingNode) -> str | None:
    """'caption' / 'transcript' / None for a heading that is really a figure
    caption or a prompt-transcript fragment (not a real section)."""
    body = _strip_picture_text(child.text).strip()
    if not body:
        return None
    first = body.splitlines()[0].lstrip("# ").strip()
    if _FIGURE_CAPTION_LINE_RE.match(first) and not _section_signature(child.title):
        return "caption"
    if _ROLE_HEADER_LINE_RE.match(first):
        return "transcript"
    if _plain_heading(child.title).strip().lower() in _BARE_ROLE_TITLES:
        return "transcript"
    return None


def _absorb_noise_nodes(node: HeadingNode) -> None:
    """Drop headings that are really figure captions or prompt-transcript
    fragments: their body merges into this node, their children (if any) are
    promoted to this node. Without this, a promoted caption like ``## General
    Alpaca Training Framework`` becomes a bogus parent that every following
    section nests under."""
    for child in list(node.children):
        _absorb_noise_nodes(child)

    merged_children: list[HeadingNode] = []
    for child in list(node.children):
        if _noise_node_reason(child) is None:
            merged_children.append(child)
            continue
        if child.text.strip():
            separator = "\n" if node.text.strip() else ""
            node.text = f"{node.text}{separator}{child.text}".strip()
        if child.page_texts:
            node.page_texts.extend(child.page_texts)
        node.page_end = max(node.page_end, child.page_end)
        node.subtree_page_end = max(node.subtree_page_end, child.subtree_page_end)
        for grandchild in child.children:
            grandchild.parent_id = node.id
        merged_children.extend(child.children)
    node.children = merged_children


def _merge_low_signal_sections(node: HeadingNode) -> None:
    node.children = [child for child in node.children if child is not None]
    for child in list(node.children):
        _merge_low_signal_sections(child)

    merged_children: list[HeadingNode] = []
    for child in list(node.children):
        if not child.children and _should_merge_child(child):
            if child.text.strip():
                separator = "\n" if node.text.strip() else ""
                node.text = f"{node.text}{separator}{child.text}".strip()
            if child.page_texts:
                node.page_texts.extend(child.page_texts)
            node.page_end = max(node.page_end, child.page_end)
            node.subtree_page_end = max(node.subtree_page_end, child.subtree_page_end)
            continue
        merged_children.append(child)
    node.children = merged_children


def _should_merge_child(child: HeadingNode) -> bool:
    title = _plain_heading(child.title)
    lowered = title.lower()
    if not title:
        return False
    if child.level <= 2:
        return False
    if len(title.split()) > 3:
        return False
    if lowered in {"characters", "timeline", "notes", "current state", "weekly planning", "output format", "output format:", "planning & contact prompts", "goal setting & reflection", "impact statement", "system prompts"}:
        return True
    if re.match(r"^(?:a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)\.?\d+(?:\.\d+)*", lowered):
        return True
    if re.match(r"^(?:appendix|附录)\s*(?:[a-z]|[一-龥]|\d)", lowered):
        return True
    if re.match(r"^(?:case|案例)\s*(?:[0-9]+|[a-z])", lowered):
        return True
    if len(title.split()) <= 2 and child.text.strip() and len(child.text.strip()) <= 220:
        return not CJK_RE.search(title)
    return False


# Small numbered leaf sections (a one-paragraph ``9.1.2 Task``) each pay a full
# extraction + synthesis round, so a survey with 60+ single-paragraph sub-
# sections multiplies API calls. Merging a small numbered leaf up one level
# collapses those rounds into one chunked pass over the parent; the sub-section's
# own heading line is kept inside the merged body so its number still anchors
# evidence for sub-section-level questions.
_FRAGMENT_MERGE_MAX_TOKENS = 500


def _should_merge_fragment_leaf(child: HeadingNode) -> bool:
    """True for a tiny sub-section that is a fragment, not a section worth its
    own extraction round: a leaf (no children), below the L2 main level, with a
    numeric signature that carries a sub-number (``number:5.4``, never a main
    ``number:5``), and a body within ``_FRAGMENT_MERGE_MAX_TOKENS``."""
    if child.children:
        return False
    if child.level <= 2:
        return False
    signature = _section_signature(child.title)
    if not (signature and signature.startswith("number:")):
        return False
    if "." not in signature:
        return False  # main section ("5"), not a sub-section ("5.4")
    return estimate_tokens(child.text) <= _FRAGMENT_MERGE_MAX_TOKENS


def _merge_fragment_leaves(node: HeadingNode) -> None:
    """Merge eligible fragment leaves into ``node``, keeping each child's
    heading line as a marker at the top of the merged body. Runs after
    ``_merge_low_signal_sections`` and before ``_normalize_levels`` (both merge
    and level-rebasing guard on the raw markdown level)."""
    node.children = [child for child in node.children if child is not None]
    for child in list(node.children):
        _merge_fragment_leaves(child)

    merged_children: list[HeadingNode] = []
    for child in list(node.children):
        if not _should_merge_fragment_leaf(child):
            merged_children.append(child)
            continue
        marker = f"{'#' * child.level} {child.title}".strip()
        body = marker if not child.text.strip() else f"{marker}\n{child.text}".strip()
        node.text = f"{node.text}\n{body}".strip() if node.text.strip() else body
        if child.page_texts:
            child.page_texts[0].text = f"{marker}\n{child.page_texts[0].text}".strip()
            node.page_texts.extend(child.page_texts)
        node.page_end = max(node.page_end, child.page_end)
        node.subtree_page_end = max(node.subtree_page_end, child.subtree_page_end)
    node.children = merged_children


def _normalize_levels(root: HeadingNode) -> None:
    """Re-base heading levels so the top-level numbered sections land at L2.

    Some papers mark main sections with ```###```/```####``` (landing at L3/L4)
    instead of ```##```/```###```. Compute the minimum markdown level among
    numbered sections and shift every section at or below that level up by the
    offset, so the top of the numbering becomes L2. Unnumbered wrappers below
    that level (e.g. the paper title at L1) keep their original level. Runs
    after ``_merge_low_signal_sections`` because that logic guards on the raw
    markdown level (```child.level <= 2```)."""
    min_level: int | None = None
    nodes: list[HeadingNode] = []

    def visit(node: HeadingNode) -> None:
        nodes.append(node)
        for child in node.children:
            visit(child)

    visit(root)
    for node in nodes:
        signature = _section_signature(node.title)
        if signature and signature.startswith("number:") and node.level != 0:
            if min_level is None or node.level < min_level:
                min_level = node.level
    if min_level is None or min_level <= 2:
        return
    offset = min_level - 2
    for node in nodes:
        if node.level >= min_level:
            node.level -= offset


def flatten_sections(root: HeadingNode) -> list[HeadingNode]:
    result: list[HeadingNode] = []

    def visit(node: HeadingNode) -> None:
        for child in node.children:
            result.append(child)
            visit(child)

    visit(root)
    return result


def is_back_matter(title: str) -> bool:
    cleaned = re.sub(r"[*_#<>/\\]+", "", title).strip()
    return bool(BACK_MATTER_RE.fullmatch(cleaned))


def _normalize_concept_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value)


def _concept_keys(name: str) -> set[str]:
    normalized = _normalize_concept_text(name)
    keys = {normalized} if normalized else set()
    for acronym in re.findall(r"[（(]\s*([A-Za-z][A-Za-z0-9_-]{1,15})\s*[)）]", name):
        keys.add(acronym.lower().replace("_", "").replace("-", ""))
    return keys


_SCRIPT_RUN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
# A space sitting between two CJK characters is a PDF extraction artefact, not a
# real word break, so it must be removed before tokenising -- otherwise the alias
# "支持向量机" can never match "支持 向量机".
_CJK_GAP_RE = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\ufaff])\s+(?=[\u3400-\u4dbf\u4e00-\u9fff\ufaff])"
)
# Joins token runs in a matching haystack. It must not appear in real text and
# must not be an ASCII word character, so the word-boundary test can treat it as
# a boundary while still keeping tokens from being welded together.
_RUN_SEPARATOR = "\x00"


def _concept_runs(value: str) -> list[str]:
    """Lowercased ASCII/CJK runs of `value`, with intra-CJK spaces dropped."""
    normalized = _CJK_GAP_RE.sub("", unicodedata.normalize("NFKC", value).lower())
    return _SCRIPT_RUN_RE.findall(normalized)


def _tokenized_concept_text(value: str) -> str:
    """Normalised text as script runs joined by a single space.

    ``_normalize_concept_text`` strips every separator, which welds neighbouring
    words together: "each agent acts" became "eachagentacts", so the alias
    "agent" matched starting *inside* "each" and a word-boundary check then
    rejected the whole string.
    """
    return " ".join(_concept_runs(value))


def _alias_occurs_in(alias_key: str, normalized_text: str) -> bool:
    """True when `alias_key` occurs as a whole token of `normalized_text`.

    A plain substring test over the stripped normalization lets short aliases
    match inside unrelated longer words. Measured on a real 1530-alias memory DB,
    ordinary prose retrieved pure noise that was then injected into every
    extraction prompt as `related_memory`: "HER" matched inside "other",
    "Position" inside "composition", "Actions" inside "interactions", "map"
    inside "checkpoint".

    Matching therefore requires that the alias is not glued to an adjacent ASCII
    word character, which is what makes "agent" safe against "agentopia".

    The text is tokenised here via `_concept_runs`, so this function accepts raw
    or pre-normalised input and cannot drift from `_tokenized_concept_text`.
    """
    if not alias_key:
        return False
    # Stored alias keys are normally already lowercased by `_normalize_concept_text`,
    # but not guaranteed (an acronym alias can keep its case). The haystack is
    # lowercased, so comparing without folding case silently drops those aliases --
    # the real DB has the key 'AI' stored uppercase.
    alias_key = unicodedata.normalize("NFKC", alias_key).lower()
    # Collapse whitespace before tokenising so the alias key and the haystack use
    # the same token sequence regardless of how the key was originally spaced.
    alias_runs = _concept_runs(re.sub(r"\s+", " ", alias_key))
    if not alias_runs:
        return False
    runs = _concept_runs(normalized_text)
    if not runs:
        return False
    # Join on a preserved separator so a multi-token alias key stays findable while
    # adjacent tokens keep an edge to test against. Joining with "" would let
    # "agent" match inside "agentopia"; separating with " " would break every
    # multi-word concept, whose alias key is one glued word
    # ("knowledgeaugmentedtransformer") that no spaced text can ever contain.
    haystack = _RUN_SEPARATOR.join(runs)
    needle = _RUN_SEPARATOR.join(alias_runs)
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        before = haystack[start - 1] if start else ""
        after = haystack[end] if end < len(haystack) else ""
        # A boundary is satisfied at a separator, so only a *word* character
        # glued to the match disqualifies it.
        if not (before.isascii() and before.isalnum()) and not (
            after.isascii() and after.isalnum()
        ):
            return True
        start = haystack.find(needle, start + 1)
    return False


class ConceptMemory:
    """Concept/alias memory used to normalise terminology while extracting.

    The database is shared between runs, so rows are stamped with the document
    they came from and `retrieve` only returns concepts belonging to the document
    being processed. Without that filter, processing paper B surfaced paper A's
    concepts into B's prompts.
    """

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                definition TEXT NOT NULL DEFAULT '',
                document_key TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS aliases (
                alias_key TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL,
                alias TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS occurrences (
                concept_id TEXT NOT NULL,
                section_id TEXT NOT NULL,
                page INTEGER,
                source_quote TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self._ensure_document_key_column()

    def _ensure_document_key_column(self) -> None:
        """Add `concepts.document_key` to databases predating document scoping.

        Existing rows default to '' and are treated as unowned, so they are never
        returned for a document-keyed lookup. CREATE TABLE IF NOT EXISTS cannot
        add a column to an existing table, hence the explicit migration.
        """
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(concepts)")
        }
        if "document_key" not in columns:
            self.connection.execute(
                "ALTER TABLE concepts ADD COLUMN document_key TEXT NOT NULL DEFAULT ''"
            )
            self.connection.commit()

    def upsert(
        self,
        name: str,
        definition: str = "",
        section_id: str = "",
        page: int | None = None,
        source_quote: str = "",
        document_key: str = "",
    ) -> str:
        keys = _concept_keys(name)
        if not keys:
            raise ValueError("empty concept")
        placeholders = ",".join("?" for _ in keys)
        row = self.connection.execute(
            f"SELECT concept_id FROM aliases WHERE alias_key IN ({placeholders}) LIMIT 1", tuple(keys)
        ).fetchone()
        concept_id = row["concept_id"] if row else f"concept-{hashlib.sha1(min(keys).encode()).hexdigest()[:12]}"
        self.connection.execute(
            "INSERT OR IGNORE INTO concepts(id, canonical_name, definition, document_key) VALUES (?, ?, ?, ?)",
            (concept_id, name, definition, document_key),
        )
        if definition:
            self.connection.execute(
                "UPDATE concepts SET definition = CASE WHEN definition = '' THEN ? ELSE definition END WHERE id = ?",
                (definition, concept_id),
            )
        for key in keys:
            self.connection.execute(
                "INSERT OR REPLACE INTO aliases(alias_key, concept_id, alias) VALUES (?, ?, ?)",
                (key, concept_id, name),
            )
        if section_id:
            self.connection.execute(
                "INSERT INTO occurrences(concept_id, section_id, page, source_quote) VALUES (?, ?, ?, ?)",
                (concept_id, section_id, page, source_quote),
            )
        self.connection.commit()
        return concept_id

    def retrieve(self, text: str, limit: int = 8, document_key: str = "") -> list[dict]:
        normalized = _tokenized_concept_text(text)
        if not normalized.strip():
            return []
        rows = self.connection.execute(
            """
            SELECT DISTINCT c.id, c.canonical_name, c.definition, c.document_key, a.alias_key
            FROM concepts c JOIN aliases a ON a.concept_id = c.id
            """
        ).fetchall()
        matches = [
            dict(row)
            for row in rows
            if _alias_occurs_in(row["alias_key"], normalized)
            and (not document_key or row["document_key"] == document_key)
        ]
        return matches[:limit]

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM concepts").fetchone()[0])

    def purge_unowned(self) -> dict:
        """Delete concepts that belong to no document (`document_key = ''`).

        Rows written before document scoping existed carry no owner, so `retrieve`
        without a `document_key` still returns them -- which is exactly the
        cross-document leak the scoping was added to stop. Measured on the real
        memory DB: 1499 of 1596 concepts were unowned, so scoping only protected
        documents processed after the change.

        The DB is a rebuildable cache, so the default remedy is to drop the unowned
        rows and let the next run repopulate them under a real document key.
        `occurrences` rows for removed concepts go too, else they orphan.
        """
        concept_ids = [
            row["id"]
            for row in self.connection.execute(
                "SELECT id FROM concepts WHERE document_key = ''"
            )
        ]
        if not concept_ids:
            return {"concepts_deleted": 0, "occurrences_deleted": 0, "aliases_deleted": 0}
        placeholders = ",".join("?" for _ in concept_ids)
        occurrences = self.connection.execute(
            f"DELETE FROM occurrences WHERE concept_id IN ({placeholders})", concept_ids
        ).rowcount
        aliases = self.connection.execute(
            f"DELETE FROM aliases WHERE concept_id IN ({placeholders})", concept_ids
        ).rowcount
        self.connection.execute(
            f"DELETE FROM concepts WHERE id IN ({placeholders})", concept_ids
        )
        self.connection.commit()
        # SQLite keeps freed pages in the file, so without this the DB stays the
        # same size on disk after deleting almost everything.
        self.connection.execute("VACUUM")
        self.connection.commit()
        return {
            "concepts_deleted": len(concept_ids),
            "occurrences_deleted": occurrences,
            "aliases_deleted": aliases,
        }

    def close(self) -> None:
        self.connection.close()


def cache_key(stage: str, model: str, prompt_version: str, content: str) -> str:
    raw = json.dumps([stage, model, prompt_version, content], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResultCache:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.hits = 0
        self.misses = 0
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)"
        )

    def get(self, key: str) -> dict | None:
        row = self.connection.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
        else:
            self.misses += 1
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO cache(key, payload, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def summary(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}


class UsageTracker:
    """Per-run and all-time API usage accounting.

    The ``usage`` table is append-only and shared across runs, so a naive
    ``SUM(...) FROM usage`` reports the lifetime total of the database, not what
    the current run spent. ``summary()`` therefore scopes to the rows written
    since this tracker was opened, and ``all_time_summary()`` exposes the
    lifetime aggregate explicitly. Both are reported in the output JSON so a
    cost estimate cannot silently mix the two.

    Tokens are always written with a UTC "+00:00" suffix, so comparing the ISO
    strings lexicographically is exactly comparing instants.
    """

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                stage TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                latency_seconds REAL NOT NULL,
                request_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Captured before the first write, so every row recorded by this tracker
        # sorts at or after it.
        self.run_started_at = datetime.now(timezone.utc).isoformat()

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        latency_seconds: float,
        request_id: str,
    ) -> None:
        self.connection.execute(
            "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                stage,
                model,
                input_tokens,
                output_tokens,
                total_tokens,
                latency_seconds,
                request_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.connection.commit()

    def _aggregate(self, since: str | None) -> dict:
        where = " WHERE created_at >= ?" if since else ""
        params: tuple = (since,) if since else ()
        totals = self.connection.execute(
            f"""
            SELECT COUNT(*), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0),
                   COALESCE(SUM(total_tokens), 0), COALESCE(SUM(latency_seconds), 0)
            FROM usage{where}
            """,
            params,
        ).fetchone()
        by_model = {}
        for row in self.connection.execute(
            f"""
            SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(total_tokens), SUM(latency_seconds)
            FROM usage{where} GROUP BY model
            """,
            params,
        ):
            by_model[row[0]] = {
                "calls": row[1],
                "input_tokens": row[2],
                "output_tokens": row[3],
                "total_tokens": row[4],
                "latency_seconds": round(row[5], 3),
            }
        return {
            "calls": totals[0],
            "input_tokens": totals[1],
            "output_tokens": totals[2],
            "total_tokens": totals[3],
            "latency_seconds": round(totals[4], 3),
            "by_model": by_model,
        }

    def summary(self) -> dict:
        """Usage recorded by this run only. Same shape as `all_time_summary`."""
        result = self._aggregate(self.run_started_at)
        result["run_started_at"] = self.run_started_at
        return result

    def all_time_summary(self) -> dict:
        """Lifetime usage of the database, across every run that shares it."""
        return self._aggregate(None)

    def close(self) -> None:
        self.connection.close()


def _response_request_id(response: object) -> str:
    """Provider-side identifier for a completion, for the usage ledger.

    The OpenAI SDK exposes it as ``id`` on the response model (``ChatCompletion``
    declares ``id`` as a field). An earlier version read ``_request_id``, which no
    real SDK object has -- so every row in the usage DB recorded an empty string
    and the column was useless. ``_request_id`` is kept as a fallback so responses
    produced by test doubles or older adapters still resolve.
    """
    for attribute in ("id", "_request_id"):
        value = getattr(response, attribute, None)
        if value:
            return str(value)
    return ""


def _quote_is_grounded(quote: str, source: str) -> bool:
    def compact(value: str) -> str:
        value = re.sub(r"<[^>]*>", "", value)
        value = unicodedata.normalize("NFKC", value).lower()
        return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value)

    compact_quote = compact(quote)
    compact_source = compact(source)
    return bool(compact_quote) and compact_quote in compact_source


def _quote_page(quote: str, page_texts: list[PageMarkdown]) -> int | None:
    for item in page_texts:
        if _quote_is_grounded(quote, item.text):
            return item.page
    pages = {item.page for item in page_texts}
    return next(iter(pages)) if len(pages) == 1 else None


def _apply_grounding(
    result: ChunkExtraction, source: str, page_texts: list[PageMarkdown] | None = None
) -> ChunkExtraction:
    for fact in result.facts:
        fact.grounded = _quote_is_grounded(fact.source_quote, source)
        if fact.grounded and page_texts:
            fact.page = _quote_page(fact.source_quote, page_texts)
        elif not fact.grounded:
            fact.confidence = min(fact.confidence, 0.5)
    return result


def _clean_table_cell(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]*>", "", value)
    value = re.sub(r"[*_`]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _parse_numeric_cell(value: str) -> tuple[float, str, str] | None:
    cleaned = _clean_table_cell(value)
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*(%?)", cleaned)
    if not match:
        return None
    return float(match.group(1)), match.group(1), match.group(2)


def _cell_is_numeric(cell: str) -> bool:
    """True when a table cell is a plain number (optionally a % and a leading
    arrow/delta glyph), e.g. "64.28", "99%", "↑2.5%". Model names like
    "4.5-Opus" or "70B" are NOT numeric, so they can stay in header rows."""
    stripped = _clean_table_cell(cell).lstrip("↑↓−+-±")
    return bool(re.fullmatch(r"\d+(?:\.\d+)?\s*%?", stripped))


_LEADING_STRAY_CLOSERS_RE = re.compile(r"^[)\]}）】」]+")
_BRACKET_PAIRS = (("(", ")"), ("（", "）"), ("[", "]"), ("【", "】"), ("{", "}"))


def _repair_header_name(name: str) -> str:
    """Clean deterministic noise left by pymupdf4llm splitting a wide header
    cell into fragment columns (``Reward (z-score)`` -> ``Reward`` / ``(z-score``
    / ``)``) that then stack across header rows. ``) Soc. (Δ)`` becomes
    ``Soc. (Δ)`` and ``(z-score Econ.`` becomes ``Econ.``. Balanced names such as
    ``Accuracy (%)`` or ``Soc. (Δ)`` and bracket-free names are left untouched.
    """
    cleaned = re.sub(_LEADING_STRAY_CLOSERS_RE, "", name).strip()
    # A dangling opener never closed within the name is a fragment of a
    # neighbouring cell — drop the fragment token that begins at it.
    for opener, closer in _BRACKET_PAIRS:
        if opener not in cleaned:
            continue
        if closer in cleaned:
            continue  # balanced — a genuine name like ``(Δ)``
        start = cleaned.find(opener)
        space = cleaned.find(" ", start)
        end = space if space != -1 else len(cleaned)
        cleaned = (cleaned[:start] + cleaned[end:]).strip()
        break
    # A stray closer with no opener in the name (its opener lived in the
    # neighbouring fragment column) is a lone noise character.
    for opener, closer in _BRACKET_PAIRS:
        if opener not in cleaned and closer in cleaned:
            cleaned = cleaned.replace(closer, "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _merge_table_headers(header_rows: list[list[str]]) -> list[str]:
    """Merge multi-row table headers into one composite header per column.

    Header rows are read top-to-bottom; empty cells are span continuations and
    are skipped. Non-empty cells stack into a space-joined name, so a two-row
    header like ``Claude`` / ``4.5-Opus`` becomes ``Claude 4.5-Opus``. When two
    columns end up with the same composite name (a table whose header repeats a
    value), a ``#N`` suffix keeps every column's identity distinct.
    """
    width = max((len(row) for row in header_rows), default=0)
    parts: list[list[str]] = [[] for _ in range(width)]
    for row in header_rows:
        padded = [*row, *("" for _ in range(width - len(row)))]
        for column, cell in enumerate(padded):
            cleaned = _clean_table_cell(cell)
            if cleaned:
                parts[column].append(cleaned)
    seen: dict[str, int] = {}
    headers: list[str] = []
    for column, part in enumerate(parts):
        raw = " ".join(part) if part else f"column_{column + 1}"
        name = _repair_header_name(raw) or f"column_{column + 1}"
        count = seen.get(name, 0)
        headers.append(name if count == 0 else f"{name} #{count + 1}")
        seen[name] = count + 1
    return headers


def _table_rows(text: str) -> list[tuple[str, list[str], list[list[str]], list[str]]]:
    lines = text.splitlines()
    tables = []
    index = 0
    last_text = ""
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("|"):
            if line:
                last_text = _clean_table_cell(line)
            index += 1
            continue
        block = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index].strip())
            index += 1
        rows = []
        for row in block:
            inner = row[1:] if row.startswith("|") else row
            inner = inner[:-1] if inner.endswith("|") else inner
            rows.append([cell.strip() for cell in inner.split("|")])
        separator = next(
            (
                row_index
                for row_index, row in enumerate(rows)
                if row and all(not cell or re.fullmatch(r":?-{3,}:?", cell) for cell in row)
            ),
            None,
        )
        if separator is None or separator == 0 or separator == len(rows) - 1:
            continue
        # Multi-row headers: merge the rows above the separator with any
        # immediately-following rows that contain no numeric cell (the second
        # row of a two-row header like ``Claude`` / ``4.5-Opus``).
        header_rows = [
            row for row in rows[:separator] if any(_clean_table_cell(cell) for cell in row)
        ]
        continuation = 0
        for row in rows[separator + 1 :]:
            if row and not any(_cell_is_numeric(cell) for cell in row):
                header_rows.append(row)
                continuation += 1
            else:
                break
        headers = _merge_table_headers(header_rows)
        data_start = separator + 1 + continuation
        title = last_text if re.search(r"\btable\s*\d+|表\s*[0-9]", last_text, re.IGNORECASE) else ""
        embedded_title = next(
            (
                _clean_table_cell(cell)
                for cell in headers
                if re.search(r"\btable\s*\d+|表\s*[0-9]", cell, re.IGNORECASE)
            ),
            "",
        )
        tables.append(
            (embedded_title or title or "Table", headers, rows[data_start:], block[data_start:])
        )
    return tables


def _row_is_config_row(table: str, metric: str, row_context: dict[str, str]) -> bool:
    """True when a numeric table row is a config/parameter value, not an
    evaluation result. Mirrors the retrieval-time config penalty
    (``_GENERIC_METRIC_COLUMNS`` / ``_CONFIG_TABLE_MARKERS``) at the source, so
    config rows never enter the fact DB as ``kind="metric"`` noise in the first
    place. A row counts as config when its numeric column has a generic label
    AND the table is flagged as config OR every row-context key is itself a
    generic label — genuine result columns like ``Total``/``Mood`` under a
    ``Model`` context survive."""
    generic_column = metric.strip().lower() in _GENERIC_METRIC_COLUMNS
    marker_text = " ".join([table, metric, *row_context.values()]).lower()
    config_marker = any(marker in marker_text for marker in _CONFIG_TABLE_MARKERS)
    generic_context = bool(row_context) and all(
        key.strip().lower() in _GENERIC_METRIC_COLUMNS for key in row_context
    )
    return generic_column and (config_marker or generic_context)


# Concept label for the per-table header fact (see `extract_table_facts`).
_TABLE_HEADER_CONCEPT = "表结构"


def extract_table_facts(section: HeadingNode) -> list[TableFact]:
    """One fact per numeric table cell, plus a single header fact per table.

    The table caption is emitted **once** per table, in a `表结构` header fact, and
    kept verbatim in `TableFact.table_caption`. It used to be spliced into every
    row's `statement`, which repeated one sentence per row: measured on two real
    runs that was 54.7% (NAFNet) and 69.1% (Agentopia) of all table-fact tokens,
    with the same caption repeated up to 150 times. Since the retrieval layer
    selects facts into a fixed token budget, that redundancy directly displaced
    other evidence -- the 2200-token catalog filled at 99.6% capacity while 91% of
    available facts were left out.
    """
    page_texts = section.page_texts or [PageMarkdown(page=section.page_start, text=section.text)]
    facts: list[TableFact] = []
    for page_text in page_texts:
        for table, headers, rows, source_rows in _table_rows(page_text.text):
            width = max(len(headers), max((len(row) for row in rows), default=0))
            headers = [*headers, *(f"column_{i + 1}" for i in range(len(headers), width))]
            numeric_columns = {
                column
                for column in range(width)
                if any(column < len(row) and _parse_numeric_cell(row[column]) for row in rows)
            }
            if not numeric_columns:
                # A textual table carries no metric values; a header fact alone
                # would add noise without evidence.
                continue
            # Prefer the bare "Table N" label. A caption without a colon used to be
            # swallowed whole, which put the caption back into every row's `table`
            # field and undid the de-duplication this function exists to do.
            label = re.match(r"(?i)\s*((?:table|\u8868)\s*\d+)\s*[:\uff1a]?", table)
            table_id = label.group(1).strip() if label else table.strip()[:120]
            table_id = table_id or table.strip()
            column_names = [_clean_table_cell(name) for name in headers]
            caption = table.strip()
            table_facts: list[TableFact] = []
            carried: dict[int, str] = {}
            for row, source_quote in zip(rows, source_rows):
                padded = [*row, *("" for _ in range(width - len(row)))]
                cleaned = [_clean_table_cell(cell) for cell in padded]
                for column, cell in enumerate(cleaned):
                    if column in numeric_columns:
                        continue
                    if cell:
                        carried[column] = cell
                    elif column in carried:
                        cleaned[column] = carried[column]
                row_context = {
                    headers[column]: cleaned[column]
                    for column in range(width)
                    if column not in numeric_columns and cleaned[column]
                }
                for column in numeric_columns:
                    parsed = _parse_numeric_cell(padded[column])
                    if not parsed:
                        continue
                    value, display_value, cell_unit = parsed
                    metric = headers[column]
                    # Config detection still sees the full caption text, so marker
                    # matching is unchanged.
                    if _row_is_config_row(table, metric, row_context):
                        # A config/parameter value is not an evaluation metric;
                        # keep it out of the fact DB entirely (the retrieval
                        # layer used to sink it after the fact).
                        continue
                    unit = "%" if "%" in metric or cell_unit == "%" else cell_unit
                    parts = [f"{key}={value}" for key, value in row_context.items()]
                    parts.append(f"{metric}={display_value}{unit}")
                    # The caption is NOT repeated here; it lives in the per-table
                    # header fact below, which is what makes this compact.
                    statement = "; ".join(parts)
                    raw_id = json.dumps(
                        [section.id, page_text.page, table, row_context, metric, display_value],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    table_facts.append(
                        TableFact(
                            fact_id=f"fact-{hashlib.sha1(raw_id.encode('utf-8')).hexdigest()[:16]}",
                            section_id=section.id,
                            table=table_id,
                            table_caption=caption,
                            row_context=row_context,
                            metric=metric,
                            value=value,
                            display_value=display_value,
                            unit=unit,
                            statement=statement,
                            source_quote=source_quote,
                            page=page_text.page,
                            concept=metric,
                            bold="**" in padded[column],
                        )
                    )
            if not table_facts:
                # Every row was a config row (or unparseable): emit nothing at all,
                # including no header, so a config table stays absent from the DB.
                continue
            header_raw = json.dumps(
                [section.id, page_text.page, "header", table_id], ensure_ascii=False, sort_keys=True
            )
            facts.append(
                TableFact(
                    fact_id=f"fact-{hashlib.sha1(header_raw.encode('utf-8')).hexdigest()[:16]}",
                    section_id=section.id,
                    table=table_id,
                    table_caption=caption,
                    row_context={},
                    metric="",
                    value=0.0,
                    display_value="",
                    statement=(
                        f"{_TABLE_HEADER_CONCEPT}：{_prefix_for_budget(caption, 200)}；"
                        f"列名：{'、'.join(name for name in column_names if name)}"
                    ),
                    source_quote=caption[:400],
                    page=page_text.page,
                    concept=_TABLE_HEADER_CONCEPT,
                    confidence=1.0,
                )
            )
            facts.extend(table_facts)
    return list({fact.fact_id: fact for fact in facts}.values())


# --- Prose metric recovery -------------------------------------------------
# `kind="metric"` used to be produced only by `extract_table_facts`, which needs
# pymupdf4llm to emit markdown pipe tables. A PDF that yields no tables therefore
# had an empty metric pool, and "使用了哪些评价指标？" degraded into citing an
# unrelated definition fact: a sentence that IS grounded in the source (so the
# claim auditor passes it) but does not answer the question.
# `extract_metric_facts` recovers evaluation metrics straight from prose so the
# metric route always has relevant evidence.
#
# Two risk classes, because metric names collide with ordinary words and with
# other technical terms. Measured on a real 75-page survey:
#   * Tier A names are unambiguous ("bleu", "accuracy", "召回率") -- a mention is
#     enough, modulo the evidence gate below.
#   * Tier B names collide with homonyms or acronyms ("mae" = masked autoencoder,
#     "iou"/"map" used about geometry, "mse" inside other identifiers) -- these
#     additionally require explicit numeric evidence.
#   * Excluded entirely: "is" (264 hits, the English copula), "oracle", "acc",
#     "cer", "der", "kappa", "ap50" -- pure false-positive generators.
_METRIC_TERMS_UNAMBIGUOUS: tuple[str, ...] = (
    "accuracy", "balanced accuracy", "precision", "recall",
    "f1-score", "f1 score", "micro-f1", "macro-f1", "f1",
    "mcc", "matthews correlation", "auroc", "auprc", "top-1 accuracy",
    "top-5 accuracy", "bleu", "rouge", "meteor", "cider", "spice", "bertscore",
    "perplexity", "ndcg", "mrr", "mean average precision", "success rate",
    "psnr", "ssim", "lpips", "clip score", "rmse", "miou",
    "jaccard", "dice coefficient", "wer", "word error rate",
    "likert", "mos", "win rate", "human evaluation", "human rating",
    # Chinese counterparts (CJK matching ignores punctuation/whitespace)
    "准确率", "精确率", "精确度", "召回率", "查准率", "查全率", "错误率",
    "f1值", "f1分数", "宏平均", "微平均", "混淆矩阵", "平均精度",
    "皮尔逊", "相关系数", "斯皮尔曼", "决定系数", "均方误差", "平均绝对误差",
    "交并比", "成功率", "导航误差", "困惑度", "评价指标", "评估指标",
)

_METRIC_TERMS_AMBIGUOUS: tuple[str, ...] = (
    "mae", "mse", "iou", "map", "map@50", "fid", "spl",
)

# Compiling the full alternation once keeps `_metric_terms_in` linear in the
# number of terms without re-scanning the sentence per term.
_METRIC_TERM_RE = re.compile(
    "|".join(
        re.escape(term)
        for term in sorted(
            (*_METRIC_TERMS_UNAMBIGUOUS, *_METRIC_TERMS_AMBIGUOUS),
            key=len,
            reverse=True,
        )
    ),
    re.IGNORECASE,
)
_AMBIGUOUS_KEYS = {term.lower() for term in _METRIC_TERMS_AMBIGUOUS}

# Evidence gate. A metric mention alone is not evidence that the document *used*
# the metric, so a stronger signal must accompany it:
#   * a percentage anywhere in the sentence ("10% on SPL") -- the strongest and
#     most common reporting signal;
#   * "<metric> of <number>" / "<metric> = <number>" ("F1 of 0.82");
#   * the metric appearing in an enumerated metric list ("metrics include
#     accuracy, precision, and F1").
# A bare number is deliberately NOT sufficient: "12-layer ViT ... initialized
# from MAE (He et al., 2022)" would otherwise enter the metric pool. A bare
# evaluation verb is NOT sufficient either: "to enhance the accuracy of video
# descriptions" is ordinary prose about accuracy, not a reported metric.
_PERCENT_EVIDENCE_RE = re.compile(r"\d\s*%")
_METRIC_VALUE_RE = re.compile(
    r"(?i)(?:of|=|:|达到|为)\s*[-+]?\d+(?:\.\d+)?\s*%?"
)
_METRIC_ENUM_RE = re.compile(
    r"(?i)(?:metric[s]?\s+(?:include|includes|are|were|used|such as)"
    r"|evaluat\w*\s+(?:with|using|by)"
    r"|measur\w*\s+(?:with|using|by)"
    r"|report\w*\s+(?:with|using|by)"
    r"|我们?(?:采用|使用|选取)|评价指标(?:包括|为|采用)|评估指标(?:包括|为)"
    r"|指标包括)"
)

# Sentence boundaries: ASCII terminators always break; a CJK terminator breaks
# only when it is not a decimal point between digits.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[!?])\s+|(?<=\.)\s+|(?<=[。！？；;])(?!\d)"
)


def _metric_terms_in(text: str) -> list[str]:
    """Evaluation-metric terms mentioned in `text`, longest first (so
    "f1-score" wins over "f1"), deduplicated case-insensitively."""
    found: dict[str, str] = {}
    for match in _METRIC_TERM_RE.finditer(text):
        term = match.group(0)
        key = term.lower()
        if term[0].isascii() and term[-1].isascii():
            # Whole-word match only: "recall" must not fire on "recalled",
            # "f1" must not fire on "f10".
            before = text[match.start() - 1] if match.start() else ""
            after = text[match.end()] if match.end() < len(text) else ""
            if (before.isalnum() and before.isascii()) or (
                after.isalnum() and after.isascii()
            ):
                continue
        found.setdefault(key, term)
    return [term for _key, term in sorted(found.items(), key=lambda kv: (-len(kv[0]), kv[0]))]


def _metric_evidence_ok(sentence: str) -> bool:
    """True when the sentence reports or explicitly enumerates evaluation metrics."""
    if _PERCENT_EVIDENCE_RE.search(sentence):
        return True
    if _METRIC_VALUE_RE.search(sentence):
        return True
    return bool(_METRIC_ENUM_RE.search(sentence))


def _metric_statement(sentence: str, terms: list[str]) -> str:
    """Chinese restatement of a metric-bearing source sentence. Numbers are kept
    verbatim so a quantitative answer stays auditable against the source."""
    numbers = re.findall(r"\d+(?:[.,]\d+)?\s*%?", sentence)
    named = "、".join(terms[:5])
    if numbers:
        return f"原文使用了评价指标 {named}，并把数值结果记为 {'、'.join(numbers[:6])}。"
    return f"原文使用了评价指标 {named}，作为该项工作的评估口径。"


# Some evaluation sentences report a benchmark comparison without naming a
# metric ("outperforms others by 6% on the 2022 OK-VQA open-domain multimodal
# task"). Those are still exactly the quantitative evidence the metric question
# needs, and the gate below is narrow enough not to admit unrelated percentages
# such as "(6% of pretraining data)".
_PERFORMANCE_CUE_RE = re.compile(
    r"(?i)\b(outperform\w*|superior|better than|improve[sd]?|gain(?:s|ed)?|"
    r"exceed\w*|higher than|lower than|reduc\w*|achiev\w*|beat(?:s|en)?)\b"
    r"|优于|超过|提升|改进|降低|领先"
)
_BENCHMARK_CONTEXT_RE = re.compile(
    r"(?i)\b(benchmark|dataset|task|leaderboard|baseline|state[- ]of[- ]the[- ]art|"
    r"sota)\b|基准|数据集|评测|任务|排行榜"
)
# Concept label for benchmark-comparison facts that carry a number but name no
# metric. Kept as a module constant so the producer and the statement builder
# cannot drift apart.
_PERFORMANCE_CONCEPT = "性能对比"


def _metric_performance_claim(sentence: str) -> bool:
    """True when a sentence makes a quantified performance claim against a task
    or benchmark, even without naming the metric."""
    return bool(
        _PERCENT_EVIDENCE_RE.search(sentence)
        and _PERFORMANCE_CUE_RE.search(sentence)
        and _BENCHMARK_CONTEXT_RE.search(sentence)
    )


def _performance_statement(sentence: str) -> str:
    numbers = re.findall(r"\d+(?:[.,]\d+)?\s*%?", sentence)
    if numbers:
        return f"原文给出了量化的性能对比结论，数值为 {'、'.join(numbers[:6])}。"
    return "原文给出了量化的性能对比结论。"


def _metric_fact_from_sentence(
    sentence: str,
    terms: list[str],
    section: "HeadingNode",
    page_texts: list[PageMarkdown],
    confidence: float,
) -> Fact | None:
    text = section.text or ""
    if not terms:
        return None
    quote = sentence[:400]
    if not _quote_is_grounded(quote, text):
        return None
    statement = (
        _performance_statement(sentence)
        if terms == [_PERFORMANCE_CONCEPT]
        else _metric_statement(sentence, terms)
    )
    fact = Fact(
        concept=terms[0],
        statement=statement,
        kind="metric",
        source_quote=quote,
        page=section.page_start,
        confidence=confidence,
    )
    fact.grounded = True
    fact.page = _quote_page(quote, page_texts) or section.page_start
    fact.fact_id = _stable_fact_id(section.id, fact)
    return fact


def extract_metric_facts(
    section: "HeadingNode",
    max_facts: int = 8,
) -> list[Fact]:
    """Deterministic `kind="metric"` facts recovered from a section's prose.

    Each fact quotes a real sentence from the section, so `_quote_is_grounded`
    accepts it and `_quote_page` can attribute a page. At most one fact per
    distinct metric term per section, which keeps the metric pool small enough
    that it cannot crowd out the other kinds during retrieval.
    """
    text = section.text or ""
    if not text.strip():
        return []
    page_texts = section.page_texts or [PageMarkdown(page=section.page_start, text=text)]
    facts: list[Fact] = []
    seen_terms: set[str] = set()
    claimed_performance = False
    for sentence in (raw.strip() for raw in _SENTENCE_SPLIT_RE.split(text)):
        if not sentence:
            continue
        terms = [term for term in _metric_terms_in(sentence) if term.lower() not in seen_terms]
        if terms:
            if not _metric_evidence_ok(sentence):
                continue
            # Tier B names colliding with acronyms/homonyms need a percentage or an
            # explicit "metric of N" construction to count as real evidence.
            if all(term.lower() in _AMBIGUOUS_KEYS for term in terms) and not (
                _PERCENT_EVIDENCE_RE.search(sentence)
                or _METRIC_VALUE_RE.search(sentence)
            ):
                continue
            fact = _metric_fact_from_sentence(
                sentence, terms, section, page_texts, confidence=0.75
            )
        elif not claimed_performance and _metric_performance_claim(sentence):
            # One benchmark-comparison fact per section is enough; without the
            # cap this channel would flood the metric pool on results-heavy papers.
            claimed_performance = True
            fact = _metric_fact_from_sentence(
                sentence, [_PERFORMANCE_CONCEPT], section, page_texts, confidence=0.7
            )
        else:
            continue
        if fact is None:
            continue
        facts.append(fact)
        seen_terms.update(term.lower() for term in terms)
        if len(facts) >= max_facts:
            break
    return facts


def _entity_keys(value: str) -> set[str]:
    normalized = _normalize_concept_text(value)
    words = re.findall(r"[A-Za-z]+", value)
    keys = {normalized}
    if len(words) >= 2:
        keys.add("".join(word[0] for word in words).lower())
    aliases = {
        "ann": {"ann", "artificialneuralnetwork", "artifcialneuralnetwork", "人工神经网络"},
        "svm": {"svm", "supportvectormachine", "支持向量机"},
        "knn": {"knn", "knearestneighbors", "k最近邻", "最近邻"},
    }
    for values in aliases.values():
        if keys & values:
            keys.update(values)
    return {key for key in keys if key}


def _context_value(context: dict[str, str], patterns: tuple[str, ...]) -> str | None:
    for key, value in context.items():
        normalized = _normalize_concept_text(key)
        if any(pattern in normalized for pattern in patterns):
            return value
    return None


def _category_value(context: dict[str, str]) -> str | None:
    for key, value in context.items():
        normalized = _normalize_concept_text(key)
        if "model" in normalized or "模型" in normalized or "classifier" in normalized:
            continue
        if "audioclass" in normalized or "class" in normalized or "类别" in normalized:
            return value
    return None


def detect_numeric_contradictions(
    section: HeadingNode, table_facts: list[TableFact]
) -> list[DetectedContradiction]:
    accuracy_facts = [
        fact
        for fact in table_facts
        if "accuracy" in _normalize_concept_text(fact.metric) or "准确率" in fact.metric
    ]
    if not accuracy_facts:
        return []
    sentence_break = re.compile(
        r"(?<=[。！？!?])|(?<!\d)(?<!\bfig)(?<!\bfigs)\.(?!\d)(?:\s+|$)|\n+",
        re.IGNORECASE,
    )
    sentences = [
        item.strip()
        for item in sentence_break.split(section.text)
        if item.strip()
    ]
    claims = [
        sentence
        for sentence in sentences
        if re.search(r"maximum|highest|最高|最大", sentence, re.IGNORECASE)
        and re.search(r"all.{0,40}class|所有.{0,20}(?:类别|类)", sentence, re.IGNORECASE)
    ]
    contradictions = []
    for claim in claims:
        claim_normalized = _normalize_concept_text(claim)
        models = {
            model
            for fact in accuracy_facts
            if (model := _context_value(fact.row_context, ("classifiermodel", "model", "分类器模型", "模型")))
        }
        claimed_model = next(
            (model for model in models if any(key in claim_normalized for key in _entity_keys(model))),
            None,
        )
        if not claimed_model:
            continue
        claim_page = _quote_page(claim, section.page_texts) or section.page_start
        claim_fact = Fact(
            concept=claimed_model,
            statement=claim,
            kind="result",
            source_quote=claim,
            page=claim_page,
            confidence=1.0,
            grounded=True,
        )
        claim_fact.fact_id = _stable_fact_id(section.id, claim_fact)
        grouped: dict[tuple[str, str], list[tuple[str, TableFact]]] = {}
        for fact in accuracy_facts:
            model = _context_value(fact.row_context, ("classifiermodel", "model", "分类器模型", "模型"))
            category = _category_value(fact.row_context)
            if model and category:
                grouped.setdefault((_normalize_concept_text(fact.table), _normalize_concept_text(category)), []).append((model, fact))
        for (_, _), candidates in grouped.items():
            claimed = next(
                (fact for model, fact in candidates if _entity_keys(model) & _entity_keys(claimed_model)),
                None,
            )
            winner_model, winner = max(candidates, key=lambda item: item[1].value)
            if not claimed or winner.value <= claimed.value + 1e-9:
                continue
            category = _category_value(winner.row_context) or "该类别"
            statement = (
                f"原文声称“{claim}”，但 {winner.table} 中 {category} 的 {winner.metric} 显示 "
                f"{claimed_model}={claimed.display_value}{claimed.unit}，低于 "
                f"{winner_model}={winner.display_value}{winner.unit}。"
            )
            contradictions.append(
                DetectedContradiction(
                    statement=statement,
                    source_fact_ids=[claim_fact.fact_id, claimed.fact_id, winner.fact_id],
                    claim_fact=claim_fact,
                )
            )
            break
    deduplicated: dict[tuple[str, ...], DetectedContradiction] = {}
    for contradiction in contradictions:
        key = tuple(contradiction.source_fact_ids[1:])
        current = deduplicated.get(key)
        if current is None or len(contradiction.statement) < len(current.statement):
            deduplicated[key] = contradiction
    return list(deduplicated.values())


def _normalize_metric_key(value: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", str(value).lower())


def detect_document_contradictions(section_results: list[dict]) -> list[dict]:
    """Deterministically find numeric table conflicts that span the document.

    Every numeric table cell is grouped by (normalized metric, normalized row
    context, normalized table caption). A group whose value differs across
    pages is a genuine contradiction — the same metric+context reported with
    two different numbers in two places of the same table. Generic config
    columns (Value/Setting/…) and correlation tables are skipped, because both
    legitimately repeat one label against many rows/targets.

    The table caption is part of the group key so that structurally identical
    tables describing different experiments (e.g. per-world z-scores in
    different simulation worlds) never look like contradictions — only the same
    table caption on different pages can conflict.
    """
    rows: list[tuple[dict, str, str, int]] = []
    for section_result in section_results:
        for fact in section_result.get("table_facts", []):
            metric = _normalize_metric_key(fact.get("metric", ""))
            if not metric or metric in _GENERIC_METRIC_COLUMNS:
                continue
            table = str(fact.get("table", ""))
            if _CORRELATION_TABLE_RE.search(table):
                continue
            rc = fact.get("row_context", {}) or {}
            rc_key = "|".join(
                f"{_normalize_metric_key(k)}={_normalize_metric_key(v)}"
                for k, v in sorted(rc.items())
            )
            rows.append(
                (
                    fact,
                    f"{metric}||{rc_key}||{_normalize_metric_key(table)}",
                    table,
                    int(fact.get("page", 0) or 0),
                )
            )

    groups: dict[str, list[tuple[dict, str, int]]] = {}
    for fact, key, table, page in rows:
        groups.setdefault(key, []).append((fact, table, page))

    contradictions: list[dict] = []
    for key, items in groups.items():
        distinct: dict[float, tuple[dict, str, int]] = {}
        for fact, table, page in items:
            value = round(float(fact.get("value", 0.0)), 2)
            if value not in distinct:
                distinct[value] = (fact, table, page)
        if len(distinct) < 2:
            continue
        low_v = min(distinct)
        high_v = max(distinct)
        low_f, low_t, low_p = distinct[low_v]
        high_f, high_t, high_p = distinct[high_v]
        if low_p == high_p and _normalize_metric_key(low_t) == _normalize_metric_key(high_t):
            # Same page AND same table: header dedup should have prevented
            # this; an unresolved extraction collision, not a contradiction.
            continue
        context = "; ".join(
            f"{k}={v}" for k, v in (low_f.get("row_context") or {}).items()
        )
        metric = low_f.get("metric", "")
        statement = (
            f"同一指标“{metric}”"
            f"{('（' + context + '）') if context else ''}"
            f"在不同位置出现冲突数值：{low_f.get('display_value', low_v)}"
            f"{low_f.get('unit', '')}（第{low_p}页，{low_t}）与 "
            f"{high_f.get('display_value', high_v)}{high_f.get('unit', '')}"
            f"（第{high_p}页，{high_t}）。"
        )
        contradictions.append(
            {
                "statement": statement,
                "source_fact_ids": [
                    str(low_f.get("fact_id", "")),
                    str(high_f.get("fact_id", "")),
                ],
            }
        )
    contradictions.sort(
        key=lambda item: (
            -len(item.get("source_fact_ids", [])),
            item.get("statement", ""),
        )
    )
    return contradictions


def _stable_fact_id(section_id: str, fact: dict | Fact) -> str:
    read = fact.get if isinstance(fact, dict) else lambda key, default="": getattr(fact, key, default)
    raw = "\n".join(
        str(read(key, "") or "")
        for key in ("page", "kind", "concept", "statement", "source_quote")
    )
    digest = hashlib.sha1(f"{section_id}\n{raw}".encode("utf-8")).hexdigest()[:16]
    return f"fact-{digest}"


def _assign_fact_ids(result: ChunkExtraction, section_id: str) -> ChunkExtraction:
    for fact in result.facts:
        fact.fact_id = _stable_fact_id(section_id, fact)
    return result


def _grounded_payload(extractions: list[ChunkExtraction]) -> list[dict]:
    return [
        {
            "summary": extraction.summary,
            "facts": [fact.model_dump() for fact in extraction.facts if fact.grounded],
        }
        for extraction in extractions
    ]


def _compact_evidence_fact(fact: dict, section_id: str) -> dict:
    return {
        "fact_id": fact["fact_id"],
        "section_id": section_id,
        "concept": _prefix_for_budget(str(fact.get("concept", "")), 80),
        "statement": _prefix_for_budget(str(fact.get("statement", "")), 320),
        "kind": fact.get("kind", "relation"),
        "source_quote": _prefix_for_budget(str(fact.get("source_quote", "")), 420),
        "page": fact.get("page"),
        "confidence": float(fact.get("confidence", 0.0)),
        "grounded": True,
    }


def _metric_facts_from_result(result: dict) -> list[Fact]:
    """Metric facts for a section result, computed from the stored section text.

    `process_section` already persists `metric_facts`, but a result dict rebuilt
    from an older output file (a resumed or hand-edited run) may not carry them,
    so the section node is reconstructed and the deterministic extractor re-run.
    Recomputation is safe because it is pure and keyed on the section id.
    """
    stored = result.get("metric_facts")
    if stored is not None:
        return [Fact.model_validate(item) for item in stored]
    try:
        node = HeadingNode(**result["section"])
    except (KeyError, TypeError, ValidationError):
        return []
    return extract_metric_facts(node)


def build_section_evidence(
    result: dict,
    required_table_fact_ids: set[str] = frozenset(),
    metric_facts: list[Fact] | None = None,
) -> dict:
    section = result["section"]
    facts = []
    for extraction in result["extractions"]:
        for fact in extraction["facts"]:
            if not fact.get("grounded"):
                continue
            enriched = dict(fact)
            enriched["fact_id"] = enriched.get("fact_id") or _stable_fact_id(section["id"], enriched)
            enriched["section_id"] = section["id"]
            facts.append(_compact_evidence_fact(enriched, section["id"]))
    for fact in (
        metric_facts if metric_facts is not None else _metric_facts_from_result(result)
    ):
        if fact.grounded and fact.fact_id:
            facts.append(_compact_evidence_fact(fact.model_dump(), section["id"]))
    table_facts = result.get("table_facts", [])
    salient_table_ids = {fact["fact_id"] for fact in table_facts if fact.get("bold")}
    salient_table_ids.update(required_table_fact_ids)
    salient_table_ids.update(
        fact_id
        for contradiction in result.get("numeric_contradictions", [])
        for fact_id in contradiction.get("source_fact_ids", [])
    )
    extrema: dict[tuple[str, str], list[dict]] = {}
    for fact in table_facts:
        extrema.setdefault((fact.get("table", ""), fact.get("metric", "")), []).append(fact)
    for group in extrema.values():
        if group:
            salient_table_ids.add(max(group, key=lambda item: float(item.get("value", 0)))["fact_id"])
            salient_table_ids.add(min(group, key=lambda item: float(item.get("value", 0)))["fact_id"])
    selected_table_facts = [fact for fact in table_facts if fact.get("fact_id") in salient_table_ids]
    for fact in [*selected_table_facts, *result.get("detected_facts", [])]:
        if fact.get("grounded") and fact.get("fact_id"):
            facts.append(_compact_evidence_fact(fact, section["id"]))
    facts = list({fact["fact_id"]: fact for fact in facts}.values())
    return {
        "section_id": section["id"],
        "title": section["title"],
        "pages": [section["page_start"], section["page_end"]],
        "summary": result["synthesis"]["summary"],
        "core_concepts": result["synthesis"]["core_concepts"],
        "key_findings": result["synthesis"]["key_findings"],
        "facts": facts,
    }


def _build_fact_index(evidence: list[dict]) -> dict[str, dict]:
    return {
        fact["fact_id"]: {**fact, "section_title": section.get("title", "")}
        for section in evidence
        for fact in section.get("facts", [])
        if fact.get("fact_id")
    }


# --- Verifiable-extraction metrics -----------------------------------------
# Provenance metrics for the understanding layer, computed from a finished run so
# quality is measurable and regressible instead of eyeballed. Every quantity here
# is deterministic: no model call, no judgement, reproducible run to run.
#
# Scope note: `unsupported_claim_rate` is the sentence-level claim audit. It
# measures grounding (is each sentence backed by a cited fact), NOT relevance (does
# the answer address the question). An answer can be fully grounded and still not
# answer its question -- that failure mode needs a benchmark, and is deliberately
# out of scope here.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。．！？!?；;])\s*|(?<=\.)\s+")


def _answer_sentences(text: str) -> list[str]:
    """Sentences of an answer, ignoring the bare insufficient-evidence marker."""
    stripped = (text or "").strip()
    if not stripped or stripped == "证据不足":
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]


def _supported_sentence_count(unsupported: list[str], surviving: int) -> int:
    """Sentences retained in the answer, floored at zero.

    The audit DELETES ungrounded sentences from `answer` and records them in
    `unsupported_claims`, so the sentences still present are the supported ones.
    The floor guards the case where `_apply_answer_provenance` appends a
    whole-answer entry to that list.
    """
    return max(0, surviving)


def _evidence_supports(fact: dict, num_cited: int) -> bool:
    """True when the cited fact carries enough evidence to count as support.

    One citing fact is enough for a single-sentence answer, but a multi-sentence
    answer needs either the same number of facts or at least one fact rich enough
    to back more than one claim (a long statement or a long verbatim quote).
    """
    if num_cited <= 1:
        return True
    if num_cited >= 2:
        return True
    statement = str(fact.get("statement", ""))
    quote = str(fact.get("source_quote", ""))
    return len(statement) >= 40 or len(quote) >= 80


def _answer_metrics(answer: dict, fact_index: dict[str, dict]) -> dict:
    """Deterministic provenance metrics for one global answer.

    Counting note: `_audit_answer_claims` MOVES ungrounded sentences out of `answer`
    into `unsupported_claims` and keeps only the grounded ones in `answer`. So the
    model's total output is `surviving + unsupported`, which is the only correct
    denominator: dividing unsupported by the surviving count alone yields rates
    above 1.0 (measured 2.50 on one answer) and makes short answers look worse than
    long ones purely because they were pruned harder.
    """
    answer_text = str(answer.get("answer", ""))
    unsupported = [str(item) for item in (answer.get("unsupported_claims") or [])]
    surviving = _answer_sentences(answer_text)
    surviving_count = len(surviving)
    model_sentence_count = surviving_count + len(unsupported)

    cited_ids = [str(entry) for entry in (answer.get("source_fact_ids") or []) if entry]
    resolved = [fact_id for fact_id in cited_ids if fact_id in fact_index]
    # `_apply_answer_provenance` already drops ids missing from the index, so this
    # is near 1.0 by construction in a normal run. It is reported anyway: a drop
    # means the output was assembled outside the normal pipeline.
    resolvable = (len(resolved) / len(cited_ids)) if cited_ids else 0.0

    support = sum(
        1
        for fact_id in resolved
        if _evidence_supports(fact_index[fact_id], model_sentence_count)
    )
    pages = {int(page) for page in (answer.get("source_pages") or [])}

    return {
        "question": str(answer.get("question", "")),
        "cited_fact_count": len(cited_ids),
        "resolvable_fact_count": len(resolved),
        "citation_resolvable_rate": round(resolvable, 4),
        "unsupported_claim_count": len(unsupported),
        # Sentences still in `answer` (i.e. grounded) and the model's total output.
        "supported_sentence_count": surviving_count,
        "model_sentence_count": model_sentence_count,
        "unsupported_claim_rate": (
            round(len(unsupported) / model_sentence_count, 4)
            if model_sentence_count
            else 0.0
        ),
        "sufficient_evidence": bool(cited_ids) and support > 0,
        "distinct_source_pages": len(pages),
    }


# --- Backcheck of audited-unsupported sentences ----------------------------
# `_audit_answer_claims` can only test a sentence against the facts that were
# actually cited for that question. A sentence that IS grounded in the document
# but whose fact the retrieval layer did not select is therefore reported as
# "unsupported" -- measured on a real run, all four flagged sentences existed in
# the fact DB, so the flag was a retrieval-coverage gap, not a fabrication.
# The recheck re-tests each flagged sentence against the WHOLE fact index and
# separates the two cases. It is a metric only: the gating behaviour of the
# pipeline is unchanged.
_VERDICT_RETRIEVAL_GAP = "retrieval_gap"
_VERDICT_UNDECIDED = "undecided"

# Coverage threshold for "the DB carries this content". Measured on real and
# synthetic claims: a faithful paraphrase scores 0.76-0.90 and a claim stitching
# three facts together scores 0.46, while a claim naming something with a
# different word scores 0.00-0.17.
#
# Everything below the threshold is reported as `undecided`, NOT as fabricated.
# Vocabulary overlap cannot separate a synonym ("加速比" where every fact says
# "speedup") from an invented sentence -- both measure 0.00 -- so a fabrication
# verdict would be a guess dressed up as a finding. `undecided` is therefore an
# upper bound on fabrication: some of those claims are real content worded
# differently. Resolving them needs the source document, which the run output
# does not retain.
_FACT_DB_COVERAGE_THRESHOLD = 0.4
_FACT_DB_NUMBER_OVERLAP_THRESHOLD = 0.6


def _claim_fact_db_coverage(
    claim: str, fact_token_sets: list[tuple[set[str], set[str]]]
) -> tuple[bool, float]:
    """Return (numbers_agree, vocabulary_coverage) for `claim` against the DB.

    Numbers are exactly what a paraphrase step corrupts, so agreement on them is
    strong independent evidence that the content is real. Coverage accumulates
    across facts so a claim that legitimately combines several facts still scores
    well (best-single-fact overlap does not).
    """
    claim_tokens = _claim_tokens(claim)
    if not claim_tokens:
        return False, 0.0
    claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim))
    numbers_agree = False
    if claim_numbers:
        for _fact_tokens, fact_numbers in fact_token_sets:
            shared = claim_numbers & fact_numbers
            if shared and (
                len(shared) >= len(claim_numbers) * _FACT_DB_NUMBER_OVERLAP_THRESHOLD
            ):
                numbers_agree = True
                break
    covered: set[str] = set()
    for fact_tokens, _fact_numbers in fact_token_sets:
        covered |= claim_tokens & fact_tokens
    return numbers_agree, len(covered) / len(claim_tokens)


def _claim_matches_fact_db(
    claim: str, fact_token_sets: list[tuple[set[str], set[str]]]
) -> bool:
    """True when the fact DB plausibly carries `claim`."""
    numbers_agree, coverage = _claim_fact_db_coverage(claim, fact_token_sets)
    return numbers_agree or coverage >= _FACT_DB_COVERAGE_THRESHOLD


def _fact_token_sets(fact_index: dict[str, dict]) -> list[tuple[set[str], set[str]]]:
    """Precompute (tokens, numbers) per fact once, so the recheck stays cheap."""
    prepared = []
    for fact in fact_index.values():
        blob = " ".join(
            str(fact.get(key, "")) for key in ("concept", "statement", "source_quote")
        )
        prepared.append(
            (_claim_tokens(blob), set(re.findall(r"\d+(?:\.\d+)?", blob)))
        )
    return prepared


def recheck_unsupported_claims(
    unsupported: list[str], fact_token_sets: list[tuple[set[str], set[str]]]
) -> list[dict]:
    """Classify each audited-unsupported sentence against the whole fact DB.

    Two verdicts, because the check genuinely cannot always decide:
      * `retrieval_gap` -- the DB carries this content, so the pipeline had the
        evidence and retrieval did not surface it. A recall problem, not an
        accuracy one.
      * `undecided` -- the DB shows no clear sign of it. This is an UPPER BOUND on
        fabrication, not a finding: a sentence invented from nothing and a real
        sentence worded with a synonym both score the same, so neither verdict
        would be honest. Reported with its coverage so it can be triaged.
    """
    verdicts = []
    for claim in unsupported:
        text = str(claim).strip()
        if not text or not _claim_tokens(text):
            continue
        numbers_agree, coverage = _claim_fact_db_coverage(text, fact_token_sets)
        if numbers_agree or coverage >= _FACT_DB_COVERAGE_THRESHOLD:
            verdict = _VERDICT_RETRIEVAL_GAP
        else:
            verdict = _VERDICT_UNDECIDED
        verdicts.append({"claim": text, "verdict": verdict, "coverage": round(coverage, 3)})
    return verdicts


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return int(ordered[index])


def run_evidence_metrics(
    section_results: list[dict], document_synthesis: object = None
) -> dict:
    """Provenance metrics for a finished run.

    Accepts what `_run` already has in hand: the per-section results plus the
    document synthesis (a `DocumentSynthesis` model or its dumped dict), so this
    adds no new coupling and no extra model calls.
    """
    if hasattr(document_synthesis, "model_dump"):
        document_synthesis = document_synthesis.model_dump()
    synthesis = document_synthesis if isinstance(document_synthesis, dict) else {}
    answers = [
        answer
        for answer in (synthesis.get("global_answers") or [])
        if isinstance(answer, dict)
    ]

    # Facts reachable as evidence: every grounded fact the sections produced,
    # across all shapes the pipeline emits.
    fact_index: dict[str, dict] = {}
    for result in section_results:
        for extraction in result.get("extractions") or []:
            for fact in extraction.get("facts") or []:
                if fact.get("grounded") and fact.get("fact_id"):
                    fact_index[str(fact["fact_id"])] = fact
        for key in ("table_facts", "detected_facts", "metric_facts"):
            for fact in result.get(key) or []:
                if fact.get("fact_id") and fact.get("grounded", True):
                    fact_index[str(fact["fact_id"])] = fact

    per_answer = [_answer_metrics(answer, fact_index) for answer in answers]

    # Backcheck the audited-unsupported sentences against the whole fact DB, so a
    # retrieval gap is not reported as if it were a fabrication.
    fact_token_sets = _fact_token_sets(fact_index)
    for item, answer in zip(per_answer, answers):
        item["unsupported_claims"] = recheck_unsupported_claims(
            [str(entry) for entry in (answer.get("unsupported_claims") or [])],
            fact_token_sets,
        )

    # Quantiles span every answer, including zero-evidence ones: excluding them
    # would hide the very case the metric exists to catch (an answer that cited
    # nothing would vanish from min/median/max instead of pulling min to 0).
    counts = [item["cited_fact_count"] for item in per_answer]
    total_cited = sum(item["cited_fact_count"] for item in per_answer)
    total_resolved = sum(item["resolvable_fact_count"] for item in per_answer)
    total_unsupported = sum(item["unsupported_claim_count"] for item in per_answer)
    # The model's total sentence output, not just the sentences that survived the
    # audit. See `_answer_metrics` for why this denominator matters.
    total_sentences = sum(item["model_sentence_count"] for item in per_answer)
    retrieval_gaps = sum(
        1
        for item in per_answer
        for verdict in item["unsupported_claims"]
        if verdict["verdict"] == _VERDICT_RETRIEVAL_GAP
    )
    undecided = sum(
        1
        for item in per_answer
        for verdict in item["unsupported_claims"]
        if verdict["verdict"] == _VERDICT_UNDECIDED
    )
    # Numbers in the overview layer that no fact carries. Deterministic and
    # false-positive free by construction (see `audit_overview_numerics`).
    overview_findings = audit_overview_numerics(synthesis, fact_index)

    return {
        "fact_count": len(fact_index),
        "answer_count": len(per_answer),
        "answers_with_evidence": sum(
            1 for item in per_answer if item["cited_fact_count"]
        ),
        "answers_with_sufficient_evidence": sum(
            1 for item in per_answer if item["sufficient_evidence"]
        ),
        "citation_precision": (
            round(total_resolved / total_cited, 4) if total_cited else 0.0
        ),
        "unsupported_claim_rate": (
            round(total_unsupported / total_sentences, 4) if total_sentences else 0.0
        ),
        # Of the audited-unsupported sentences: how many the fact DB actually
        # carries (retrieval gap -- a recall problem), and how many it cannot
        # account for (`undecided`, an UPPER BOUND on fabrication: it also contains
        # real sentences worded with a synonym, which overlap cannot distinguish).
        "unsupported_retrieval_gaps": retrieval_gaps,
        "unsupported_undecided": undecided,
        # Overview-layer numbers with no supporting fact. Must stay 0; a non-zero
        # value means the synthesis layer invented or mis-stated a figure.
        "overview_unsupported_numbers": len(overview_findings),
        "overview_numeric_findings": overview_findings,
        "median_cited_facts": _quantile(counts, 0.5),
        "min_cited_facts": _quantile(counts, 0.0),
        "max_cited_facts": _quantile(counts, 1.0),
        "total_cited_facts": total_cited,
        "total_unsupported_claims": total_unsupported,
        "per_answer": per_answer,
    }


def _collect_source_fact_ids(items: list[dict]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        ids.update(str(value) for value in item.get("source_fact_ids", []) if value)
        ids.update(
            str(fact["fact_id"])
            for fact in item.get("facts", [])
            if fact.get("fact_id")
        )
    return ids


def fallback_group_synthesis(items: list[dict]) -> GroupSynthesis:
    def collect(*keys: str, limit: int = 12) -> list[str]:
        values = []
        for item in items:
            for key in keys:
                value = item.get(key, [])
                values.extend(str(entry) for entry in _ensure_list(value) if entry)
        return list(dict.fromkeys(values))[:limit]

    facts = [fact for item in items for fact in item.get("facts", []) if fact.get("fact_id")]
    summaries = collect("summary", limit=8)
    if not summaries:
        summaries = [str(fact.get("statement", "")) for fact in facts[:8] if fact.get("statement")]
    source_fact_ids = []
    for item in items:
        source_fact_ids.extend(str(value) for value in item.get("source_fact_ids", []) if value)
        source_fact_ids.extend(str(fact["fact_id"]) for fact in item.get("facts", []) if fact.get("fact_id"))
    numeric_claims = [
        str(fact.get("statement", ""))
        for fact in facts
        if re.search(r"\d", str(fact.get("statement", "")))
    ]
    concepts = [*collect("concepts", "core_concepts"), *[str(fact.get("concept", "")) for fact in facts]]
    findings = [*collect("findings", "key_findings"), *[str(fact.get("statement", "")) for fact in facts if fact.get("kind") == "result"]]
    return GroupSynthesis(
        summary=_prefix_for_budget("；".join(summaries), 240),
        concepts=list(dict.fromkeys(value for value in concepts if value))[:12],
        findings=list(dict.fromkeys(value for value in findings if value))[:12],
        numeric_claims=list(dict.fromkeys(value for value in numeric_claims if value))[:12],
        limitations=collect("limitations", limit=8),
        future_work=collect("future_work", limit=8),
        contradictions=collect("contradictions", limit=8),
        source_fact_ids=list(dict.fromkeys(source_fact_ids))[:32],
        source_sections=[],
        source_pages=[],
    )


def _build_document_summary(groups: list[GroupSynthesis], contradictions: list[dict]) -> str:
    summaries = [group.summary for group in groups if group.summary]
    findings = [value for group in groups for value in group.findings if value][:8]
    limitations = [value for group in groups for value in group.limitations if value][:4]
    future_work = [value for group in groups for value in group.future_work if value][:4]
    numeric_claims = [value for group in groups for value in group.numeric_claims if value][:4]

    parts: list[str] = []
    if summaries:
        parts.append("；".join(dict.fromkeys(summaries)))
    if findings:
        parts.append("关键结论包括：" + "；".join(dict.fromkeys(findings[:4])))
    if numeric_claims:
        parts.append("主要定量结果包括：" + "；".join(dict.fromkeys(numeric_claims[:3])))
    if limitations:
        parts.append("局限包括：" + "；".join(dict.fromkeys(limitations)))
    if future_work:
        parts.append("未来工作包括：" + "；".join(dict.fromkeys(future_work)))

    combined = "。".join(part for part in parts if part)
    return _prefix_for_budget(combined, 300)


def fallback_document_overview(
    groups: list[GroupSynthesis],
    contradictions: list[dict],
    title: str | None = None,
) -> DocumentOverview:
    findings = list(
        dict.fromkeys(
            [
                *[value for group in groups for value in group.numeric_claims if value],
                *[value for group in groups for value in group.findings if value],
            ]
        )
    )[:10]
    limitations = list(dict.fromkeys(value for group in groups for value in group.limitations))[:10]
    future_work = list(dict.fromkeys(value for group in groups for value in group.future_work))[:10]
    contradiction_text = list(
        dict.fromkeys(
            [
                *[item.get("statement", "") for item in contradictions],
                *[value for group in groups for value in group.contradictions],
            ]
        )
    )
    summaries = [group.summary for group in groups if group.summary]
    summary_text = _build_document_summary(groups, contradictions)
    if not summary_text and summaries:
        summary_text = _prefix_for_budget("；".join(summaries), 300)
    return DocumentOverview(
        title=title or "Document",
        executive_summary=summary_text,
        research_question="证据不足",
        methods=[],
        datasets=[],
        metrics=[],
        key_findings=findings,
        limitations=limitations,
        future_work=future_work,
        contradictions=[value for value in contradiction_text if value][:10],
    )


def _apply_group_provenance(
    result: GroupSynthesis, fact_index: dict[str, dict], allowed_fact_ids: set[str]
) -> GroupSynthesis:
    result.source_fact_ids = list(
        dict.fromkeys(
            fact_id
            for fact_id in result.source_fact_ids
            if fact_id in allowed_fact_ids and fact_id in fact_index
        )
    )
    if not result.source_fact_ids:
        result.source_fact_ids = [
            fact_id for fact_id in fact_index if fact_id in allowed_fact_ids
        ][:32]
    sources = [fact_index[fact_id] for fact_id in result.source_fact_ids]
    result.source_sections = sorted({str(source["section_id"]) for source in sources})
    result.source_pages = sorted(
        {int(source["page"]) for source in sources if source.get("page") is not None}
    )
    return result


def _apply_answer_provenance(
    answer: GlobalAnswer, fact_index: dict[str, dict], allowed_fact_ids: set[str]
) -> GlobalAnswer:
    answer.source_fact_ids = list(
        dict.fromkeys(
            fact_id
            for fact_id in answer.source_fact_ids
            if fact_id in allowed_fact_ids and fact_id in fact_index
        )
    )
    sources = [fact_index[fact_id] for fact_id in answer.source_fact_ids]
    answer.source_sections = sorted({str(source["section_id"]) for source in sources})
    answer.source_pages = sorted(
        {int(source["page"]) for source in sources if source.get("page") is not None}
    )
    if not sources:
        if answer.answer.strip() != "证据不足":
            answer.unsupported_claims = list(
                dict.fromkeys([*answer.unsupported_claims, answer.answer])
            )
            answer.answer = "证据不足"
        answer.confidence = 0.0
    else:
        evidence_confidence = sum(float(source.get("confidence", 0.0)) for source in sources) / len(sources)
        answer.confidence = min(answer.confidence, round(evidence_confidence, 3))
        evidence_text = "\n".join(
            str(source.get(key, ""))
            for source in sources
            for key in ("statement", "source_quote")
            if source.get(key)
        )
        if evidence_text:
            supported, unsupported = _audit_answer_claims(answer.answer, evidence_text)
            answer.unsupported_claims = unsupported
            if supported:
                answer.answer = "".join(supported)
            else:
                answer.answer = "证据不足"
                answer.source_fact_ids = []
                answer.source_sections = []
                answer.source_pages = []
                answer.confidence = 0.0
    _strip_english_leak(answer)
    return answer


def _claim_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    ascii_tokens = set(re.findall(r"[a-z0-9]+(?:\.\d+)?%?", normalized))
    cjk = "".join(CJK_RE.findall(normalized))
    return ascii_tokens | {cjk[index : index + 2] for index in range(max(0, len(cjk) - 1))}


# A run of 4+ ASCII words (each 3+ letters) is verbatim English prose, not a short
# metric/proper-noun token. Hyphenated or digit-bearing names break the pattern, so
# "CoSER Test", "Tulu 3", "long-horizon" never match; "life reward training" (3
# words) stays too.
_ENGLISH_PROSE_RUN = re.compile(r"[A-Za-z]{3,}(?: [A-Za-z]{3,}){3,}")


def _apply_document_title(result: DocumentOverview, candidate: str) -> DocumentOverview:
    """Deterministic backstop for the overview title.

    The overview prompt tells the model to keep the original document title
    verbatim, but a flash model can still mangle or invent one. If the returned
    title shares almost no tokens with the candidate (e.g. an invented
    "多模态代理论坛" next to an English candidate), replace it with the
    candidate -- but only when the candidate is actually consistent with the
    content the model summarized. That consistency check guards against the
    heading tree promoting an author byline or journal name to a heading: the
    model would echo it faithfully (high overlap), and we must not force a wrong
    title. A verbatim or same-language variant shares enough tokens and is kept;
    a cross-language translation of an English candidate scores ~0.2 and is
    overridden back to the original -- the intended behavior for a title.
    """
    if not candidate:
        return result
    answer_tokens = _claim_tokens(result.title)
    if not answer_tokens:
        result.title = candidate
        return result
    candidate_tokens = _claim_tokens(candidate)
    overlap = len(answer_tokens & candidate_tokens) / len(answer_tokens)
    coverage = len(answer_tokens & candidate_tokens) / len(candidate_tokens)
    # overlap 高但 coverage 低 => 答案是候选标题的前缀/片段（模型截断回显，
    # 如候选 "AGENT AI: Surveying..." 被回成 "AGENT AI:"），此时保留会得到一个
    # 残缺标题，必须用候选覆盖；cross-language 翻译 overlap ~0.2 由 overlap<0.5 分支处理。
    truncated = overlap >= 0.5 and coverage < 0.5
    if overlap < 0.5 or truncated:
        if candidate_tokens & _claim_tokens(
            f"{result.executive_summary} {result.research_question}"
        ):
            result.title = candidate
    return result


def _strip_english_leak(answer: GlobalAnswer) -> GlobalAnswer:
    """Deterministic backstop against English source text leaking into answers.

    The prompts forbid pasting English statements/quote into a Chinese answer, but
    if the model copies a verbatim English run anyway, cut the answer at the start
    of the English block and move the leaked tail into unsupported_claims for
    review. A run of only 2-3 English words (a metric or proper noun like "CoSER
    Test" or "life reward training") is left untouched; a fully-English answer keeps
    its content (visible in unsupported_claims) rather than being silently deleted.
    """
    if not CJK_RE.search(answer.question):
        return answer
    if answer.answer.strip() in ("", "证据不足"):
        return answer
    text = answer.answer
    match = _ENGLISH_PROSE_RUN.search(text)
    if match is None:
        return answer
    start = match.start()
    # Extend the cut back to the start of the English block so no partial English
    # prefix (e.g. "Environment model data is ") is left in the answer.
    while start > 0:
        previous = text[start - 1]
        if previous.isascii() and (previous.isalpha() or previous.isspace()):
            start -= 1
        else:
            break
    leaked = text[start:].strip()
    kept = text[:start].rstrip("；;，, ").strip()
    if not kept or not CJK_RE.search(kept):
        # Nothing Chinese survived the cut (the whole answer was leaked English, or
        # only a stub like "Table 8:" remains). An English-only fragment is not a
        # usable Chinese answer, so fall back to "证据不足" and keep the leak visible.
        if kept and leaked:
            answer.unsupported_claims = list(
                dict.fromkeys([*answer.unsupported_claims, kept, leaked])
            )
        else:
            answer.unsupported_claims = list(
                dict.fromkeys([*answer.unsupported_claims, leaked])
            )
        answer.answer = "证据不足"
        answer.source_fact_ids = []
        answer.source_sections = []
        answer.source_pages = []
        answer.confidence = 0.0
        return answer
    answer.answer = kept
    answer.unsupported_claims = list(
        dict.fromkeys([*answer.unsupported_claims, leaked])
    )
    answer.confidence = round(answer.confidence * 0.5, 3)
    return answer


def _audit_answer_claims(answer: str, evidence: str) -> tuple[list[str], list[str]]:
    claims = [
        claim
        for claim in re.findall(r"[^。！？!?；;]+[。！？!?；;]?", answer)
        if claim.strip()
    ]
    if not claims:
        return [], []

    compact_evidence = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", unicodedata.normalize("NFKC", evidence).lower())
    evidence_tokens = _claim_tokens(evidence)
    supported = []
    unsupported = []
    for claim in claims:
        compact_claim = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", unicodedata.normalize("NFKC", claim).lower())
        claim_tokens = _claim_tokens(claim)
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        direct = len(compact_claim) >= 6 and compact_claim in compact_evidence
        coverage = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
        number_match = not claim_numbers or claim_numbers <= evidence_numbers
        partial = coverage >= 0.35 and number_match and len(claim_tokens & evidence_tokens) >= 1
        is_supported = (direct or coverage >= 0.65 or partial) and number_match
        (supported if is_supported else unsupported).append(claim)

    # Mixed answers: keep every evidence-backed clause in the answer and report the
    # unsupported clauses for human review instead of discarding the whole answer.
    # Pure fabrication stays isolated from the returned answer and is visible in
    # unsupported_claims, so we no longer trade away real evidence for one bad clause.
    return supported, unsupported


_NO_CONTRADICTION_RE = re.compile(r"未发现|no contradiction", re.IGNORECASE)


def _match_contradiction_fact_ids(
    statement: str, fact_index: dict[str, dict], top: int = 4
) -> list[str]:
    """Match a contradiction statement to the facts that most plausibly support it.

    Numbers are the strongest signal (conflicting scores co-occur in the same
    sentence), so a statement carrying digits only considers facts that share at
    least one number; otherwise we fall back to token overlap.
    """
    statement_numbers = set(re.findall(r"\d+(?:\.\d+)?", statement))
    statement_tokens = _claim_tokens(statement)
    scored = []
    for fact_id, fact in fact_index.items():
        evidence = "\n".join(
            str(fact.get(key, ""))
            for key in ("concept", "statement", "source_quote")
            if fact.get(key)
        )
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        shared_numbers = statement_numbers & evidence_numbers
        if statement_numbers and not shared_numbers:
            continue
        overlap = len(statement_tokens & _claim_tokens(evidence))
        if not statement_numbers and overlap < 2:
            continue
        scored.append(
            (len(shared_numbers), overlap, float(fact.get("confidence", 0.0)), fact_id)
        )
    scored.sort(reverse=True)
    return [fact_id for *_, fact_id in scored[:top]]


def _build_contradiction_answer(
    question: str,
    contradictions: list[dict],
    fact_index: dict[str, dict],
) -> GlobalAnswer:
    """Build the global contradiction answer deterministically.

    Contradiction statements come from the deterministic numeric detector or the
    verified overview layer, so they are trusted as-is: the per-claim audit is
    bypassed here (it would reject hand-written multi-number sentences). Provenance
    is recovered from matched facts; statements without any supporting fact keep a
    0.5 confidence ceiling instead of being dropped to "证据不足".
    """
    statements = []
    source_fact_ids = []
    for item in contradictions:
        if not item.get("statement"):
            continue
        statement = item["statement"]
        if _NO_CONTRADICTION_RE.search(statement):
            continue
        statements.append(statement)
        fact_ids = [
            str(fact_id)
            for fact_id in item.get("source_fact_ids", [])
            if fact_id in fact_index
        ]
        if not fact_ids:
            fact_ids = _match_contradiction_fact_ids(statement, fact_index)
        source_fact_ids.extend(fact_ids)
    if not statements:
        return GlobalAnswer(
            question=question,
            answer="证据不足",
            source_fact_ids=[],
            source_sections=[],
            source_pages=[],
            confidence=0.0,
        )
    source_fact_ids = list(dict.fromkeys(source_fact_ids))
    sources = [fact_index[fact_id] for fact_id in source_fact_ids]
    if sources:
        confidence = sum(
            float(source.get("confidence", 0.0)) for source in sources
        ) / len(sources)
    else:
        confidence = 0.5
    return GlobalAnswer(
        question=question,
        answer="；".join(statements),
        source_fact_ids=source_fact_ids,
        source_sections=sorted(
            {str(source["section_id"]) for source in sources if source.get("section_id")}
        ),
        source_pages=sorted(
            {int(source["page"]) for source in sources if source.get("page") is not None}
        ),
        confidence=confidence,
    )


# --- Overview numeric attribution ------------------------------------------
# The overview layer concatenates facts into one prose paragraph, and in doing so
# can pair a number with the wrong subject. Measured on a real run: the executive
# summary said the model reached "BLEU 41.8 ... 训练十二小时", but 41.8 is the BIG
# model's score (trained 3.5 days) while 12 hours is the BASE model's budget -- both
# of which the extraction layer had recorded correctly as separate facts. Nothing
# upstream verified the pairing, so the conflation reached the output unchallenged.
#
# This audit applies the same standard the answer layer already uses: a number must
# be traceable to the evidence. It is deterministic and read-only.
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Units are spelled many ways in a Chinese restatement of an English source
# ("100,000 steps" -> "10万步", "3.5 days" -> "3.5天", "36M" -> "3600万"). Comparing
# bare digit strings flagged a correct "3600万" against a source "36M", so both
# sides are also compared as a scale-free mantissa.
_SCALE_SUFFIXES = {
    "k": 1e3, "thousand": 1e3, "千": 1e3,
    "m": 1e6, "million": 1e6, "万": 1e4, "百万": 1e6,
    "b": 1e9, "billion": 1e9, "亿": 1e8, "十亿": 1e9,
}


def _numeric_forms(value: str) -> set[str]:
    """Normalised numeric forms for comparison.

    A number written with a scale word yields ONLY its scaled absolute value, so
    "36M" and "3600万" both become "36000000". Emitting the bare mantissa as well
    would make "3600万" also claim the form "3600", which matches nothing and was
    reported as an unsupported number in real output.
    """
    forms: set[str] = set()
    for match in _NUMBER_TOKEN_RE.finditer(value):
        raw = match.group(0).replace(",", "")
        tail = value[match.end(): match.end() + 3].strip().lower()
        scaled = None
        try:
            number = float(raw)
        except ValueError:
            number = None
        if number is not None:
            for suffix, scale in _SCALE_SUFFIXES.items():
                if tail.startswith(suffix):
                    scaled = round(number * scale)
                    break
        if scaled is not None:
            forms.add(str(scaled))
            continue
        forms.add(raw)
        if number is not None and number.is_integer():
            forms.add(str(int(number)))
    return forms


def audit_overview_numerics(
    overview: object, fact_index: dict[str, dict]
) -> list[dict]:
    """Numbers in the overview that no fact in the evidence carries.

    This is the unambiguous half of the problem: a number appearing in the
    overview that exists nowhere in the evidence is either invented or an
    arithmetic slip. Deterministic, no false-positive risk from phrasing.

    What this deliberately does NOT attempt: detecting a number paired with the
    WRONG subject. A sentence such as "NAFNet reaches 33.69 dB on GoPro and
    40.30 dB on SIDD" legitimately carries several numbers from several facts, and
    a naive "no single fact holds them all" test flags correct sentences like that
    constantly (measured: 12 findings across 7 real runs, essentially all correct).
    Catching a true mispairing needs subject/entity comparison, which is not
    decidable from token overlap; the answer layer's evidence audit is the guard
    that applies there.
    """
    if hasattr(overview, "model_dump"):
        overview = overview.model_dump()
    if not isinstance(overview, dict):
        return []

    all_forms: set[str] = set()
    for fact in fact_index.values():
        all_forms |= _numeric_forms(" ".join(
            str(fact.get(key, "")) for key in ("concept", "statement", "source_quote")
        ))

    fields: list[tuple[str, str]] = []
    if overview.get("executive_summary"):
        fields.append(("executive_summary", str(overview["executive_summary"])))
    if overview.get("research_question"):
        fields.append(("research_question", str(overview["research_question"])))
    for key in ("methods", "datasets", "metrics", "key_findings",
                "limitations", "future_work", "contradictions"):
        for item in overview.get(key) or []:
            fields.append((key, str(item)))

    findings: list[dict] = []
    for key, text in fields:
        for sentence in _answer_sentences(text) or [text]:
            missing = {
                number
                for number in _numeric_forms(sentence)
                if number not in all_forms
            }
            if missing:
                findings.append({
                    "field": key,
                    "kind": "unsupported_number",
                    "numbers": sorted(missing),
                    "sentence": sentence[:200],
                })
    return findings


def _merge_overview_contradictions(
    detected: list[dict],
    overview_statements: list[str],
    fact_index: dict[str, dict],
) -> list[dict]:
    """Combine the deterministic numeric contradictions with the overview's findings.

    Overview statements are matched back to facts so they carry provenance; the
    no-contradiction sentinel is filtered out, and detected entries win dedup ties.
    """
    merged = list(detected)
    seen = {item.get("statement") for item in merged if item.get("statement")}
    for statement in overview_statements:
        statement = str(statement)
        if not statement or _NO_CONTRADICTION_RE.search(statement):
            continue
        if statement in seen:
            continue
        fact_ids = _match_contradiction_fact_ids(statement, fact_index)
        merged.append({"statement": statement, "source_fact_ids": fact_ids})
        seen.add(statement)
    return merged


def _recover_answer_fact_ids(
    answer: GlobalAnswer,
    candidate_facts: list[dict],
    fact_index: dict[str, dict],
    max_ids: int = 3,
) -> list[str]:
    if max_ids <= 0 or answer.answer.strip() == "证据不足":
        return []
    candidate_ids = {
        str(fact["fact_id"])
        for fact in candidate_facts
        if fact.get("fact_id") in fact_index
    }
    existing_ids = {
        fact_id
        for fact_id in answer.source_fact_ids
        if fact_id in candidate_ids
    }
    answer_tokens = _claim_tokens(answer.answer)
    answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer.answer))
    strict_number_question = any(
        term in answer.question for term in ("定量", "数字", "准确率", "错误率", "主要结果")
    )
    section_title = str(answer.question).lower()
    scored = []
    for position, compact_fact in enumerate(candidate_facts):
        fact_id = str(compact_fact.get("fact_id", ""))
        if fact_id not in candidate_ids:
            continue
        fact = fact_index[fact_id]
        evidence = "\n".join(
            str(fact.get(key, ""))
            for key in ("concept", "statement", "source_quote")
            if fact.get(key)
        )
        evidence_tokens = _claim_tokens(evidence)
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        matched_numbers = answer_numbers & evidence_numbers
        title = re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", str(fact.get("section_title", "")).lower())
        is_related_work = any(term in title for term in ("related work", "literature", "background", "相关工作", "文献综述", "背景"))
        if strict_number_question and answer_numbers and evidence_numbers and not matched_numbers:
            continue
        if strict_number_question and is_related_work:
            continue
        overlap = len(answer_tokens & evidence_tokens)
        coverage = overlap / max(1, len(answer_tokens))
        if not strict_number_question and overlap >= 2 and coverage >= 0.12:
            coverage = max(coverage, 0.2)
        compact_answer = re.sub(
            r"[^a-z0-9\u3400-\u9fff]+", "", unicodedata.normalize("NFKC", answer.answer).lower()
        )
        compact_evidence = re.sub(
            r"[^a-z0-9\u3400-\u9fff]+", "", unicodedata.normalize("NFKC", evidence).lower()
        )
        direct = bool(
            min(len(compact_answer), len(compact_evidence)) >= 6
            and (compact_answer in compact_evidence or compact_evidence in compact_answer)
        )
        if not direct and coverage < 0.08 and overlap < 2:
            continue
        scored.append(
            (
                len(matched_numbers),
                int(direct),
                int(fact_id in existing_ids),
                coverage,
                overlap,
                float(fact.get("confidence", 0.0)),
                -position,
                fact_id,
            )
        )
    selected = []
    for *_, fact_id in sorted(scored, reverse=True):
        selected.append(fact_id)
        evidence = "\n".join(
            str(fact_index[selected_id].get(key, ""))
            for selected_id in selected
            for key in ("statement", "source_quote")
            if fact_index[selected_id].get(key)
        )
        selected_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        supported, unsupported = _audit_answer_claims(answer.answer, evidence)
        if supported and not unsupported and answer_numbers <= selected_numbers:
            break
        if len(selected) >= max_ids:
            break
    if not selected:
        return []
    evidence = "\n".join(
        str(fact_index[fact_id].get(key, ""))
        for fact_id in selected
        for key in ("statement", "source_quote")
        if fact_index[fact_id].get(key)
    )
    selected_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
    supported, _ = _audit_answer_claims(answer.answer, evidence)
    if not supported or not answer_numbers <= selected_numbers:
        return []
    return selected


def _finalize_answer_provenance(
    answer: GlobalAnswer,
    candidate_facts: list[dict],
    fact_index: dict[str, dict],
    allowed_fact_ids: set[str],
) -> GlobalAnswer:
    original_answer = answer.answer
    answer.source_fact_ids = _recover_answer_fact_ids(answer, candidate_facts, fact_index)
    if not answer.source_fact_ids:
        if any(term in answer.question for term in ("定量", "评价指标", "比较了哪些")):
            limit = 1
        elif any(term in answer.question for term in ("核心研究", "如何解释")):
            limit = 2
        else:
            limit = 3
        selected = []
        statements = []
        for candidate in candidate_facts:
            fact_id = str(candidate.get("fact_id", ""))
            fact = fact_index.get(fact_id)
            statement = str((fact or {}).get("statement", "")).strip()
            if fact_id not in allowed_fact_ids or not statement or statement in statements:
                continue
            selected.append(fact_id)
            statements.append(statement)
            if len(selected) >= limit:
                break
        if selected:
            answer.source_fact_ids = selected
            answer.answer = "；".join(statements)
            answer.confidence = sum(
                float(fact_index[fact_id].get("confidence", 0.0)) for fact_id in selected
            ) / len(selected)
            if original_answer.strip() not in ("", "证据不足", answer.answer):
                answer.unsupported_claims = list(
                    dict.fromkeys([*answer.unsupported_claims, original_answer])
                )
    return _apply_answer_provenance(answer, fact_index, allowed_fact_ids)


def batch_by_token_budget(items: list[dict], max_tokens: int = 3200) -> list[list[dict]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    batches: list[list[dict]] = []
    current: list[dict] = []
    for item in items:
        candidate = [*current, item]
        serialized = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        if current and estimate_tokens(serialized) > max_tokens:
            batches.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def shard_section_evidence(section: dict, max_tokens: int = 3200) -> list[dict]:
    if max_tokens <= 200:
        raise ValueError("max_tokens must be greater than 200")
    header = {key: value for key, value in section.items() if key != "facts"}
    header["summary"] = _prefix_for_budget(str(header.get("summary", "")), max_tokens // 5)
    header["core_concepts"] = list(header.get("core_concepts", []))[:20]
    header["key_findings"] = list(header.get("key_findings", []))[:20]
    header_tokens = estimate_tokens(json.dumps(header, ensure_ascii=False, separators=(",", ":")))
    fact_budget = max(100, max_tokens - header_tokens - 80)
    batches = batch_by_token_budget(section.get("facts", []), fact_budget)
    if not batches:
        batches = [[]]
    shards = []
    for index, facts in enumerate(batches):
        shard = {
            **header,
            "shard": [index + 1, len(batches)],
            "facts": facts,
        }
        if estimate_tokens(json.dumps(shard, ensure_ascii=False)) > max_tokens:
            raise ValueError("single evidence fact exceeds the reduce token budget")
        shards.append(shard)
    return shards


def build_document_request(
    groups: list[GroupSynthesis],
    fact_catalog: list[dict],
    questions: list[dict],
    contradictions: list[dict],
    max_tokens: int = 2400,
) -> tuple[dict, set[str]]:
    if max_tokens <= 500:
        raise ValueError("max_tokens must be greater than 500")
    required_ids = list(
        dict.fromkeys(
            fact_id
            for contradiction in contradictions
            for fact_id in contradiction.get("source_fact_ids", [])
        )
    )
    catalog_by_id = {fact["fact_id"]: fact for fact in fact_catalog if fact.get("fact_id")}
    priority = {"method": 0, "dataset": 1, "metric": 2, "result": 3, "limitation": 4}
    ordered_ids = [
        *required_ids,
        *[
            fact["fact_id"]
            for fact in sorted(fact_catalog, key=lambda item: priority.get(item.get("kind", ""), 5))
            if fact.get("fact_id") not in required_ids
        ],
    ]
    # Evidence shrinks across attempts so a rich document can still fit the budget
    # instead of crashing the whole run. Attempt 1 mirrors the previous behavior.
    attempts = [
        (160, {"concepts": 8, "findings": 8, "numeric_claims": 8, "limitations": 6, "future_work": 6, "contradictions": 6}),
        (100, {"concepts": 5, "findings": 5, "numeric_claims": 5, "limitations": 4, "future_work": 4, "contradictions": 4}),
        (60, {"concepts": 3, "findings": 3, "numeric_claims": 3, "limitations": 2, "future_work": 2, "contradictions": 2}),
        (40, {"concepts": 2, "findings": 2, "numeric_claims": 2, "limitations": 1, "future_work": 1, "contradictions": 1}),
    ]
    for summary_len, lens in attempts:
        evidence = [
            {
                "summary": _prefix_for_budget(group.summary, summary_len),
                "concepts": group.concepts[: lens["concepts"]],
                "findings": group.findings[: lens["findings"]],
                "numeric_claims": group.numeric_claims[: lens["numeric_claims"]],
                "limitations": group.limitations[: lens["limitations"]],
                "future_work": group.future_work[: lens["future_work"]],
                "contradictions": group.contradictions[: lens["contradictions"]],
            }
            for group in groups
        ]
        payload = {
            "evidence": evidence,
            "fact_catalog": [],
            "deterministic_contradictions": contradictions,
            "questions": questions,
        }
        selected: set[str] = set()
        for fact_id in ordered_ids:
            fact = catalog_by_id.get(fact_id)
            if not fact:
                continue
            compact_fact = {
                "fact_id": fact_id,
                "statement": _prefix_for_budget(str(fact.get("statement", "")), 120),
                "kind": fact.get("kind", "relation"),
            }
            candidate = {**payload, "fact_catalog": [*payload["fact_catalog"], compact_fact]}
            if estimate_tokens(json.dumps(candidate, ensure_ascii=False)) <= max_tokens:
                payload = candidate
                selected.add(fact_id)
            elif fact_id in required_ids:
                compact_fact["statement"] = _prefix_for_budget(compact_fact["statement"], 40)
                payload["fact_catalog"].append(compact_fact)
                selected.add(fact_id)
        if estimate_tokens(json.dumps(payload, ensure_ascii=False)) <= max_tokens:
            return payload, selected
    # Absolute last resort: even the most trimmed evidence plus the mandatory
    # contradiction facts cannot fit. Return the minimal payload rather than crash —
    # the model accepts a slightly larger request, and a complete run beats a crash.
    return payload, selected


def build_question_request(
    question: str,
    fact_catalog: list[dict],
    contradictions: list[dict],
    max_tokens: int = 2200,
) -> tuple[dict, set[str]]:
    routes = [
        (("研究问题", "核心"), {"method", "relation", "result"}, ("aim", "purpose", "goal", "目标", "目的", "研究")),
        (("数据集", "材料"), {"dataset"}, ("dataset", "数据集", "sample", "样本", "urbansound", "sound event")),
        (("主要方法", "采用"), {"method", "formula", "classification"}, ("method", "feature", "preprocess", "方法", "特征", "预处理")),
        (("比较了", "模型"), {"classification", "method", "relation", "advantage", "disadvantage"}, ("model", "classifier", "模型", "分类器", "mfcc", "stft")),
        (("评价指标", "指标"), {"metric", "definition"}, ("accuracy", "precision", "recall", "f1", "mcc", "指标", "准确率", "召回率")),
        (("定量结果", "最重要"), {"result", "metric", "example"}, ("accuracy", "result", "准确率", "结果", "%", "mcc")),
        (("如何解释", "解释"), {"relation", "result", "advantage"}, ("because", "due", "reason", "原因", "因为", "解释")),
        (("局限", "不足"), {"limitation", "disadvantage"}, ("limitation", "disadvantage", "limited", "局限", "不足", "缺点")),
        (("未来", "改进"), {"limitation", "result"}, ("future", "next aim", "goal", "improve", "未来", "下一个目标", "改进", "后续", "展望")),
    ]
    desired_kinds: set[str] = set()
    keywords: tuple[str, ...] = ()
    lowered_question = question.lower()
    for triggers, kinds, terms in routes:
        if any(trigger.lower() in lowered_question for trigger in triggers):
            desired_kinds = kinds
            keywords = terms
            break

    # The future-work route is the only one whose kind whitelist must be a hard
    # filter rather than a soft score: kind_relevance gives "method" 0 here, but
    # the additive keyword/section bonuses can still lift a method-implementation
    # fact above the score>0 cutoff, letting implementation prose leak into
    # "未来工作或改进方向是什么？" answers. Gate on the *resolved* route (its
    # desired_kinds set is unique among all nine routes), not raw question text,
    # so a question that merely contains 未来/改进 but routes elsewhere (e.g.
    # "未来工作中计划使用哪些数据集？") is untouched. Skip the gate when the
    # document has no limitation/result facts at all, so such a run does not
    # degrade into an empty catalog / "证据不足".
    future_route = desired_kinds == {"limitation", "result"}
    has_limitation_or_result = any(
        fact.get("kind") in {"limitation", "result"} for fact in fact_catalog
    )

    section_title_routes = [
        (("数据集", "材料"), ("dataset", "data set", "material", "数据集", "材料")),
        (("主要方法", "采用"), ("method", "feature", "preprocess", "方法", "特征", "预处理")),
        (("评价指标", "指标"), ("experiment", "evaluation", "result", "实验", "评估", "结果")),
        (("局限", "不足"), ("limitation", "discussion", "conclusion", "局限", "讨论", "结论")),
        (("未来", "改进"), ("future", "conclusion", "未来", "展望", "结论")),
        (("研究问题", "核心"), ("abstract", "introduction", "摘要", "引言")),
        (("如何解释", "解释"), ("discussion", "conclusion", "讨论", "结论")),
    ]

    def section_relevance(fact: dict) -> int:
        title = re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", str(fact.get("section_title", "")).lower())
        value = 0
        for triggers, terms in section_title_routes:
            if any(trigger.lower() in lowered_question for trigger in triggers):
                value += 4 * sum(term in title for term in terms)
                break
        if "定量" in question or "最重要" in question:
            if any(
                term in title
                for term in ("result", "discussion", "conclusion", "abstract", "experiment", "结果", "讨论", "结论", "摘要", "实验")
            ):
                value += 24
            if any(term in title for term in ("related work", "literature", "background", "相关工作", "文献综述", "背景")):
                value -= 60
        if any(term in lowered_question for term in ("未来", "改进")):
            if any(term in title for term in ("future", "未来", "展望")):
                value += 40
            if any(term in title for term in ("related work", "literature", "相关工作", "文献综述")):
                value -= 40
        return value

    def kind_relevance(fact: dict) -> int:
        kind = fact.get("kind")
        if "定量" in question or "最重要" in question:
            return {"result": 32, "metric": 12, "example": 8}.get(kind, 0)
        if any(term in question for term in ("数据集", "材料")):
            return 32 if kind == "dataset" else 0
        if any(term in question for term in ("评价指标", "指标")):
            return {"metric": 32, "definition": 12, "method": 6}.get(kind, 0)
        if any(term in question for term in ("比较了", "模型")):
            return {"classification": 32, "method": 12, "relation": 10}.get(kind, 0)
        if any(term in question for term in ("局限", "不足")):
            return {"limitation": 32, "disadvantage": 14}.get(kind, 0)
        if any(term in question for term in ("未来", "改进")):
            return {"limitation": 24, "result": 10}.get(kind, 0)
        return 10 if kind in desired_kinds else 0

    def score(fact: dict) -> tuple[int, float]:
        if (
            future_route
            and has_limitation_or_result
            and fact.get("kind") not in {"limitation", "result"}
        ):
            # Hard exclusion: a future-work answer may only cite limitation or
            # result facts. Zero here is absolute -- the selection loop drops any
            # fact with score <= 0, so no keyword/section bonus can rescue a
            # method-implementation fact into the catalog.
            return 0, float(fact.get("confidence", 0.0))
        text = f"{fact.get('concept', '')} {fact.get('statement', '')}".lower()
        title = re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", str(fact.get("section_title", "")).lower())
        value = kind_relevance(fact)
        if fact.get("kind") == "metric":
            # A row of a result/correlation table. Statements no longer carry a
            # "Table N:" prefix (the caption moved to a single per-table header
            # fact), so read the `table` field; the inline marker check keeps
            # hand-built or legacy fact dicts working.
            statement_lower = str(fact.get("statement", "")).lstrip().lower()
            table_row = bool(str(fact.get("table", "")).strip()) or (
                re.match(r"(?:table|\u8868)\s*\d*\s*[:\uff1a]", statement_lower) is not None
            )
            generic_column = (
                str(fact.get("concept", "")).strip().lower() in _GENERIC_METRIC_COLUMNS
            )
            config_marker = any(marker in text for marker in _CONFIG_TABLE_MARKERS)
            if generic_column or config_marker:
                # A row of a config/parameter table is not an evaluation metric;
                # it pollutes every question, so sink it below any real evidence.
                value -= 120
            elif table_row and any(trigger in question for trigger in ("\u8bc4\u4ef7\u6307\u6807", "\u6307\u6807")):
                # Raw correlation/result-table dump is weak evidence for "which
                # metrics were used" \u2014 prefer metric definitions over the dump.
                value -= 40
        value += sum(3 for keyword in keywords if keyword in text)
        if "定量" in question and re.search(r"\d", text):
            value += 5
        if "定量" in question or "最重要" in question:
            if any(
                term in text
                for term in ("respectively", "overall", "outperform", "分别", "总体", "优于", "两个数据集")
            ):
                value += 20
            if any(term in text for term in ("table ", "audio class", "类别=")):
                value -= 8
            if any(term in title for term in ("related work", "literature", "background", "相关工作", "文献综述", "背景")):
                value -= 40
        value += section_relevance(fact)
        return value, float(fact.get("confidence", 0.0))

    ranked = sorted(fact_catalog, key=score, reverse=True)
    payload = {
        "fact_catalog": [],
        "deterministic_contradictions": contradictions if "矛盾" in question else [],
        "questions": [{"index": 1, "question": question}],
    }
    selected: set[str] = set()
    for fact in ranked:
        fact_id = fact.get("fact_id")
        if not fact_id or score(fact)[0] <= 0:
            continue
        compact_fact = {
            "fact_id": fact_id,
            "concept": _prefix_for_budget(str(fact.get("concept", "")), 40),
            "statement": _prefix_for_budget(str(fact.get("statement", "")), 180),
            "kind": fact.get("kind", "relation"),
            "section_title": _prefix_for_budget(str(fact.get("section_title", "")), 40),
        }
        candidate = {**payload, "fact_catalog": [*payload["fact_catalog"], compact_fact]}
        if estimate_tokens(json.dumps(candidate, ensure_ascii=False)) > max_tokens:
            continue
        payload = candidate
        selected.add(str(fact_id))
    return payload, selected


class LongDocumentPipeline:
    def __init__(
        self,
        api_key: str,
        memory_path: str | Path,
        cache_path: str | Path,
        usage_path: str | Path,
        flash_model: str = "deepseek-v4-flash",
        pro_model: str = "deepseek-v4-pro",
        concurrency: int = 4,
        content_retry_attempts: int = 3,
        content_retry_base_seconds: float = 1.0,
        extract_max_output_tokens: int = 4096,
        reduce_max_output_tokens: int = 2200,
        document_max_output_tokens: int = 3000,
        document_title: str | None = None,
        document_key: str = "",
    ):
        # Retry layers multiply: the SDK retries transport/5xx/429 errors
        # `sdk_max_retries` times, and the application retries invalid structured
        # output `content_retry_attempts` times on top. Worst case for one logical
        # request is therefore sdk_max_retries * content_retry_attempts attempts,
        # plus the truncation-recovery calls. Keep both visible instead of letting
        # the product be a hidden constant.
        self.sdk_max_retries = 3
        self.request_timeout_seconds = 120.0
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=self.request_timeout_seconds,
            max_retries=self.sdk_max_retries,
        )
        self.memory = ConceptMemory(memory_path)
        self.cache = ResultCache(cache_path)
        self.usage = UsageTracker(usage_path)
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.content_retry_attempts = max(1, content_retry_attempts)
        self.content_retry_base_seconds = max(0.0, content_retry_base_seconds)
        self.extract_max_output_tokens = max(256, extract_max_output_tokens)
        self.reduce_max_output_tokens = max(256, reduce_max_output_tokens)
        self.document_max_output_tokens = max(256, document_max_output_tokens)
        self.document_title = document_title
        self.document_key = document_key
        self._document_fact_index: dict[str, dict] = {}
        self.group_fallbacks = 0
        # Truncation recovery budget: when a response is cut off mid-stream (or the
        # model burned all output tokens on reasoning), retry with thinking disabled.
        self.truncation_recovery_attempts = max(1, getattr(self, "truncation_recovery_attempts", 3))

    def _extract_json_payload(self, content: str) -> dict:
        if not content or not content.strip():
            raise ValueError("empty model response")
        text = content.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            payload = self._find_best_json_object(text)
            if payload is None:
                raise ValueError(
                    f"unable to parse JSON payload from model response: {error.msg}"
                ) from error
        if not isinstance(payload, dict):
            raise ValueError("model response must be a JSON object")
        return payload

    def _find_best_json_object(self, text: str) -> dict | None:
        decoder = json.JSONDecoder()
        candidates: list[tuple[int, object]] = []
        start = 0
        while True:
            idx = text.find("{", start)
            if idx == -1:
                break
            try:
                payload, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                start = idx + 1
                continue
            if isinstance(payload, dict):
                candidates.append((end - idx, payload))
            start = idx + 1
        if not candidates:
            return None
        # Prefer the longest successfully decoded JSON object.
        return max(candidates, key=lambda item: item[0])[1]

    def _iter_strings(self, value: object):
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, dict):
            for subvalue in value.values():
                yield from self._iter_strings(subvalue)
            return
        if isinstance(value, (list, tuple, set)):
            for subvalue in value:
                yield from self._iter_strings(subvalue)
            return
        if hasattr(value, "model_dump"):
            try:
                yield from self._iter_strings(value.model_dump())
            except Exception:
                pass
            return
        if hasattr(value, "__dict__"):
            try:
                yield from self._iter_strings(vars(value))
            except Exception:
                pass
            return

    def _find_json_in_choice(self, choice: object) -> str | dict | None:
        for text in self._iter_strings(choice):
            if not isinstance(text, str):
                continue
            candidate = text.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                payload = self._find_best_json_object(candidate)
                if payload is not None:
                    return payload
        return None

    def _normalize_choice_content(self, choice: object) -> str | dict | None:
        def is_json_like(value: object) -> bool:
            return isinstance(value, str) and value.strip().lstrip().startswith(("{", "["))

        if hasattr(choice, "message") and getattr(choice, "message", None) is not None:
            message = getattr(choice, "message")
            content = getattr(message, "content", None)
            if isinstance(content, (str, dict)) and str(content).strip():
                return content
            text = getattr(message, "text", None)
            if isinstance(text, str) and text.strip():
                return text
            reasoning = getattr(message, "reasoning_content", None)
            if isinstance(reasoning, str) and is_json_like(reasoning):
                return reasoning
        if hasattr(choice, "text"):
            content = getattr(choice, "text", None)
            if isinstance(content, str) and content.strip():
                return content
        if hasattr(choice, "content"):
            content = getattr(choice, "content", None)
            if isinstance(content, (str, dict)) and str(content).strip():
                return content
        if hasattr(choice, "delta"):
            delta = getattr(choice, "delta")
            if isinstance(delta, dict):
                content = delta.get("content") or delta.get("text")
                if isinstance(content, str) and content.strip():
                    return content
        if isinstance(choice, dict):
            for value in choice.values():
                if isinstance(value, dict) and value:
                    return value
                if isinstance(value, str) and value.strip() and is_json_like(value):
                    return value
        return self._find_json_in_choice(choice)

    def _log_model_response(
        self,
        stage: str,
        model: str,
        attempt: int,
        response: object,
        choice: object,
        content: object,
    ) -> None:
        try:
            root = Path(__file__).resolve().parent.parent
            debug_dir = root / "outputs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / "model_response_debug.log"
            with path.open("a", encoding="utf-8") as handle:
                handle.write("---\n")
                handle.write(f"{datetime.now(timezone.utc).isoformat()} {stage} {model} attempt={attempt}\n")
                handle.write(f"choice={repr(choice)}\n")
                handle.write(f"content={repr(content)}\n")
                handle.write(f"response={repr(response)}\n")
                handle.write("---\n\n")
        except Exception as exc:
            try:
                fallback = Path(__file__).resolve().parent.parent / "outputs" / "model_response_debug_error.log"
                with fallback.open("a", encoding="utf-8") as handle:
                    handle.write(f"log failure: {repr(exc)}\n")
            except Exception:
                pass

    async def _request_json(
        self,
        stage: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
        max_output_tokens: int = 1800,
        reasoning_effort: str = "low",
    ) -> dict:
        # Truncation recovery with thinking disabled. When a response is cut off
        # (finish_reason == "length"), the model usually burned the whole output
        # budget on reasoning_content, so content never started. Re-issuing the same
        # request with reasoning_effort="none" frees the full budget for content and
        # is far more reliable than a continuation seeded from a reasoning dump.
        recovery_attempts = getattr(self, "truncation_recovery_attempts", 3)
        last_error: Exception | None = None
        for attempt in range(self.content_retry_attempts):
            async with self.semaphore:
                # Time only the API call. Starting the clock before acquiring the
                # semaphore charged this row for concurrency queueing as well, which
                # inflated every reported latency and made "slow stage" conclusions
                # unreliable (measured: rows over 200s where the call itself was not).
                started = time.perf_counter()
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    reasoning_effort=reasoning_effort,
                )
                elapsed = time.perf_counter() - started
            self._record_usage(stage, model, response, elapsed)
            try:
                choice = response.choices[0]
                content = self._normalize_choice_content(choice)
                self._log_model_response(stage, model, attempt + 1, response, choice, content)
                truncated = getattr(choice, "finish_reason", None) == "length"
                if truncated:
                    # Model ran out of output budget. The content is usable only if
                    # it happens to parse as complete JSON (rare — e.g. the budget
                    # ended exactly at the closing brace). Otherwise recover: continue
                    # from the latest partial content when available (the larger
                    # continuation budget lets the JSON finish), else regenerate with
                    # thinking off so the whole budget goes to content. A response
                    # that truncates again becomes the seed for the next attempt, so
                    # partial content is never thrown away.
                    payload = self._parse_payload(content, response_model)
                    if payload is not None:
                        return payload
                    seed = choice
                    continued = False
                    for rec_attempt in range(recovery_attempts):
                        seed_text = self._message_content(seed)
                        if seed_text and not continued:
                            recovered, seed = await self._continue_truncated(
                                stage, model, seed_text, system, user,
                                response_model, max_output_tokens,
                            )
                            continued = True
                        else:
                            recovered, seed = await self._regenerate_truncated(
                                stage, model, system, user, response_model,
                                max_output_tokens,
                            )
                        if recovered is not None:
                            return recovered
                    raise ValueError(
                        "response truncated and recovery attempts exhausted"
                    )
                if content is None:
                    raise ValueError("empty model response")
                payload = self._parse_payload(content, response_model)
                if payload is None:
                    raise ValueError("unable to parse JSON payload from model response")
                return payload
            except (json.JSONDecodeError, ValidationError, ValueError, IndexError) as error:
                last_error = error
                if attempt + 1 < self.content_retry_attempts:
                    await asyncio.sleep(self.content_retry_base_seconds * (2**attempt))
        raise RuntimeError(
            f"{stage} returned invalid structured output after {self.content_retry_attempts} attempts"
        ) from last_error

    def _record_usage(self, stage: str, model: str, response: object, elapsed: float) -> None:
        """Record one API call.

        `elapsed` must already be the call duration (the caller starts the clock
        after acquiring the semaphore); this function does not time anything itself
        so a caller cannot accidentally include queueing again.
        """
        usage = getattr(response, "usage", None)
        self.usage.record(
            stage,
            model,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            int(getattr(usage, "total_tokens", 0) or 0),
            max(0.0, float(elapsed)),
            _response_request_id(response),
        )

    def _message_content(self, response: object) -> str:
        """Raw string content of a response (may be partial JSON).

        Accepts either a full response (choices[0]) or a bare choice, since recovery
        seeds alternate between the two.
        """
        choice = getattr(response, "choices", [None])[0]
        if choice is None:
            choice = response
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str) and content.strip():
            return content.strip()
        return ""

    async def _continue_truncated(
        self,
        stage: str,
        model: str,
        partial: str,
        system: str,
        user: str,
        response_model: type[BaseModel] | None,
        max_output_tokens: int,
    ) -> tuple[dict | None, object]:
        """Continue a truncated JSON response from the exact partial content, with
        thinking off and a larger budget. Returns (payload, response)."""
        cont_user = (
            "The previous assistant response was truncated mid-JSON. "
            "Continue the previous output exactly where it stopped so the combined "
            "messages form a single valid JSON object. Only output the JSON continuation."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": partial},
            {"role": "user", "content": cont_user},
        ]
        async with self.semaphore:
            started = time.perf_counter()
            cont_resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=min(max_output_tokens * 2, 4096),
                reasoning_effort="none",
            )
            elapsed = time.perf_counter() - started
        self._record_usage(stage + "_cont", model, cont_resp, elapsed)
        cont_choice = getattr(cont_resp, "choices", [None])[0]
        cont_content = self._message_content(cont_resp)
        self._log_model_response(stage + "_cont", model, 0, cont_resp, cont_choice, cont_content)
        if cont_content:
            # The continuation outputs only the tail of the JSON. Parsing it alone
            # always fails, so parse the partial and the tail combined.
            payload = self._parse_payload(partial + cont_content, response_model)
            if payload is not None:
                return payload, cont_resp
        payload = self._payload_from_response(cont_resp, response_model)
        return payload, cont_resp

    async def _regenerate_truncated(
        self,
        stage: str,
        model: str,
        system: str,
        user: str,
        response_model: type[BaseModel] | None,
        max_output_tokens: int,
    ) -> tuple[dict | None, object]:
        """Re-issue the full JSON request with thinking off so the whole output
        budget goes to content. Returns (payload, response)."""
        async with self.semaphore:
            started = time.perf_counter()
            regen_resp = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=max(2048, max_output_tokens),
                reasoning_effort="none",
            )
            elapsed = time.perf_counter() - started
        self._record_usage(stage + "_regen", model, regen_resp, elapsed)
        self._log_model_response(
            stage + "_regen", model, 0, regen_resp, getattr(regen_resp, "choices", [None])[0], None
        )
        return self._payload_from_response(regen_resp, response_model), regen_resp

    def _parse_payload(self, content: object, response_model: type[BaseModel] | None) -> dict | None:
        """Return a validated payload parsed from content, or None if unusable."""
        if content is None:
            return None
        try:
            if isinstance(content, dict):
                payload = content
            elif isinstance(content, str):
                payload = self._extract_json_payload(content)
            else:
                return None
            if response_model:
                response_model.model_validate(payload)
            return payload
        except (json.JSONDecodeError, ValidationError, ValueError, IndexError):
            return None

    def _payload_from_response(self, response: object, response_model: type[BaseModel] | None) -> dict | None:
        choice = getattr(response, "choices", [None])[0]
        if choice is None:
            return None
        content = self._normalize_choice_content(choice)
        return self._parse_payload(content, response_model)

    async def extract_chunk(self, section: HeadingNode, chunk: TextChunk) -> ChunkExtraction:
        key = cache_key("extract", self.flash_model, EXTRACTION_PROMPT_VERSION, chunk.text)
        cached = self.cache.get(key)
        if cached:
            result = _apply_grounding(ChunkExtraction.model_validate(cached), chunk.text, chunk.page_texts)
            return _assign_fact_ids(result, section.id)
        memory = self.memory.retrieve(chunk.text, document_key=getattr(self, "document_key", ""))
        system = """你是严格的长文事实抽取器。只抽取原文明确支持的信息，并输出 JSON：
{"summary":"本块摘要","facts":[{"concept":"概念","statement":"事实","kind":"definition|formula|method|result|limitation|advantage|disadvantage|example|relation|classification|dataset|metric","source_quote":"原文逐字引用","page":1,"confidence":0.95}],"new_concepts":["概念"]}
source_quote 必须是原文中的连续文本；不得补充原文没有的因果关系。直接逐字证据通常使用 0.8-1.0 的置信度，不要机械照抄示例值。
statement 必须用中文概括原文语义：原文是英文时翻译成中文，只保留专有名词、指标名、模型名等原文术语，禁止整句照抄英文原文。
每个块最多输出 20 条事实；候选超过 20 条时优先保留信息量最大、置信度最高的。source_quote 尽量简短（约 80 字以内），足以在原文中定位即可。
kind 判定要点：文本提到用于评估该工作的指标（accuracy、precision、recall、F1、AUC、BLEU、ROUGE、METEOR、PSNR、SSIM、mIoU、RMSE、MAE、success rate、SPL、perplexity、准确率、召回率、F1 值、皮尔逊相关系数等）时，必须标为 metric；指标出现在正文叙述里（不限于表格）同样要抽取，concept 用指标名，statement 保留该指标对应的数值。仅当文本确实是评估口径时才用 metric，泛泛出现的普通词（如 "accuracy" 仅表示"准确性"）不要标 metric。"""
        user = json.dumps(
            {
                "section": section.title,
                "page_range": [section.page_start, section.page_end],
                "related_memory": memory,
                "text": chunk.text,
            },
            ensure_ascii=False,
        )
        try:
            payload = await self._request_json(
                "chunk_extract",
                self.flash_model,
                system,
                user,
                response_model=ChunkExtraction,
                max_output_tokens=self.extract_max_output_tokens,
                reasoning_effort="none",
            )
        except RuntimeError as error:
            if self.pro_model and self.pro_model != self.flash_model:
                payload = await self._request_json(
                    "chunk_extract",
                    self.pro_model,
                    system,
                    user,
                    response_model=ChunkExtraction,
                    max_output_tokens=self.extract_max_output_tokens,
                    reasoning_effort="none",
                )
            else:
                raise
        result = ChunkExtraction.model_validate(payload)
        result = _apply_grounding(result, chunk.text, chunk.page_texts)
        result = _assign_fact_ids(result, section.id)
        self.cache.put(key, result.model_dump())
        return result

    async def synthesize_section(
        self, section: HeadingNode, extractions: list[ChunkExtraction]
    ) -> SectionSynthesis:
        compact = _grounded_payload(extractions)
        content = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        model = self.pro_model if len(extractions) > 1 else self.flash_model
        key = cache_key("section_reduce", model, REDUCE_PROMPT_VERSION, f"{section.title}\n{content}")
        cached = self.cache.get(key)
        if cached:
            return SectionSynthesis.model_validate(cached)
        system = """你是章节级归纳器。只能根据输入事实归纳，不得新增事实。输出 JSON：
{"title":"章节标题","summary":"章节摘要","core_concepts":["概念"],"key_findings":["关键结论"],"source_pages":[1]}
summary 和 key_findings 用中文概括；输入事实为英文时先翻译成中文再概括，禁止整句照抄英文。"""
        payload = await self._request_json(
            "section_reduce",
            model,
            system,
            json.dumps({"title": section.title, "facts": compact}, ensure_ascii=False),
            response_model=SectionSynthesis,
            max_output_tokens=self.reduce_max_output_tokens,
            reasoning_effort="none",
        )
        result = SectionSynthesis.model_validate(payload)
        self.cache.put(key, result.model_dump())
        return result

    def _group_reduce_model(self) -> str:
        return self.flash_model

    def _document_reduce_model(self) -> str:
        return self.flash_model

    def _global_answer_model(self) -> str:
        return self.flash_model

    async def _reduce_group(self, items: list[dict], level: int, index: int) -> GroupSynthesis:
        content = json.dumps(items, ensure_ascii=False, sort_keys=True)
        model = self._group_reduce_model()
        key = cache_key(
            "document_group_reduce",
            model,
            GROUP_PROMPT_VERSION,
            f"{level}:{index}:{content}",
        )
        cached = self.cache.get(key)
        if cached:
            return _apply_group_provenance(
                GroupSynthesis.model_validate(cached),
                self._document_fact_index,
                _collect_source_fact_ids(items),
            )
        system = """你是全文分层归纳器。输入只包含已通过原文验证的事实或上一层摘要。
只能压缩和合并输入，不得增加外部知识。输出 JSON：
{"summary":"组摘要","concepts":["概念"],"findings":["结论"],"numeric_claims":["数字结论"],"limitations":["局限"],"future_work":["未来工作"],"contradictions":["矛盾"],"source_fact_ids":["fact-id"],"source_sections":[],"source_pages":[]}
source_fact_ids 必须来自输入，只保留直接支持组摘要的最少证据，覆盖研究问题、方法、数据、指标、结果、局限、未来工作和矛盾所需事实，最多32条。章节和页码留空，由程序反推。"""
        system += "\n必须压缩：summary 不超过200字；其余数组每项不超过60字，concepts/findings/numeric_claims 各最多12项，其他数组各最多8项。"
        try:
            payload = await self._request_json(
                "document_group_reduce",
                model,
                system,
                content,
                response_model=GroupSynthesis,
                max_output_tokens=self.reduce_max_output_tokens,
                reasoning_effort="none",
            )
            result = GroupSynthesis.model_validate(payload)
        except RuntimeError:
            self.group_fallbacks += 1
            result = fallback_group_synthesis(items)
        result = _apply_group_provenance(
            result,
            self._document_fact_index,
            _collect_source_fact_ids(items),
        )
        self.cache.put(key, result.model_dump())
        return result

    async def _synthesize_overview(
        self,
        document_request: dict,
        groups: list[GroupSynthesis],
        detected_contradictions: list[dict],
    ) -> DocumentOverview:
        request = {**document_request, "questions": []}
        if self.document_title:
            request["candidate_title"] = self.document_title
        content = json.dumps(request, ensure_ascii=False, sort_keys=True)
        model = self._document_reduce_model()
        key = cache_key("document_overview", model, DOCUMENT_PROMPT_VERSION, content)
        cached = self.cache.get(key)
        if cached:
            result = DocumentOverview.model_validate(cached)
            return _apply_document_title(result, self.document_title)
        system = """你是全文总览归纳器。只能使用输入证据，不得增加外部知识。输出 JSON：
{"title":"文档标题","executive_summary":"全文摘要","research_question":"研究问题","methods":["方法"],"datasets":["数据集"],"metrics":["指标"],"key_findings":["结论"],"limitations":["局限"],"future_work":["未来工作"],"contradictions":["矛盾"]}
executive_summary 不超过300字，各数组最多10项。
要求：
1. executive_summary 必须写成一段连贯的中文叙述（200-300字）：先以一句总起句说明文档主题与研究对象，再按 研究问题→方法→主要结果→局限与未来工作 的顺序展开，句与句之间用"首先/此外/最后/总体而言"等衔接词过渡；禁止分点罗列，禁止用分号或顿号把零散事实堆叠成清单。
2. title：优先使用输入 evidence 中的 candidate_title 字段原样保留（禁止翻译、改写或添加副标题）；candidate_title 缺失时才自行给出文档标题。
3. 论文含未来/展望/结论章节证据时 future_work 必须填写，不允许空数组。
4. contradictions 只列证据支持的跨章节数字或结论冲突。
5. 每个数字必须与它所属的主体写在一起，禁止把不同事实里的数字拼进同一个论断。证据里的数字各自绑定着主体（例如"base 模型 12 小时"与"big 模型 3.5 天"、"base 模型 BLEU 27.3"与"big 模型 BLEU 41.8"）：不得把某个主体的成绩与另一个主体的训练时长/配置写成一句话。若无法确定数字归属，就不要写这个数字。
6. 其余字段（executive_summary 及各数组）用中文填写：英文证据先翻译成中文再概括，禁止整句英文，只保留专有名词/指标名/模型名；title 保留原语言。"""
        try:
            payload = await self._request_json(
                "document_overview",
                model,
                system,
                json.dumps(request, ensure_ascii=False),
                response_model=DocumentOverview,
                max_output_tokens=min(1800, self.document_max_output_tokens),
            )
            result = DocumentOverview.model_validate(payload)
        except RuntimeError:
            result = fallback_document_overview(
                groups, detected_contradictions, title=self.document_title
            )
        result = _apply_document_title(result, self.document_title)
        self.cache.put(key, result.model_dump())
        return result

    async def _answer_global_question(
        self,
        question: str,
        fact_catalog: list[dict],
        detected_contradictions: list[dict],
    ) -> GlobalAnswer:
        if ("矛盾" in question or "contradiction" in question.lower()) and detected_contradictions:
            return _build_contradiction_answer(
                question, detected_contradictions, self._document_fact_index
            )
        request, allowed_fact_ids = build_question_request(
            question, fact_catalog, detected_contradictions
        )
        content = json.dumps(request, ensure_ascii=False, sort_keys=True)
        model = self._global_answer_model()
        key = cache_key("global_answer", model, DOCUMENT_PROMPT_VERSION, content)
        cached = self.cache.get(key)
        if cached:
            answer = GlobalAnswer.model_validate(cached)
            return _finalize_answer_provenance(
                answer,
                request["fact_catalog"],
                self._document_fact_index,
                allowed_fact_ids,
            )
        system = """你是全文证据问答器。只回答一个问题，只能使用 fact_catalog 中的事实；证据不足时回答“证据不足”。输出 JSON：
{"question":"原问题","answer":"不超过100字的答案","source_fact_ids":["最多3个fact-id"],"source_sections":[],"source_pages":[],"confidence":0.85}
source_fact_ids 必须来自 fact_catalog；章节和页码留空，由程序反推。
要求：
1. 用自己的话概括，禁止把 source_quote 或原始表格整段贴进答案。
2. 答案必须全用问题的语言（问题为中文时答案必须全中文）：若证据的 statement 是英文，先用中文转述其含义再写进答案；禁止把英文 statement 或 source_quote 原样写进答案，只能保留短小的专有名词、指标名、模型名。
3. 问“未来工作”时只答论文明确提到的未来方向，不要复述方法实现细节。
4. 定量答案必须保留事实中的具体数字。"""
        try:
            payload = await self._request_json(
                "global_answer",
                model,
                system,
                json.dumps(request, ensure_ascii=False),
                response_model=GlobalAnswer,
                max_output_tokens=min(1000, self.document_max_output_tokens),
            )
            answer = GlobalAnswer.model_validate(payload)
            answer.question = question
        except RuntimeError:
            answer = GlobalAnswer(
                question=question,
                answer="证据不足",
                source_fact_ids=[],
                source_sections=[],
                source_pages=[],
                confidence=0.0,
            )
        _finalize_answer_provenance(
            answer,
            request["fact_catalog"],
            self._document_fact_index,
            allowed_fact_ids,
        )
        self.cache.put(key, answer.model_dump())
        return answer

    async def synthesize_document(
        self, section_results: list[dict], max_tokens: int = 3200
    ) -> DocumentSynthesis:
        # Deterministic cross-chapter scan runs before evidence is built so its
        # source facts are forced into the evidence index (and thus keep their
        # provenance through the answer pipeline).
        document_contradictions = detect_document_contradictions(section_results)
        detected_contradictions = [
            item
            for section_result in section_results
            for item in section_result.get("numeric_contradictions", [])
        ]
        detected_contradictions.extend(document_contradictions)
        required_fact_ids = {
            str(fact_id)
            for item in document_contradictions
            for fact_id in item.get("source_fact_ids", [])
        }
        section_evidence = [
            build_section_evidence(result, required_fact_ids) for result in section_results
        ]
        evidence = [
            shard
            for section in section_evidence
            for shard in shard_section_evidence(section, max_tokens)
        ]
        self._document_fact_index = _build_fact_index(evidence)
        batches = batch_by_token_budget(evidence, max_tokens)
        groups = await asyncio.gather(
            *(self._reduce_group(batch, 0, index) for index, batch in enumerate(batches))
        )
        level = 1
        while (
            len(groups) > 1
            and estimate_tokens(json.dumps([group.model_dump() for group in groups], ensure_ascii=False))
            > max_tokens
            and level <= 4
        ):
            group_dicts = [group.model_dump() for group in groups]
            batches = batch_by_token_budget(group_dicts, max_tokens)
            groups = await asyncio.gather(
                *(self._reduce_group(batch, level, index) for index, batch in enumerate(batches))
            )
            level += 1

        compact = [group.model_dump() for group in groups]
        group_fact_ids = _collect_source_fact_ids(compact)
        group_fact_ids.update(
            fact_id
            for contradiction in detected_contradictions
            for fact_id in contradiction.get("source_fact_ids", [])
            if fact_id in self._document_fact_index
        )
        full_fact_catalog = [
            {
                "fact_id": fact_id,
                "concept": fact.get("concept", ""),
                "statement": fact.get("statement", ""),
                "kind": fact.get("kind", "relation"),
                "section_id": fact.get("section_id"),
                "section_title": fact.get("section_title", ""),
                "page": fact.get("page"),
                "confidence": fact.get("confidence", 0.0),
            }
            for fact_id, fact in self._document_fact_index.items()
        ]
        fact_catalog = [
            fact for fact in full_fact_catalog if fact["fact_id"] in group_fact_ids
        ]
        questions = [{"index": index + 1, "question": question} for index, question in enumerate(GLOBAL_QUESTIONS)]
        document_request, allowed_fact_ids = build_document_request(
            groups,
            fact_catalog,
            questions,
            detected_contradictions,
            max_tokens=min(max_tokens, 2400),
        )
        bundle_scope = {"document_request": document_request}
        if self.document_title:
            # 候选标题决定 bundle 的 title，必须折进缓存键，否则标题随代码演进变化时
            # document_overview 走不到（bundle 命中即短路返回），仍回旧标题。
            bundle_scope["candidate_title"] = self.document_title
        content = json.dumps(bundle_scope, ensure_ascii=False, sort_keys=True)
        key = cache_key("document_bundle", self.flash_model, DOCUMENT_PROMPT_VERSION, content)
        cached = self.cache.get(key)
        if cached:
            result = DocumentSynthesis.model_validate(cached)
            # bundle 里缓存的 title 可能来自候选标题修正前的旧版模型输出（截断回显），
            # 命中后同样过一遍确定性兜底，保证展示 title 恒等于候选标题。
            result = _apply_document_title(result, self.document_title)
            effective_contradictions = _merge_overview_contradictions(
                detected_contradictions, result.contradictions, self._document_fact_index
            )
            for answer in result.global_answers:
                if "矛盾" in answer.question or "contradiction" in answer.question.lower():
                    replacement = _build_contradiction_answer(
                        answer.question,
                        effective_contradictions,
                        self._document_fact_index,
                    )
                    if effective_contradictions and replacement.answer != "证据不足":
                        answer.answer = replacement.answer
                        answer.source_fact_ids = replacement.source_fact_ids
                        answer.source_sections = replacement.source_sections
                        answer.source_pages = replacement.source_pages
                        answer.confidence = replacement.confidence
                        answer.unsupported_claims = []
                    continue
                question_request, question_fact_ids = build_question_request(
                    answer.question, full_fact_catalog, detected_contradictions
                )
                _finalize_answer_provenance(
                    answer,
                    question_request["fact_catalog"],
                    self._document_fact_index,
                    question_fact_ids,
                )
            return result
        overview = await self._synthesize_overview(
            document_request, groups, detected_contradictions
        )
        effective_contradictions = _merge_overview_contradictions(
            detected_contradictions, overview.contradictions, self._document_fact_index
        )
        answers = await asyncio.gather(
            *(
                self._answer_global_question(
                    question,
                    full_fact_catalog,
                    effective_contradictions,
                )
                for question in GLOBAL_QUESTIONS
            )
        )
        result = DocumentSynthesis(**overview.model_dump(), global_answers=answers)
        if detected_contradictions:
            statements = list(
                dict.fromkeys(item["statement"] for item in detected_contradictions if item.get("statement"))
            )
            result.contradictions = [
                *statements,
                *[
                    item
                    for item in result.contradictions
                    if not re.search(r"未发现|no contradiction", item, re.IGNORECASE)
                    and item not in statements
                ],
            ]
        self.cache.put(key, result.model_dump())
        return result

    def stats(self) -> dict:
        return {
            # `usage` is this run only; `all_time_usage` is the lifetime total of
            # the usage DB. Keeping them separate stops a cost estimate from
            # silently reporting the whole history.
            "usage": self.usage.summary(),
            "all_time_usage": self.usage.all_time_summary(),
            "cache": self.cache.summary(),
            "group_fallbacks": getattr(self, "group_fallbacks", 0),
            # Retry/timeout configuration, so a slow or expensive run can be
            # explained without reading the code. `latency_seconds` in `usage` is
            # per API call and EXCLUDES semaphore queueing.
            "config": {
                "request_timeout_seconds": getattr(self, "request_timeout_seconds", None),
                "sdk_max_retries": getattr(self, "sdk_max_retries", None),
                "content_retry_attempts": getattr(self, "content_retry_attempts", None),
                "max_attempts_per_request": (
                    getattr(self, "sdk_max_retries", 0)
                    * getattr(self, "content_retry_attempts", 0)
                ),
            },
        }

    async def process_section(
        self, section: HeadingNode, max_tokens: int = 3200, overlap_tokens: int = 180
    ) -> dict:
        chunks = split_text(
            section.text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            page_start=section.page_start,
            page_end=section.page_end,
            page_texts=section.page_texts,
        )
        extractions = await asyncio.gather(*(self.extract_chunk(section, chunk) for chunk in chunks))
        # Deterministic prose metric recovery. Table-derived metric facts need
        # markdown tables from pymupdf4llm; when a PDF yields none, the metric pool
        # is empty and "使用了哪些评价指标？" has nothing relevant to cite. Adding
        # these as a synthetic extraction makes them visible to the section reduce,
        # the evidence index, and the global-answer retrieval alike.
        metric_facts = extract_metric_facts(section)
        if metric_facts:
            extractions = [
                *extractions,
                ChunkExtraction(
                    summary=f"确定性抽取的评价指标事实（{len(metric_facts)} 条）",
                    facts=metric_facts,
                    new_concepts=[],
                ),
            ]
        table_facts = extract_table_facts(section)
        numeric_contradictions = detect_numeric_contradictions(section, table_facts)
        for extraction in extractions:
            for fact in extraction.facts:
                self.memory.upsert(
                    fact.concept,
                    fact.statement if fact.kind == "definition" else "",
                    section.id,
                    fact.page,
                    fact.source_quote,
                    getattr(self, "document_key", ""),
                )
        synthesis = await self.synthesize_section(section, extractions)
        return {
            "section": section.model_dump(exclude={"children"}),
            "chunks": [chunk.model_dump() for chunk in chunks],
            "extractions": [item.model_dump() for item in extractions],
            "table_facts": [fact.model_dump() for fact in table_facts],
            "metric_facts": [fact.model_dump() for fact in metric_facts],
            "detected_facts": [
                item.claim_fact.model_dump() for item in numeric_contradictions
            ],
            "numeric_contradictions": [
                item.model_dump(exclude={"claim_fact"}) for item in numeric_contradictions
            ],
            "synthesis": synthesis.model_dump(),
        }

    async def close(self) -> None:
        await self.client.close()
        self.memory.close()
        self.cache.close()
        self.usage.close()


def _is_toc_section(title: str) -> bool:
    """True for table-of-contents style sections (their text is a page-number
    listing, not paper content, so they must be dropped along with their body)."""
    cleaned = _plain_heading(title).strip().lower()
    return cleaned in {
        "contents",
        "table of contents",
        "toc",
        "list of figures",
        "list of tables",
        "list of algorithms",
        "目录",
        "内容",
        "图目录",
        "表目录",
    } or cleaned.startswith("contents:")


def preprocess_pdf(pdf_path: str | Path, include_back_matter: bool = False) -> tuple[HeadingNode, list[HeadingNode]]:
    pages = extract_pdf_pages(pdf_path)
    root = build_heading_tree(pages)
    sections = deduplicate_sections([
        section
        for section in flatten_sections(root)
        if section.text.strip()
        and not _is_toc_section(section.title)
        and (include_back_matter or not is_back_matter(section.title))
    ])
    return root, sections


def _manifest(sections: list[HeadingNode], max_tokens: int, overlap_tokens: int) -> dict:
    rows = []
    for section in sections:
        chunks = split_text(
            section.text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            page_start=section.page_start,
            page_end=section.page_end,
            page_texts=section.page_texts,
        )
        rows.append(
            {
                "id": section.id,
                "title": section.title,
                "level": section.level,
                "parent_id": section.parent_id,
                "pages": [section.page_start, section.page_end],
                "tokens": estimate_tokens(section.text),
                "chunks": len(chunks),
                "max_chunk_tokens": max(chunk.token_estimate for chunk in chunks),
            }
        )
    return {"section_count": len(rows), "sections": rows}


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def report_evidence_metrics(output_path: str | Path) -> dict:
    """Print (and return) provenance metrics for an existing run output.

    Supports the dev loop: metrics can be recomputed over a stored run without
    paying for another full API pass, which is what makes them usable as a
    regression gate.
    """
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = run_evidence_metrics(
        data.get("sections") or [], data.get("document_synthesis")
    )
    print(f"# provenance metrics: {path}")
    print(
        f"facts={metrics['fact_count']} answers={metrics['answer_count']} "
        f"with_evidence={metrics['answers_with_evidence']} "
        f"sufficient={metrics['answers_with_sufficient_evidence']} "
        f"citation_precision={metrics['citation_precision']} "
        f"unsupported_rate={metrics['unsupported_claim_rate']} "
        f"cited(min/median/max)={metrics['min_cited_facts']}/"
        f"{metrics['median_cited_facts']}/{metrics['max_cited_facts']}"
    )
    print(
        f"unsupported breakdown: retrieval_gap={metrics['unsupported_retrieval_gaps']} "
        f"undecided={metrics['unsupported_undecided']} "
        f"(total {metrics['total_unsupported_claims']}; undecided bounds fabrication)"
    )
    print(
        f"overview unsupported numbers: {metrics['overview_unsupported_numbers']} "
        f"(must be 0)"
    )
    for finding in metrics["overview_numeric_findings"]:
        print(f"    [{finding['kind']}] {finding['field']} {finding['numbers']}")
        print(f"       {finding['sentence']}")
    print(f"\n{'#':>2} {'cited':>5} {'resolv':>6} {'suff':>5} {'kept':>4} {'unsup':>5} "
          f"{'rate':>5} {'pages':>5}  question")
    for index, item in enumerate(metrics["per_answer"], 1):
        print(
            f"{index:>2} {item['cited_fact_count']:>5} "
            f"{item['resolvable_fact_count']:>6} "
            f"{str(item['sufficient_evidence']):>5} "
            f"{item['supported_sentence_count']:>4} "
            f"{item['unsupported_claim_count']:>5} "
            f"{item['unsupported_claim_rate']:>5.2f} "
            f"{item['distinct_source_pages']:>5}  {item['question']}"
        )
    return metrics


async def _run(args: argparse.Namespace) -> None:
    _load_env_file()
    if not args.pdf:
        raise ValueError("a PDF path is required (or use --metrics on an existing run output)")
    root, sections = preprocess_pdf(args.pdf, args.include_back_matter)
    if args.section:
        terms = [term.lower() for term in args.section]
        sections = [section for section in sections if any(term in section.title.lower() for term in terms)]
    if args.max_sections:
        sections = sections[: args.max_sections]
    if not sections:
        raise ValueError("no matching sections")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(sections, args.max_tokens, args.overlap_tokens)
    if args.dry_run:
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), **{k: manifest[k] for k in ("section_count",)}}, ensure_ascii=False))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing environment variable: {args.api_key_env}")
    pipeline = LongDocumentPipeline(
        api_key=api_key,
        memory_path=args.memory_db,
        cache_path=args.cache_db,
        usage_path=args.usage_db,
        flash_model=args.flash_model,
        pro_model=args.pro_model,
        concurrency=args.concurrency,
        content_retry_attempts=args.content_retries,
        content_retry_base_seconds=args.retry_base_seconds,
        extract_max_output_tokens=args.extract_max_output_tokens,
        reduce_max_output_tokens=args.reduce_max_output_tokens,
        document_max_output_tokens=args.document_max_output_tokens,
        document_title=_candidate_document_title(root),
        document_key=args.document_key or _document_key(args.pdf),
    )
    try:
        results = []
        for index, section in enumerate(sections, 1):
            print(f"[{index}/{len(sections)}] {section.title}")
            results.append(await pipeline.process_section(section, args.max_tokens, args.overlap_tokens))
            output.write_text(
                json.dumps(
                    {
                        "sections": results,
                        "document_synthesis": None,
                        "stats": pipeline.stats(),
                        "evidence_metrics": run_evidence_metrics(results, None),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        document_synthesis = None
        if len(results) > 1 and not args.skip_document_reduce:
            print(f"[Global] reducing {len(results)} sections")
            document_synthesis = await pipeline.synthesize_document(results, args.reduce_max_tokens)
        metrics = run_evidence_metrics(results, document_synthesis)
        output.write_text(
            json.dumps(
                {
                    "sections": results,
                    "document_synthesis": document_synthesis.model_dump() if document_synthesis else None,
                    "stats": pipeline.stats(),
                    "evidence_metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "[Metrics] facts={fact_count} answers={answer_count} "
            "with_evidence={answers_with_evidence} sufficient={answers_with_sufficient_evidence} "
            "citation_precision={citation_precision} unsupported_rate={unsupported_claim_rate} "
            "median_cited={median_cited_facts}".format(**metrics)
        )
    finally:
        await pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="P2G V2 long-document pipeline")
    parser.add_argument("pdf", nargs="?", default="")
    parser.add_argument("--output", default="p2g_v2_output.json")
    parser.add_argument("--memory-db", default="p2g_v2_memory.sqlite3")
    parser.add_argument("--cache-db", default="p2g_v2_cache.sqlite3")
    parser.add_argument("--usage-db", default="p2g_v2_usage.sqlite3")
    parser.add_argument(
        "--document-key",
        default="",
        help="Override the document identity used to scope concept memory "
        "(defaults to a hash of the PDF content).",
    )
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--overlap-tokens", type=int, default=180)
    parser.add_argument("--section", action="append")
    parser.add_argument("--max-sections", type=int)
    parser.add_argument("--include-back-matter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-document-reduce", action="store_true")
    parser.add_argument("--reduce-max-tokens", type=int, default=3200)
    parser.add_argument("--flash-model", default="deepseek-v4-flash")
    parser.add_argument("--pro-model", default="deepseek-v4-pro")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--content-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--extract-max-output-tokens", type=int, default=4096)
    parser.add_argument("--reduce-max-output-tokens", type=int, default=2200)
    parser.add_argument("--document-max-output-tokens", type=int, default=3000)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--metrics",
        metavar="RUN_OUTPUT_JSON",
        default="",
        help="Print provenance metrics for an existing run output and exit "
        "(no PDF or API key needed).",
    )
    parser.add_argument(
        "--purge-unowned-memory",
        action="store_true",
        help="Delete concept-memory rows that belong to no document (written before "
        "document scoping existed), then exit. These rows leak across documents.",
    )
    args = parser.parse_args()
    if args.purge_unowned_memory:
        memory = ConceptMemory(args.memory_db)
        try:
            result = memory.purge_unowned()
            print(f"purged unowned memory rows: {result}")
        finally:
            memory.close()
        return
    if args.metrics:
        report_evidence_metrics(args.metrics)
        return
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
