"""Benchmark one PDF end to end: wall-clock, tokens, and quality.

Kept as a real tool rather than a scratch script: the previous performance numbers
were measured with a throwaway copy, so they could not be reproduced without
rewriting it. Reports both full end-to-end wall time (including PDF preprocessing)
and pipeline wall time (excluding preprocessing); file timestamps are not enough
for either claim. `--extract-only` isolates chunk extraction and skips all
section/group/document reductions and QA so response-schema experiments do not
need to pay for downstream stages.
The section-level concurrency of `_run` is mirrored so the same scheduling is under
test; `--section-concurrency 1` reproduces the old serial behaviour.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts"))

import p2g_v2  # noqa: E402


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _extract_only_metrics(section_results: list[dict]) -> dict:
    chunks = []
    facts = []
    for result in section_results:
        chunk_records = result.get("chunks", [])
        extraction_records = result.get("extractions", [])
        for index, chunk in enumerate(chunk_records):
            extraction = extraction_records[index] if index < len(extraction_records) else {}
            chunk_facts = extraction.get("facts", [])
            chunks.append({"index": chunk.get("index"), "fact_count": len(chunk_facts)})
            facts.extend(chunk_facts)
    quote_lengths = sorted(len(str(fact.get("source_quote", ""))) for fact in facts)
    count = len(facts)
    p90_index = max(0, min(count - 1, math.ceil(0.90 * count) - 1)) if count else 0
    grounded = sum(bool(fact.get("grounded")) for fact in facts)
    return {
        "chunk_count": len(chunks),
        "fact_count": count,
        "grounded_fact_count": grounded,
        "grounding_rate": round(grounded / count, 4) if count else 0.0,
        "chunks_at_fact_cap_20": sum(chunk["fact_count"] >= 20 for chunk in chunks),
        "quote_chars_total": sum(quote_lengths),
        "quote_chars_mean": round(sum(quote_lengths) / count, 1) if count else 0.0,
        "quote_chars_p50": quote_lengths[count // 2] if count else 0,
        "quote_chars_p90": quote_lengths[p90_index] if count else 0,
        "quote_chars_max": max(quote_lengths, default=0),
        "quote_over_80_count": sum(length > 80 for length in quote_lengths),
        "per_chunk_fact_counts": [chunk["fact_count"] for chunk in chunks],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--tag", required=True, help="output name suffix")
    parser.add_argument("--section-concurrency", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument(
        "--group-batch-tokens",
        type=int,
        default=p2g_v2.DEFAULT_GROUP_BATCH_TOKENS,
        help="Token budget for each hierarchical group-reduce batch.",
    )
    parser.add_argument("--out-dir", default="outputs/perf")
    parser.add_argument(
        "--global-answer-reasoning-effort",
        choices=("low", "none"),
        default="low",
        help="reasoning effort for initial global-answer requests only",
    )
    parser.add_argument(
        "--cache-label",
        default="unspecified",
        help="descriptive cache-state label; does not flush the OS filesystem cache",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="benchmark chunk extraction only; skip section reduction and document QA",
    )
    args = parser.parse_args()

    p2g_v2._load_env_file()
    pdf = Path(args.pdf)
    if not pdf.is_absolute():
        pdf = PROJECT_ROOT / pdf
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set (checked env and .env)")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}.json"
    udb = out_dir / f"{args.tag}_usage.sqlite3"
    cdb = out_dir / f"{args.tag}_cache.sqlite3"
    mdb = out_dir / f"{args.tag}_memory.sqlite3"
    for path in (out, udb, cdb, mdb):
        path.unlink(missing_ok=True)

    pdf_size_bytes = pdf.stat().st_size
    run_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_label": args.cache_label,
        "git_revision": _git_revision(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "pdf_name": pdf.name,
        "pdf_size_bytes": pdf_size_bytes,
        "python_version": platform.python_version(),
        "pymupdf4llm_version": _distribution_version("pymupdf4llm"),
        "pymupdf_version": _distribution_version("PyMuPDF"),
        "openai_version": _distribution_version("openai"),
        "pydantic_version": _distribution_version("pydantic"),
    }

    benchmark_started = time.perf_counter()
    preprocess_started = time.perf_counter()
    root, sections = p2g_v2.preprocess_pdf(str(pdf), False)
    preprocess_elapsed = time.perf_counter() - preprocess_started
    print(f"sections={len(sections)} parse={preprocess_elapsed:.1f}s", flush=True)

    pipeline = p2g_v2.LongDocumentPipeline(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        memory_path=mdb,
        cache_path=cdb,
        usage_path=udb,
        concurrency=args.concurrency,
        section_concurrency=args.section_concurrency,
        group_batch_tokens=args.group_batch_tokens,
        document_title=p2g_v2._candidate_document_title(root),
        document_key=p2g_v2._document_key(pdf),
        global_answer_reasoning_effort=args.global_answer_reasoning_effort,
    )

    started = time.perf_counter()
    try:
        results: list[dict | None] = [None] * len(sections)
        gate = asyncio.Semaphore(max(1, args.section_concurrency))
        done = 0
        lock = asyncio.Lock()

        async def run_section(index: int, section) -> None:
            nonlocal done
            async with gate:
                if args.extract_only:
                    chunks = p2g_v2.split_text(
                        section.text,
                        max_tokens=args.max_tokens,
                        overlap_tokens=180,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        page_texts=section.page_texts,
                    )
                    extractions = await asyncio.gather(
                        *(pipeline.extract_chunk(section, chunk) for chunk in chunks)
                    )
                    for extraction in extractions:
                        for fact in extraction.facts:
                            pipeline.memory.upsert(
                                fact.concept,
                                fact.statement if fact.kind == "definition" else "",
                                section.id,
                                fact.page,
                                fact.source_quote,
                                getattr(pipeline, "document_key", ""),
                            )
                    results[index] = {
                        "section": section.model_dump(exclude={"children"}),
                        "chunks": [chunk.model_dump() for chunk in chunks],
                        "extractions": [item.model_dump() for item in extractions],
                    }
                else:
                    results[index] = await pipeline.process_section(
                        section, args.max_tokens, 180
                    )
            async with lock:
                done += 1
                if done % 5 == 0 or done == len(sections):
                    print(f"  [{done}/{len(sections)}] "
                          f"{time.perf_counter() - started:.0f}s", flush=True)

        await asyncio.gather(*(run_section(i, s) for i, s in enumerate(sections)))
        extract_elapsed = time.perf_counter() - started
        section_results = [item for item in results if item is not None]
        synthesis = None
        if not args.extract_only:
            synthesis = await pipeline.synthesize_document(section_results, args.max_tokens)
        finished = time.perf_counter()
        pipeline_elapsed = finished - started
        end_to_end_elapsed = finished - benchmark_started
        stats = pipeline.stats()
    finally:
        await pipeline.close()

    metrics = (
        _extract_only_metrics(section_results)
        if args.extract_only
        else p2g_v2.run_evidence_metrics(section_results, synthesis.model_dump())
    )
    run_metadata["pdf_sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    out.write_text(json.dumps({
        "run_metadata": run_metadata,
        "sections": section_results,
        "document_synthesis": synthesis.model_dump() if synthesis is not None else None,
        "evidence_metrics": (
            metrics if args.extract_only
            else {k: v for k, v in metrics.items() if k != "per_answer"}
        ),
        "timing": {
            "parse_seconds": round(preprocess_elapsed, 2),
            "extract_seconds": round(extract_elapsed, 2),
            "synthesis_seconds": round(pipeline_elapsed - extract_elapsed, 2),
            "pipeline_wall_seconds": round(pipeline_elapsed, 2),
            "end_to_end_seconds": round(end_to_end_elapsed, 2),
        },
        "config": {
            "concurrency": args.concurrency,
            "section_concurrency": args.section_concurrency,
            "group_batch_tokens": args.group_batch_tokens,
            "global_answer_reasoning_effort": args.global_answer_reasoning_effort,
            "extract_only": args.extract_only,
        },
        "stats": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    usage = stats.get("usage") or {}
    print(f"\n=== {args.tag}  (API并发={args.concurrency}, 章节并发={args.section_concurrency}) ===")
    print(f"  PDF 解析      : {preprocess_elapsed:8.1f} s")
    print(f"  抽取阶段      : {extract_elapsed:8.1f} s")
    print(f"  归纳+问答     : {pipeline_elapsed - extract_elapsed:8.1f} s")
    print(f"  流水线墙钟    : {pipeline_elapsed:8.1f} s (不含 PDF 解析)")
    print(f"  端到端墙钟    : {end_to_end_elapsed:8.1f} s (含 PDF 解析)")
    print(f"  调用          : {usage.get('calls')}")
    print(f"  输入 token    : {usage.get('input_tokens'):,}")
    print(f"  输出 token    : {usage.get('output_tokens'):,}")
    print(f"  总 token      : {usage.get('total_tokens'):,}")
    if args.extract_only:
        print(f"  chunks={metrics['chunk_count']} facts={metrics['fact_count']} "
              f"grounded={metrics['grounded_fact_count']} "
              f"grounding={metrics['grounding_rate']:.3f} "
              f"quote_chars(p90/max)={metrics['quote_chars_p90']}/"
              f"{metrics['quote_chars_max']}")
    else:
        print(f"  facts={metrics['fact_count']} "
              f"with_evidence={metrics['answers_with_evidence']}/10 "
              f"unsupported={metrics['unsupported_claim_rate']:.3f} "
              f"overview_unsup={metrics['overview_unsupported_numbers']}")


if __name__ == "__main__":
    asyncio.run(main())
