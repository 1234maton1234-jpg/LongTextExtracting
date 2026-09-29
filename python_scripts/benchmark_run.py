"""Benchmark one PDF end to end: wall-clock, tokens, and quality.

Kept as a real tool rather than a scratch script: the previous performance numbers
were measured with a throwaway copy, so they could not be reproduced without
rewriting it. Reports wall-clock measured here, because the CLI does not print its
own timing and inferring it from file timestamps is not good enough for a claim.

The section-level concurrency of `_run` is mirrored so the same scheduling is under
test; `--section-concurrency 1` reproduces the old serial behaviour.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts"))

import p2g_v2  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--tag", required=True, help="output name suffix")
    parser.add_argument("--section-concurrency", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--out-dir", default="outputs/perf")
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
        document_title=p2g_v2._candidate_document_title(root),
        document_key=p2g_v2._document_key(pdf),
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
        synthesis = await pipeline.synthesize_document(section_results, args.max_tokens)
        finished = time.perf_counter()
        stats = pipeline.stats()
    finally:
        await pipeline.close()

    metrics = p2g_v2.run_evidence_metrics(section_results, synthesis.model_dump())
    out.write_text(json.dumps({
        "sections": section_results,
        "document_synthesis": synthesis.model_dump(),
        "evidence_metrics": {k: v for k, v in metrics.items() if k != "per_answer"},
        "timing": {
            "parse_seconds": round(preprocess_elapsed, 2),
            "extract_seconds": round(extract_elapsed, 2),
            "synthesis_seconds": round(finished - started - extract_elapsed, 2),
            "wall_seconds": round(finished - started, 2),
        },
        "config": {
            "concurrency": args.concurrency,
            "section_concurrency": args.section_concurrency,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    usage = stats.get("usage") or {}
    print(f"\n=== {args.tag}  (API并发={args.concurrency}, 章节并发={args.section_concurrency}) ===")
    print(f"  PDF 解析      : {preprocess_elapsed:8.1f} s")
    print(f"  抽取阶段      : {extract_elapsed:8.1f} s")
    print(f"  归纳+问答     : {finished - started - extract_elapsed:8.1f} s")
    print(f"  总墙钟        : {finished - started:8.1f} s")
    print(f"  调用          : {usage.get('calls')}")
    print(f"  输入 token    : {usage.get('input_tokens'):,}")
    print(f"  输出 token    : {usage.get('output_tokens'):,}")
    print(f"  总 token      : {usage.get('total_tokens'):,}")
    print(f"  facts={metrics['fact_count']} "
          f"with_evidence={metrics['answers_with_evidence']}/10 "
          f"unsupported={metrics['unsupported_claim_rate']:.3f} "
          f"overview_unsup={metrics['overview_unsupported_numbers']}")


asyncio.run(main())
