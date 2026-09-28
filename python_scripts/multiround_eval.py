"""Multi-round evaluation: repeat the synthesis phase N times and report variance.

Motivation
----------
Every judgement about this pipeline so far has rested on ONE run per document. A
single sample cannot separate a systematic defect from sampling noise: the same
code produced 4 reported errors on one run and 0 on another for the same paper.
This tool repeats the non-deterministic synthesis stage (section reduce -> group
reduce -> overview -> 10 questions) N times over the SAME extracted facts, so the
only thing varying is the model's own sampling.

Why reusing the extraction layer is legitimate
---------------------------------------------
Fact extraction is LLM work and therefore noisy, but it is noisy *in the same way
every round*. Holding it fixed isolates the variance of the layers that produce
the final answer, which is what any quality conclusion is about. It also cuts the
cost per round roughly in half (no chunk-extraction calls).

What it reports
---------------
* per-round provenance metrics (unsupported rate, evidence coverage, citations)
* mean / stdev / min / max across rounds, so a difference can be judged against
  the spread instead of against a single number
* answers that appear in only some rounds (unstable answers), with the round count
* candidate number-mispairing sentences for manual review. This is a SCREEN, not a
  verdict: a sentence legitimately may combine numbers from several facts, so the
  flagged text must be read by hand.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts"))

from p2g_v2 import (  # noqa: E402
    GLOBAL_QUESTIONS,
    LongDocumentPipeline,
    build_section_evidence,
    run_evidence_metrics,
    shard_section_evidence,
    _build_fact_index,
    _numeric_forms,
    audit_overview_numerics,
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _sentence_numbers(text: str) -> set[str]:
    return _numeric_forms(text)


def screen_mispairings(synthesis: dict, fact_index: dict[str, dict]) -> list[dict]:
    """Sentences that pair *substantive* numbers no single fact carries together.

    A SCREEN for human review, not a decision: combining numbers from different
    facts is normal in a synthesis paragraph. It is the only deterministic handle
    on "number attached to the wrong subject" -- the defect class that reached the
    output in a real run -- so it is worth reading, but it must stay quiet enough
    to read.

    Only numbers with 2+ significant digits count, and a sentence is flagged only
    when it carries at least one number of 3+ digits. Identifier and concentration
    noise such as `RAW264.7`, `P<0.05`, `1x10^7`, `GraphPad Prism`, `4-6 weeks`
    otherwise floods the screen. Measured on a real run, this requirement cut the
    candidate count sharply without losing the case that matters.
    """
    def substantive(value: str) -> set[str]:
        return {
            form for form in _numeric_forms(value)
            if len(re.sub(r"\D", "", form)) >= 2
        }

    fact_forms = []
    for fact in fact_index.values():
        fact_forms.append(substantive(" ".join(
            str(fact.get(k, "")) for k in ("concept", "statement", "source_quote")
        )))
    flagged = []
    fields = [("executive_summary", synthesis.get("executive_summary") or "")]
    for key in ("research_question", "methods", "datasets", "metrics",
                "key_findings", "limitations", "future_work", "contradictions"):
        value = synthesis.get(key)
        if isinstance(value, str):
            fields.append((key, value))
        else:
            fields.extend((key, item) for item in (value or []))
    for key, text in fields:
        for sentence in re.split(r"(?<=[。．！？!?；;])\s*", str(text)):
            sentence = sentence.strip()
            numbers = substantive(sentence)
            if not any(len(re.sub(r"\D", "", n)) >= 3 for n in numbers):
                continue
            if not any(numbers <= forms for forms in fact_forms):
                flagged.append({"field": key, "numbers": sorted(numbers),
                                "sentence": sentence[:200]})
    return flagged


def _summarise(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "stdev": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


async def run_rounds(
    run_json: Path,
    rounds: int,
    concurrency: int,
    reduce_max_tokens: int,
    flash_model: str,
    pro_model: str,
    out_dir: Path,
) -> dict:
    data = json.loads(run_json.read_text(encoding="utf-8"))
    sections = data.get("sections") or []
    if not sections:
        raise ValueError(f"{run_json.name} has no sections")
    label = run_json.stem
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set (checked env and .env)")

    per_round = []
    for rnd in range(1, rounds + 1):
        tmp = Path(tempfile.mkdtemp())
        pipeline = LongDocumentPipeline(
            api_key=api_key,
            memory_path=tmp / "m.sqlite3",
            cache_path=tmp / "c.sqlite3",
            usage_path=tmp / "u.sqlite3",
            flash_model=flash_model,
            pro_model=pro_model,
            concurrency=concurrency,
            document_key=f"eval-{label}-{rnd}",
        )
        pipeline.document_title = (data.get("document_synthesis") or {}).get("title")
        try:
            synthesis = await pipeline.synthesize_document(
                sections, max_tokens=reduce_max_tokens
            )
            dumped = synthesis.model_dump()
            stats = pipeline.stats()
        finally:
            await pipeline.close()
            shutil.rmtree(tmp, ignore_errors=True)

        metrics = run_evidence_metrics(sections, dumped)
        evidence = [build_section_evidence(r) for r in sections]
        index = _build_fact_index(
            [s for sec in evidence for s in shard_section_evidence(sec, 3200)]
        )
        mispair = screen_mispairings(dumped, index)
        record = {
            "round": rnd,
            # `per_answer` is kept on purpose. Without the sentence counts a change in
            # the headline rate cannot be split into "more unsupported sentences" versus
            # "fewer sentences overall", and the latter moves this ratio with no quality
            # change at all (a shorter answer is penalised for being short).
            "metrics": metrics,
            "mispair_candidates": mispair,
            "answers": {a["question"]: a["answer"] for a in dumped["global_answers"]},
            "overview": {
                "executive_summary": dumped.get("executive_summary"),
                "key_findings": dumped.get("key_findings") or [],
            },
            "usage": stats.get("usage"),
        }
        per_round.append(record)
        print(
            f"  [{label}] round {rnd}/{rounds}: "
            f"unsupported_rate={metrics['unsupported_claim_rate']:.3f} "
            f"with_evidence={metrics['answers_with_evidence']}/10 "
            f"cited_median={metrics['median_cited_facts']} "
            f"overview_unsupported={metrics['overview_unsupported_numbers']} "
            f"mispair_screen={len(mispair)}"
        )

    (out_dir / f"{label}_rounds.json").write_text(
        json.dumps(per_round, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"label": label, "rounds": per_round}


def report(results: list[dict], rounds: int) -> None:
    print("\n" + "=" * 78)
    print("多轮评测结果")
    print("=" * 78)
    header = f"{'论文':22s} {'unsup率 均值±σ':>18s} {'范围':>14s} {'有证据':>8s} " \
             f"{'引用中位':>8s} {'概述无据数':>10s} {'待审数':>7s}"
    print(header)
    for item in results:
        ms = [r["metrics"] for r in item["rounds"]]
        rate = _summarise([m["unsupported_claim_rate"] for m in ms])
        wev = _summarise([m["answers_with_evidence"] for m in ms])
        med = _summarise([m["median_cited_facts"] for m in ms])
        ov = _summarise([m["overview_unsupported_numbers"] for m in ms])
        mp = _summarise([len(r["mispair_candidates"]) for r in item["rounds"]])
        print(f"{item['label'][:22]:22s} "
              f"{rate['mean']:>10.3f}±{rate['stdev']:<6.3f} "
              f"{rate['min']:>6.2f}-{rate['max']:<6.2f} "
              f"{wev['mean']:>8.1f} {med['mean']:>8.1f} {ov['mean']:>10.1f} "
              f"{mp['mean']:>7.1f}")

    print("\n" + "=" * 78)
    print("答案稳定性（10 题在各轮之间是否一致）")
    print("=" * 78)
    for item in results:
        rounds_data = item["rounds"]
        n = len(rounds_data)
        print(f"\n### {item['label']}  ({n} 轮)")
        for q in GLOBAL_QUESTIONS:
            texts = [r["answers"].get(q, "") for r in rounds_data]
            unique = {t for t in texts if t}
            stable = "稳定" if len(unique) == 1 else f"**{len(unique)} 种**"
            print(f"  [{stable:>8s}] {q}")
            if len(unique) > 1:
                for t in list(unique)[:3]:
                    print(f"            · {t[:110]}")

    print("\n" + "=" * 78)
    print("数字-主体错配筛查（需人工阅读，不是判定）")
    print("=" * 78)
    for item in results:
        seen = set()
        print(f"\n### {item['label']}")
        for r in item["rounds"]:
            for cand in r["mispair_candidates"]:
                key = cand["sentence"]
                if key in seen:
                    continue
                seen.add(key)
                print(f"  r{r['round']} [{cand['field']}] {cand['numbers']}")
                print(f"      {key}")
        if not seen:
            print("  （无候选）")

    print("\n" + "=" * 78)
    print("总用量")
    print("=" * 78)
    grand_calls = grand_tokens = 0
    for item in results:
        calls = sum((r["usage"] or {}).get("calls", 0) for r in item["rounds"])
        tokens = sum((r["usage"] or {}).get("total_tokens", 0) for r in item["rounds"])
        grand_calls += calls
        grand_tokens += tokens
        print(f"  {item['label']:26s} calls={calls:>4} tokens={tokens:>8}")
    print(f"  {'合计':26s} calls={grand_calls:>4} tokens={grand_tokens:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="stored run output JSON file(s)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--reduce-max-tokens", type=int, default=3200)
    parser.add_argument("--flash-model", default="deepseek-v4-flash")
    parser.add_argument("--pro-model", default="deepseek-v4-pro")
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    _load_env()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.runs]

    results = []
    for path in paths:
        print(f"\n########## {path.name} ##########")
        results.append(asyncio.run(run_rounds(
            path, args.rounds, args.concurrency, args.reduce_max_tokens,
            args.flash_model, args.pro_model, out_dir,
        )))
    report(results, args.rounds)
    print(f"\n逐轮明细已写入 {out_dir}/")


if __name__ == "__main__":
    main()
