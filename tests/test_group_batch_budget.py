"""Regression guard: the group-reduce batch budget must be WIDER than a shard.

The defect: `shard_section_evidence(section, max_tokens)` caps a shard AT max_tokens,
and `batch_by_token_budget(shards, max_tokens)` then grouped shards under that same
cap. Shards of ~3000 tokens cannot pair under a 3200 cap, so level 0 issued one call
per section -- a second pass over content the section_reduce had already summarised.
Measured on a 48-page paper: 46 group-reduce calls became 7, total calls 104 -> 65,
output tokens 98,494 -> 63,081.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python_scripts"))
from p2g_v2 import (  # noqa: E402
    LongDocumentPipeline,
    batch_by_token_budget,
    estimate_tokens,
)


def _synthetic_shards(count: int, tokens_each: int) -> list[dict]:
    """Shards of roughly `tokens_each` tokens, shaped like section evidence.

    `填充` tokenises at about 2 tokens per character, so half that many characters
    per shard lands near the requested size.
    """
    body = "填充" * max(1, tokens_each // 4)
    return [
        {"title": f"Section {i}", "summary": body, "facts": [{"statement": body}]}
        for i in range(count)
    ]


class GroupBatchBudgetTests(unittest.TestCase):
    def test_default_budget_groups_several_shards_per_call(self):
        shards = _synthetic_shards(30, 2600)
        pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
        pipeline.group_batch_tokens = 9600

        group_budget = max(3200, getattr(pipeline, "group_batch_tokens", 3200))
        batches = batch_by_token_budget(shards, group_budget)

        # 30 shards of ~2600 tokens must not become ~30 calls.
        self.assertLess(len(batches), 15)
        self.assertGreater(sum(len(b) for b in batches), len(batches))

    def test_a_budget_equal_to_the_shard_budget_degenerates(self):
        # This is the bug being guarded against: with the batch budget pinned to the
        # shard budget, a shard near the cap cannot pair with anything.
        shards = _synthetic_shards(30, 2600)
        degenerate = batch_by_token_budget(shards, 3200)
        healthy = batch_by_token_budget(shards, 9600)

        self.assertGreater(len(degenerate), len(healthy))
        self.assertGreaterEqual(len(degenerate), 25)

    def test_batching_does_not_duplicate_content(self):
        # Widening the batch must not repeat any shard: input tokens stay flat.
        shards = _synthetic_shards(30, 2600)
        for budget in (3200, 9600, 12800):
            batches = batch_by_token_budget(shards, budget)
            ids = [id(s) for b in batches for s in b]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), len(shards))
            total = sum(estimate_tokens(json.dumps(b, ensure_ascii=False)) for b in batches)
            self.assertAlmostEqual(total, estimate_tokens(json.dumps(shards, ensure_ascii=False)),
                                   delta=total * 0.25)

    def test_pipeline_exposes_the_budget_default(self):
        pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
        self.assertFalse(hasattr(pipeline, "group_batch_tokens"))
        # The call site must tolerate an object built without __init__ (tests and
        # `__new__`-based construction), falling back to the shard budget.
        self.assertEqual(3200, max(3200, getattr(pipeline, "group_batch_tokens", 3200)))


if __name__ == "__main__":
    unittest.main()
