"""Regression guard: the group-reduce payload carries only fields it reads.

The section_reduce stage already consumed `source_quote`, `confidence`, `grounded`
and `section_id` for every fact. Re-sending them to the hierarchical group stage is
re-reading consumed input: measured at 33% of that payload (68,255 -> 46,010 tokens on
a 48-page paper) and 17% of end-to-end input tokens.

Kept as a switch because it changes what the model sees, so it must stay testable and
revertible: `--full-group-payload` restores the previous behaviour.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python_scripts"))
from p2g_v2 import _GROUP_PAYLOAD_FIELDS, _lean_group_items  # noqa: E402


class LeanGroupPayloadTests(unittest.TestCase):
    def test_fact_fields_are_trimmed_to_what_the_stage_reads(self):
        items = [{
            "title": "Section 1",
            "summary": "s",
            "facts": [{
                "fact_id": "f1", "concept": "c", "statement": "st", "kind": "result",
                "page": 3, "source_quote": "quote", "section_id": "s1",
                "confidence": 0.9, "grounded": True, "section_title": "Section 1",
            }],
        }]

        lean = _lean_group_items(items)
        fact = lean[0]["facts"][0]

        self.assertEqual(
            sorted(_GROUP_PAYLOAD_FIELDS),
            sorted(k for k in fact if k in _GROUP_PAYLOAD_FIELDS),
        )
        for dropped in ("source_quote", "section_id", "confidence", "grounded",
                        "section_title"):
            self.assertNotIn(dropped, fact)

    def test_section_header_fields_are_preserved(self):
        # The header carries the summary/concepts the stage does need.
        items = [{"title": "T", "summary": "sum", "core_concepts": ["a"],
                  "key_findings": ["b"], "facts": []}]

        lean = _lean_group_items(items)

        self.assertEqual("T", lean[0]["title"])
        self.assertEqual("sum", lean[0]["summary"])
        self.assertEqual(["b"], lean[0]["key_findings"])

    def test_upper_level_group_dicts_pass_through_untouched(self):
        # Above level 0 the batch holds GroupSynthesis dicts, not shards with "facts".
        group = {"summary": "g", "concepts": ["x"], "findings": ["y"],
                 "source_fact_ids": ["f1"]}

        self.assertEqual([group], _lean_group_items([group]))

    def test_trimming_does_not_change_the_number_of_facts(self):
        items = [{"title": "T", "facts": [{"fact_id": f"f{i}", "statement": "s",
                                           "source_quote": "q"} for i in range(5)]}]

        lean = _lean_group_items(items)

        self.assertEqual(5, len(lean[0]["facts"]))


if __name__ == "__main__":
    unittest.main()
