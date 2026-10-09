"""Regression guards for the future-work routing.

Measured defect (reported by a reviewer on a maths paper): `future_work` listed work
the document had ALREADY done. Investigating it exposed a second, larger defect: the
future-work ANSWER cited ordinary limitations lifted from the Abstract and
Introduction, and never surfaced what the paper proposed to do next -- even though
those facts were extracted and indexed.

Two mechanisms are guarded here:
  1. `_FORWARD_ITEM_RE` must fire on open/planned phrasings and NOT on the lookalikes
     that real runs produced ("可证明", "可推广", "we expect", "进行进一步研究").
  2. The future-work route's hard kind-exclusion must exempt those facts, because the
     extracted `kind` is unreliable -- the Attention paper's real future directions
     were stored as `kind="method"` and were being dropped before scoring.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python_scripts"))
from p2g_v2 import (  # noqa: E402
    build_question_request,
    GLOBAL_QUESTIONS,
    _FORWARD_ITEM_RE,
    _FORWARD_TITLE_RE,
    _forward_work_boost,
    remove_completed_extension_future_items,
    remove_completed_arctic_curve_future_items,
)

FUTURE_QUESTION = next(q for q in GLOBAL_QUESTIONS if "未来" in q)


class ForwardItemDetectionTests(unittest.TestCase):
    def test_open_and_planned_phrasings_are_recognised(self):
        for text in (
            "开放问题7.1：为命题4.7提供组合证明。",
            "作者将寻找该命题的组合证明留作开放问题。",
            "作者计划将 Transformer 扩展到涉及文本以外的模态。",
            "Open problem 7.3. We define their interactions using a vertex model.",
            "We plan to extend the Transformer to other modalities.",
            "This remains an open question for future work.",
            "m6A 诱导 HCC 骨转移的确切机制尚不清楚。",
            "针对该调控因子的详细研究仍然缺乏，尚待阐明。",
        ):
            self.assertTrue(_FORWARD_ITEM_RE.search(text), text)

    def test_lookalikes_that_real_runs_produced_are_not_recognised(self):
        # Each of these was observed firing on a real run and was NOT a future item.
        for text in (
            "ERF 可视化可证明 MambaIRv2 中提出的非因果建模策略的有效性。",
            "该行解释可推广到一般 k：给定两个元组分拆。",
            "t=0 时红色镶嵌的北极曲线也预期为 Tracy-Widom 涨落。",
            "筛选出14个基因进行进一步研究。",
            "该研究提出注意力状态空间恢复方法，旨在解决 Mamba 的因果性问题。",
            "本文为开放获取文章，遵循 CC BY 4.0 许可。",
            "We evaluate on the open vocabulary segmentation benchmark.",
        ):
            self.assertIsNone(_FORWARD_ITEM_RE.search(text), text)


class CompletedWorkBackcheckTests(unittest.TestCase):
    def test_removes_appendix_curve_step_only_with_same_page_computation(self):
        future_items = [
            "北极曲线的其他部分可类似处理。",
            "一般 t 下涨落的精确分布仍是开放问题。",
        ]
        facts = [
            {
                "kind": "method",
                "section_title": "Appendix A: Proof of Theorem 1.4",
                "statement": "北极曲线的其他部分可类似处理。",
                "source_quote": "The other portions of the arctic curve can be done similarly.",
                "page": 44,
            },
            {
                "kind": "example",
                "section_title": "Appendix A: Proof of Theorem 1.4",
                "statement": "图25展示模拟与计算的北极曲线。",
                "source_quote": "Figure 25: A simulation and computed arctic curve for a hexagon.",
                "page": 44,
            },
        ]

        self.assertEqual(
            ["一般 t 下涨落的精确分布仍是开放问题。"],
            remove_completed_arctic_curve_future_items(future_items, facts),
        )

    def test_keeps_appendix_curve_step_without_same_page_result(self):
        future_items = ["北极曲线的其他部分可类似处理。"]
        facts = [{
            "kind": "method",
            "section_title": "Appendix A",
            "statement": "北极曲线的其他部分可类似处理。",
            "source_quote": "The other portions of the arctic curve can be done similarly.",
            "page": 44,
        }, {
            "kind": "example",
            "section_title": "Appendix A",
            "statement": "另一页展示计算的北极曲线。",
            "source_quote": "Figure 25: A computed arctic curve.",
            "page": 45,
        }]

        self.assertEqual(
            future_items, remove_completed_arctic_curve_future_items(future_items, facts)
        )

    def test_removes_extension_already_documented_later(self):
        future_items = [
            "将把这两种模型推广到k-铺砖，并在k=1时恢复命题2.2的结果。",
            "寻找命题4.7的组合证明。",
        ]
        facts = [{
            "fact_id": "done-k-extension",
            "kind": "method",
            "section_title": "2.4 Extending the models to k-tilings",
            "statement": "前节的两个模型可以扩展到 k-tilings。",
            "source_quote": "The two models in the previous section can be extended to k-tilings.",
        }]

        filtered = remove_completed_extension_future_items(future_items, facts)

        self.assertEqual(["寻找命题4.7的组合证明。"], filtered)

    def test_keeps_plan_when_document_has_no_completion_evidence(self):
        future_items = ["We will generalize these two models to k-tilings."]

        self.assertEqual(
            future_items,
            remove_completed_extension_future_items(future_items, []),
        )

    def test_does_not_remove_extension_without_completed_evidence(self):
        future_items = ["作者计划将两种模型推广到k-铺砖。"]
        facts = [{
            "kind": "limitation",
            "section_title": "7 Open Problems",
            "statement": "k-tilings 尚未推广到此情形。",
            "source_quote": "We plan to extend the models to k-tilings.",
        }]

        filtered = remove_completed_extension_future_items(future_items, facts)

        self.assertEqual(future_items, filtered)


class ForwardWorkBoostTests(unittest.TestCase):
    def test_open_item_in_a_forward_section_is_boosted_hardest(self):
        fact = {"statement": "开放问题7.2：计算北极曲线是困难的开放问题。",
                "source_quote": "Open problem 7.2.", "section_title": "7 Conclusion and open questions"}
        other = {**fact, "section_title": "4.5 Combining the two models"}

        self.assertEqual(70, _forward_work_boost(fact))
        # A mid-paper "we leave this for later" is scoping, not future work.
        self.assertEqual(25, _forward_work_boost(other))

    def test_ordinary_limitations_get_no_boost(self):
        fact = {"statement": "这种固有的顺序性阻碍了训练样本内的并行化。",
                "source_quote": "This inherently sequential nature precludes parallelization.",
                "section_title": "1 Introduction"}
        self.assertEqual(0, _forward_work_boost(fact))


class FutureWorkRouteTests(unittest.TestCase):
    def _facts(self, kinds_and_statements):
        facts = []
        for i, (kind, statement, section) in enumerate(kinds_and_statements):
            facts.append({
                "fact_id": f"f{i}", "concept": "c", "statement": statement,
                "kind": kind, "section_title": section,
                "source_quote": statement, "confidence": 0.9,
            })
        return facts

    def test_a_method_kind_future_direction_beats_ordinary_limitations(self):
        # The exact shape of the Attention defect: the genuine direction is stored as
        # `method`, and the hard kind-exclusion used to drop it outright.
        facts = self._facts([
            ("method", "作者计划将 Transformer 扩展到涉及文本以外的输入和输出模态的问题。",
             "7 Conclusion"),
            ("method", "作者计划研究局部受限的注意力机制，以高效处理图像、音频和视频。",
             "7 Conclusion"),
            ("limitation", "这种固有的顺序性阻碍了训练样本内的并行化。", "1 Introduction"),
            ("limitation", "顺序计算的根本约束仍然存在。", "1 Introduction"),
            ("limitation", "操作数随距离增长使得学习远距离依赖更加困难。", "2 Background"),
        ])

        payload, allowed = build_question_request(
            FUTURE_QUESTION, facts, [], max_tokens=600)

        self.assertIn("f0", allowed)
        self.assertIn("f1", allowed)

    def test_the_exemption_does_not_readmit_implementation_details(self):
        # A method fact with no open/planned phrasing must stay excluded.
        facts = self._facts([
            ("limitation", "队列样本量需要增加。", "Conclusion"),
            ("method", "使用自注意力替换循环层，编码器由 6 层堆叠组成。", "3 Model Architecture"),
        ])

        _, allowed = build_question_request(FUTURE_QUESTION, facts, [], max_tokens=600)

        self.assertIn("f0", allowed)
        self.assertNotIn("f1", allowed)


if __name__ == "__main__":
    unittest.main()
