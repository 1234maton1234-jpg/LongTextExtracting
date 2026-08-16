import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python_scripts"))

from p2g_v2 import (  # noqa: E402
    HeadingNode,
    ConceptMemory,
    ChunkExtraction,
    DocumentSynthesis,
    GLOBAL_QUESTIONS,
    Fact,
    GlobalAnswer,
    GroupSynthesis,
    LongDocumentPipeline,
    UsageTracker,
    PageMarkdown,
    _apply_answer_provenance,
    _absorb_noise_nodes,
    _build_fact_index,
    _apply_grounding,
    _apply_group_provenance,
    _audit_answer_claims,
    _build_contradiction_answer,
    _finalize_answer_provenance,
    _match_contradiction_fact_ids,
    _merge_overview_contradictions,
    _merge_table_headers,
    _normalize_levels,
    _repair_header_name,
    _row_is_config_row,
    _recover_answer_fact_ids,
    _strip_english_leak,
    _table_rows,
    build_heading_tree,
    _merge_low_signal_sections,
    _merge_fragment_leaves,
    _should_merge_fragment_leaf,
    build_document_request,
    build_question_request,
    cache_key,
    detect_document_contradictions,
    detect_numeric_contradictions,
    deduplicate_sections,
    estimate_tokens,
    extract_table_facts,
    fallback_document_overview,
    fallback_group_synthesis,
    flatten_sections,
    is_back_matter,
    _is_toc_section,
    _noise_node_reason,
    _strip_picture_text,
    shard_section_evidence,
    split_text,
    _quote_is_grounded,
    _grounded_payload,
    batch_by_token_budget,
    build_section_evidence,
)


class HeadingTreeTests(unittest.TestCase):
    def test_builds_nested_tree_and_preserves_pages(self):
        pages = [
            PageMarkdown(page=1, text="# 第一章\n章节导语\n## 第一节\n第一节正文"),
            PageMarkdown(page=2, text="### 细节\n细节正文"),
        ]

        root = build_heading_tree(pages)
        sections = {node.title: node for node in flatten_sections(root)}

        self.assertEqual(sections["第一章"].level, 1)
        self.assertEqual(sections["第一节"].parent_id, sections["第一章"].id)
        self.assertEqual(sections["细节"].parent_id, sections["第一节"].id)
        self.assertEqual(sections["细节"].page_start, 2)
        self.assertIn("细节正文", sections["细节"].text)
        self.assertEqual(sections["第一节"].page_end, 1)
        self.assertEqual(sections["第一节"].subtree_page_end, 2)

    def test_preserves_page_fragments_for_exact_fact_pages(self):
        pages = [
            PageMarkdown(page=7, text="# 结果\n第一页结果"),
            PageMarkdown(page=8, text="第二页结果为 99.37%。"),
        ]

        section = flatten_sections(build_heading_tree(pages))[0]
        fact = Fact(
            concept="准确率",
            statement="第二页结果为 99.37%。",
            kind="result",
            source_quote="第二页结果为 99.37%。",
            page=99,
            confidence=0.9,
        )
        result = _apply_grounding(ChunkExtraction(summary="", facts=[fact]), section.text, section.page_texts)

        self.assertEqual([(item.page, item.text) for item in section.page_texts], [(7, "第一页结果"), (8, "第二页结果为 99.37%。")])
        self.assertTrue(result.facts[0].grounded)
        self.assertEqual(result.facts[0].page, 8)

    def test_rejects_formula_table_and_sentence_lines_as_headings(self):
        pages = [
            PageMarkdown(
                page=1,
                text=(
                    "# 1 Introduction\n正文\n"
                    "###### TP Precision = (6) TP + FP\n公式说明\n"
                    "###### 表6 城市声音数据集结果\n表格说明\n"
                    "###### 表�7（续）\n续表说明\n"
                    "###### 这是一个被误判成标题的完整句子。\n句子说明\n"
                    "# 2 Method\n方法正文"
                ),
            )
        ]

        sections = flatten_sections(build_heading_tree(pages))

        self.assertEqual([section.title for section in sections], ["1 Introduction", "2 Method"])
        self.assertIn("TP Precision", sections[0].text)
        self.assertIn("表6", sections[0].text)
        self.assertIn("表�7", sections[0].text)

    def test_rejects_generic_prompt_labels_as_headings(self):
        pages = [
            PageMarkdown(
                page=1,
                text=(
                    "# 1 Introduction\n正文\n"
                    "###### Characters\n角色说明\n"
                    "###### Timeline\n时间线说明\n"
                    "###### Notes\n备注说明\n"
                    "###### Output Format:\n格式说明\n"
                    "###### Weekly Planning\n周计划说明\n"
                    "# 2 Method\n方法正文"
                ),
            )
        ]

        sections = flatten_sections(build_heading_tree(pages))

        self.assertEqual([section.title for section in sections], ["1 Introduction", "2 Method"])
        self.assertIn("Characters", sections[0].text)
        self.assertIn("Timeline", sections[0].text)
        self.assertIn("Output Format:", sections[0].text)
        self.assertIn("Weekly Planning", sections[0].text)

    def test_merges_low_signal_child_sections_into_parent(self):
        root = HeadingNode(id="root", title="Document", level=0, page_start=1, page_end=1, subtree_page_end=1)
        parent = HeadingNode(id="parent", title="System Prompts", level=2, page_start=1, page_end=1, subtree_page_end=1)
        root.children.append(parent)
        child = HeadingNode(id="child", title="Characters", level=3, page_start=1, page_end=1, subtree_page_end=1)
        child.text = "角色说明"
        child.page_texts = [PageMarkdown(page=1, text="角色说明")]
        parent.children.append(child)

        _merge_low_signal_sections(root)

        self.assertEqual(parent.children, [])
        self.assertIn("角色说明", parent.text)

    def test_merges_small_numbered_leaf_into_parent(self):
        root = HeadingNode(id="root", title="Document", level=0, page_start=1, page_end=1, subtree_page_end=1)
        parent = HeadingNode(id="s91", title="9.1 Evaluation", level=3, page_start=1, page_end=1, subtree_page_end=1)
        root.children.append(parent)
        child = HeadingNode(id="s912", title="9.1.2 Task", level=4, page_start=2, page_end=2, subtree_page_end=2)
        child.text = "该子节描述一个具体任务。"
        child.page_texts = [PageMarkdown(page=2, text="该子节描述一个具体任务。")]
        parent.children.append(child)

        _merge_fragment_leaves(root)

        self.assertEqual(parent.children, [])
        self.assertIn("#### 9.1.2 Task", parent.text)
        self.assertIn("该子节描述一个具体任务。", parent.text)
        self.assertEqual(parent.page_end, 2)

    def test_keeps_main_numbered_section(self):
        root = HeadingNode(id="root", title="Document", level=0, page_start=1, page_end=1, subtree_page_end=1)
        parent = HeadingNode(id="p", title="Back Matter", level=2, page_start=1, page_end=1, subtree_page_end=1)
        root.children.append(parent)
        main = HeadingNode(id="c6", title="6 Conclusion", level=3, page_start=1, page_end=1, subtree_page_end=1)
        main.text = "结论内容。"
        main.page_texts = [PageMarkdown(page=1, text="结论内容。")]
        parent.children.append(main)

        self.assertFalse(_should_merge_fragment_leaf(main))  # number:6, no sub-number
        _merge_fragment_leaves(root)
        self.assertEqual(parent.children, [main])

    def test_keeps_large_numbered_leaf(self):
        root = HeadingNode(id="root", title="Document", level=0, page_start=1, page_end=1, subtree_page_end=1)
        parent = HeadingNode(id="s91", title="9.1 Evaluation", level=3, page_start=1, page_end=1, subtree_page_end=1)
        root.children.append(parent)
        body = "这是一个内容充实的子节，包含大量细节描述。" * 45
        child = HeadingNode(id="s912", title="9.1.2 Task", level=4, page_start=2, page_end=2, subtree_page_end=2)
        child.text = body
        child.page_texts = [PageMarkdown(page=2, text=body)]
        parent.children.append(child)

        self.assertGreater(estimate_tokens(body), 500)
        _merge_fragment_leaves(root)
        self.assertEqual(parent.children, [child])

    def test_keeps_section_with_children_and_unnumbered_leaf(self):
        root = HeadingNode(id="root", title="Document", level=0, page_start=1, page_end=1, subtree_page_end=1)
        parent = HeadingNode(id="s91", title="9.1 Evaluation", level=3, page_start=1, page_end=1, subtree_page_end=1)
        root.children.append(parent)
        body = "这是一个内容充实的子节，包含大量细节描述。" * 45
        child = HeadingNode(id="s911", title="9.1.1 Long Task", level=4, page_start=2, page_end=2, subtree_page_end=2)
        child.text = body
        child.page_texts = [PageMarkdown(page=2, text=body)]
        parent.children.append(child)
        unnumbered = HeadingNode(id="rw", title="Related Work", level=3, page_start=3, page_end=3, subtree_page_end=3)
        unnumbered.text = "相关工作介绍。"
        unnumbered.page_texts = [PageMarkdown(page=3, text="相关工作介绍。")]
        parent.children.append(unnumbered)

        _merge_fragment_leaves(root)

        # 9.1 itself has children -> kept; 9.1.1 Long Task too large -> kept;
        # Related Work has no numeric signature -> kept.
        self.assertEqual([c.title for c in parent.children], ["9.1.1 Long Task", "Related Work"])

    def test_keeps_low_level_numbered_leaf(self):
        root = HeadingNode(id="root", title="Document", level=0, page_start=1, page_end=1, subtree_page_end=1)
        parent = HeadingNode(id="s6", title="6 Evaluation", level=2, page_start=1, page_end=1, subtree_page_end=1)
        root.children.append(parent)
        sub = HeadingNode(id="s61", title="6.1 Data", level=2, page_start=1, page_end=1, subtree_page_end=1)
        sub.text = "数据说明。"
        sub.page_texts = [PageMarkdown(page=1, text="数据说明。")]
        parent.children.append(sub)

        self.assertFalse(_should_merge_fragment_leaf(sub))  # level <= 2
        _merge_fragment_leaves(root)
        self.assertEqual(parent.children, [sub])

    def test_merge_fragment_leaves_via_build_heading_tree(self):
        pages = [
            PageMarkdown(
                page=1,
                text=(
                    "# 1 Introduction\n导言内容。\n"
                    "## 1.1 Related Work\n相关工作介绍。\n"
                    "### 1.1.1 Task\n" + ("这是子节的内容。" * 30) + "\n"
                    "# 2 Conclusion\n总结。"
                ),
            )
        ]
        sections = flatten_sections(build_heading_tree(pages))
        self.assertEqual(
            [section.title for section in sections],
            ["1 Introduction", "1.1 Related Work", "2 Conclusion"],
        )
        related = sections[1]
        self.assertIn("### 1.1.1 Task", related.text)
        self.assertIn("这是子节的内容。", related.text)

    def test_rejects_sentence_like_and_case_style_headings(self):
        pages = [
            PageMarkdown(
                page=1,
                text=(
                    "# 1 Introduction\n正文\n"
                    "###### This is a full sentence.\n句子说明\n"
                    "###### Case 1: A case study title\n案例说明\n"
                    "###### Impact Statement\n影响说明\n"
                    "###### System Prompts\n提示词说明\n"
                    "# 2 Method\n方法正文"
                ),
            )
        ]

        sections = flatten_sections(build_heading_tree(pages))

        self.assertEqual([section.title for section in sections], ["1 Introduction", "2 Method"])
        self.assertIn("This is a full sentence.", sections[0].text)
        self.assertIn("Case 1: A case study title", sections[0].text)
        self.assertIn("Impact Statement", sections[0].text)
        self.assertIn("System Prompts", sections[0].text)

    def test_rejects_appendix_and_case_subheadings(self):
        pages = [
            PageMarkdown(
                page=1,
                text=(
                    "# 1 Introduction\n正文\n"
                    "###### A.1. Pipeline Overview\n附录说明\n"
                    "###### E.1. Behavioral Cases\n案例说明\n"
                    "# 2 Method\n方法正文"
                ),
            )
        ]

        sections = flatten_sections(build_heading_tree(pages))

        self.assertEqual([section.title for section in sections], ["1 Introduction", "2 Method"])
        self.assertIn("A.1. Pipeline Overview", sections[0].text)
        self.assertIn("E.1. Behavioral Cases", sections[0].text)

    def test_rejects_publication_running_headers(self):
        pages = [
            PageMarkdown(
                page=1,
                text="# 1 Introduction\n正文\n###### Discover Internet of Things\n页眉\n###### 发现物联网\n页眉中文",
            )
        ]

        sections = flatten_sections(build_heading_tree(pages))

        self.assertEqual([section.title for section in sections], ["1 Introduction"])
        self.assertIn("Discover Internet of Things", sections[0].text)

    def test_merges_numbered_bilingual_duplicate_sections(self):
        pages = [
            PageMarkdown(
                page=1,
                text="# 1 Introduction\nEnglish body\n# 1�引言\n中文正文\n# 2 Method\nMethod body",
            )
        ]

        sections = deduplicate_sections(flatten_sections(build_heading_tree(pages)))

        self.assertEqual([section.title for section in sections], ["1 Introduction", "2 Method"])
        self.assertIn("English body", sections[0].text)
        self.assertIn("中文正文", sections[0].text)


class TokenChunkTests(unittest.TestCase):
    def test_splits_single_long_paragraph_with_bounded_overlap(self):
        text = "机器学习用于音频分类。" * 180
        chunks = split_text(text, max_tokens=120, overlap_tokens=20)

        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(estimate_tokens(chunk.text) <= 120 for chunk in chunks))
        self.assertEqual(chunks[0].overlap_text, "")
        self.assertTrue(all(chunk.overlap_text for chunk in chunks[1:]))
        self.assertTrue(all(chunk.text.startswith(chunk.overlap_text) for chunk in chunks[1:]))


class ConceptMemoryTests(unittest.TestCase):
    def test_merges_acronyms_and_case_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ConceptMemory(Path(temp_dir) / "memory.sqlite3")
            mcc_long = memory.upsert("马修斯相关系数 (MCC)")
            mcc_short = memory.upsert("MCC")
            urban_mixed = memory.upsert("UrbanSound8K")
            urban_lower = memory.upsert("urbansound8k")
            memory.upsert("支持向量机")

            self.assertEqual(mcc_long, mcc_short)
            self.assertEqual(urban_mixed, urban_lower)
            self.assertEqual(memory.count(), 3)
            self.assertEqual(len(memory.retrieve("支持 向量机适合分类")), 1)
            memory.close()


class UtilityTests(unittest.TestCase):
    def test_routes_intermediate_document_reduce_to_flash(self):
        pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
        pipeline.flash_model = "flash"
        pipeline.pro_model = "pro"

        self.assertEqual(pipeline._group_reduce_model(), "flash")
        self.assertEqual(pipeline._document_reduce_model(), "flash")
        self.assertEqual(pipeline._global_answer_model(), "flash")

    def test_group_fallback_preserves_grounded_facts_without_new_claims(self):
        items = [
            {
                "summary": "章节摘要",
                "core_concepts": ["MFCC"],
                "key_findings": ["ANN准确率更高"],
                "facts": [
                    {
                        "fact_id": "fact-a",
                        "concept": "ANN",
                        "statement": "ANN总体准确率为91.41%",
                        "kind": "result",
                        "page": 20,
                    }
                ],
            }
        ]

        result = fallback_group_synthesis(items)

        self.assertIn("章节摘要", result.summary)
        self.assertIn("fact-a", result.source_fact_ids)
        self.assertIn("ANN总体准确率为91.41%", result.numeric_claims)

    def test_extracts_structured_numeric_facts_from_markdown_table(self):
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=16,
                        text=(
                            "# 8 Discussion\n**Table 6** Results\n"
                            "|Classifier model|Audio class|Accuracy (%)|Precision (%)|\n"
                            "|---|---|---|---|\n"
                            "|ANN|Air conditioner|98.73|95.94|\n"
                            "||Car horn|**99.31**|87.84|\n"
                            "|Support Vector Machine (rbf)|Air conditioner|97.08|89.34|\n"
                            "||Car horn|**99.37**|87.84|"
                        ),
                    )
                ]
            )
        )[0]

        facts = extract_table_facts(section)
        svm_accuracy = next(
            fact
            for fact in facts
            if fact.metric == "Accuracy (%)"
            and fact.row_context.get("Classifier model") == "Support Vector Machine (rbf)"
            and fact.row_context.get("Audio class") == "Car horn"
        )

        self.assertEqual(svm_accuracy.value, 99.37)
        self.assertEqual(svm_accuracy.unit, "%")
        self.assertEqual(svm_accuracy.page, 16)
        self.assertTrue(svm_accuracy.bold)
        self.assertTrue(svm_accuracy.fact_id.startswith("fact-"))

    def test_detects_universal_maximum_claim_contradicted_by_table(self):
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=16,
                        text=(
                            "# 8 Discussion\n**Table 6** Results\n"
                            "|Classifier model|Audio class|Accuracy (%)|\n"
                            "|---|---|---|\n"
                            "|ANN|Air conditioner|98.73|\n"
                            "||Car horn|99.31|\n"
                            "|Support Vector Machine (rbf)|Air conditioner|97.08|\n"
                            "||Car horn|99.37|\n"
                            "Similarly, the KNN model performs poorly for another class. "
                            "The Artificial Neural Network model has achieved maximum accuracy for all the audio sample classes."
                        ),
                    )
                ]
            )
        )[0]

        contradictions = detect_numeric_contradictions(section, extract_table_facts(section))

        self.assertEqual(len(contradictions), 1)
        self.assertIn("ANN=99.31", contradictions[0].statement)
        self.assertIn("99.31", contradictions[0].statement)
        self.assertIn("99.37", contradictions[0].statement)
        self.assertGreaterEqual(len(contradictions[0].source_fact_ids), 2)

    def test_document_synthesis_normalizes_scalar_list_fields(self):
        payload = {
            "title": "文档",
            "executive_summary": "摘要",
            "research_question": "问题",
            "methods": "方法",
            "datasets": "数据集",
            "metrics": "指标",
            "key_findings": "结论",
            "limitations": "局限",
            "future_work": "未来工作",
            "contradictions": "未发现矛盾",
            "global_answers": [
                {
                    "question": question,
                    "answer": "证据不足",
                    "source_fact_ids": [],
                    "source_sections": [],
                    "source_pages": [],
                    "confidence": 0.0,
                }
                for question in GLOBAL_QUESTIONS
            ],
        }

        result = DocumentSynthesis.model_validate(payload)

        self.assertEqual(result.methods, ["方法"])
        self.assertEqual(result.contradictions, ["未发现矛盾"])

    def test_batches_evidence_within_budget(self):
        items = [{"text": "知识点" * 20, "index": index} for index in range(6)]
        batches = batch_by_token_budget(items, max_tokens=100)
        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(len(batch) for batch in batches), len(items))
        self.assertTrue(
            all(estimate_tokens(__import__("json").dumps(batch, ensure_ascii=False)) <= 100 for batch in batches)
        )

    def test_shards_oversized_section_evidence_within_budget(self):
        evidence = {
            "section_id": "s1",
            "title": "大表格章节",
            "pages": [1, 8],
            "summary": "摘要",
            "core_concepts": ["指标"],
            "key_findings": ["结论"],
            "facts": [
                {
                    "fact_id": f"fact-{index}",
                    "statement": f"模型{index}的准确率为{90 + index / 100:.2f}%",
                    "source_quote": f"|模型{index}|{90 + index / 100:.2f}|",
                    "kind": "metric",
                    "page": 2,
                    "confidence": 1.0,
                    "grounded": True,
                    "section_id": "s1",
                }
                for index in range(100)
            ],
        }

        shards = shard_section_evidence(evidence, max_tokens=500)

        self.assertGreater(len(shards), 1)
        self.assertEqual(sum(len(shard["facts"]) for shard in shards), 100)
        self.assertTrue(
            all(estimate_tokens(json.dumps(shard, ensure_ascii=False)) <= 500 for shard in shards)
        )

    def test_final_document_request_is_budgeted_and_keeps_required_evidence(self):
        groups = [
            GroupSynthesis(
                summary="摘要" * 100,
                concepts=[f"概念{i}" for i in range(20)],
                findings=[f"结论{i}" for i in range(20)],
                numeric_claims=[f"数值{i}=99%" for i in range(20)],
                limitations=["局限"] * 10,
                future_work=["未来"] * 10,
                contradictions=[],
                source_fact_ids=[f"fact-{i}" for i in range(100)],
                source_sections=[],
                source_pages=[],
            )
        ]
        catalog = [
            {
                "fact_id": f"fact-{i}",
                "statement": f"事实{i}" * 20,
                "kind": "result",
            }
            for i in range(100)
        ]

        payload, allowed = build_document_request(
            groups,
            catalog,
            [{"index": 1, "question": "问题"}],
            [{"statement": "矛盾", "source_fact_ids": ["fact-98", "fact-99"]}],
            max_tokens=1000,
        )

        self.assertLessEqual(estimate_tokens(json.dumps(payload, ensure_ascii=False)), 1000)
        self.assertTrue({"fact-98", "fact-99"} <= allowed)

    def test_document_request_degrades_gracefully_when_budget_impossible(self):
        # A rich document whose evidence alone exceeds a tiny budget must shrink
        # evidence and still return a request — never raise and kill the run.
        groups = [
            GroupSynthesis(
                summary="摘要" * 80,
                concepts=[f"概念{i}" for i in range(30)],
                findings=[f"关键结论{i}：这是一条非常长的中文结论用于占据更多预算。{i}" for i in range(30)],
                numeric_claims=[f"数值{i}=99%且描述很长" for i in range(30)],
                limitations=[f"局限{i}：详细描述" for i in range(20)],
                future_work=[f"未来工作{i}：详细描述" for i in range(20)],
                contradictions=[],
                source_fact_ids=[f"fact-{i}" for i in range(60)],
                source_sections=[],
                source_pages=[],
            )
            for _ in range(6)
        ]
        catalog = [
            {
                "fact_id": f"fact-{i}",
                "statement": f"事实{i}" * 30,
                "kind": "result",
            }
            for i in range(60)
        ]

        payload, allowed = build_document_request(
            groups,
            catalog,
            [{"index": 1, "question": "问题"}],
            [{"statement": "矛盾", "source_fact_ids": ["fact-58", "fact-59"]}],
            max_tokens=600,
        )

        self.assertTrue({"fact-58", "fact-59"} <= allowed)
        self.assertGreater(len(payload["fact_catalog"]), 0)

    def test_question_request_retrieves_relevant_fact_kinds_within_budget(self):
        facts = [
            {"fact_id": "fact-dataset", "statement": "使用UrbanSound8K数据集", "kind": "dataset"},
            {"fact_id": "fact-method", "statement": "采用MFCC特征", "kind": "method"},
            {"fact_id": "fact-metric", "statement": "评价指标包括准确率和F1", "kind": "metric"},
        ]

        payload, allowed = build_question_request(
            "文档使用了哪些数据集或材料？", facts, [], max_tokens=600
        )

        self.assertIn("fact-dataset", allowed)
        self.assertLessEqual(estimate_tokens(json.dumps(payload, ensure_ascii=False)), 600)

    def test_quantitative_request_prefers_current_results_over_related_work(self):
        facts = [
            {
                "fact_id": "fact-related",
                "statement": "QGC分类器的错误率为13.6%",
                "kind": "result",
                "section_title": "2 Related work",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-current",
                "statement": "ANN在两个数据集上的准确率分别为91.41%和91.27%，优于其他模型",
                "kind": "result",
                "section_title": "Abstract",
                "confidence": 0.9,
            },
            {
                "fact_id": "fact-table-detail",
                "statement": "Table 6: SVM在Car horn类别的Accuracy为99.37%",
                "kind": "result",
                "section_title": "7 Results",
                "confidence": 1.0,
            },
        ]

        payload, _ = build_question_request("最重要的定量结果是什么？", facts, [], max_tokens=600)

        self.assertEqual(payload["fact_catalog"][0]["fact_id"], "fact-current")
        self.assertNotIn("fact-related", {fact["fact_id"] for fact in payload["fact_catalog"]})

    def test_metric_question_skips_config_table_rows(self):
        # A config/parameter table row (generic "Value" column, kind="metric")
        # must never outrank genuine metric facts for "使用了哪些评价指标？".
        facts = [
            {
                "fact_id": "fact-config-row",
                "concept": "Value",
                "statement": "Table 8: Key configuration parameters used in the simulation.: Parameter=maxpossessions; Description=Max items an agent can hold before discarding; Value=50",
                "kind": "metric",
                "section_title": "C. Experiment Settings",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-real-metric",
                "concept": "RoleEval",
                "statement": "RoleEval从知识角度客观评估LLM。",
                "kind": "metric",
                "section_title": "C.3 Metrics Definition",
                "confidence": 0.95,
            },
            {
                "fact_id": "fact-metric-def",
                "concept": "Fulfillment Metrics",
                "statement": "满足度指标包括四个维度的年度快照：fulfillment_mood、fulfillment_material等。",
                "kind": "definition",
                "section_title": "C.3 Metrics Definition",
                "confidence": 0.9,
            },
        ]

        payload, allowed = build_question_request("使用了哪些评价指标？", facts, [], max_tokens=600)

        self.assertIn("fact-real-metric", allowed)
        self.assertIn("fact-metric-def", allowed)
        self.assertNotIn("fact-config-row", allowed)

    def test_config_table_row_penalized_for_quantitative_question_too(self):
        # Config values are not quantitative "results" either.
        config = {
            "fact_id": "fact-config",
            "concept": "Value",
            "statement": "Table 8: Key configuration parameters used in the simulation.: Parameter=maxpossessions; Description=Max items an agent can hold before discarding; Value=50",
            "kind": "metric",
            "section_title": "C. Experiment Settings",
            "confidence": 1.0,
        }

        payload, allowed = build_question_request(
            "最重要的定量结果是什么？", [config], [], max_tokens=600
        )

        self.assertNotIn("fact-config", allowed)

    def test_correlation_table_row_kept_for_quantitative_not_for_metric(self):
        # A raw correlation-table row is weak evidence for "which metrics were
        # used" but stays available for the quantitative question.
        row = {
            "fact_id": "fact-corr-row",
            "concept": "#",
            "statement": "Table 9: Top-10 metrics correlated with social reward, sorted by mean Pearson's |r|.: #=1; Arcane Academy=0.42",
            "kind": "metric",
            "section_title": "5.2 Analysis",
            "confidence": 1.0,
        }

        _, metric_allowed = build_question_request(
            "使用了哪些评价指标？", [row], [], max_tokens=600
        )
        self.assertNotIn("fact-corr-row", metric_allowed)

        _, quant_allowed = build_question_request(
            "最重要的定量结果是什么？", [row], [], max_tokens=600
        )
        self.assertIn("fact-corr-row", quant_allowed)

    def test_fact_index_preserves_section_title_for_retrieval(self):
        index = _build_fact_index(
            [
                {
                    "title": "7 Results",
                    "facts": [{"fact_id": "fact-result", "statement": "结果"}],
                }
            ]
        )

        self.assertEqual(index["fact-result"]["section_title"], "7 Results")

    def test_answer_fact_ids_recover_from_directly_supporting_allowlist_fact(self):
        answer = GlobalAnswer(
            question="最重要的定量结果是什么？",
            answer="ANN在两个数据集上的准确率分别为91.41%和91.27%。",
            source_fact_ids=["fact-forged"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        candidates = [
            {
                "fact_id": "fact-current",
                "statement": "ANN在两个数据集上的准确率分别为91.41%和91.27%。",
            },
            {"fact_id": "fact-related", "statement": "QGC分类器的错误率为13.6%。"},
        ]
        fact_index = {
            fact["fact_id"]: {
                **fact,
                "source_quote": fact["statement"],
                "section_id": "s1",
                "page": 1,
                "confidence": 1.0,
            }
            for fact in candidates
        }

        recovered = _recover_answer_fact_ids(answer, candidates, fact_index)

        self.assertEqual(recovered, ["fact-current"])

    def test_answer_fact_id_recovery_rejects_mismatched_numbers(self):
        answer = GlobalAnswer(
            question="最重要的定量结果是什么？",
            answer="ANN准确率为91.41%。",
            source_fact_ids=[],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        candidates = [{"fact_id": "fact-related", "statement": "QGC错误率为13.6%。"}]
        fact_index = {
            "fact-related": {
                **candidates[0],
                "source_quote": candidates[0]["statement"],
                "section_id": "s1",
                "page": 1,
                "confidence": 1.0,
            }
        }

        self.assertEqual(_recover_answer_fact_ids(answer, candidates, fact_index), [])

    def test_answer_fact_id_recovery_rejects_related_work_for_quantitative_questions(self):
        answer = GlobalAnswer(
            question="最重要的定量结果是什么？",
            answer="ANN准确率为91.41%。",
            source_fact_ids=[],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        candidates = [
            {
                "fact_id": "fact-related",
                "statement": "ANN准确率为91.41%。",
                "kind": "result",
                "section_title": "2 Related work",
            }
        ]
        fact_index = {
            "fact-related": {
                **candidates[0],
                "source_quote": candidates[0]["statement"],
                "section_id": "s1",
                "page": 1,
                "confidence": 1.0,
            }
        }

        self.assertEqual(_recover_answer_fact_ids(answer, candidates, fact_index), [])

    def test_answer_falls_back_to_grounded_candidate_when_paraphrase_is_unsupported(self):
        answer = GlobalAnswer(
            question="文档有哪些局限？",
            answer="模型补充了证据中不存在的局限。",
            source_fact_ids=["fact-forged"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        candidates = [{"fact_id": "fact-limit", "statement": "环境声音随机且难以分类。"}]
        fact_index = {
            "fact-limit": {
                **candidates[0],
                "source_quote": "环境声音随机且难以分类。",
                "section_id": "s1",
                "page": 1,
                "confidence": 0.8,
            }
        }

        result = _finalize_answer_provenance(
            answer, candidates, fact_index, {"fact-limit"}
        )

        self.assertEqual(result.answer, "环境声音随机且难以分类。")
        self.assertEqual(result.source_fact_ids, ["fact-limit"])
        self.assertEqual(result.confidence, 0.8)

    def test_section_evidence_only_contains_grounded_facts(self):
        result = {
            "section": {"id": "s1", "title": "章节", "page_start": 1, "page_end": 2},
            "extractions": [
                {
                    "summary": "摘要",
                    "facts": [
                        {"statement": "可信", "grounded": True},
                        {"statement": "不可信", "grounded": False},
                    ],
                }
            ],
            "synthesis": {"summary": "章节摘要", "core_concepts": ["概念"], "key_findings": ["结论"]},
        }
        evidence = build_section_evidence(result)
        self.assertEqual([fact["statement"] for fact in evidence["facts"]], ["可信"])
        self.assertTrue(evidence["facts"][0]["fact_id"].startswith("fact-"))
        self.assertEqual(evidence["facts"][0]["section_id"], "s1")

    def test_answer_sources_are_derived_from_valid_fact_ids(self):
        answer = GlobalAnswer(
            question="问题",
            answer="答案",
            source_fact_ids=["fact-valid", "fact-forged"],
            source_sections=["wrong-section"],
            source_pages=[99],
            confidence=1.0,
        )
        fact_index = {
            "fact-valid": {
                "section_id": "section-1",
                "page": 3,
                "confidence": 0.82,
            }
        }

        _apply_answer_provenance(answer, fact_index, {"fact-valid"})

        self.assertEqual(answer.source_fact_ids, ["fact-valid"])
        self.assertEqual(answer.source_sections, ["section-1"])
        self.assertEqual(answer.source_pages, [3])
        self.assertEqual(answer.confidence, 0.82)

    def test_group_provenance_falls_back_to_allowed_facts_when_model_returns_none(self):
        result = GroupSynthesis(
            summary="摘要",
            concepts=[],
            findings=[],
            numeric_claims=[],
            limitations=[],
            future_work=[],
            contradictions=[],
            source_fact_ids=[],
            source_sections=[],
            source_pages=[],
        )
        fact_index = {
            "fact-a": {"section_id": "s1", "page": 1},
            "fact-b": {"section_id": "s2", "page": 2},
        }

        _apply_group_provenance(result, fact_index, {"fact-a", "fact-b"})

        self.assertEqual(result.source_fact_ids, ["fact-a", "fact-b"])
        self.assertEqual(result.source_sections, ["s1", "s2"])

    def test_answer_audit_keeps_supported_clause_and_reports_unsupported_clause(self):
        answer = GlobalAnswer(
            question="问题",
            answer="ANN总体准确率为91.41%。月球由奶酪组成。",
            source_fact_ids=["fact-valid"],
            source_sections=[],
            source_pages=[],
            confidence=1.0,
        )
        fact_index = {
            "fact-valid": {
                "fact_id": "fact-valid",
                "statement": "ANN总体准确率为91.41%。",
                "source_quote": "ANN|91.41",
                "section_id": "section-1",
                "page": 20,
                "confidence": 1.0,
            }
        }

        result = _apply_answer_provenance(answer, fact_index, {"fact-valid"})

        self.assertEqual(result.answer, "ANN总体准确率为91.41%。")
        self.assertEqual(result.source_fact_ids, ["fact-valid"])
        self.assertEqual(result.source_sections, ["section-1"])
        self.assertEqual(result.source_pages, [20])
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.unsupported_claims, ["月球由奶酪组成。"])

    def test_answer_provenance_keeps_partially_supported_multi_clause_answers(self):
        answer = GlobalAnswer(
            question="文档有哪些局限？",
            answer="模型对噪声环境的鲁棒性较弱，且训练成本较高。",
            source_fact_ids=["fact-valid"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        fact_index = {
            "fact-valid": {
                "fact_id": "fact-valid",
                "statement": "模型对噪声环境的鲁棒性较弱。",
                "source_quote": "模型对噪声环境的鲁棒性较弱。",
                "section_id": "section-1",
                "page": 10,
                "confidence": 0.85,
            }
        }

        result = _apply_answer_provenance(answer, fact_index, {"fact-valid"})

        self.assertEqual(result.answer, "模型对噪声环境的鲁棒性较弱，且训练成本较高。")
        self.assertEqual(result.source_fact_ids, ["fact-valid"])
        self.assertEqual(result.confidence, 0.85)

    def test_audit_answer_claims_splits_mixed_by_clause(self):
        supported, unsupported = _audit_answer_claims(
            "A被证实。B纯属虚构。", "A被证实。"
        )
        self.assertEqual(supported, ["A被证实。"])
        self.assertEqual(unsupported, ["B纯属虚构。"])

        supported, unsupported = _audit_answer_claims("A被证实。", "A被证实。")
        self.assertEqual(supported, ["A被证实。"])
        self.assertEqual(unsupported, [])

        supported, unsupported = _audit_answer_claims("B纯属虚构。", "A被证实。")
        self.assertEqual(supported, [])
        self.assertEqual(unsupported, ["B纯属虚构。"])

    def test_apply_answer_provenance_keeps_evidence_backed_limitations(self):
        answer = GlobalAnswer(
            question="文档有哪些局限？",
            answer="回合制LLM生成与人类实时感知不同；环境与现实社会对齐困难；内存增长是瓶颈；模拟均为虚构；LLM易产生幻觉。计算开销大。",
            source_fact_ids=["fact-limit-a", "fact-limit-b"],
            source_sections=[],
            source_pages=[],
            confidence=1.0,
        )
        fact_index = {
            "fact-limit-a": {
                "fact_id": "fact-limit-a",
                "statement": "LLM易产生幻觉。",
                "source_quote": "LLM易产生幻觉。",
                "section_id": "section-1",
                "page": 16,
                "confidence": 0.8,
            },
            "fact-limit-b": {
                "fact_id": "fact-limit-b",
                "statement": "计算开销大。",
                "source_quote": "计算开销大。",
                "section_id": "section-1",
                "page": 17,
                "confidence": 0.85,
            },
        }

        result = _apply_answer_provenance(answer, fact_index, {"fact-limit-a", "fact-limit-b"})

        self.assertEqual(result.answer, "LLM易产生幻觉。计算开销大。")
        self.assertEqual(result.source_fact_ids, ["fact-limit-a", "fact-limit-b"])
        self.assertEqual(result.source_pages, [16, 17])
        self.assertEqual(
            result.unsupported_claims,
            ["回合制LLM生成与人类实时感知不同；", "环境与现实社会对齐困难；", "内存增长是瓶颈；", "模拟均为虚构；"],
        )

    def test_strip_english_leak_truncates_mixed_answer(self):
        # Chinese question + Chinese prose followed by a verbatim English statement:
        # the answer is cut at the start of the English block, the tail is reported.
        answer = GlobalAnswer(
            question="文档使用了哪些数据集或材料？",
            answer="实验创建了三个虚构世界。将响应按50:50混合。Environment model data is collected separately via random sampling, with high-frequency activity types downsampled.",
            source_fact_ids=["fact-a"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        result = _strip_english_leak(answer)
        self.assertEqual(result.answer, "实验创建了三个虚构世界。将响应按50:50混合。")
        self.assertEqual(
            result.unsupported_claims,
            ["Environment model data is collected separately via random sampling, with high-frequency activity types downsampled."],
        )
        self.assertAlmostEqual(result.confidence, 0.45)

    def test_strip_english_leak_keeps_short_metric_runs(self):
        # A 3-word English run (metric/proper noun) inside a Chinese answer stays.
        answer = GlobalAnswer(
            question="最重要的定量结果是什么？",
            answer="life reward training在CoSER Test上整体提升15.6%，拟人化提升23.7%。",
            source_fact_ids=["fact-b"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        result = _strip_english_leak(answer)
        self.assertEqual(
            result.answer,
            "life reward training在CoSER Test上整体提升15.6%，拟人化提升23.7%。",
        )
        self.assertEqual(result.unsupported_claims, [])

    def test_strip_english_leak_keeps_english_question(self):
        # Non-Chinese question is outside this backstop's scope.
        answer = GlobalAnswer(
            question="What datasets are used?",
            answer="Environment model data is collected separately via random sampling.",
            source_fact_ids=["fact-c"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        result = _strip_english_leak(answer)
        self.assertEqual(
            result.answer,
            "Environment model data is collected separately via random sampling.",
        )

    def test_strip_english_leak_flags_fully_english_answer(self):
        # A fully-English answer has no Chinese content to salvage: degrade to
        # "证据不足" and keep the whole leak visible for review.
        answer = GlobalAnswer(
            question="使用了哪些评价指标？",
            answer="Anthropomorphism evaluates whether agents behave in a human-like manner, covering self-identity and emotional depth.",
            source_fact_ids=["fact-d"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        result = _strip_english_leak(answer)
        self.assertEqual(result.answer, "证据不足")
        self.assertEqual(
            result.unsupported_claims,
            ["Anthropomorphism evaluates whether agents behave in a human-like manner, covering self-identity and emotional depth."],
        )
        self.assertEqual(result.source_fact_ids, [])
        self.assertEqual(result.confidence, 0.0)

    def test_strip_english_leak_degrades_english_stub(self):
        # Retrieval returned a table/config fact, the model wrote only a stub
        # ("Table 8:") and the tail was English. A stub with no Chinese content
        # must not survive as an answer.
        answer = GlobalAnswer(
            question="使用了哪些评价指标？",
            answer="Table 8: Key configuration parameters used in the simulation.: Parameter=maxpossessions; Description=Max items an agent can hold before discarding; Value=50",
            source_fact_ids=["fact-60ca678bcd38fd02"],
            source_sections=["C. Experiment Settings"],
            source_pages=[38],
            confidence=0.9,
        )
        result = _strip_english_leak(answer)
        self.assertEqual(result.answer, "证据不足")
        self.assertEqual(
            result.unsupported_claims,
            [
                "Table 8:",
                "Key configuration parameters used in the simulation.: Parameter=maxpossessions; Description=Max items an agent can hold before discarding; Value=50",
            ],
        )
        self.assertEqual(result.source_fact_ids, [])
        self.assertEqual(result.source_sections, [])
        self.assertEqual(result.source_pages, [])
        self.assertEqual(result.confidence, 0.0)

    def test_strip_english_leak_keeps_evidence_insufficient(self):
        answer = GlobalAnswer(
            question="不同章节之间是否存在矛盾？",
            answer="证据不足",
            source_fact_ids=[],
            source_sections=[],
            source_pages=[],
            confidence=0.0,
        )
        result = _strip_english_leak(answer)
        self.assertEqual(result.answer, "证据不足")

    def test_future_question_request_excludes_method_section_method_facts(self):
        facts = [
            {
                "fact_id": "fact-method-methods",
                "statement": "Communication in Agentopia proceeds in rounds.",
                "kind": "method",
                "section_title": "3 Methods",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-limitation-conclusion",
                "statement": "未来可扩展更长时间尺度训练。",
                "kind": "limitation",
                "section_title": "8 Conclusions",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-result-future",
                "statement": "后续计划将代理数量扩展到1000。",
                "kind": "result",
                "section_title": "8 Conclusions and future work",
                "confidence": 1.0,
            },
        ]

        _, allowed = build_question_request(GLOBAL_QUESTIONS[8], facts, [], max_tokens=600)

        self.assertNotIn("fact-method-methods", allowed)
        self.assertIn("fact-limitation-conclusion", allowed)
        self.assertIn("fact-result-future", allowed)

    def test_merge_table_headers_joins_two_row_header(self):
        # ``Claude`` / ``4.5-Opus`` becomes ``Claude 4.5-Opus`` and stays distinct
        # from ``Claude 4.5-Sonnet`` — the two Claude columns are NOT conflated.
        headers = _merge_table_headers(
            [
                ["", "Claude", "Gemini", "Qwen3.5", "Claude", "Qwen3.5"],
                ["", "4.5-Opus", "3-Pro", "397B-Agentopia", "4.5-Sonnet", "397B"],
            ]
        )
        self.assertEqual(headers[1], "Claude 4.5-Opus")
        self.assertEqual(headers[4], "Claude 4.5-Sonnet")
        self.assertEqual(len(set(headers)), len(headers))

    def test_merge_table_headers_dedupes_repeated_names(self):
        # A table whose header repeats a name gets a #N suffix per column, so
        # every column keeps a distinct identity (no false metric collisions).
        headers = _merge_table_headers(
            [["Metric", "Base", "Tuned", "Base", "Tuned", "Base", "Tuned"]]
        )
        self.assertEqual(headers[1], "Base")
        self.assertEqual(headers[3], "Base #2")
        self.assertEqual(headers[5], "Base #3")
        self.assertEqual(headers[2], "Tuned")
        self.assertEqual(headers[6], "Tuned #3")
        self.assertEqual(len(set(headers)), len(headers))

    def test_repair_header_name_fixes_fragment_noise(self):
        # pymupdf4llm splits a wide header cell into fragment columns that then
        # stack into names carrying a dangling bracket from a neighbour.
        self.assertEqual(_repair_header_name("(z-score Econ."), "Econ.")
        self.assertEqual(_repair_header_name(") Soc. (Δ)"), "Soc. (Δ)")
        self.assertEqual(_repair_header_name("nt (z-scor Esteem"), "nt Esteem")
        self.assertEqual(_repair_header_name("e) Material"), "e Material")

    def test_repair_header_name_keeps_balanced_and_clean_names(self):
        # Balanced or bracket-free names are genuine headers — never touched.
        self.assertEqual(_repair_header_name("Accuracy (%)"), "Accuracy (%)")
        self.assertEqual(_repair_header_name("Reward (z-score)"), "Reward (z-score)")
        self.assertEqual(_repair_header_name("Soc. (Δ)"), "Soc. (Δ)")
        self.assertEqual(_repair_header_name("Claude 4.5-Opus"), "Claude 4.5-Opus")
        self.assertEqual(_repair_header_name("Reward Subj."), "Reward Subj.")

    def test_merge_table_headers_repairs_fragment_cells(self):
        headers = _merge_table_headers([["Model", "Reward", "(z-score", ")", "Mood"]])
        self.assertEqual(headers[0], "Model")
        self.assertEqual(headers[1], "Reward")
        self.assertEqual(headers[4], "Mood")
        self.assertNotIn("(z-score", headers)
        self.assertNotIn(")", headers)

    def test_table_facts_skip_config_rows_at_source(self):
        # A config table's numeric "Value" column must not enter the fact DB as
        # kind="metric" noise — it is skipped at extraction, not just at
        # retrieval time.
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=5,
                        text=(
                            "# 8 Discussion\n"
                            "**Table 8** Key configuration parameters used in the simulation.\n"
                            "|Setting|Value|\n"
                            "|---|---|\n"
                            "|max_seq_len|4096|\n"
                            "|learning_rate|0.1|\n"
                        ),
                    )
                ]
            )
        )[0]
        self.assertEqual(extract_table_facts(section), [])

    def test_table_facts_skip_config_rows_without_table_marker(self):
        # Even without a config-ish caption, a generic "Value" column over an
        # all-generic context is still a config row, not a result.
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=5,
                        text=(
                            "# 8 Discussion\n"
                            "**Table 5** Model settings\n"
                            "|Setting|Value|\n"
                            "|---|---|\n"
                            "|max_seq_len|4096|\n"
                        ),
                    )
                ]
            )
        )[0]
        self.assertEqual(extract_table_facts(section), [])

    def test_table_facts_keep_generic_column_under_model_context(self):
        # A generic-named result column (Total/Mood) under a "Model" context is
        # a genuine result, not a config row — it must survive the source skip.
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=40,
                        text=(
                            "# 8 Discussion\n"
                            "**Table 13** Model comparison: reward and fulfillment metrics\n"
                            "|Model|Total|Mood|\n"
                            "|---|---|---|\n"
                            "|Qwen3.5-27B|10|5|\n"
                        ),
                    )
                ]
            )
        )[0]
        facts = extract_table_facts(section)
        metrics = {fact.metric for fact in facts}
        self.assertIn("Total", metrics)
        self.assertIn("Mood", metrics)

    def test_table_rows_consumes_two_row_header_into_data_start(self):
        # Table 3 in the real paper: two header rows, only one separator. The
        # second (variant) header row must be merged into the headers, not left
        # as a data row.
        markdown = (
            "Table 3: Performance of various LLMs\n"
            "||Claude|Gemini|Qwen3.5|Claude|Qwen3.5|CoSER|GPT-5|\n"
            "|---|---|---|---|---|---|---|---|\n"
            "||4.5-Opus|3-Pro|397B-Agentopia|4.5-Sonnet|397B|70B|Mini|\n"
            "|Anthropomorphism|64.28|60.42|49.67|36.02|40.16|31.16|24.60|\n"
            "|Average|62.43|61.80|49.16|45.21|42.51|35.95|32.97|\n"
        )
        tables = _table_rows(markdown)
        self.assertEqual(len(tables), 1)
        table, headers, rows, source_rows = tables[0]
        self.assertEqual(headers[1], "Claude 4.5-Opus")
        self.assertEqual(headers[4], "Claude 4.5-Sonnet")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "Anthropomorphism")
        self.assertEqual(len(source_rows), 2)

    def test_document_contradiction_flags_cross_page_same_cell(self):
        # Same metric + row context, SAME table caption, different pages,
        # different values -> the genuine contradiction a doc-level scan is for
        # (a table repeated across pages with conflicting numbers).
        sections = [
            {
                "section": {"id": "s1", "title": "5.3 Training"},
                "table_facts": [
                    {
                        "fact_id": "f-low",
                        "metric": "Accuracy",
                        "value": 91.41,
                        "display_value": "91.41",
                        "unit": "%",
                        "table": "Table 4: Results of all models",
                        "row_context": {"Model": "SVM"},
                        "kind": "metric",
                        "page": 16,
                    }
                ],
            },
            {
                "section": {"id": "s2", "title": "D. Additional"},
                "table_facts": [
                    {
                        "fact_id": "f-high",
                        "metric": "Accuracy",
                        "value": 91.37,
                        "display_value": "91.37",
                        "unit": "%",
                        "table": "Table 4: Results of all models",
                        "row_context": {"Model": "SVM"},
                        "kind": "metric",
                        "page": 17,
                    }
                ],
            },
        ]
        result = detect_document_contradictions(sections)
        self.assertEqual(len(result), 1)
        self.assertIn("f-low", result[0]["source_fact_ids"])
        self.assertIn("f-high", result[0]["source_fact_ids"])
        self.assertIn("91.41", result[0]["statement"])
        self.assertIn("91.37", result[0]["statement"])

    def test_document_contradiction_skips_cross_table_metric_legit_difference(self):
        # Identical table structure, same metric+row, but DIFFERENT table
        # captions (e.g. per-world z-scores) is a legitimate difference, not a
        # contradiction.
        sections = [
            {
                "section": {"id": "s1", "title": "D. Additional"},
                "table_facts": [
                    {
                        "fact_id": "f-world-a",
                        "metric": "Soc. (Δ)",
                        "value": 0.13,
                        "display_value": "+0.13",
                        "table": "Table 13: Model comparison reward and fulfillment metrics ( The Campus )",
                        "row_context": {"Model": "Qwen3.5-397B"},
                        "kind": "metric",
                        "page": 40,
                    }
                ],
            },
            {
                "section": {"id": "s2", "title": "D. Additional"},
                "table_facts": [
                    {
                        "fact_id": "f-world-b",
                        "metric": "Soc. (Δ)",
                        "value": 0.05,
                        "display_value": "+0.05",
                        "table": "Table 18: Model comparison reward and fulfillment metrics ( Arcane Academy )",
                        "row_context": {"Model": "Qwen3.5-397B"},
                        "kind": "metric",
                        "page": 53,
                    }
                ],
            },
        ]
        self.assertEqual(detect_document_contradictions(sections), [])

    def test_document_contradiction_skips_generic_and_correlation(self):
        # Config "Value" columns and correlation tables are not contradictions.
        sections = [
            {
                "section": {"id": "s1", "title": "C. Experiment"},
                "table_facts": [
                    {
                        "fact_id": "f-config-low",
                        "metric": "Value",
                        "value": 50.0,
                        "display_value": "50",
                        "table": "Table 8: Key configuration",
                        "row_context": {"Parameter": "max_possessions"},
                        "kind": "metric",
                        "page": 38,
                    },
                    {
                        "fact_id": "f-config-high",
                        "metric": "Value",
                        "value": 10.0,
                        "display_value": "10",
                        "table": "Table 8: Key configuration",
                        "row_context": {"Parameter": "max_possessions"},
                        "kind": "metric",
                        "page": 39,
                    },
                    {
                        "fact_id": "f-corr",
                        "metric": "Pearson's r",
                        "value": 0.19,
                        "display_value": "0.19",
                        "table": "Table 9: Top-10 metrics correlated with social reward",
                        "row_context": {"Metric": "n_likes"},
                        "kind": "metric",
                        "page": 40,
                    },
                ],
            }
        ]
        self.assertEqual(detect_document_contradictions(sections), [])

    def test_document_contradiction_same_value_not_flagged(self):
        sections = [
            {
                "section": {"id": "s1", "title": "5.3"},
                "table_facts": [
                    {
                        "fact_id": "f1",
                        "metric": "Accuracy",
                        "value": 91.41,
                        "display_value": "91.41",
                        "table": "Table 2",
                        "row_context": {"Model": "SVM"},
                        "kind": "metric",
                        "page": 10,
                    },
                    {
                        "fact_id": "f2",
                        "metric": "Accuracy",
                        "value": 91.41,
                        "display_value": "91.41",
                        "table": "Table 3",
                        "row_context": {"Model": "SVM"},
                        "kind": "metric",
                        "page": 11,
                    },
                ],
            }
        ]
        self.assertEqual(detect_document_contradictions(sections), [])

    def test_build_section_evidence_forces_required_table_facts(self):
        result = {
            "section": {"id": "s1", "title": "5.3", "page_start": 10, "page_end": 11},
            "extractions": [],
            "table_facts": [
                {
                    "fact_id": "f-boring",
                    "metric": "Accuracy",
                    "value": 91.41,
                    "display_value": "91.41",
                    "table": "Table 2",
                    "row_context": {"Model": "SVM"},
                    "kind": "metric",
                    "page": 10,
                    "grounded": True,
                    "bold": False,
                    "concept": "Accuracy",
                    "statement": "Table 2: Model=SVM; Accuracy=91.41",
                }
            ],
            "numeric_contradictions": [],
            "detected_facts": [],
            "synthesis": {
                "summary": "s",
                "core_concepts": [],
                "key_findings": [],
            },
        }
        evidence = build_section_evidence(result, required_table_fact_ids={"f-boring"})
        ids = {fact["fact_id"] for fact in evidence["facts"]}
        self.assertIn("f-boring", ids)

    def test_match_contradiction_fact_ids_by_number_and_tokens(self):
        statement = "Anthropomorphism准确率在3.1节为64.28%，但在4.2节为36.02%"
        fact_index = {
            "f1": {"statement": "Anthropomorphism得分为64.28。", "source_quote": "64.28", "confidence": 0.9},
            "f2": {"statement": "Anthropomorphism得分为36.02。", "source_quote": "36.02", "confidence": 0.9},
            "f3": {"statement": "总体准确率为99.99。", "source_quote": "99.99", "confidence": 0.9},
        }

        matched = _match_contradiction_fact_ids(statement, fact_index)

        self.assertIn("f1", matched)
        self.assertIn("f2", matched)
        self.assertNotIn("f3", matched)

    def test_build_contradiction_answer_matched_populates_provenance(self):
        contradictions = [
            {
                "statement": "Claude's Anthropomorphism score reported as both 64.28 and 36.02",
                "source_fact_ids": ["f1"],
            }
        ]
        fact_index = {
            "f1": {
                "section_id": "s1",
                "page": 3,
                "confidence": 0.8,
                "statement": "Anthropomorphism reported 64.28 and 36.02",
            }
        }

        result = _build_contradiction_answer(
            "不同章节之间是否存在数字或结论矛盾？", contradictions, fact_index
        )

        self.assertIn("64.28", result.answer)
        self.assertEqual(result.source_fact_ids, ["f1"])
        self.assertEqual(result.source_sections, ["s1"])
        self.assertEqual(result.source_pages, [3])
        self.assertEqual(result.confidence, 0.8)

    def test_build_contradiction_answer_unmatched_keeps_answer_with_0_5_confidence(self):
        contradictions = [{"statement": "某处声称X但另一处声称Y。", "source_fact_ids": []}]

        result = _build_contradiction_answer(
            "不同章节之间是否存在数字或结论矛盾？", contradictions, {}
        )

        self.assertEqual(result.answer, "某处声称X但另一处声称Y。")
        self.assertEqual(result.source_fact_ids, [])
        self.assertEqual(result.source_sections, [])
        self.assertEqual(result.source_pages, [])
        self.assertEqual(result.confidence, 0.5)

    def test_build_contradiction_answer_filters_no_contradiction_sentinel(self):
        contradictions = [
            {"statement": "未发现矛盾", "source_fact_ids": []},
            {"statement": "no contradiction found", "source_fact_ids": []},
        ]

        result = _build_contradiction_answer(
            "不同章节之间是否存在数字或结论矛盾？", contradictions, {}
        )

        self.assertEqual(result.answer, "证据不足")
        self.assertEqual(result.confidence, 0.0)

    def test_merge_overview_contradictions_merges_and_dedups(self):
        detected = [{"statement": "A", "source_fact_ids": ["f1"]}]
        overview_statements = ["A", "Claude在A测试64.28%，在B测试36.02%", "未发现明显矛盾"]
        fact_index = {
            "f1": {"section_id": "s1", "page": 1, "confidence": 0.9},
            "f2": {
                "section_id": "s2",
                "page": 2,
                "confidence": 0.9,
                "statement": "Anthropomorphism得分为64.28。",
                "source_quote": "64.28",
            },
        }

        merged = _merge_overview_contradictions(detected, overview_statements, fact_index)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["statement"], "A")
        self.assertEqual(merged[1]["statement"], "Claude在A测试64.28%，在B测试36.02%")
        self.assertEqual(merged[1]["source_fact_ids"], ["f2"])

    def test_answer_global_question_short_circuit_uses_effective_contradictions(self):
        pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
        pipeline._document_fact_index = {
            "f1": {
                "section_id": "s1",
                "page": 3,
                "confidence": 0.8,
                "statement": "Anthropomorphism reported 64.28 and 36.02",
            }
        }
        effective = [
            {
                "statement": "Claude's Anthropomorphism score reported as both 64.28 and 36.02",
                "source_fact_ids": ["f1"],
            }
        ]

        answer = asyncio.run(
            pipeline._answer_global_question(GLOBAL_QUESTIONS[9], [], effective)
        )

        self.assertIn("64.28", answer.answer)
        self.assertEqual(answer.source_fact_ids, ["f1"])
        self.assertEqual(answer.source_sections, ["s1"])
        self.assertEqual(answer.confidence, 0.8)

    def test_finalize_provenance_number_gate_replaces_fabricated_clause_answer(self):
        answer = GlobalAnswer(
            question="最重要的定量结果是什么？",
            answer="ANN准确率为91.41%。SVM准确率为13.6%。",
            source_fact_ids=["fact-ann"],
            source_sections=[],
            source_pages=[],
            confidence=0.9,
        )
        candidates = [{"fact_id": "fact-ann", "statement": "ANN准确率为91.41%。"}]
        fact_index = {
            "fact-ann": {
                **candidates[0],
                "source_quote": "ANN准确率为91.41%。",
                "section_id": "s1",
                "page": 1,
                "confidence": 0.9,
            }
        }

        result = _finalize_answer_provenance(
            answer, candidates, fact_index, {"fact-ann"}
        )

        # 答案含一条证据外数字子句（13.6%）：数字门把整条恢复驳回，fallback 用
        # 有证据的候选句替换整个答案，而不是保留带伪造子句的版本。
        self.assertEqual(result.answer, "ANN准确率为91.41%。")
        self.assertEqual(result.source_fact_ids, ["fact-ann"])
        self.assertEqual(result.unsupported_claims, [])

    def test_global_question_set_is_complete(self):
        self.assertEqual(len(GLOBAL_QUESTIONS), 10)

    def test_grounding_ignores_pdf_markdown_layout_tags(self):
        source = "The mathematical<br>expression for<br>probability"
        quote = "The mathematical expression for probability"
        self.assertTrue(_quote_is_grounded(quote, source))

    def test_reduce_payload_excludes_ungrounded_facts(self):
        common = {
            "concept": "概念",
            "statement": "陈述",
            "kind": "result",
            "source_quote": "引用",
            "confidence": 0.9,
        }
        extraction = ChunkExtraction(
            summary="摘要",
            facts=[Fact(grounded=True, **common), Fact(grounded=False, **common)],
        )
        payload = _grounded_payload([extraction])
        self.assertEqual(len(payload[0]["facts"]), 1)

    def test_document_overview_summary_is_built_from_group_evidence(self):
        groups = [
            GroupSynthesis(
                summary="系统使用奖励信号训练智能体并观察长期行为。",
                concepts=["奖励", "智能体"],
                findings=["模型在长期互动中表现出稳定行为。"],
                numeric_claims=["准确率达到91.41%"],
                limitations=["噪声环境下鲁棒性较弱。"],
                future_work=["扩展更长时间尺度训练。"],
                contradictions=[],
                source_fact_ids=[],
                source_sections=[],
                source_pages=[],
            )
        ]

        overview = fallback_document_overview(groups, [])

        self.assertIn("奖励", overview.executive_summary)
        self.assertIn("长期", overview.executive_summary)
        self.assertIn("模型在长期互动中表现出稳定行为。", overview.key_findings)
        self.assertIn("准确率达到91.41%", overview.key_findings)


class UsageTrackerTests(unittest.TestCase):
    def test_summarizes_calls_and_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = UsageTracker(Path(temp_dir) / "usage.sqlite3")
            tracker.record("extract", "flash", 100, 20, 120, 1.5, "req-1")
            tracker.record("reduce", "pro", 200, 40, 240, 2.5, "req-2")
            summary = tracker.summary()
            self.assertEqual(summary["calls"], 2)
            self.assertEqual(summary["input_tokens"], 300)
            self.assertEqual(summary["output_tokens"], 60)
            self.assertEqual(summary["by_model"]["pro"]["calls"], 1)
            tracker.close()

    def test_fact_supports_advantage_and_disadvantage_tags(self):
        common = {
            "concept": "测试概念",
            "statement": "测试陈述",
            "source_quote": "测试原文",
            "confidence": 0.9,
        }
        self.assertEqual(Fact(kind="advantage", **common).kind, "advantage")
        self.assertEqual(Fact(kind="disadvantage", **common).kind, "disadvantage")
        self.assertEqual(Fact(kind="classification", **common).kind, "classification")
        self.assertEqual(Fact(kind="unexpected-label", **common).kind, "relation")

    def test_cache_key_tracks_model_prompt_and_content(self):
        base = cache_key("extract", "flash", "v1", "same content")
        self.assertNotEqual(base, cache_key("extract", "pro", "v1", "same content"))
        self.assertNotEqual(base, cache_key("extract", "flash", "v2", "same content"))
        self.assertEqual(base, cache_key("extract", "flash", "v1", "same content"))

    def test_detects_back_matter(self):
        self.assertTrue(is_back_matter("References"))
        self.assertTrue(is_back_matter("利益冲突"))
        self.assertFalse(is_back_matter("分类模型"))

    def test_back_matter_matches_author_biographies(self):
        # Survey papers end with author pages that are not paper content.
        self.assertTrue(is_back_matter("Author Biographies"))
        self.assertTrue(is_back_matter("Biographies"))
        self.assertTrue(is_back_matter("About the Authors"))
        self.assertFalse(is_back_matter("Biographical Consistency of agents"))

    def test_is_toc_section_detects_contents(self):
        # A "Contents" node's body is a page-number listing, not paper content.
        self.assertTrue(_is_toc_section("**Contents**"))
        self.assertTrue(_is_toc_section("Table of Contents"))
        self.assertTrue(_is_toc_section("目录"))
        self.assertTrue(_is_toc_section("List of Figures"))
        self.assertFalse(_is_toc_section("Contents and Discussion of Results"))
        self.assertFalse(_is_toc_section("2 Agent AI Integration"))

    def test_strip_picture_text_removes_ocr_block(self):
        # The OCR text pymupdf4llm reads out of a rasterized figure lives inside
        # an HTML comment block; it is not paper prose and must be removed.
        text = (
            "## General Alpaca Training Framework\n"
            "<!-- Start of picture text -->\n"
            "Pre-trained<br>Open-source LLM Alpaca<br>Larger model\n"
            "<!-- End of picture text -->\n"
            "Figure 28: The training recipe used to train the Alpaca model."
        )
        cleaned = _strip_picture_text(text)
        self.assertNotIn("Pre-trained", cleaned)
        self.assertNotIn("Start of picture text", cleaned)
        self.assertIn("Figure 28", cleaned)
        self.assertIn("General Alpaca Training Framework", cleaned)

    def test_strip_picture_text_strips_img_and_keeps_mark_content(self):
        text = '<img src="fig1.png"/> real prose <mark>highlighted</mark> tail'
        cleaned = _strip_picture_text(text)
        self.assertNotIn("<img", cleaned)
        self.assertNotIn("<mark>", cleaned)
        self.assertNotIn("</mark>", cleaned)
        self.assertIn("highlighted", cleaned)
        self.assertIn("real prose", cleaned)

    def test_strip_picture_text_handles_unclosed_block(self):
        # Some pages start a picture block but never close it; everything from
        # the opening marker onward is picture text.
        text = "intro\n<!-- Start of picture text -->\nOCR junk\n"
        self.assertEqual(_strip_picture_text(text), "intro")

    def test_noise_node_reason_figure_caption_heading(self):
        # A heading whose body is a "Figure N:" caption is a promoted caption.
        node = HeadingNode(
            id="n1", title="General Alpaca Training Framework", level=2,
            page_start=46, page_end=46, subtree_page_end=46, parent_id="root",
            text="Figure 28: The training recipe used to train the Alpaca model.",
        )
        self.assertEqual(_noise_node_reason(node), "caption")

    def test_noise_node_reason_ignores_numbered_section_starting_with_figure(self):
        # A numbered section that happens to open with a figure reference keeps
        # its identity (the caption rule is skipped when a numeric signature
        # exists).
        node = HeadingNode(
            id="n2", title="3.2 Agent Transformer Definition", level=3,
            page_start=15, page_end=16, subtree_page_end=16, parent_id="root",
            text="Figure 4: The architecture. Then the model is trained.",
        )
        self.assertIsNone(_noise_node_reason(node))

    def test_noise_node_reason_transcript_heading(self):
        # A heading whose body opens with a prompt role header, and a heading
        # whose bare title is a role word, are transcript fragments.
        prompt = HeadingNode(
            id="n3", title="GPT-4-Vision", level=3,
            page_start=71, page_end=71, subtree_page_end=71, parent_id="root",
            text="**Prompt:** These are frames of a video of a player.",
        )
        self.assertEqual(_noise_node_reason(prompt), "transcript")
        output = HeadingNode(
            id="n4", title="Output:", level=3,
            page_start=71, page_end=75, subtree_page_end=75, parent_id="root",
            text="The player begins by running around the map.",
        )
        self.assertEqual(_noise_node_reason(output), "transcript")

    def test_absorb_noise_nodes_promotes_children_of_caption_heading(self):
        # The real problem: a promoted figure-caption heading becomes a bogus
        # parent that every following section nests under. Absorbing it must
        # promote its children to the grandparent and keep its body there.
        root = HeadingNode(
            id="root", title="Document", level=0, page_start=1, page_end=1,
            subtree_page_end=1, parent_id=None,
        )
        caption = HeadingNode(
            id="cap", title="General Alpaca Training Framework", level=2,
            page_start=46, page_end=46, subtree_page_end=55, parent_id="root",
            text="Figure 28: The training recipe.",
        )
        real = HeadingNode(
            id="real", title="8 Continuous and Self-improvement", level=3,
            page_start=49, page_end=50, subtree_page_end=50, parent_id="cap",
            text="Models keep improving.",
        )
        caption.children = [real]
        root.children = [caption]

        _absorb_noise_nodes(root)

        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].id, "real")
        self.assertEqual(root.children[0].parent_id, "root")
        self.assertIn("Figure 28", root.text)

    def test_absorb_noise_nodes_transcript_children_kept(self):
        # A transcript fragment has no children; its body merges into the parent
        # and the fragment disappears as a section.
        parent = HeadingNode(
            id="p", title="Appendix", level=3, page_start=69, page_end=69,
            subtree_page_end=69, parent_id="root", text="prose",
        )
        output = HeadingNode(
            id="o", title="Output:", level=3, page_start=71, page_end=75,
            subtree_page_end=75, parent_id="p",
            text="The player begins by running around the map.",
        )
        parent.children = [output]
        _absorb_noise_nodes(parent)
        self.assertEqual(parent.children, [])
        self.assertIn("The player begins", parent.text)


class LevelNormalizationTests(unittest.TestCase):
    def _node(self, node_id, title, level, parent, children=(), page=1):
        node = HeadingNode(
            id=node_id, title=title, level=level, page_start=page, page_end=page,
            subtree_page_end=page, parent_id=parent, text="body",
        )
        node.children = list(children)
        return node

    def test_rebases_survey_style_numbered_sections_to_l2(self):
        # The survey paper marks main sections with ### / ####, landing at L3/L4.
        sub = self._node("sub", "1.1 Motivation", 4, "intro")
        intro = self._node("intro", "1 Introduction", 3, "title", [sub])
        abstract = self._node("abs", "Abstract", 3, "title")
        title = self._node("title", "Survey Title", 1, "root", [abstract, intro])
        root = self._node("root", "Document", 0, None, [title])

        _normalize_levels(root)

        levels = {node.id: node.level for node in flatten_sections(root)}
        self.assertEqual(levels["title"], 1)      # unnumbered wrapper keeps L1
        self.assertEqual(levels["intro"], 2)      # top numbered section -> L2
        self.assertEqual(levels["sub"], 3)        # subsection -> L3
        self.assertEqual(levels["abs"], 2)        # unnumbered at the min also shifts

    def test_leaves_l2_based_paper_unchanged(self):
        # Agentopia uses ## / ###: min numbered level is already 2 -> no shift.
        sub = self._node("sub", "3.1 Agent Design", 3, "intro")
        intro = self._node("intro", "3 Agentopia", 2, "title", [sub])
        title = self._node("title", "Agentopia: ...", 1, "root", [intro])
        root = self._node("root", "Document", 0, None, [title])

        _normalize_levels(root)

        levels = {node.id: node.level for node in flatten_sections(root)}
        self.assertEqual(levels["intro"], 2)
        self.assertEqual(levels["sub"], 3)

    def test_no_numbered_sections_no_change(self):
        unnumbered = self._node("unc", "Chapter", 3, "title")
        title = self._node("title", "Report", 1, "root", [unnumbered])
        root = self._node("root", "Document", 0, None, [title])

        _normalize_levels(root)

        self.assertEqual(root.children[0].children[0].level, 3)

    def test_heading_tree_integration_rebases_numbered_sections(self):
        # Keep bodies long so the low-signal merge (<=220 chars) and the
        # fragment merge (<=500 tokens) do not absorb the sections before level
        # normalization runs.
        pages = [
            PageMarkdown(page=1, text=(
                "# The Title\n"
                "### 1 Introduction\n" + ("intro text " * 60) + "\n"
                "#### 1.1 Motivation\n" + ("motivation text " * 200) + "\n"
                "### 2 Method\n" + ("method text " * 60)
            )),
        ]
        root = build_heading_tree(pages)
        sections = {node.title: node for node in flatten_sections(root)}
        self.assertEqual(sections["The Title"].level, 1)
        self.assertEqual(sections["1 Introduction"].level, 2)
        self.assertEqual(sections["1.1 Motivation"].level, 3)
        self.assertEqual(sections["2 Method"].level, 2)
        self.assertEqual(
            sections["1.1 Motivation"].parent_id,
            sections["1 Introduction"].id,
        )


class RequestRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_empty_json_and_schema_invalid_responses(self):
        valid = {
            "summary": "摘要",
            "facts": [
                {
                    "concept": "概念",
                    "statement": "事实",
                    "kind": "result",
                    "source_quote": "原文",
                    "page": 1,
                    "confidence": 0.9,
                }
            ],
            "new_concepts": [],
        }
        contents = ["", "{}", json.dumps(valid, ensure_ascii=False)]

        class FakeCompletions:
            def __init__(self):
                self.calls = 0
                self.kwargs = []

            async def create(self, **kwargs):
                self.kwargs.append(kwargs)
                content = contents[self.calls]
                self.calls += 1
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    _request_id=f"req-{self.calls}",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
            completions = FakeCompletions()
            pipeline.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
            pipeline.semaphore = asyncio.Semaphore(1)
            pipeline.usage = UsageTracker(Path(temp_dir) / "usage.sqlite3")
            pipeline.content_retry_attempts = 3
            pipeline.content_retry_base_seconds = 0

            payload = await pipeline._request_json(
                "chunk_extract",
                "flash",
                "system",
                "user",
                response_model=ChunkExtraction,
            )

            self.assertEqual(payload["summary"], "摘要")
            self.assertEqual(completions.calls, 3)
            self.assertTrue(all(item["max_tokens"] == 1800 for item in completions.kwargs))
            self.assertTrue(all(item["reasoning_effort"] == "low" for item in completions.kwargs))
            self.assertEqual(pipeline.usage.summary()["calls"], 3)
            pipeline.usage.close()

    async def _make_recovery_pipeline(self, scripted_responses):
        """Build a pipeline whose completions.create replays scripted responses.

        Each scripted response is (finish_reason, content).  The fake records every
        create() call (messages, max_tokens, reasoning_effort) so tests can assert on
        the recovery path.  The caller is responsible for pipeline.usage.close().
        """

        class FakeCompletions:
            def __init__(self):
                self.calls = 0
                self.kwargs = []

            async def create(self, **kwargs):
                self.kwargs.append(kwargs)
                finish, content = scripted_responses[self.calls]
                self.calls += 1
                message = SimpleNamespace(content=content, reasoning_content="思考散文")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message, finish_reason=finish)],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                    _request_id=f"req-{self.calls}",
                )

        self._usage_dir = tempfile.mkdtemp()
        pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
        completions = FakeCompletions()
        pipeline.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        pipeline.semaphore = asyncio.Semaphore(1)
        pipeline.usage = UsageTracker(Path(self._usage_dir) / "usage.sqlite3")
        pipeline.content_retry_attempts = 3
        pipeline.content_retry_base_seconds = 0
        pipeline.truncation_recovery_attempts = 2
        return pipeline, completions

    async def _cleanup_usage(self, pipeline) -> None:
        pipeline.usage.close()
        shutil.rmtree(self._usage_dir, ignore_errors=True)

    async def test_truncated_empty_content_reissues_with_thinking_off(self):
        # content never started (empty) + reasoning burned the budget -> regenerate
        # with reasoning_effort="none" and NO assistant seed.
        valid = {
            "summary": "摘要",
            "facts": [],
            "new_concepts": [],
        }
        pipeline, completions = await self._make_recovery_pipeline(
            [("length", ""), ("stop", json.dumps(valid, ensure_ascii=False))]
        )

        payload = await pipeline._request_json(
            "chunk_extract", "flash", "system", "user", response_model=ChunkExtraction
        )

        self.assertEqual(payload["summary"], "摘要")
        self.assertEqual(completions.calls, 2)
        # regen call: no assistant seed, thinking off, full budget
        regen = completions.kwargs[1]
        self.assertEqual(regen["reasoning_effort"], "none")
        self.assertEqual(len(regen["messages"]), 2)
        self.assertEqual([m["role"] for m in regen["messages"]], ["system", "user"])
        self.assertNotIn("思考散文", [m.get("content", "") for m in regen["messages"]])
        self.assertEqual(regen["max_tokens"], max(2048, 1800))
        # every create() is recorded
        self.assertEqual(pipeline.usage.summary()["calls"], 2)
        await self._cleanup_usage(pipeline)

    async def test_truncated_partial_content_continues_from_content_not_reasoning(self):
        # partial JSON content exists -> continuation seeded from exact content,
        # thinking off, larger budget.
        valid = {
            "summary": "摘要",
            "facts": [
                {
                    "concept": "概念",
                    "statement": "事实",
                    "kind": "result",
                    "source_quote": "原文",
                    "page": 1,
                    "confidence": 0.9,
                }
            ],
            "new_concepts": [],
        }
        partial = '{"summary":"部分","facts":['
        pipeline, completions = await self._make_recovery_pipeline(
            [("length", partial), ("stop", json.dumps(valid, ensure_ascii=False))]
        )

        payload = await pipeline._request_json(
            "chunk_extract", "flash", "system", "user", response_model=ChunkExtraction
        )

        self.assertEqual(payload["summary"], "摘要")
        self.assertEqual(completions.calls, 2)
        cont = completions.kwargs[1]
        self.assertEqual(cont["reasoning_effort"], "none")
        self.assertEqual(cont["max_tokens"], min(1800 * 2, 4096))
        # the assistant seed must be the partial JSON content, not reasoning/'length'
        self.assertEqual(cont["messages"][2]["role"], "assistant")
        self.assertEqual(cont["messages"][2]["content"], partial)
        await self._cleanup_usage(pipeline)

    async def test_regen_partial_chains_into_continuation(self):
        # Empty content -> regen (thinking off) also truncates but NOW leaves partial
        # JSON -> the next recovery attempt continues from that partial instead of
        # discarding it and regenerating the same insufficient budget again.
        valid = {
            "summary": "摘要",
            "facts": [],
            "new_concepts": [],
        }
        pipeline, completions = await self._make_recovery_pipeline(
            [
                ("length", ""),
                ("length", '{"summary":"部分","facts":['),
                ("stop", json.dumps(valid, ensure_ascii=False)),
            ]
        )

        payload = await pipeline._request_json(
            "chunk_extract", "flash", "system", "user", response_model=ChunkExtraction
        )

        self.assertEqual(payload["summary"], "摘要")
        self.assertEqual(completions.calls, 3)
        # call[1] = regen (none, no seed); call[2] = continuation seeded from call[1]'s partial
        self.assertEqual(completions.kwargs[1]["reasoning_effort"], "none")
        self.assertEqual(len(completions.kwargs[1]["messages"]), 2)
        cont = completions.kwargs[2]
        self.assertEqual(cont["reasoning_effort"], "none")
        self.assertEqual(len(cont["messages"]), 4)
        self.assertEqual(cont["messages"][2]["role"], "assistant")
        self.assertEqual(cont["messages"][2]["content"], '{"summary":"部分","facts":[')
        await self._cleanup_usage(pipeline)

    async def test_continuation_tail_combined_with_partial_parses(self):
        # A real continuation outputs only the tail of the JSON, which cannot be
        # parsed standalone — the recovery must combine partial + tail before parsing.
        partial = (
            '{"summary":"部分","facts":[{"concept":"c1","statement":"s1",'
            '"kind":"result","source_quote":"q1","page":1,"confidence":0.9}],'
            '"new_concepts":'
        )
        tail = '["a"]}'
        pipeline, completions = await self._make_recovery_pipeline(
            [("length", partial), ("stop", tail)]
        )

        payload = await pipeline._request_json(
            "chunk_extract", "flash", "system", "user", response_model=ChunkExtraction
        )

        self.assertEqual(payload["summary"], "部分")
        self.assertEqual(len(payload["facts"]), 1)
        self.assertEqual(payload["new_concepts"], ["a"])
        self.assertEqual(completions.calls, 2)
        cont = completions.kwargs[1]
        self.assertEqual(cont["reasoning_effort"], "none")
        self.assertEqual(len(cont["messages"]), 4)
        self.assertEqual(cont["messages"][2]["content"], partial)
        await self._cleanup_usage(pipeline)

    async def test_continuation_also_truncated_falls_back_to_regenerate(self):
        # continuation also truncates -> next recovery attempt re-issues from scratch.
        valid = {
            "summary": "摘要",
            "facts": [],
            "new_concepts": [],
        }
        pipeline, completions = await self._make_recovery_pipeline(
            [
                ("length", '{"summary":"部分","facts":['),
                ("length", '{"summary":"部分2","facts":['),
                ("stop", json.dumps(valid, ensure_ascii=False)),
            ]
        )

        payload = await pipeline._request_json(
            "chunk_extract", "flash", "system", "user", response_model=ChunkExtraction
        )

        self.assertEqual(payload["summary"], "摘要")
        self.assertEqual(completions.calls, 3)
        # call[0] = original (low), call[1] = continuation (none, seeded), call[2] = regen (none, no seed)
        self.assertEqual(completions.kwargs[0]["reasoning_effort"], "low")
        self.assertEqual(completions.kwargs[1]["reasoning_effort"], "none")
        self.assertEqual(len(completions.kwargs[1]["messages"]), 4)
        self.assertEqual(completions.kwargs[2]["reasoning_effort"], "none")
        self.assertEqual(len(completions.kwargs[2]["messages"]), 2)
        await self._cleanup_usage(pipeline)


if __name__ == "__main__":
    unittest.main()
