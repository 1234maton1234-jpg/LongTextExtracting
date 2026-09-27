import asyncio
import contextlib
import io
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
    TextChunk,
    ConceptMemory,
    ResultCache,
    ChunkExtraction,
    DocumentOverview,
    DocumentSynthesis,
    GLOBAL_QUESTIONS,
    Fact,
    GlobalAnswer,
    GroupSynthesis,
    LongDocumentPipeline,
    UsageTracker,
    PageMarkdown,
    _apply_answer_provenance,
    _apply_document_title,
    _absorb_noise_nodes,
    _candidate_document_title,
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
    extract_metric_facts,
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
    _response_request_id,
    _alias_occurs_in,
    _grounded_payload,
    batch_by_token_budget,
    build_section_evidence,
    run_evidence_metrics,
    report_evidence_metrics,
    _answer_sentences,
    _quantile,
    _TABLE_HEADER_CONCEPT,
    _claim_matches_fact_db,
    _claim_tokens,
    _numeric_forms,
    audit_overview_numerics,
)


class MetricFactExtractionTests(unittest.TestCase):
    """Prose metric recovery: `kind="metric"` must not depend on markdown tables.

    Regression guard for the defect where a PDF yielding no pipe tables left the
    metric pool empty, so "使用了哪些评价指标？" answered with an unrelated but
    grounded definition sentence.
    """

    def _section(self, text, page_start=1, page_end=None, page_texts=None, section_id="node-metric"):
        return HeadingNode(
            id=section_id,
            title="评价指标",
            level=1,
            page_start=page_start,
            page_end=page_end if page_end is not None else page_start,
            subtree_page_end=page_end if page_end is not None else page_start,
            text=text,
            page_texts=page_texts if page_texts is not None else [
                PageMarkdown(page=page_start, text=text)
            ],
        )

    def test_recovers_metric_reported_with_percentage(self):
        section = self._section(
            "Evaluation on a VLN benchmark shows our RCM model outperforms "
            "previous methods by 10% on SPL and achieves state-of-the-art performance."
        )

        facts = extract_metric_facts(section)

        self.assertEqual(1, len(facts))
        self.assertEqual("metric", facts[0].kind)
        self.assertEqual("SPL", facts[0].concept)
        self.assertTrue(facts[0].grounded)
        self.assertIn("10%", facts[0].statement)
        self.assertEqual(1, facts[0].page)

    def test_recovers_chinese_metric(self):
        section = self._section("实验采用准确率和召回率作为评价指标，准确率达到 91.4%。")

        facts = extract_metric_facts(section)

        self.assertTrue(facts)
        self.assertEqual("metric", facts[0].kind)
        self.assertIn(facts[0].concept, {"准确率", "召回率", "评价指标"})
        self.assertTrue(all(fact.grounded for fact in facts))

    def test_recovers_benchmark_comparison_without_named_metric(self):
        # "outperforms others by 6% on the 2022 OK-VQA open-domain multimodal task"
        # names no metric, but it is still the quantitative evidence the metric
        # question needs.
        section = self._section(
            "We introduce the Knowledge Augmented Transformer (KAT), which "
            "outperforms others by 6% on the 2022 OK-VQA open-domain multimodal task."
        )

        facts = extract_metric_facts(section)

        self.assertEqual(1, len(facts))
        self.assertEqual("metric", facts[0].kind)
        self.assertIn("6%", facts[0].statement)

    def test_ignores_bare_mention_without_evidence(self):
        # No number, no evaluation framing: not evidence that the metric was used.
        section = self._section(
            "Common uses for retrieving and analysing data must be respectful of all users."
        )

        self.assertEqual([], extract_metric_facts(section))

    def test_ignores_english_copula_and_geographic_map(self):
        # "is" and "map" are the worst false-positive generators in real papers.
        section = self._section(
            "The player begins by running around the map, passing through different "
            "checkpoints, and the agent is able to interact with several capture points."
        )

        self.assertEqual([], extract_metric_facts(section))

    def test_ignores_acronym_homonym_without_numeric_evidence(self):
        # "MAE" here is the masked autoencoder, not mean absolute error.
        section = self._section(
            "We use a 12-layer ViT initialized from MAE (He et al., 2022) "
            "(ImageNet-1K without labels) as our backbone."
        )

        self.assertEqual([], extract_metric_facts(section))

    def test_ambiguous_metric_accepted_when_value_present(self):
        section = self._section("Our model reaches an mIoU of 0.82 on the test set.")

        facts = extract_metric_facts(section)

        self.assertTrue(facts)
        self.assertEqual("miou", facts[0].concept.lower())

    def test_ignores_metric_word_used_as_ordinary_prose(self):
        # "accuracy" here is ordinary prose about accuracy, not a reported metric:
        # an evaluation verb alone must not admit it.
        section = self._section(
            "To enhance the accuracy of video descriptions, we employ GPT-4V to "
            "caption frames, while audio and its transcriptions are sourced elsewhere."
        )

        self.assertEqual([], extract_metric_facts(section))

    def test_accepts_enumerated_metric_list_without_numbers(self):
        section = self._section(
            "Evaluation metrics include accuracy, precision, recall and F1."
        )

        facts = extract_metric_facts(section)

        # One fact per sentence, named after the longest matched term, so that a
        # dense metric list cannot flood the evidence pool.
        self.assertEqual(1, len(facts))
        self.assertEqual("metric", facts[0].kind)
        self.assertEqual("precision", facts[0].concept.lower())
        self.assertIn("precision", facts[0].statement)

    def test_does_not_match_metric_names_inside_longer_tokens(self):
        # "recalled" must not fire "recall"; "f10" must not fire "f1".
        section = self._section(
            "The agent recalled the f10 configuration from memory last week."
        )

        self.assertEqual([], extract_metric_facts(section))

    def test_deduplicates_by_term_and_respects_cap(self):
        sentence = " ".join(
            f"We report accuracy of {value}.0% on benchmark {index}."
            for index, value in enumerate(range(60, 75))
        )
        section = self._section(sentence)

        facts = extract_metric_facts(section, max_facts=3)

        self.assertLessEqual(len(facts), 3)
        concepts = [fact.concept.lower() for fact in facts]
        self.assertEqual(len(concepts), len(set(concepts)))
        self.assertEqual(len({fact.fact_id for fact in facts}), len(facts))

    def test_fact_ids_are_stable_across_calls(self):
        text = "Evaluation on the VLN benchmark reports 10% on SPL."
        first = extract_metric_facts(self._section(text))
        second = extract_metric_facts(self._section(text))

        self.assertEqual(
            [fact.fact_id for fact in first], [fact.fact_id for fact in second]
        )

    def test_metric_facts_reach_question_retrieval_without_tables(self):
        # End-to-end through the evidence builder: a table-free section must still
        # put metric evidence in front of the "评价指标" route.
        text = (
            "Evaluation on a VLN benchmark dataset shows that our RCM model "
            "significantly outperforms previous methods by 10% on SPL."
        )
        section = self._section(text)
        section_dict = section.model_dump(exclude={"children"})
        result = {
            "section": section_dict,
            "extractions": [{"summary": "s", "facts": []}],
            "table_facts": [],
            "synthesis": {
                "summary": "s",
                "core_concepts": [],
                "key_findings": [],
                "source_pages": [1],
                "title": "t",
            },
        }

        evidence = build_section_evidence(result)
        metric_evidence = [f for f in evidence["facts"] if f["kind"] == "metric"]
        self.assertTrue(metric_evidence)

        catalog = [
            {**fact, "section_title": evidence["title"]} for fact in evidence["facts"]
        ]
        _, allowed = build_question_request(
            "使用了哪些评价指标？", catalog, [], max_tokens=600
        )

        self.assertIn(metric_evidence[0]["fact_id"], allowed)


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

    def test_alias_must_match_a_whole_token(self):
        # Substring matching retrieved pure noise on a real 1530-alias DB: these
        # exact pairs were junk `related_memory` injected into extraction prompts.
        for alias, text in [
            ("her", "the other agents were there"),
            ("position", "the composition of the dataset"),
            ("actions", "rich interactions between agents"),
            ("map", "running around the checkpoint"),
            ("agent", "Agentopia is a society simulator"),
        ]:
            with self.subTest(alias=alias, text=text):
                self.assertFalse(_alias_occurs_in(alias, text))

    def test_whole_token_match_still_detects_real_mentions(self):
        self.assertTrue(_alias_occurs_in("agent", "each agent acts"))
        self.assertTrue(_alias_occurs_in("ai", "an AI system"))
        # A stray space between CJK characters must not split the token.
        self.assertTrue(_alias_occurs_in("支持向量机", "支持 向量机适合分类"))

    def test_alias_match_folds_case(self):
        # The real DB stores the acronym alias key uppercase ('AI').
        self.assertTrue(_alias_occurs_in("AI", "an ai system"))
        self.assertTrue(_alias_occurs_in("ai", "an AI system"))

    def test_retrieve_scopes_to_the_current_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ConceptMemory(Path(temp_dir) / "memory.sqlite3")
            memory.upsert("Agentopia", "多智能体社会", document_key="doc-a")
            memory.upsert("Knowledge Augmented Transformer", "KAT", document_key="doc-b")

            a = [row["canonical_name"] for row in memory.retrieve("Agentopia", document_key="doc-a")]
            b = [row["canonical_name"] for row in memory.retrieve("Agentopia", document_key="doc-b")]

            self.assertEqual(["Agentopia"], a)
            self.assertEqual([], b)  # must not leak across documents
            memory.close()

    def test_legacy_rows_without_owner_do_not_leak_into_a_scoped_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ConceptMemory(Path(temp_dir) / "memory.sqlite3")
            # Rows written before document scoping existed carry no owner. They
            # stay reachable for an unscoped lookup, but must never appear inside a
            # specific document's retrieval -- that is the cross-document leak.
            memory.upsert("Legacy")

            self.assertEqual(1, len(memory.retrieve("Legacy Concept")))
            self.assertEqual(
                [], memory.retrieve("Legacy Concept", document_key="doc-any")
            )
            # An empty document key is "unscoped", so it does not claim the row.
            self.assertEqual(1, len(memory.retrieve("Legacy Concept", document_key="")))
            memory.close()

    def test_alias_match_keeps_cjk_substring_semantics(self):
        # CJK is written without spaces; a gap in extracted text is an artefact.
        self.assertTrue(_alias_occurs_in("支持向量机", "本文使用支持向量机进行分类"))
        # A CJK token that merely *starts* with the alias still matches, because
        # segmenting Chinese needs a dictionary and the pre-existing behaviour
        # relies on substring semantics here.
        self.assertTrue(_alias_occurs_in("支持向量机", "支持向量机器"))

    def test_purge_unowned_removes_legacy_rows_and_their_children(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ConceptMemory(Path(temp_dir) / "memory.sqlite3")
            # One legacy (unowned) concept with an occurrence, one owned concept.
            memory.upsert("Legacy", section_id="s-old", page=3, source_quote="q")
            memory.upsert("Owned", document_key="doc-a", section_id="s-new", page=1)

            self.assertEqual(2, memory.count())
            result = memory.purge_unowned()

            self.assertEqual(1, result["concepts_deleted"])
            self.assertEqual(1, result["occurrences_deleted"])
            self.assertGreaterEqual(result["aliases_deleted"], 1)
            self.assertEqual(1, memory.count())
            # The legacy concept is gone from every table, not just `concepts`.
            self.assertEqual(
                0,
                memory.connection.execute(
                    "SELECT COUNT(*) FROM concepts WHERE document_key = ''"
                ).fetchone()[0],
            )
            self.assertEqual(
                0,
                memory.connection.execute(
                    "SELECT COUNT(*) FROM occurrences WHERE section_id = 's-old'"
                ).fetchone()[0],
            )
            # The owned concept survives and is still retrievable.
            self.assertEqual(
                1, len(memory.retrieve("Owned", document_key="doc-a"))
            )
            memory.close()

    def test_purge_unowned_is_a_noop_when_everything_is_owned(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ConceptMemory(Path(temp_dir) / "memory.sqlite3")
            memory.upsert("Owned", document_key="doc-a")

            result = memory.purge_unowned()

            self.assertEqual(0, result["concepts_deleted"])
            self.assertEqual(1, memory.count())
            memory.close()

    def test_purge_unowned_clears_cross_document_leak(self):
        # The purge exists because unowned rows were returned for ANY lookup,
        # which is the cross-document leak. After it, an unscoped lookup of a
        # legacy concept finds nothing.
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ConceptMemory(Path(temp_dir) / "memory.sqlite3")
            memory.upsert("Agentopia")

            self.assertEqual(1, len(memory.retrieve("Agentopia is a simulator")))
            memory.purge_unowned()

            self.assertEqual([], memory.retrieve("Agentopia is a simulator"))
            memory.close()

    def test_multi_word_alias_requires_all_tokens_adjacent(self):
        # A multi-word concept's alias key is glued into one word, so it must not
        # match a partial phrase. This strictness is what suppresses retrieval noise.
        self.assertFalse(
            _alias_occurs_in("knowledgeaugmentedtransformer", "the knowledge transformer")
        )

    def test_known_limitation_multi_word_alias_is_unmatchable(self):
        # KNOWN LIMITATION, asserted so it cannot regress silently into noise.
        #
        # `_concept_keys` glues a multi-word name into one key ("Knowledge Augmented
        # Transformer" -> "knowledgeaugmentedtransformer"), while matching works on
        # token runs separated by a boundary marker. The glued needle therefore
        # never appears in the separated haystack, so *multi-word* concept aliases
        # effectively never match. Single-word aliases ("agent", "AI", "KAT") and
        # CJK aliases are unaffected.
        #
        # The consequence is a recall loss for multi-word terminology, not a
        # precision loss, and it is the error direction a verifiable-extraction
        # project wants. Fixing it properly means tokenising the alias key at
        # storage time (changing concept identity, alias_key PRIMARY KEY collisions
        # and therefore the existing DB), so it is deliberately left as-is.
        self.assertFalse(
            _alias_occurs_in(
                "knowledgeaugmentedtransformer",
                "the Knowledge Augmented Transformer (KAT)",
            )
        )


class EvidenceMetricsTests(unittest.TestCase):
    """Deterministic provenance metrics over a finished run.

    These are the regression signals for the understanding layer: without them,
    "the 评价指标 answer degenerated" was only visible by reading output by hand.
    """

    def _section(self, facts, section_id="s1", title="T"):
        return {
            "section": {
                "id": section_id, "title": title,
                "page_start": 1, "page_end": 1,
            },
            "extractions": [{"summary": "s", "facts": facts}],
            "table_facts": [],
            "detected_facts": [],
            "metric_facts": [],
            "numeric_contradictions": [],
            "synthesis": {
                "summary": "s", "core_concepts": [], "key_findings": [],
                "source_pages": [1], "title": title,
            },
        }

    def _fact(self, fact_id, page=1, grounded=True, statement="一句足够长的陈述内容用来检验证据是否充分"):
        return {
            "fact_id": fact_id, "concept": "C", "statement": statement,
            "kind": "definition", "source_quote": "q", "page": page,
            "confidence": 0.9, "grounded": grounded,
        }

    def _answer(self, question, answer, fact_ids, unsupported=None, pages=None):
        return {
            "question": question, "answer": answer,
            "source_fact_ids": list(fact_ids),
            "source_sections": ["s1"] if fact_ids else [],
            "source_pages": list(pages if pages is not None else ([1] if fact_ids else [])),
            "confidence": 0.8,
            "unsupported_claims": list(unsupported or []),
        }

    def test_sentence_split_ignores_the_insufficient_marker(self):
        self.assertEqual([], _answer_sentences("证据不足"))
        self.assertEqual([], _answer_sentences(""))
        self.assertEqual(["一。", "二。"], _answer_sentences("一。二。"))

    def test_quantile_spans_zero_citation_answers(self):
        # Excluding zero-evidence answers would hide the failure the metric exists
        # to catch, so min must be able to reach 0.
        self.assertEqual(0, _quantile([], 0.5))
        self.assertEqual(0, _quantile([0, 1, 5], 0.0))
        self.assertEqual(5, _quantile([0, 1, 5], 1.0))
        self.assertEqual(1, _quantile([0, 1, 5], 0.5))

    def test_fully_grounded_run_reports_perfect_precision(self):
        results = [self._section([self._fact("f1"), self._fact("f2", page=2)])]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "一个句子的回答。", ["f1"]),
                self._answer("Q2", "另一个回答。", ["f1", "f2"], pages=[1, 2]),
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(2, m["fact_count"])
        self.assertEqual(1.0, m["citation_precision"])
        self.assertEqual(0.0, m["unsupported_claim_rate"])
        self.assertEqual(2, m["answers_with_evidence"])
        self.assertEqual(2, m["answers_with_sufficient_evidence"])
        self.assertEqual(1, m["min_cited_facts"])
        self.assertEqual(2, m["max_cited_facts"])

    def test_answer_without_evidence_is_counted_as_insufficient(self):
        results = [self._section([self._fact("f1")])]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "有据。", ["f1"]),
                self._answer("Q2", "证据不足", []),
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(1, m["answers_with_evidence"])
        self.assertEqual(1, m["answers_with_sufficient_evidence"])
        self.assertEqual(0, m["min_cited_facts"])
        self.assertFalse(m["per_answer"][1]["sufficient_evidence"])

    def test_citation_precision_drops_when_an_id_does_not_resolve(self):
        results = [self._section([self._fact("f1")])]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "回答。", ["f1", "f-ghost"]),
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(0.5, m["citation_precision"])
        self.assertEqual(0.5, m["per_answer"][0]["citation_resolvable_rate"])
        self.assertEqual(2, m["per_answer"][0]["cited_fact_count"])
        self.assertEqual(1, m["per_answer"][0]["resolvable_fact_count"])

    def test_unsupported_claim_rate_uses_the_models_total_output(self):
        # The audit DELETES ungrounded sentences from `answer` and records them in
        # `unsupported_claims`, so real output has them disjoint: `answer` holds
        # only the grounded sentences. The denominator is therefore
        # surviving + unsupported = the model's total output.
        # (Verified against real runs: answer sentences and unsupported_claims
        # never overlap.)
        results = [self._section([self._fact("f1")])]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "有据。", ["f1"], unsupported=["无据甲。", "无据乙。"]),
            ]
        }

        m = run_evidence_metrics(results, synthesis)
        item = m["per_answer"][0]

        self.assertEqual(1, item["supported_sentence_count"])   # kept in `answer`
        self.assertEqual(3, item["model_sentence_count"])       # 1 kept + 2 moved out
        self.assertEqual(
            item["model_sentence_count"],
            item["supported_sentence_count"] + item["unsupported_claim_count"],
        )
        self.assertEqual(2, m["total_unsupported_claims"])
        self.assertAlmostEqual(2 / 3, item["unsupported_claim_rate"], places=4)
        self.assertLessEqual(item["unsupported_claim_rate"], 1.0)

    def test_unsupported_claim_rate_never_exceeds_one(self):
        # A heavily pruned answer: 1 sentence survived, 5 were dropped. The old
        # denominator produced 5/1 = 5.0.
        results = [self._section([self._fact("f1")])]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "唯一保留的句子。", ["f1"],
                             unsupported=[f"被剔除的句子{i}。" for i in range(5)]),
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertAlmostEqual(5 / 6, m["per_answer"][0]["unsupported_claim_rate"], places=4)
        self.assertLessEqual(m["unsupported_claim_rate"], 1.0)

    def test_ungrounded_facts_are_not_evidence(self):
        results = [self._section([self._fact("f1", grounded=False)])]
        synthesis = {"global_answers": [self._answer("Q1", "回答。", ["f1"])]}

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(0, m["fact_count"])       # not reachable as evidence
        self.assertEqual(0.0, m["citation_precision"])

    def test_metric_and_table_facts_count_as_reachable_evidence(self):
        section = self._section([])
        section["metric_facts"] = [self._fact("m1")]
        section["table_facts"] = [{**self._fact("t1"), "kind": "metric"}]
        results = [section]
        synthesis = {"global_answers": [self._answer("Q1", "回答。", ["m1", "t1"])]}

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(2, m["fact_count"])
        self.assertEqual(1.0, m["citation_precision"])

    def test_handles_a_run_without_document_synthesis(self):
        results = [self._section([self._fact("f1")])]

        m = run_evidence_metrics(results, None)

        self.assertEqual(0, m["answer_count"])
        self.assertEqual(0, m["answers_with_evidence"])
        self.assertEqual(0.0, m["citation_precision"])
        self.assertEqual([], m["per_answer"])

    def test_accepts_a_model_instance_as_well_as_a_dict(self):
        results = [self._section([self._fact("f1")])]
        model = DocumentSynthesis(
            title="T",
            executive_summary="s",
            research_question="q",
            methods=[], datasets=[], metrics=[], key_findings=[],
            limitations=[], future_work=[], contradictions=[],
            global_answers=[
                GlobalAnswer(
                    question=question,
                    answer="回答。",
                    source_fact_ids=["f1"],
                    source_sections=["s1"],
                    source_pages=[1],
                    confidence=0.8,
                )
                for question in GLOBAL_QUESTIONS
            ],
        )

        m = run_evidence_metrics(results, model)

        self.assertEqual(10, m["answer_count"])
        self.assertEqual(10, m["answers_with_evidence"])
        self.assertEqual(1.0, m["citation_precision"])


    def test_report_writes_a_readable_table_for_an_existing_run(self):
        # The dev loop needs to recompute metrics over a stored run without paying
        # for another API pass, so this path must work from JSON alone.
        results = [self._section([self._fact("f1")])]
        synthesis = {"global_answers": [self._answer("Q1", "回答。", ["f1"])]}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run.json"
            path.write_text(
                json.dumps({"sections": results, "document_synthesis": synthesis},
                           ensure_ascii=False),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                metrics = report_evidence_metrics(path)
            output = buffer.getvalue()

        self.assertEqual(1, metrics["answer_count"])
        self.assertIn("citation_precision=1.0", output)
        self.assertIn("provenance metrics", output)

    def test_report_rejects_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            report_evidence_metrics("does-not-exist.json")


    def test_table_caption_is_emitted_once_not_per_row(self):
        # The caption used to be spliced into every row's statement, repeating one
        # sentence per row (measured: 55-69% of all table-fact tokens, the same
        # caption up to 150 times). It must now appear in exactly one header fact.
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=3,
                        text=(
                            "# 5 Results\n"
                            "**Table 2** Denoising results on SIDD\n"
                            "|Model|PSNR|\n"
                            "|---|---|\n"
                            "|Baseline|39.85|\n"
                            "|NAFNet|39.96|\n"
                        ),
                    )
                ]
            )
        )[0]

        facts = extract_table_facts(section)
        headers = [f for f in facts if f.concept == _TABLE_HEADER_CONCEPT]
        rows = [f for f in facts if f.concept != _TABLE_HEADER_CONCEPT]

        self.assertEqual(1, len(headers))
        self.assertEqual(2, len(rows))
        self.assertIn("Denoising results on SIDD", headers[0].statement)
        for row in rows:
            self.assertNotIn("Denoising results on SIDD", row.statement)
            self.assertEqual("Table 2", row.table)
        # The caption is still carried on every fact as a field, for provenance.
        self.assertTrue(all("Denoising results on SIDD" in f.table_caption for f in facts))

    def test_table_header_is_not_emitted_for_a_config_only_table(self):
        # A config table contributes no metric facts, so it must contribute no
        # header fact either -- otherwise the header alone leaks into the DB.
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=5,
                        text=(
                            "# 8 Discussion\n"
                            "**Table 8** Key configuration parameters\n"
                            "|Setting|Value|\n"
                            "|---|---|\n"
                            "|max_seq_len|4096|\n"
                        ),
                    )
                ]
            )
        )[0]
        self.assertEqual([], extract_table_facts(section))

    def test_table_row_metric_facts_are_compact(self):
        # A row statement must be the row's own content, not a caption plus content.
        section = flatten_sections(
            build_heading_tree(
                [
                    PageMarkdown(
                        page=3,
                        text=(
                            "# 5 Results\n"
                            "**Table 6** Image Denoising Results on SIDD\n"
                            "|Model|PSNR|\n"
                            "|---|---|\n"
                            "|NAFNet|40.30|\n"
                        ),
                    )
                ]
            )
        )[0]
        rows = [f for f in extract_table_facts(section) if f.concept != _TABLE_HEADER_CONCEPT]
        self.assertEqual(1, len(rows))
        # The row keeps its own context (Model=NAFNet) but never the caption.
        self.assertIn("PSNR=40.30", rows[0].statement)
        self.assertNotIn("Denoising", rows[0].statement)
        self.assertEqual("Table 6", rows[0].table)

    def test_retrieval_treats_explicit_table_field_as_a_table_row(self):
        # The "Table N:" prefix used to drive this; statements no longer carry it,
        # so the explicit field must count on its own.
        row = {
            "fact_id": "fact-tbl",
            "concept": "#",
            "statement": "#=1; Arcane Academy=0.42",
            "kind": "metric",
            "table": "Table 9",
            "section_title": "5.2 Analysis",
            "confidence": 1.0,
        }
        _, metric_allowed = build_question_request(
            "使用了哪些评价指标？", [row], [], max_tokens=600
        )
        self.assertNotIn("fact-tbl", metric_allowed)
        _, quant_allowed = build_question_request(
            "最重要的定量结果是什么？", [row], [], max_tokens=600
        )
        self.assertIn("fact-tbl", quant_allowed)


    def test_overview_audit_flags_a_number_absent_from_the_evidence(self):
        overview = {
            "executive_summary": "模型在某个基准上达到 99.9 dB 的最优性能。",
            "methods": [], "datasets": [], "metrics": [], "key_findings": [],
            "limitations": [], "future_work": [], "contradictions": [],
        }
        index = {
            "f1": {"concept": "PSNR", "statement": "在该基准上达到 40.30 dB。",
                   "source_quote": "40.30 dB"},
        }

        findings = audit_overview_numerics(overview, index)

        self.assertEqual(1, len(findings))
        self.assertEqual("unsupported_number", findings[0]["kind"])
        self.assertIn("99.9", findings[0]["numbers"])
        self.assertEqual("executive_summary", findings[0]["field"])

    def test_overview_audit_passes_when_every_number_is_supported(self):
        overview = {
            "executive_summary": "在 GoPro 达到 33.69 dB，在 SIDD 达到 40.30 dB。",
            "key_findings": ["超此前最佳 0.38 dB"],
            "methods": [], "datasets": [], "metrics": [],
            "limitations": [], "future_work": [], "contradictions": [],
        }
        index = {
            "f1": {"statement": "GoPro 33.69 dB。", "source_quote": "33.69 dB"},
            "f2": {"statement": "SIDD 40.30 dB。", "source_quote": "40.30 dB"},
            "f3": {"statement": "超过此前最佳 0.38 dB。", "source_quote": "0.38 dB"},
        }

        self.assertEqual([], audit_overview_numerics(overview, index))

    def test_overview_audit_does_not_flag_numbers_borrowed_across_facts(self):
        # A sentence legitimately combining numbers from several facts must NOT be
        # flagged: a "no single fact holds them all" rule fired on ~every real
        # multi-number sentence, which made it useless as a signal.
        overview = {
            "executive_summary": "GoPro 达 33.69 dB、SIDD 达 40.30 dB，均超 SOTA。",
            "methods": [], "datasets": [], "metrics": [], "key_findings": [],
            "limitations": [], "future_work": [], "contradictions": [],
        }
        index = {
            "f1": {"statement": "GoPro 33.69 dB。", "source_quote": "33.69 dB"},
            "f2": {"statement": "SIDD 40.30 dB。", "source_quote": "40.30 dB"},
        }

        self.assertEqual([], audit_overview_numerics(overview, index))

    def test_overview_audit_normalises_scale_words(self):
        # A real case: the answer said "3600万" where the source says "36M". Bare
        # digit comparison reported that correct statement as unsupported.
        self.assertIn("36000000", _numeric_forms("36M sentences"))
        index = {"f1": {"statement": "英法数据集包含 36M 句子。",
                        "source_quote": "36M sentences"}}
        overview = {
            "executive_summary": "英法数据集包含 3600万 句子。",
            "methods": [], "datasets": [], "metrics": [], "key_findings": [],
            "limitations": [], "future_work": [], "contradictions": [],
        }

        self.assertEqual([], audit_overview_numerics(overview, index))

    def test_overview_audit_scans_every_overview_field(self):
        overview = {
            "executive_summary": "", "research_question": "是否使用 12345 个样本？",
            "methods": ["采用 54321 步训练"], "datasets": [], "metrics": [],
            "key_findings": [], "limitations": [], "future_work": [], "contradictions": [],
        }

        findings = audit_overview_numerics(overview, {})

        fields = {f["field"] for f in findings}
        self.assertIn("research_question", fields)
        self.assertIn("methods", fields)
        self.assertEqual(2, len(findings))

    def test_overview_audit_tolerates_a_model_and_empty_input(self):
        self.assertEqual([], audit_overview_numerics(None, {}))
        self.assertEqual([], audit_overview_numerics({}, {}))
        overview = DocumentOverview(
            title="T", executive_summary="达到 1.5 dB 提升。", research_question="q",
            methods=[], datasets=[], metrics=[], key_findings=[],
            limitations=[], future_work=[], contradictions=[],
        )
        self.assertEqual([], audit_overview_numerics(overview, {
            "f1": {"statement": "提升 1.5 dB", "source_quote": "1.5 dB"},
        }))

    def test_metrics_report_the_overview_number_audit(self):
        results = [self._section([self._fact("f1", statement="达到 40.30 dB。")])]
        synthesis = {
            "executive_summary": "达到 99.9 dB。",
            "global_answers": [self._answer("Q1", "回答。", ["f1"])],
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(1, m["overview_unsupported_numbers"])
        self.assertEqual(1, len(m["overview_numeric_findings"]))

    def test_backcheck_separates_retrieval_gap_from_undecided(self):
        # `_audit_answer_claims` only tests against the facts actually cited, so a
        # sentence grounded in the document but not retrieved is reported as
        # "unsupported". The backcheck re-tests it against the whole DB and marks
        # what the DB carries as a retrieval gap.
        results = [
            self._section([
                self._fact("f1", statement="窗口注意力仅在固定大小的局部窗口内应用自注意力，但缺乏全局信息。"),
                self._fact("f2", statement="在256×256输入下，36块延迟39.1 ms。"),
            ])
        ]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "回答。", ["f1"], unsupported=[
                    "窗口注意力缺乏全局信息；",
                    "该模型在量子化学基准上取得了最优结果。",
                ]),
            ]
        }

        m = run_evidence_metrics(results, synthesis)
        by_claim = {v["claim"]: v["verdict"] for v in m["per_answer"][0]["unsupported_claims"]}

        self.assertEqual("retrieval_gap", by_claim["窗口注意力缺乏全局信息；"])
        self.assertEqual(1, m["unsupported_retrieval_gaps"])
        self.assertEqual(1, m["unsupported_undecided"])

    def test_backcheck_never_claims_fabrication(self):
        # A synonym and an invented sentence both score 0.00 on vocabulary overlap,
        # so the check must not label either one fabricated. `undecided` is an
        # upper bound on fabrication, not a verdict.
        results = [
            self._section([self._fact("f1", statement="column_1=NAFNet; speedup=1.09×; SIDD PSNR=39.96")])
        ]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "回答。", ["f1"], unsupported=[
                    "消融中还涉及加速比。",              # synonym for "speedup"
                    "该模型在量子化学基准上取得了最优结果。",  # invented
                ])
            ]
        }

        m = run_evidence_metrics(results, synthesis)
        verdicts = m["per_answer"][0]["unsupported_claims"]

        self.assertEqual(2, m["unsupported_undecided"])
        self.assertEqual(0, m["unsupported_retrieval_gaps"])
        self.assertTrue(all(v["verdict"] == "undecided" for v in verdicts))
        # The coverage is reported so the case can be triaged by hand.
        self.assertTrue(all("coverage" in v for v in verdicts))

    def test_backcheck_accepts_a_claim_that_combines_several_facts(self):
        # A claim stitching three facts together has low overlap with any single
        # fact; coverage must accumulate across facts or it would look fabricated.
        results = [
            self._section([
                self._fact("f1", statement="窗口注意力缺乏全局信息。"),
                self._fact("f2", statement="通道注意力聚合全局信息与通道交互。"),
                self._fact("f3", statement="局部增强前馈网络增强局部信息捕获能力。"),
            ])
        ]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "回答。", ["f1"], unsupported=[
                    "还涉及通道注意力、窗口多头自注意力、局部增强前馈网络等已有方法。"
                ])
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual("retrieval_gap", m["per_answer"][0]["unsupported_claims"][0]["verdict"])

    def test_backcheck_uses_number_agreement_as_evidence(self):
        # A numeric claim whose numbers all appear in one fact is real content even
        # when the wording is entirely different.
        results = [
            self._section([
                self._fact("f1", statement="该配置的数值为 40.30 与 0.28。"),
            ])
        ]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "回答。", ["f1"], unsupported=["指标达到了 40.30，提升 0.28。"])
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual("retrieval_gap", m["per_answer"][0]["unsupported_claims"][0]["verdict"])

    def test_backcheck_breakdown_sums_to_total_unsupported(self):
        results = [
            self._section([self._fact("f1", statement="窗口注意力缺乏全局信息。")])
        ]
        synthesis = {
            "global_answers": [
                self._answer("Q1", "回答。", ["f1"], unsupported=[
                    "窗口注意力缺乏全局信息。",
                    "完全无关的量子化学结论。",
                ])
            ]
        }

        m = run_evidence_metrics(results, synthesis)

        self.assertEqual(
            m["total_unsupported_claims"],
            m["unsupported_retrieval_gaps"] + m["unsupported_undecided"],
        )

    def test_claim_matches_fact_db_threshold_behaviour(self):
        # Directly pin the boundary so a future tuning change is deliberate.
        facts = [(_claim_tokens("窗口注意力仅在局部窗口内应用自注意力，但缺乏全局信息。"), set())]

        # A close paraphrase clears the threshold.
        self.assertTrue(_claim_matches_fact_db("窗口注意力缺乏全局信息。", facts))
        # Unrelated wording does not, and must not be called fabricated elsewhere.
        self.assertFalse(_claim_matches_fact_db("量子化学基准上的最优结果。", facts))
        # An empty claim carries no evidence either way.
        self.assertFalse(_claim_matches_fact_db("", facts))


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

    def test_future_question_request_blocks_method_facts_with_future_markers(self):
        # The hard kind gate must exclude method facts even when they carry future
        # keywords or sit in a future-work section (the additive keyword +40
        # section bonus used to lift them above the score>0 cutoff).
        facts = [
            {
                "fact_id": "fact-method-future-section",
                "statement": "未来的工作将扩展该方法的并行训练策略。",
                "kind": "method",
                "section_title": "6 Future Work",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-method-keywords",
                "statement": "该方法的目标是改进推理性能，后续计划部署。",
                "kind": "method",
                "section_title": "3 Methods",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-limitation",
                "statement": "当前方法在噪声环境下鲁棒性有限。",
                "kind": "limitation",
                "section_title": "5 Discussion",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-result",
                "statement": "未来计划将智能体数量扩展到 1000。",
                "kind": "result",
                "section_title": "8 Conclusion",
                "confidence": 1.0,
            },
        ]

        _, allowed = build_question_request(GLOBAL_QUESTIONS[8], facts, [], max_tokens=600)

        self.assertNotIn("fact-method-future-section", allowed)
        self.assertNotIn("fact-method-keywords", allowed)
        self.assertIn("fact-limitation", allowed)
        self.assertIn("fact-result", allowed)

    def test_future_question_request_fallback_when_no_limitation_or_result(self):
        # The gate falls back to the soft scorer when the document has zero
        # limitation/result facts, so such a run does not degrade into an empty
        # catalog / "证据不足". The plain method fact still scores 0.
        facts = [
            {
                "fact_id": "fact-method-future-section",
                "statement": "未来的工作将扩展并行训练策略。",
                "kind": "method",
                "section_title": "6 Future Work",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-method-plain",
                "statement": "方法采用 MFCC 特征。",
                "kind": "method",
                "section_title": "3 Methods",
                "confidence": 1.0,
            },
        ]

        _, allowed = build_question_request(GLOBAL_QUESTIONS[8], facts, [], max_tokens=600)

        self.assertIn("fact-method-future-section", allowed)
        self.assertNotIn("fact-method-plain", allowed)

    def test_future_gate_tied_to_route_not_raw_text(self):
        # The gate keys off the *resolved* route, not raw question text: a
        # question that merely contains 未来/改进 but routes to the dataset
        # branch must keep its own kind mix untouched.
        facts = [
            {
                "fact_id": "fact-dataset",
                "statement": "未来工作中计划使用 UrbanSound8K 数据集。",
                "kind": "dataset",
                "section_title": "3 Methods",
                "confidence": 1.0,
            },
            {
                "fact_id": "fact-method",
                "statement": "方法采用 MFCC 特征。",
                "kind": "method",
                "section_title": "3 Methods",
                "confidence": 1.0,
            },
        ]

        _, allowed = build_question_request(
            "未来工作中计划使用哪些数据集或材料？", facts, [], max_tokens=600
        )

        self.assertIn("fact-dataset", allowed)
        self.assertNotIn("fact-method", allowed)

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

    def test_summary_reports_only_the_current_run(self):
        # The usage DB is append-only and shared between runs, so `summary()` must
        # not report the lifetime total: a prior run's calls must be excluded while
        # `all_time_summary()` still counts them.
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "usage.sqlite3"

            first = UsageTracker(db)
            first.record("chunk_extract", "flash", 1000, 500, 1500, 3.0, "req-a")
            first.record("chunk_extract", "flash", 1000, 500, 1500, 3.0, "req-b")
            first.close()

            second = UsageTracker(db)
            second.record("section_reduce", "pro", 200, 40, 240, 2.5, "req-c")

            current = second.summary()
            self.assertEqual(current["calls"], 1)
            self.assertEqual(current["total_tokens"], 240)
            self.assertEqual(set(current["by_model"]), {"pro"})
            self.assertIn("run_started_at", current)

            lifetime = second.all_time_summary()
            self.assertEqual(lifetime["calls"], 3)
            self.assertEqual(lifetime["total_tokens"], 3240)
            self.assertEqual(lifetime["by_model"]["flash"]["calls"], 2)
            self.assertEqual(lifetime["by_model"]["pro"]["calls"], 1)
            self.assertNotIn("run_started_at", lifetime)
            second.close()

    def test_summary_is_empty_for_a_run_that_records_nothing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "usage.sqlite3"
            writer = UsageTracker(db)
            writer.record("chunk_extract", "flash", 10, 5, 15, 0.1, "req-1")
            writer.close()

            idle = UsageTracker(db)
            summary = idle.summary()
            self.assertEqual(summary["calls"], 0)
            self.assertEqual(summary["total_tokens"], 0)
            self.assertEqual(summary["by_model"], {})
            self.assertEqual(idle.all_time_summary()["calls"], 1)
            idle.close()

    def test_stats_separates_run_usage_from_lifetime_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "usage.sqlite3"
            previous = UsageTracker(db)
            previous.record("chunk_extract", "flash", 1000, 500, 1500, 3.0, "req-old")
            previous.close()

            pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
            pipeline.usage = UsageTracker(db)
            pipeline.cache = ResultCache(Path(temp_dir) / "cache.sqlite3")
            pipeline.group_fallbacks = 0
            pipeline.usage.record("global_answer", "flash", 100, 10, 110, 1.0, "req-new")

            stats = pipeline.stats()

            self.assertEqual(stats["usage"]["calls"], 1)
            self.assertEqual(stats["usage"]["total_tokens"], 110)
            self.assertEqual(stats["all_time_usage"]["calls"], 2)
            self.assertEqual(stats["all_time_usage"]["total_tokens"], 1610)
            pipeline.usage.close()
            pipeline.cache.close()

    def test_stats_reports_retry_configuration(self):
        # A slow or expensive run must be explainable without reading the code, so
        # the retry/timeout settings and their worst-case product are surfaced.
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
            pipeline.usage = UsageTracker(Path(temp_dir) / "usage.sqlite3")
            pipeline.cache = ResultCache(Path(temp_dir) / "cache.sqlite3")
            pipeline.request_timeout_seconds = 120.0
            pipeline.sdk_max_retries = 3
            pipeline.content_retry_attempts = 3

            config = pipeline.stats()["config"]

            self.assertEqual(120.0, config["request_timeout_seconds"])
            self.assertEqual(3, config["sdk_max_retries"])
            self.assertEqual(3, config["content_retry_attempts"])
            self.assertEqual(9, config["max_attempts_per_request"])
            pipeline.usage.close()
            pipeline.cache.close()

    def test_response_request_id_prefers_sdk_id_attribute(self):
        # Real SDK responses carry `id`. The old code read `_request_id`, which no
        # SDK object defines, so the usage column was always empty.
        response = SimpleNamespace(id="chatcmpl-abc123", _request_id="legacy-id")

        self.assertEqual("chatcmpl-abc123", _response_request_id(response))

    def test_response_request_id_falls_back_and_tolerates_missing(self):
        self.assertEqual("legacy-id", _response_request_id(SimpleNamespace(_request_id="legacy-id")))
        self.assertEqual("", _response_request_id(SimpleNamespace()))
        self.assertEqual("", _response_request_id(SimpleNamespace(id="", _request_id="")))
        self.assertEqual("", _response_request_id(SimpleNamespace(id=None)))

    def test_sdk_exposes_the_id_attribute_this_code_reads(self):
        # Guard against the attribute name drifting away from the installed SDK.
        from openai.types.chat import ChatCompletion

        self.assertIn("id", ChatCompletion.model_fields)

    def test_recorded_usage_carries_the_provider_request_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
            pipeline.usage = UsageTracker(Path(temp_dir) / "usage.sqlite3")

            response = SimpleNamespace(
                id="chatcmpl-real-1",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
            pipeline._record_usage("chunk_extract", "flash", response, 0.0)

            row = pipeline.usage.connection.execute(
                "SELECT request_id FROM usage"
            ).fetchone()
            self.assertEqual("chatcmpl-real-1", row[0])
            pipeline.usage.close()

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


class DocumentTitleTests(unittest.TestCase):
    def _node(self, node_id, title, level, parent, children=(), page=1):
        node = HeadingNode(
            id=node_id, title=title, level=level, page_start=page, page_end=page,
            subtree_page_end=page, parent_id=parent, text="body",
        )
        node.children = list(children)
        return node

    def _overview(self, title):
        return DocumentOverview(
            title=title,
            executive_summary="本文综述了 Agent AI 在多模态交互方向的研究进展。",
            research_question="Agent AI 如何实现多模态交互？",
            methods=[], datasets=[], metrics=[], key_findings=[],
            limitations=[], future_work=[], contradictions=[],
        )

    def test_candidate_document_title_returns_paper_title(self):
        title = self._node(
            "title", "Agent AI: Surveying the Horizons of Multimodal Interaction", 1, "root"
        )
        abstract = self._node("abs", "Abstract", 2, "root")
        intro = self._node("intro", "1 Introduction", 2, "root")
        root = self._node("root", "Document", 0, None, [title, abstract, intro])

        self.assertEqual(
            _candidate_document_title(root),
            "Agent AI: Surveying the Horizons of Multimodal Interaction",
        )

    def test_candidate_document_title_accepts_level2_title(self):
        # A title at raw ## stays at L2 when _normalize_levels finds min_level==2
        # (no shift) -- never require a specific level.
        title = self._node("title", "Agentopia: A Study of Long-Horizon Behavior", 2, "root")
        intro = self._node("intro", "3 Agentopia", 2, "root")
        root = self._node("root", "Document", 0, None, [title, intro])

        self.assertEqual(
            _candidate_document_title(root),
            "Agentopia: A Study of Long-Horizon Behavior",
        )

    def test_candidate_document_title_rejects_front_matter(self):
        # _section_signature only catches Abstract/摘要 and numbered sections, so
        # Introduction must be rejected by the front-matter blacklist.
        abstract = self._node("abs", "Abstract", 1, "root")
        intro = self._node("intro", "Introduction", 1, "root")
        numbered = self._node("num", "1 Introduction", 1, "root")
        root = self._node("root", "Document", 0, None, [abstract, intro, numbered])

        self.assertIsNone(_candidate_document_title(root))

    def test_candidate_document_title_skips_toc_and_back_matter(self):
        toc = self._node("toc", "Contents", 1, "root")
        title = self._node("title", "The Real Title", 1, "root")
        refs = self._node("refs", "References", 1, "root")
        root = self._node("root", "Document", 0, None, [toc, title, refs])

        self.assertEqual(_candidate_document_title(root), "The Real Title")

    def test_candidate_document_title_rejects_running_header(self):
        header = self._node("hdr", "Discover Internet of Things", 1, "root")
        root = self._node("root", "Document", 0, None, [header])

        self.assertIsNone(_candidate_document_title(root))

    def test_candidate_document_title_joins_split_title_lines(self):
        # The Agent AI PDF lays the title out as two sibling heading nodes;
        # the first line alone ("AGENT AI:") is truncated.
        line1 = self._node("t1", "AGENT AI:", 1, "root")
        line2 = self._node("t2", "SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION", 1, "root")
        abstract = self._node("abs", "Abstract", 2, "root")
        root = self._node("root", "Document", 0, None, [line1, line2, abstract])

        self.assertEqual(
            _candidate_document_title(root),
            "AGENT AI: SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION",
        )

    def test_candidate_document_title_does_not_join_section_after_plain_title(self):
        # No trailing colon -> never treat the next sibling as a continuation,
        # otherwise "My Paper" would swallow a following unnumbered section.
        title = self._node("title", "My Paper", 1, "root")
        related = self._node("rel", "Related Work", 1, "root")
        root = self._node("root", "Document", 0, None, [title, related])

        self.assertEqual(_candidate_document_title(root), "My Paper")

    def test_candidate_document_title_joins_then_stops_at_section(self):
        # Joining continues past plausible title lines but stops at a numbered
        # section (filtered by _section_signature).
        line1 = self._node("t1", "Agent AI:", 1, "root")
        line2 = self._node("t2", "Surveying the Horizons of Multimodal Interaction", 1, "root")
        intro = self._node("intro", "1 Introduction", 2, "root")
        root = self._node("root", "Document", 0, None, [line1, line2, intro])

        self.assertEqual(
            _candidate_document_title(root),
            "Agent AI: Surveying the Horizons of Multimodal Interaction",
        )

    def test_candidate_document_title_does_not_join_across_pages(self):
        line1 = self._node("t1", "Agent AI:", 1, "root", page=1)
        line2 = self._node("t2", "Surveying the Horizons", 1, "root", page=2)
        root = self._node("root", "Document", 0, None, [line1, line2])

        self.assertEqual(_candidate_document_title(root), "Agent AI:")

    def test_apply_document_title_overrides_hallucinated_title(self):
        result = self._overview("多模态代理论坛")

        _apply_document_title(result, "Agent AI: Surveying the Horizons of Multimodal Interaction")

        self.assertEqual(result.title, "Agent AI: Surveying the Horizons of Multimodal Interaction")

    def test_apply_document_title_keeps_verbatim_and_same_language_variant(self):
        # English candidate with the verbatim English title, and a Chinese
        # candidate with a same-language paraphrase: both share enough tokens and
        # must be kept.
        for candidate, kept in (
            (
                "Agent AI: Surveying the Horizons of Multimodal Interaction",
                "Agent AI: Surveying the Horizons of Multimodal Interaction",
            ),
            ("面向多模态智能代理的综述", "多模态智能代理综述"),
        ):
            result = self._overview(kept)
            _apply_document_title(result, candidate)
            self.assertEqual(result.title, kept)

    def test_apply_document_title_overrides_faithful_translation(self):
        # A faithful Chinese translation of an English candidate scores ~0.2 token
        # overlap and is overridden back to the original -- the intended behavior
        # per the "title keeps its original language" decision.
        result = self._overview("Agent AI：多模态交互方向综述")

        _apply_document_title(result, "Agent AI: Surveying the Horizons of Multimodal Interaction")

        self.assertEqual(result.title, "Agent AI: Surveying the Horizons of Multimodal Interaction")

    def test_apply_document_title_overrides_truncated_prefix(self):
        # The heading tree joins "AGENT AI:" + "SURVEYING..." into one candidate,
        # but a model that echoes only the first line passes the old overlap>=0.5
        # rule (answer tokens are a 100% subset of the candidate). The coverage
        # rule must catch it and restore the full title.
        result = self._overview("AGENT AI:")

        _apply_document_title(result, "Agent AI: Surveying the Horizons of Multimodal Interaction")

        self.assertEqual(result.title, "Agent AI: Surveying the Horizons of Multimodal Interaction")

    def test_apply_document_title_skips_override_when_candidate_absent_from_summary(self):
        # A byline/journal name the heading tree promoted to a heading passes the
        # filters, but the model's content never mentions it -- the consistency
        # check must refuse to force it.
        result = self._overview("多模态代理论坛")

        _apply_document_title(result, "John Doe, MIT")

        self.assertEqual(result.title, "多模态代理论坛")

    def test_fallback_document_overview_accepts_title(self):
        overview = fallback_document_overview([], [], title="My Paper")
        self.assertEqual(overview.title, "My Paper")

        overview = fallback_document_overview([], [])
        self.assertEqual(overview.title, "Document")


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

    async def _make_extract_pipeline(self, scripted_responses):
        """Minimal pipeline wired for extract_chunk: adds cache/memory/flash-model
        on top of the recovery helper so the full prompt + first-call kwargs are
        visible via FakeCompletions."""
        pipeline, completions = await self._make_recovery_pipeline(scripted_responses)
        pipeline.flash_model = "flash"
        # Keep the pro model distinct from None: adding metric facts makes a
        # section carry more than one extraction, which routes the section reduce
        # to the pro model.
        pipeline.pro_model = "pro"
        pipeline.extract_max_output_tokens = 4096
        pipeline.reduce_max_output_tokens = 2200
        self._extract_dir = tempfile.mkdtemp()
        pipeline.cache = ResultCache(Path(self._extract_dir) / "cache.sqlite3")
        pipeline.memory = ConceptMemory(Path(self._extract_dir) / "memory.sqlite3")
        return pipeline, completions

    async def _cleanup_extract(self, pipeline) -> None:
        pipeline.cache.close()
        pipeline.memory.close()
        await self._cleanup_usage(pipeline)
        shutil.rmtree(self._extract_dir, ignore_errors=True)

    async def test_recorded_latency_excludes_semaphore_queueing(self):
        # The clock must start after the semaphore is acquired. Starting it before
        # charged every row for concurrency queueing, which inflated reported
        # latency and made slow-stage conclusions unreliable.
        import time as _time

        class SlowCompletions:
            async def create(self, **kwargs):
                await asyncio.sleep(0.05)
                return SimpleNamespace(
                    id="req-1",
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"summary":"s","facts":[],"new_concepts":[]}'
                        ),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        self._usage_dir = tempfile.mkdtemp()
        pipeline = LongDocumentPipeline.__new__(LongDocumentPipeline)
        pipeline.client = SimpleNamespace(chat=SimpleNamespace(completions=SlowCompletions()))
        pipeline.semaphore = asyncio.Semaphore(1)
        pipeline.usage = UsageTracker(Path(self._usage_dir) / "usage.sqlite3")
        pipeline.content_retry_attempts = 1
        pipeline.content_retry_base_seconds = 0
        pipeline.truncation_recovery_attempts = 1
        pipeline.document_key = ""

        # Occupy the semaphore so the request must queue first.
        await pipeline.semaphore.acquire()

        async def release_later():
            await asyncio.sleep(0.15)
            pipeline.semaphore.release()

        asyncio.ensure_future(release_later())

        started = _time.perf_counter()
        await pipeline._request_json(
            "chunk_extract", "flash", "sys", "user", response_model=ChunkExtraction
        )
        wall = _time.perf_counter() - started

        recorded = pipeline.usage.connection.execute(
            "SELECT latency_seconds FROM usage"
        ).fetchone()[0]
        # Wall time includes the queue; the recorded latency must not.
        self.assertGreater(wall, 0.15, "test did not actually queue")
        self.assertLess(recorded, 0.12, f"latency leaked queue time: {recorded}")
        self.assertGreater(recorded, 0.0)
        await self._cleanup_usage(pipeline)

    async def test_recovery_calls_record_a_duration_not_a_timestamp(self):
        # Regression guard: after `_record_usage` switched from taking a start
        # timestamp to taking an elapsed duration, two recovery call sites kept
        # passing `started`, so their rows recorded ~7,000,000 seconds. Every
        # call site must record a plausible duration.
        valid = {"summary": "摘要", "facts": [], "new_concepts": []}
        pipeline, completions = await self._make_recovery_pipeline(
            [("length", '{"summary":"部分'), ("stop", json.dumps(valid, ensure_ascii=False))]
        )

        await pipeline._request_json(
            "chunk_extract", "flash", "system", "user", response_model=ChunkExtraction
        )

        rows = pipeline.usage.connection.execute(
            "SELECT stage, latency_seconds FROM usage"
        ).fetchall()
        self.assertGreaterEqual(len(rows), 2, "expected a main call plus a recovery call")
        # The point of the test is the recovery path, so require it to have run.
        stages = {stage for stage, _ in rows}
        self.assertTrue(
            any(stage.endswith("_cont") or stage.endswith("_regen") for stage in stages),
            f"recovery path not exercised; stages={stages}",
        )
        for stage, latency in rows:
            with self.subTest(stage=stage):
                # A real API call is seconds, never the value of perf_counter().
                self.assertGreaterEqual(latency, 0.0)
                self.assertLess(latency, 600.0, f"{stage} recorded {latency}s")
        await self._cleanup_usage(pipeline)

    async def test_extract_chunk_initial_call_thinks_off(self):
        # The first flash extraction call must not burn the output budget on
        # reasoning: reasoning_effort="none", max_tokens aligned to the pipeline
        # budget (4096), while all later stages keep their default "low".
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
        pipeline, completions = await self._make_extract_pipeline(
            [("stop", json.dumps(valid, ensure_ascii=False))]
        )
        section = HeadingNode(
            id="s1", title="方法", level=2,
            page_start=1, page_end=1, subtree_page_end=1,
        )
        chunk = TextChunk(index=0, text="原文", token_estimate=10)

        result = await pipeline.extract_chunk(section, chunk)

        self.assertEqual(len(result.facts), 1)
        self.assertEqual(completions.calls, 1)
        first = completions.kwargs[0]
        self.assertEqual(first["reasoning_effort"], "none")
        self.assertEqual(first["max_tokens"], pipeline.extract_max_output_tokens)
        # result is written to cache so a second call would hit (prompt unchanged)
        self.assertEqual(pipeline.cache.summary()["misses"], 1)
        await self._cleanup_extract(pipeline)

    async def test_extract_chunk_prompt_caps_output(self):
        # The extraction prompt must bound output size so dense chunks stay under
        # the budget: a fact-count cap and a short source_quote instruction.
        valid = {
            "summary": "摘要",
            "facts": [],
            "new_concepts": [],
        }
        pipeline, completions = await self._make_extract_pipeline(
            [("stop", json.dumps(valid, ensure_ascii=False))]
        )
        section = HeadingNode(
            id="s1", title="方法", level=2,
            page_start=1, page_end=1, subtree_page_end=1,
        )
        chunk = TextChunk(index=0, text="原文", token_estimate=10)

        await pipeline.extract_chunk(section, chunk)

        system_prompt = completions.kwargs[0]["messages"][0]["content"]
        self.assertIn("最多输出 20 条", system_prompt)
        self.assertIn("source_quote 尽量简短", system_prompt)
        await self._cleanup_extract(pipeline)

    async def test_extract_chunk_prompt_requests_prose_metrics(self):
        # The prompt must ask for kind="metric" explicitly and say that metrics in
        # body prose count, not just table rows. Otherwise the LLM never emits the
        # kind and the metric route depends solely on markdown table parsing.
        valid = {"summary": "摘要", "facts": [], "new_concepts": []}
        pipeline, completions = await self._make_extract_pipeline(
            [("stop", json.dumps(valid, ensure_ascii=False))]
        )
        section = HeadingNode(
            id="s1", title="方法", level=2,
            page_start=1, page_end=1, subtree_page_end=1,
        )
        chunk = TextChunk(index=0, text="原文", token_estimate=10)

        await pipeline.extract_chunk(section, chunk)

        system_prompt = completions.kwargs[0]["messages"][0]["content"]
        self.assertIn("metric", system_prompt)
        self.assertIn("正文叙述", system_prompt)
        self.assertIn("准确率", system_prompt)
        await self._cleanup_extract(pipeline)

    async def test_process_section_adds_prose_metric_facts(self):
        # Full section flow: the LLM contributes no metric facts, yet the section
        # result must still carry deterministic prose metric facts, and the
        # evidence builder must surface them as kind="metric".
        metric_text = (
            "Evaluation on a VLN benchmark dataset shows that our RCM model "
            "significantly outperforms previous methods by 10% on SPL."
        )
        valid = {"summary": "摘要", "facts": [], "new_concepts": []}
        synthesis = {
            "title": "结果",
            "summary": "章节摘要",
            "core_concepts": [],
            "key_findings": [],
            "source_pages": [1],
        }
        pipeline, _ = await self._make_extract_pipeline(
            [
                ("stop", json.dumps(valid, ensure_ascii=False)),
                ("stop", json.dumps(synthesis, ensure_ascii=False)),
            ]
        )
        section = HeadingNode(
            id="s-metric", title="结果", level=2,
            page_start=1, page_end=1, subtree_page_end=1,
            text=metric_text,
            page_texts=[PageMarkdown(page=1, text=metric_text)],
        )

        result = await pipeline.process_section(section)

        metric_facts = result["metric_facts"]
        self.assertEqual(1, len(metric_facts))
        self.assertEqual("SPL", metric_facts[0]["concept"])
        self.assertEqual("metric", metric_facts[0]["kind"])
        # The synthetic extraction must be visible to the section reduce too.
        self.assertEqual(2, len(result["extractions"]))
        self.assertIn("确定性抽取的评价指标事实", result["extractions"][-1]["summary"])

        evidence = build_section_evidence(result)
        metric_evidence = [f for f in evidence["facts"] if f["kind"] == "metric"]
        self.assertEqual(1, len(metric_evidence))
        self.assertEqual(1, metric_evidence[0]["page"])
        self.assertTrue(metric_evidence[0]["grounded"])
        await self._cleanup_extract(pipeline)

    async def _make_reduce_pipeline(self, scripted_responses):
        """Minimal pipeline wired for the reduce stages (section + group): adds
        cache/flash-model/reduce budget so the first-call kwargs are visible."""
        pipeline, completions = await self._make_recovery_pipeline(scripted_responses)
        pipeline.flash_model = "flash"
        pipeline.pro_model = None
        pipeline.reduce_max_output_tokens = 2200
        pipeline.group_fallbacks = 0
        pipeline._document_fact_index = {}
        self._reduce_dir = tempfile.mkdtemp()
        pipeline.cache = ResultCache(Path(self._reduce_dir) / "cache.sqlite3")
        return pipeline, completions

    async def _cleanup_reduce(self, pipeline) -> None:
        pipeline.cache.close()
        await self._cleanup_usage(pipeline)
        shutil.rmtree(self._reduce_dir, ignore_errors=True)

    async def test_synthesize_section_initial_call_thinks_off(self):
        # Section reduce is mechanical summarization with hard output caps; the
        # first call must not burn the 2200 budget on reasoning_content.
        valid = {
            "title": "方法",
            "summary": "摘要",
            "core_concepts": ["概念"],
            "key_findings": ["结论"],
            "source_pages": [1],
        }
        pipeline, completions = await self._make_reduce_pipeline(
            [("stop", json.dumps(valid, ensure_ascii=False))]
        )
        section = HeadingNode(
            id="s1", title="方法", level=2,
            page_start=1, page_end=1, subtree_page_end=1,
        )
        extraction = ChunkExtraction.model_validate(
            {
                "summary": "摘要",
                "facts": [
                    {
                        "concept": "概念",
                        "statement": "事实",
                        "kind": "result",
                        "source_quote": "原文",
                        "page": 1,
                        "confidence": 0.9,
                        "grounded": True,
                    }
                ],
                "new_concepts": [],
            }
        )

        result = await pipeline.synthesize_section(section, [extraction])

        self.assertEqual(result.summary, "摘要")
        self.assertEqual(completions.calls, 1)
        first = completions.kwargs[0]
        self.assertEqual(first["reasoning_effort"], "none")
        self.assertEqual(first["max_tokens"], pipeline.reduce_max_output_tokens)
        await self._cleanup_reduce(pipeline)

    async def test_reduce_group_initial_call_thinks_off(self):
        # Document group reduce merged 81 calls at 81% reasoning (42% truncation)
        # on the v224 run; the first call must run with reasoning_effort="none".
        valid = {
            "summary": "组摘要",
            "concepts": ["概念"],
            "findings": ["结论"],
            "numeric_claims": [],
            "limitations": [],
            "future_work": [],
            "contradictions": [],
            "source_fact_ids": ["f1"],
            "source_sections": [],
            "source_pages": [],
        }
        pipeline, completions = await self._make_reduce_pipeline(
            [("stop", json.dumps(valid, ensure_ascii=False))]
        )
        items = [{"fact_id": "f1", "section_id": "s1", "concept": "概念", "statement": "事实"}]

        result = await pipeline._reduce_group(items, 0, 0)

        self.assertEqual(result.summary, "组摘要")
        self.assertEqual(completions.calls, 1)
        first = completions.kwargs[0]
        self.assertEqual(first["reasoning_effort"], "none")
        self.assertEqual(first["max_tokens"], pipeline.reduce_max_output_tokens)
        await self._cleanup_reduce(pipeline)

    async def test_synthesize_overview_injects_candidate_title_and_backstop(self):
        overview_json = {
            "title": "多模态代理论坛",
            "executive_summary": "本文综述了 Agent AI 在多模态交互方向的研究进展，涵盖方法、基准与伦理讨论。",
            "research_question": "Agent AI 如何通过生成式 AI 实现现实无关训练并推动 AGI 发展？",
            "methods": ["KAT"],
            "datasets": ["CuisineWorld"],
            "metrics": ["准确率"],
            "key_findings": ["基础模型在具身 AI 中性能有限"],
            "limitations": ["音频整合不足"],
            "future_work": ["多模态整合", "工具使用"],
            "contradictions": [],
        }
        pipeline, completions = await self._make_reduce_pipeline(
            [("stop", json.dumps(overview_json, ensure_ascii=False))]
        )
        pipeline.document_max_output_tokens = 3000
        pipeline.document_title = "Agent AI: Surveying the Horizons of Multimodal Interaction"
        document_request = {
            "evidence": [],
            "fact_catalog": [],
            "deterministic_contradictions": [],
            "questions": [],
        }

        result = await pipeline._synthesize_overview(document_request, [], [])

        self.assertEqual(completions.calls, 1)
        self.assertEqual(
            result.title,
            "Agent AI: Surveying the Horizons of Multimodal Interaction",
        )
        system = completions.kwargs[0]["messages"][0]["content"]
        self.assertIn("candidate_title", system)
        self.assertIn("衔接词", system)
        self.assertIn("分点", system)
        await self._cleanup_reduce(pipeline)

    async def test_synthesize_overview_cached_reapplies_backstop(self):
        overview_json = {
            "title": "多模态代理论坛",
            "executive_summary": "本文综述了 Agent AI 在多模态交互方向的研究进展。",
            "research_question": "Agent AI 如何实现多模态交互？",
            "methods": [],
            "datasets": [],
            "metrics": [],
            "key_findings": [],
            "limitations": [],
            "future_work": [],
            "contradictions": [],
        }
        pipeline, completions = await self._make_reduce_pipeline(
            [("stop", json.dumps(overview_json, ensure_ascii=False))]
        )
        pipeline.document_max_output_tokens = 3000
        pipeline.document_title = "Agent AI: Surveying the Horizons of Multimodal Interaction"
        document_request = {
            "evidence": [],
            "fact_catalog": [],
            "deterministic_contradictions": [],
            "questions": [],
        }

        first = await pipeline._synthesize_overview(document_request, [], [])
        second = await pipeline._synthesize_overview(document_request, [], [])

        self.assertEqual(completions.calls, 1)  # second call served from cache
        self.assertEqual(first.title, pipeline.document_title)
        self.assertEqual(second.title, pipeline.document_title)
        await self._cleanup_reduce(pipeline)

    def _group_response(self):
        return ("stop", json.dumps(
            {
                "summary": "组摘要",
                "concepts": ["概念"],
                "findings": ["结论"],
                "numeric_claims": [],
                "limitations": [],
                "future_work": [],
                "contradictions": [],
                "source_fact_ids": [],
                "source_sections": [],
                "source_pages": [],
            },
            ensure_ascii=False,
        ))

    def _overview_response(self, title):
        return ("stop", json.dumps(
            {
                "title": title,
                "executive_summary": "本文综述了 Agent AI 系统在多模态交互方向的研究进展。",
                "research_question": "Agent AI 系统如何实现多模态交互？",
                "methods": [],
                "datasets": [],
                "metrics": [],
                "key_findings": [],
                "limitations": [],
                "future_work": [],
                "contradictions": [],
            },
            ensure_ascii=False,
        ))

    def _answer_response(self):
        return ("stop", json.dumps(
            {
                "question": "问题",
                "answer": "答案",
                "source_fact_ids": [],
                "source_sections": [],
                "source_pages": [],
                "confidence": 0.8,
            },
            ensure_ascii=False,
        ))

    def _section_results(self, title):
        return [
            {
                "section": {
                    "id": "s1",
                    "title": title,
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "subtree_page_end": 1,
                    "parent_id": None,
                    "children": [],
                },
                "extractions": [
                    {
                        "summary": "摘要",
                        "facts": [
                            {
                                "concept": "Agent AI",
                                "statement": "Agent AI 系统能够感知并在不同领域行动。",
                                "kind": "result",
                                "source_quote": "原文",
                                "page": 1,
                                "confidence": 0.9,
                                "grounded": True,
                            }
                        ],
                        "new_concepts": [],
                    }
                ],
                "table_facts": [],
                "detected_facts": [],
                "numeric_contradictions": [],
                "synthesis": {
                    "summary": "摘要",
                    "core_concepts": ["概念"],
                    "key_findings": ["结论"],
                },
            }
        ]

    async def test_synthesize_document_bundle_reapplies_title_backstop(self):
        # synthesize_document short-circuits on the document_bundle cache before
        # the overview stage, so a truncated title in the cached bundle must be
        # corrected at read time, and the bundle key must fold the candidate so a
        # changed candidate forces a fresh bundle instead of returning a stale title.
        full_title = "Agent AI: Surveying the Horizons of Multimodal Interaction"
        # 1 group reduce + 1 overview (truncated prefix) + 10 global answers, then
        # a fresh overview after the candidate changes. The third call reuses the
        # same sections, so reduce and all 10 answers are cache hits; only the
        # document_bundle (whose key folds the candidate) misses.
        scripted = [self._group_response(), self._overview_response("AGENT AI:")]
        scripted += [self._answer_response() for _ in range(10)]
        scripted += [self._overview_response("AGENT AI:")]
        pipeline, completions = await self._make_reduce_pipeline(scripted)
        pipeline.document_max_output_tokens = 3000
        pipeline.document_title = full_title
        section_results = self._section_results(full_title)

        first = await pipeline.synthesize_document(section_results)
        self.assertEqual(completions.calls, 12)  # 1 reduce + 1 overview + 10 answers
        self.assertEqual(first.title, full_title)  # fresh-path backstop restores the prefix

        second = await pipeline.synthesize_document(section_results)
        self.assertEqual(completions.calls, 12)  # reduce + bundle cache hits, no new API calls
        self.assertEqual(second.title, full_title)  # bundle-hit path re-applies the backstop

        # A changed candidate must invalidate the bundle cache key.
        other_title = "Agent AI: A Revised Survey"
        pipeline.document_title = other_title
        third = await pipeline.synthesize_document(section_results)
        self.assertEqual(completions.calls, 13)  # +1 fresh overview; reduce + answers still cached
        self.assertEqual(third.title, other_title)
        await self._cleanup_reduce(pipeline)


if __name__ == "__main__":
    unittest.main()
