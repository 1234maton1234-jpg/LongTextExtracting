# P2G V2.19 长文理解测试包

这个版本用于测试 PDF 超长文本理解，不负责生成 Galgame。它会把 PDF 转为 Markdown，建立标题树，按约 3200 Token 分块并保留约 180 Token 重叠，再抽取带原文引用和页码的事实，最后执行分层全文归纳与 10 项全局问答。

## 相比 V2.14 已解决的问题

1. **模型响应截断不再导致整段中断**：`finish_reason="length"` 截断时先尝试解析完整内容，否则关闭思考重新生成，或从已产出的部分 JSON 续写（续写尾部与部分内容拼接后解析，失败自动接力下一段）。抽取输出预算从 1800 提高到 4000 Token。之前会崩在附录密集章节（如 "B. System Design Details"）的完整运行，现在能跑完所有章节并产出总览与全部 10 项全局问答。

2. **全局问答答案质量**：局限类答案按句子保留证据（不再因一句无证据内容整条误杀）；"未来工作"问题只检索局限与结论章节而非方法实现；"矛盾"问题确定性列出总览发现的跨章节冲突并附出处，不再答"证据不足"；提示词强制用自己的话概括、禁止粘贴原文引用。

3. **英文原文不再泄漏进答案**：抽取层约 18% 的事实的 `statement` 直接照抄英文原文导致答案混英文。四层提示词统一要求英文证据先翻译成中文再概括（只保留专有名词/指标名/模型名），并加答案层确定性英文兜底：中文问题下连续 4 个以上英文单词时截断到英文块之前、泄漏内容移入 `unsupported_claims`；整段英文（或只剩 "Table 8:" 这类碎片）时答案降级为"证据不足"。

4. **指标类问题检索路由正确**：配置/参数表的数值行被标成 `kind="metric"` 会压过真实指标定义（Q5 曾答出 "Table 8:"）。现在检索打分对配置列名（Value/Setting/Parameter/…）和含配置标记的表格事实确定性罚分，相关表转储在"评价指标"类问题上降权，Q5 恢复为真实指标事实。

5. **矛盾检测从"偶然发现"升级为"确定性扫描"**：
   - 多行表头合并（`_merge_table_headers`）：Table 3 的 `Claude | Gemini` + `4.5-Opus | 3-Pro` 两行表头按列拼接，列身份不再丢失，64.28/36.02 这类"伪矛盾"从源头消失。
   - 文档级数值矛盾扫描（`detect_document_contradictions`）：按（指标、行上下文、表标题）分组，只判"同一张表跨页冲突"为真矛盾，不同表的同结构数值（如各模拟世界的 z-score 表）不再误报。Agentopia 误报从 29 个收敛到 0。

6. **结构去噪**（换第二篇论文实测暴露的排版问题）：
   - 目录类章节整节丢弃（Contents/Table of Contents/List of Figures/目录…，正文是页码表不是内容）。
   - 作者简介并入 back-matter 过滤。
   - 图片 OCR 文本（`<!-- Start of picture text -->` HTML 块）与 `<img>`/`<mark>` 标签在建树前剥离，只留真实正文。
   - 图注/提示词转储节点不再当章节（`_absorb_noise_nodes`）：正文首行是 `Figure N:`/`Table N:` 且无编号签名、或正文首行是 `**Prompt:**` 等角色头、或裸标题是 `Output:`/`Input:` 等角色词的节点，正文并入父节点、子树提升。Figure 28 图注 "General Alpaca Training Framework" 不再被当成 L2 章节吞掉 6.6.3 到附录 F 的整棵子树；综述章节数从 83 收敛到 80。
   - 章节层级归一化（`_normalize_levels`）：用 `###`/`####` 表示主/子章节的论文，把"编号章节的最小层级"整体平移到 L2。综述层级从 `{L1:1, L3:14, L4:65}` 归一到 `{L1:1, L2:14, L3:65}`。
7. 表格事实源头净化（V2.20）：
   - 配置表数值行不再进入事实库（`_row_is_config_row`）：数值列名是通用标签（Value/Setting/Parameter/…）且表含配置标记或行上下文键全为通用标签时，该行在抽取源头就被跳过，不再以 `kind="metric"` 写进事实库和证据索引（此前 V2.18 只在检索打分时压掉）。真实结果表的通用列名（如 `Total`/`Mood`）在 `Model` 上下文下仍保留。
   - 碎片表头清洗（`_repair_header_name`）：pymupdf4llm 把宽表头拆成跨列碎片（`Reward (z-score)` → `Reward / (z-score / )`）再堆叠成 `(z-score Econ.`、`) Soc. (Δ)` 这类带悬空括号的噪声指标名；合并表头时对悬空开/闭括号做确定性清理。真实输出中 67 个指标名里的 4 个噪声名被修复。

## 已验证基线（当前状态）

- **Agentopia**（25 节含附录）完整运行：10/10 全局问答全部带可定位页码出处、零幻觉、零英文泄漏；"矛盾"问题升级为 3 条可溯源矛盾（9 个事实、跨 6 页）；未来工作 4 条、局限 10 条。
- **Agent AI 综述**（80 页、80 节）实测：10/10 问答全部有真实出处、无幻觉；综述类论文无数值表格时"矛盾"问题正确答"证据不足"。
- 单元测试 **87 个全部通过**。

## 仍存在的问题

1. 附录提示词正文（如 GPT-4V 游戏场景描述）仍被抽取成低价值 "example" 事实，有 token 浪费，但与问答路由无冲突。
2. 图片 OCR 被整块剥离后，纯图表内的文字不再入事实库；综述实测问答未受影响，待用图表密集论文验证是否削弱"图表内容"类问题。
3. "未来工作"问题偶发仍混入方法实现描述，检索层可再收紧事实 kind 白名单。
4. 高层摘要（`executive_summary`）是事实拼接式归纳，连贯性可提升。
5. 综述类论文子节碎片化：65 个 L3 子节里不少是单段小节（如 "9.1.2 Task" 334 token），每节触发一次抽取调用；需要"碎片章节合并"从源头收敛章节数。
6. 截断恢复率偏高：chunk_extract 381 次调用中约 26% 触发重生成/续写，机制正常但每次重新请求 API。

## 环境要求

- Windows 10/11
- 64 位 Python 3.10-3.12
- 可访问 DeepSeek API
- 自己的 DeepSeek API Key

默认模型：块抽取使用 `deepseek-v4-flash`，多块章节归纳使用 `deepseek-v4-pro`，其余全文阶段使用 Flash。默认并发为 4；如果账号限流，请改为 1 或 2。

## 一键安装与测试

在解压目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\test.ps1
```

预期结果为 `87 tests ... OK`。

## 先做免费 Dry Run

Dry Run 只解析标题和分块，不请求 API：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 `
  -PdfPath "D:\docs\your-long-document.pdf" -DryRun
```

结果默认写入 `outputs\p2g_v2_output.json`。先检查章节数、分块数和 `max_chunk_tokens` 是否合理。

## 完整运行

复制配置模板并填入你自己的 Key：

```powershell
Copy-Item .env.example .env
notepad .env
```

然后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 `
  -PdfPath "D:\docs\your-long-document.pdf" -Concurrency 4
```

也可以临时降低并发或覆盖模型：

```powershell
.\run.ps1 -PdfPath "D:\docs\paper.pdf" -Concurrency 2 `
  -FlashModel "deepseek-v4-flash" -ProModel "deepseek-v4-pro"
```

额外的 Python 参数可以放在命令末尾，例如只测部分章节：

```powershell
.\run.ps1 -PdfPath "D:\docs\paper.pdf" --section "Results" --section "Conclusion"
```

## 重点看什么

最终 JSON 的 `document_synthesis.global_answers` 是长文理解测试重点：

- 回答是否覆盖研究问题、数据、方法、指标、定量结果、解释、局限和未来工作。
- `source_fact_ids` 是否非空，`source_pages` 是否能定位回原文。
- 定量回答中的数字是否与对应事实完全一致。
- `contradictions` 是否能发现跨页数字或结论冲突。
- `stats.usage` 可查看请求数、Token 和耗时。

系统优先检索当前论文的 Abstract / Results / Discussion / Conclusion，降低 Related Work 对定量结论的干扰。模型返回空、伪造或不支持答案的事实 ID 时，程序只从本次问题的证据白名单中最多选择 3 条；仍无法支持时会直接使用已验证事实作保守答案。

全局问答的证据审计按句子粒度执行：有证据的句子保留为答案，无证据的句子移入 `unsupported_claims` 供审查，不再因混入一句无证据内容而整条丢弃。"矛盾"问题不经过句子审计，直接把确定性数值矛盾与总览发现的矛盾合并列出，并对每条语句尝试匹配回事实以附带页码出处。

## 文件与隐私

- `.env`、PDF、输出 JSON、SQLite 缓存都只保存在本机。
- 不要把填有真实 Key 的 `.env` 发给别人。
- `outputs\p2g_v2_cache.sqlite3` 可用于断点式复跑；删除后会重新请求 API。
- 发送问题时请附上报错文本、Python 版本、PDF 页数和运行参数，不要附 API Key。
