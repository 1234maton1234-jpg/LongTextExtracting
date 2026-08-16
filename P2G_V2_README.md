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
   - 碎片表头清洗（`_repair_header_name`）：pymupdf4llm 把宽表头拆成跨列碎片（`Reward (z-score)` → `Reward / (z-score / )`）再堆叠成 `(z-score Econ.`、`) Soc. (Δ)` 这类带悬空括号的噪声指标名；合并表头时对悬空开/闭括号做确定性清理。全量复跑确认：噪声指标名 0 残留，`(z-score Econ.`→`Econ.`、`e) Material`→`e Material`、`nt (z-scor Esteem`→`nt Esteem`、`) Soc. (Δ)`→`Soc. (Δ)`。
8. 碎片章节合并（V2.21）：综述类论文里大量单段编号子节（如 "9.1.2 Task" 334 token）每节触发一次抽取+归纳调用。现在小体积（≤500 token）且无子节点的编号子节并入父节（`_merge_fragment_leaves`），子节标题保留为父节正文首行标记，小节号仍能锚定"小节级问题"的证据；主编号节（无子号）与字母前缀附录（如 `D.12`）不参与。合并需在 `_normalize_levels` 之前运行，避免层级重基准干扰合并判定。
9. 抽取截断率从源头降低（V2.22）：块抽取输出此前无上限——提示词不限制事实条数、`source_quote` 逐字引用，稠密块轻松顶满 4000 输出预算后 `finish_reason="length"` 触发重生成/续写（人工社会智能综述实测 32 次初始调用 12 次截断，37.5%）。现在三管齐下：
   - 抽取提示词加输出上限（`EXTRACTION_PROMPT_VERSION` 升到 v2.9）：每块最多 20 条事实、`source_quote` 尽量简短（约 80 字以内），从源头让输出可预测地落在预算内；
   - 首次抽取调用 `reasoning_effort="none"`：整个输出预算全部给 JSON 内容，杜绝"思考烧预算"模式（与续写/重生成路径一致）；
   - 抽取输出预算对齐到供应商上限 4096（CLI 默认 4000→4096）。
   恢复机制（续写/关思考重生成）保留作兜底。同一份综述复跑实测：chunk_extract 初始截断率 **37.5% → 0%**，恢复调用 **18 → 0**；抽取族总调用 50→29；事实总量反而 234→296（关思考后全部预算用于内容 + 更短引用），10/10 问答全部中文、带页码、零英文泄漏、零幻觉、零缺失引用。

## 已验证基线（当前状态）

- **Agentopia**（25 节含附录）完整运行：10/10 全局问答全部带可定位页码出处、零幻觉、零英文泄漏；"矛盾"问题升级为 3 条可溯源矛盾（9 个事实、跨 6 页）；未来工作 4 条、局限 10 条。
- **V2.20 表事实净化复跑**（同一份 Agentopia PDF，缓存重跑约 16 分钟，本次仅 117 次 API 调用）：Table 8 配置行 22 条 `Value` 指标事实**从源头清零**；表头碎片噪声 `(z-score Econ.`/`e) Material`/`nt (z-scor Esteem`/`) Soc. (Δ)` **0 残留**，替换为 `Econ.`/`e Material`/`nt Esteem`/`Soc. (Δ)`；10/10 问答依旧全部带页码出处，"评价指标"问题确认回答 RoleEval/InCharacter/LifeChoice 等真实指标而非 "Table 8:"；"未来工作"问题不再混入局限内容。
- **Agent AI 综述**（80 页、80 节）实测：10/10 问答全部有真实出处、无幻觉；综述类论文无数值表格时"矛盾"问题正确答"证据不足"。
- **V2.22 人工社会智能综述复跑**（`Artificial Social Intelligence: A Comparative and Holistic View` 中译版，新论文无缓存，本次 134 次 API 调用）：碎片合并仅处理 `1.2 文章概述`（唯一 ≤500 token 的编号叶子节，因源 markdown 层级较深被并入其直接父节 `1.1`），标记 `###### **1.2** 文章概述` 保留、总 token 零丢失；10/10 问答全部中文作答、9/10 带页码、零英文泄漏、Q10 无数值表格正确答"证据不足"、executive_summary 连贯。该 PDF 标题结构较乱（1.2 层级深于 1.1、`4 讨论`被误嵌进`工作记忆`、`5 结论`被误嵌进`4 讨论`、期刊刊头 `CAAI 人工智能研究` 被当章节），均为 #7 前后的预存问题，但流水线仍产出全部有出处的答案。
- **V2.21 碎片章节合并复跑**（同一份 Agentopia PDF，缓存重跑约 19 分钟，本次仅 114 次 API 调用）：`5.4 Computational Cost` 并入 `5. Experiments`、总 token 零丢失、子节标记保留在父节正文首行、附录 `D.12` 不受影响；10/10 问答全部带页码、零英文泄漏。与 V2.20 相比 5 题完全一致，差异题均无事实回归：Q3（方法）证据更全、Q9（未来工作）不再混入局限内容、Q10（矛盾）从 2 条升级为 3 条且更精确、Q6 数字一致仅措辞微调、Q8（局限）覆盖略减为 LLM 采样差异。
- **V2.23 抽取截断优化复跑**（同一份人工社会智能综述 PDF，提示词版本号 v2.8→v2.9 使全部块缓存失效，39→29 次抽取调用全量重抽，约 10 分钟）：chunk_extract 初始调用 29 次 **0 次截断（0%）**，续写/重生成恢复调用 **0 次**；对比修复前 v222 同一份综述 32 次初始调用 12 次截断（37.5%）、恢复调用 18 次。事实总量 234→296（+62），问答 10/10 全部中文作答、9/10 带页码、Q10 正确答"证据不足"、零英文泄漏、零幻觉、答案引用 0 缺失（271 条 grounded 事实全部可解析）。唯一差异为 Q6（"最重要的定量结果"）本次抽到定性表述而非 v222 的年龄里程碑数字——该里程碑事实仍完整存在于 v223 事实库（5 条 vs v222 的 3 条），属 LLM 采样差异而非覆盖损失。
- 单元测试 **101 个全部通过**。

## 仍存在的问题

1. 附录提示词正文（如 GPT-4V 游戏场景描述）仍被抽取成低价值 "example" 事实，有 token 浪费，但与问答路由无冲突。
2. 图片 OCR 被整块剥离后，纯图表内的文字不再入事实库；综述实测问答未受影响，待用图表密集论文验证是否削弱"图表内容"类问题。
3. "未来工作"问题偶发仍混入方法实现描述，检索层可再收紧事实 kind 白名单。
4. 高层摘要（`executive_summary`）是事实拼接式归纳，连贯性可提升。
5. 综述类论文子节碎片化：已实现"碎片章节合并"（`_merge_fragment_leaves`：≤500 token 的无子节点编号子节并入父节，子节标题保留为父节正文首行的标记）。Agentopia 与人工社会智能综述两篇实测都只合并了 1 个碎片节、内容零丢失、标记保留；但两篇论文都不是碎片化结构（综述的 `1.2` 因源 markdown 层级异常深还只并入其直接父节），调用数收敛的收益与对"小节级问题"定位精度的影响，仍需真正有 60+ 单段编号子节的综述（如 Agent AI 综述原 PDF）实测。
6. ~~截断恢复率偏高~~（V2.22 已修复，见已解决问题 #9）：块抽取输出已加事实上限 + 关思考 + 预算对齐 4096，同一综述复跑初始截断率 37.5%→0%、恢复调用 18→0。恢复机制仍保留作兜底；其余阶段（section_reduce/group_reduce/global_answer）未改动，截断率维持原状。

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

预期结果为 `101 tests ... OK`。

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
