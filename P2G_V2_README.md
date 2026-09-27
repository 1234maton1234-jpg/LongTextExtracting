# P2G V2.19 长文理解测试包

这个版本用于测试 PDF 超长文本理解，不负责生成 Galgame。它会把 PDF 转为 Markdown，建立标题树，按约 3200 Token 分块并保留约 180 Token 重叠，再抽取带原文引用和页码的事实，最后执行分层全文归纳与 10 项全局问答。

## 当前能力

- **结构理解**：目录、图注、提示词转储、图片 OCR 文本等噪声在建树前剥离；章节层级归一化；≤500 token 的碎片子节并入父节（子节标题保留为父节正文首行标记，小节号仍能锚定"小节级问题"的证据）。
- **截断不中断**：`finish_reason="length"` 截断时先尝试解析完整内容，否则关闭思考重新生成，或从已产出的部分 JSON 续写（拼接后解析、失败自动接力）。输出加事实上限 + 关思考 + 预算对齐后，抽取与归纳阶段实测截断 0 次。
- **token 高效**：抽取与归纳阶段首次调用 `reasoning_effort="none"`，整个输出预算全部给内容，杜绝"思考烧预算"；reduce 族总 token 实测下降约 34%，无质量回退。恢复机制保留作兜底。
- **可靠答案**：10 项全局问答按句子粒度审计证据（无证据句子移入 `unsupported_claims`，不再整条误杀）；答案强制中文转述、禁止粘贴原文引用，英文泄漏有确定性兜底；"矛盾"问题确定性列出跨章节冲突并附页码出处；证据不足时保守作答。
- **表格净化**：配置表数值行与跨列碎片表头噪声在事实库源头剔除，不再以 `kind="metric"` 进入证据索引，不压过真实指标定义。
- **指标不依赖表格**：`kind="metric"` 不再只由 markdown 表格解析产出。PDF 抽不出表格时（实测一份 75 页综述管道表格数为 0），原先指标池为空、`评价指标` 问题只能引用无关的 definition 事实作答；现在有两条补充路径：抽取提示词明确要求把正文叙述中的评估指标（accuracy / F1 / BLEU / PSNR / success rate / 准确率 / 召回率 等）标为 `metric`；同时 `extract_metric_facts` 做确定性兜底，从正文句子中回收指标事实，并要求句中出现百分比、"指标 of 数值"或评估性措辞才算证据，专门排除 "is"、"map"、"MAE"（遮蔽自编码器）这类同形词误报。

## 实测验证

- 多篇不同形态的 PDF（实验论文、英文综述、结构较乱的中译综述）全流程实测：10 项全局问答**句子级审计零无据陈述**、零英文泄漏、答案带可定位页码出处、"矛盾"问题可溯源。
  > 注意口径：句子级审计验证的是"这句话能否在引用的事实里找到依据"（grounding），**不验证答案是否切题**（relevance）。一条完全答非所问但确有原文依据的句子同样会通过审计。指标类问题的答非所问已由正文兜底路径修复；其余问题类别仍需人工抽查切题性。
- 截断与恢复：抽取与归纳阶段实测 0 截断、0 恢复调用；reduce 族 reasoning token 由约 134K 降至约 0.3K（仅总览保留思考），总 token 降幅约 34%。
- 单元测试 **176 个全部通过**。

## 仍存在的问题

1. 纯图表内文字随图片 OCR 一起剥离，不再入事实库；待用图表密集的论文验证是否削弱"图表内容"类问题的回答。（指标类问题已由正文兜底路径覆盖，见上文；但图表独有、正文从未提及的数值仍会丢失。）
2. "未来工作"问题偶发仍混入方法实现描述，检索层可再收紧事实 kind 白名单。
3. 高层摘要（`executive_summary`）为事实拼接式归纳，连贯性可提升；概述层文档标题偶发误译（采样波动）。

## 环境要求

- Windows 10/11
- 64 位 Python 3.10-3.13
- 可访问 DeepSeek API
- 自己的 DeepSeek API Key

默认模型：块抽取使用 `deepseek-v4-flash`，多块章节归纳使用 `deepseek-v4-pro`，其余全文阶段使用 Flash。默认并发为 4；如果账号限流，请改为 1 或 2。

## 一键安装与测试

在解压目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\test.ps1
```

预期结果为 `176 tests ... OK`。

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
- `stats.usage` 可查看**本次运行**的请求数、Token 和耗时（含 `run_started_at` 起始时间戳）；`stats.all_time_usage` 是 usage 数据库的**历史累计**值。两者分开统计——usage 库是追加共享的，早先版本把累计值当成本次运行数据，会严重高估单次成本。`stats.cache` 的 `hits`/`misses` 本身就是本次运行的计数。
- `stats.config` 给出超时与重试配置：`request_timeout_seconds`、`sdk_max_retries`、`content_retry_attempts`，以及最坏情况乘积 `max_attempts_per_request`（SDK 重试与应用层重试会**相乘**）。

> ⚠️ **`latency_seconds` 的含义**：它是**每次 API 调用的净耗时**，**不含**并发信号量排队等待。早先版本把时钟起点放在获取信号量**之前**，导致该字段混入排队时间：实测同一 stage 的吞吐（tok/s）在修正前后相差一个数量级（`document_group_reduce` 62.6 → 1087 tok/s，Token 数几乎相同）。因此**修正前的 latency 数据不能用来判断"哪个阶段慢"**，据此推断超时风险也不可靠。修正后在 NAFNet 全篇实测：80 次调用、单次最长 **8.52s**、延迟合计 241.8s、**0 次**触及 120s 超时。

## 溯源指标（可回归的质量度量）

输出 JSON 的 `evidence_metrics` 给出本次运行的确定性溯源指标，**不额外调用任何模型**，同一份输入必然得到同一结果：

| 字段 | 含义 |
|------|------|
| `fact_count` | 可作为证据的事实总数（仅计 `grounded=true`，含表格/指标/矛盾派生事实） |
| `answers_with_evidence` | 10 问中**至少引用 1 条事实**的题数——低于 10 就说明有题目无证据可依 |
| `citation_precision` | 引用的 fact_id 中能在事实索引里解析到的比例 |
| `unsupported_claim_rate` | 无据句子占**模型总输出句子**的比例（= 无据 /（保留 + 无据））。分母**必须**包含被剔除的句子：句子级审计会把无据句**从 `answer` 里删掉**并记入 `unsupported_claims`，只用保留句数做分母会算出 >1.0 的"比例"（实测 2.50），且短答案因为被删得多反而显得更差 |
| `supported_sentence_count` / `model_sentence_count` | 保留在 `answer` 里的句数 / 模型总输出句数（两者之差即被剔除的句数） |
| `unsupported_retrieval_gaps` | 无据句中**事实库里其实有**的条数——即"检索没送到"，属召回问题 |
| `unsupported_undecided` | 无据句中事实库无法解释的条数——**是幻觉上界，不是幻觉判定**（见下文） |
| `min/median/max_cited_facts` | 每题引用事实数的分布（`min` **包含**零引用的题，否则会掩盖"某题完全没证据"这一最该被抓的情况） |
| `sufficient_evidence` | 该题是否达到证据下限（至少 1 条可支持本答案的引用事实） |
| `per_answer` | 逐题明细，含每条无据句的 `verdict` 与 `coverage`，便于定位是哪一问退化 |

**读法上的三个要点：**

1. `citation_precision` 在本系统中**接近恒定 1.0**：`_recover_answer_fact_ids` 会主动剔除不在证据白名单里的 fact_id，所以它天然无法下降。它的作用是**发现异常**（例如输出被流水线之外的程序拼装过），**不是**质量卖点。
2. `unsupported_claim_rate` 度量的是**接地性**（每句话是否在引用的事实里有依据），**不度量切题性**（答案是否回答了问题）。一条完全答非所问但确有原文依据的句子同样会得 0 分——切题性需要基准答案才能评，不在本指标范围内。
3. 句子级审计**只对照该题引用到的事实**，所以"事实上库里有、但检索没送到"的句子也会被判无据。`unsupported_retrieval_gaps` / `unsupported_undecided` 就是对这批句子的**回查分类**：重新拿**整个事实库**去匹配。

**关于回查的边界（重要）**：回查用词元覆盖率判定，只能可靠地区分"库里有"（覆盖率 ≥0.4 或数字吻合）。**它不会下"幻觉"结论**——因为同义词（如正文写 `speedup`、答案写"加速比"）与凭空编造在词元重叠上**都是 0.00**，无法区分。所以 `unsupported_undecided` 是**幻觉上界**：里面既有真的编造，也有换了措辞的真实内容。要彻底判定需要原文（运行输出不保留），或让模型自证。

不重跑流水线也能对已有输出结算指标（无需 PDF 与 API Key）：

```powershell
.\.venv\Scripts\python.exe python_scripts\p2g_v2.py --metrics outputs\p2g_v2_output.json
```

系统优先检索当前论文的 Abstract / Results / Discussion / Conclusion，降低 Related Work 对定量结论的干扰。模型返回空、伪造或不支持答案的事实 ID 时，程序只从本次问题的证据白名单中最多选择 3 条；仍无法支持时会直接使用已验证事实作保守答案。

全局问答的证据审计按句子粒度执行：有证据的句子保留为答案，无证据的句子移入 `unsupported_claims` 供审查，不再因混入一句无证据内容而整条丢弃。"矛盾"问题不经过句子审计，直接把确定性数值矛盾与总览发现的矛盾合并列出，并对每条语句尝试匹配回事实以附带页码出处。

## 文件与隐私

- `.env`、PDF、输出 JSON、SQLite 缓存都只保存在本机。
- 不要把填有真实 Key 的 `.env` 发给别人。
- `outputs\p2g_v2_cache.sqlite3` 可用于断点式复跑；删除后会重新请求 API。
- 发送问题时请附上报错文本、Python 版本、PDF 页数和运行参数，不要附 API Key。
