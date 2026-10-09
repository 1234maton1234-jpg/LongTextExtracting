# 交接文档：P2G V2.19 长文本提取项目

> 用途：迁移到新对话继续开发时，把这份文档整个贴给新会话即可。
> 历史核验基线：HEAD `e8b1a5f`；`main` 与 `origin/main` 对齐；本次 `.\test.ps1` 曾通过 **217 tests OK**；当前本地接入修复后 `.\test.ps1` 为 **224 tests OK**。生成本文档时 `HANDOFF.md` 本身尚未跟踪，因此工作区并非干净。

---

## 0. 一句话背景

这是一个**独立的长文本（PDF）理解与事实提取项目**，不绑定 paper2gal。核心目标：PDF → Markdown → 标题树 → 分块 → LLM 事实抽取 → 分层归纳 → 10 项全局问答，并在**确定性可验证**的前提下省 token、保质量。

用户是研究生，有导师/课题组。GitHub 账号 `1234maton1234-jpg`（Free），仓库 `LongTextExtracting`（private）：

```
git@github.com:1234maton1234-jpg/LongTextExtracting.git   # SSH 可用；HTTPS 被重置/阻断
branch: main
```

**所有提交都直接推 `main`**（用户确认过）。

---

## 1. 环境（重要，容易踩）

| 项 | 值 |
|---|---|
| 工作目录 | `F:\P2G-V2.19-friend-test` |
| **虚拟环境目录名** | **`longtext`**（不是 `.venv`！已从 `.venv` 改名，见 commit `6b8b931`） |
| 解释器 | `.\longtext\Scripts\python.exe`，Python 3.13.9 |
| 依赖安装 | `powershell -ExecutionPolicy Bypass -File .\setup.ps1`（接受 Python 3.10–3.13） |
| 测试 | `powershell -ExecutionPolicy Bypass -File .\test.ps1`，应得 **224 tests OK** |
| API Key | `.env` 中的 `DEEPSEEK_API_KEY`（35 字符，**永不打印**） |

**必须设 `$env:PYTHONUTF8="1"`**，否则控制台中文乱码。

**关键陷阱**：`p2g_v2` 的 import 要 30–160 秒（`pymupdf4llm` 冷启动 + 首次磁盘缓存）。**每次新起 python 进程都要付这个成本**，不是脚本有问题。批量跑多篇论文时应放进同一个进程。

**沙箱**：策略是 `danger-full-access`，审批提示已禁用 —— **不要传 `sandbox_permissions`**（请求会被直接拒绝）。

---

## 2. 代码结构

单文件主程序 + 两个工具 + 5 个测试文件。

```
python_scripts/p2g_v2.py          5333 行  主程序（单模块）
python_scripts/multiround_eval.py  369 行  多轮评测工具
python_scripts/benchmark_run.py    206 行  单篇性能基准工具
tests/test_p2g_v2.py              206 个测试（主测试）
tests/test_future_work_routing.py   6 个
tests/test_group_batch_budget.py    4 个
tests/test_lean_group_payload.py    4 个
tests/test_section_concurrency.py   3 个
                                  ─────
                          合计    223 个
```

Prompt 版本常量（改提示词必须 bump，否则旧缓存命中）：
```python
EXTRACTION_PROMPT_VERSION = "v2.10"
REDUCE_PROMPT_VERSION     = "v2.9"
GROUP_PROMPT_VERSION      = "v2.11"
DOCUMENT_PROMPT_VERSION   = "v2.22"   # 定量公式候选排序（正文优先、附录软降权）
```

---

## 3. 本会话已完成的工作（按 commit 顺序）

### 3.1 修复类

| commit | 内容 |
|---|---|
| `0745010` | **评价指标误识别**：LLM 把动词 `Recall`（"回顾"）、`误差棒`、`浓度` 标成 `kind="metric"`。加 `metric_fact_is_evidenced()` 确定性门槛，无指标证据者**在指标题路由降权 40 分**（不删不降级，真指标不误杀） |
| `51240d0` | **未来工作题答错**：真实未来方向被抽成 `kind="method"` 后**被硬性 kind 白名单在打分前丢弃**，答案退回 Introduction 的普通局限。现豁免"源码标注为开放/计划"的事实并加分（前瞻章节 +70，其他 +25） |
| `406bc15` | **续写失败丢弃部分输出**：恢复链路第一次续写失败后丢弃其输出、重发整个提示词。实测 `global_answer_regen` 输入/输出比 **15.9:1**。现把已有部分作为 assistant 轮带上，只要求补完 |

### 3.2 性能 / token 类（历史单次/少量实测记录，不能代替当前复测）

| commit | 内容 | 实测效果 |
|---|---|---|
| `f8da222` | **分层归纳真正生效**：分组预算必须**大于**单块预算，否则每块顶到上限、永远放不进同一批。新增 `--group-batch-tokens`（历史默认 9600） | 调用 −37.5%、输出 token −36%、耗时 −32%；`document_group_reduce` 46→7 次 |
| `4cda5d9` | **分组归纳不重读已消费输入**：只发 `fact_id`/`concept`/`statement`/`kind`/`page`。新增 `--full-group-payload` 可回退 | payload −33%、端到端输入 −17.4%；8 配对轮未检出质量代价 |
| `9ff51eb` | **章节并行处理**：`_run` 原先串行遍历章节。新增 `--section-concurrency`（默认 4）。每节内部仍串行；每次模型调用仍走共享信号量 | 总墙钟 205.3s → 80.0s（2.6×），抽取阶段 3.4×，token 不变 |
| `e8b1a5f` | **性能基准工具** `benchmark_run.py` + 并发调参到饱和点 | 见 §5 |

### 3.3 工具类

| commit | 内容 |
|---|---|
| `378f5cd` | `multiround_eval.py`：多轮评测（均值±σ、答案跨轮稳定性、数字-主体错配筛查）。同时修掉数字归一化三处误报 |
| `47a4e01` | 多轮工具保留 `per_answer`（区分"无据句变多"与"答案变短"）；README 记录实测方差 |
| `2a670dc` | 多轮工具新增 `--full`（连抽取一起重跑），PDF 从 `inputs/` 按名字自动匹配 |

### 3.4 ⚠️ 一个被**回退**的改动（重要教训）

`dad39ec` 曾把目录里的 `section_title` 去掉（省 22% 目录 token、每问多放 4–6 条事实）。**后来 `406bc15` 按实测恢复了。**

14 配对轮 A/B（同事实、逐轮配对、两臂交替顺序）：

| | 答案中被剔除句数 | 占模型总输出 |
|---|---|---|
| 带 `section_title` | **44 / 294** | 0.150 |
| 去掉 | **69 / 302** | 0.229 |

去掉后无据句 **1.57×**；配对差 +0.079、t=+2.03、95% CI **[−0.005, +0.162]**。区间触及 0，是**判断而非已证效应**，但方向对"要发布的省 token"是错的 → 恢复。

**判据（贯穿全会话）**：省 token 分两类 ——
- **模型会读的字段**（如 `section_title`）：动它必须配对测量，宁可恢复
- **纯记账字段 / 重复投递**（如分组 payload 的 `source_quote`、重试重发）：风险低，直接测量确认即可

---

## 4. 必须遵守的方法学纪律

这是本会话最重要的产出，**新会话务必沿用**。

### 4.1 单次运行不能下任何结论

本节中的样本量、标准差、功效估算及配对检验数据均为交接时记录的历史结果；当前工作区的忽略产物不足以完整重建所有原始轮次，继续引用时应视为**待复算的实验记录**，而非当前可复现基线。

实测（交接记录：SegEarth-OV，258 条事实，24 次独立采样）：无据率均值 0.210、σ 0.074、范围 0.050–0.318。

- 要检出 **0.05** 的差异，每臂约需 **35 轮以上**（配对协议下 14 轮能检出约 0.10）
- **3 轮完全不够**（3 轮标准误约 0.043）
- 同一篇论文 10 题里常有 **6–8 题**跨轮答案不一致

### 4.2 报告必须区分三种措辞

- **【已判定】**：差异 > 2× 合并标准误
- **【未判定】**：差异在噪声内，明确说"检不出"
- **【直接测量】**：token 数、调用数、事实数 —— 不受采样影响，可直接用

### 4.3 数据管线

- 时间预算必须**先算功效再跑**：配对 σ≈0.145 时，检出 0.05 需 67 轮、0.10 需 17 轮、0.15 需 8 轮
- 长时任务**写 append-only journal**（本会话有两次后台任务丢失输出，白跑）
- 后台任务用 `Start-Process -RedirectStandardOutput <file>` 分离启动，**不要**指望 `job_output` 能一直等到
- **重定向到文件时 Python 是块缓冲**：stdout 只在结束时 flush，中途看不到进度 —— 用 `flush=True` 或另写 journal

### 4.4 我曾犯过的错（不要再犯）

1. 用 3 轮 A/B 宣布"指标改善，超出噪声" → 后来 14 轮证明是噪声
2. 用单次运行的 0.098 去比另一状态的 0.25，宣称"回归" → 事实数与 kind 分布一致，说明是采样波动
3. **脚本里的符号检验判据写反**（把 `off<=on` 当 on 胜）→ 统计必须复核
4. 把临时脚本删了，导致性能数字不可复现 → 现已固化成 `benchmark_run.py`

---

## 5. 当前性能与 token 数据（同一份 48 页 aztec 文档）

### 5.1 优化累计

⚠️ **两组数据口径不同，不可混比**：下面第一组是**流水线计时/阶段延迟**（不含 PDF 解析）；第二组来自 `benchmark_run.py`，分别报告**流水线墙钟**和**端到端墙钟**。旧版 JSON 的 `wall_seconds` 实际只含抽取+归纳问答，见下方纠正说明。旧版 CLI measurement 同样不含 PDF 解析。

**口径 A：抽取 + 归纳 + 问答（不含解析）** —— 来自 `--tag sc8` 等基准与早期 CLI 运行

| 版本 | 调用 | 总 token | 阶段耗时 |
|---|---|---|---|
| 最初（章节串行，旧分批） | 104 | 308,835 | 381.8 s（Σ 阶段延迟） |
| 分批 + payload 瘦身 | 67 | 205,697 | 277.9 s（Σ 阶段延迟） |
| + 章节并行（并发 4） | ~68 | ~204,000 | 53.6 + 26.4 = **80.0 s** |
| + 章节并行 8 / API 16 | 68 | ~206,000 | 30.5 + 25.1 = **55.6 s** |

**口径 B：解析与流水线分别计时** —— 当前 `benchmark_run.py` 在 PDF 解析结束后才启动流水线计时。现存旧版 `outputs/perf/*.json` 的 `wall_seconds` **不含解析**，仅是抽取+归纳问答；端到端耗时需另将 `parse_seconds` 加上该值。更新后的脚本分别写 `pipeline_wall_seconds` 与 `end_to_end_seconds`。

| API 配置 | 解析 | 抽取 | 归纳+问答 | 流水线墙钟（不含解析） | 端到端估算（含解析） |
|---|---:|---:|---:|---:|---:|
| 章节 8 / API 8 | 223.8 s | 36.5 s | 26.6 s | 63.1 s | **286.8 s** |
| 章节 16 / API 8 | 118.6 s | 35.2 s | 26.7 s | 61.9 s | **180.5 s** |
| 章节 8 / API 16 | 127.1 s | 30.5 s | 25.1 s | 55.6 s | **182.7 s** |

**效果解读**：token 从 308,835 降到约 **206,000（−33%）**。耗时不可直接把旧版 **381.8 s 阶段延迟总和**与当前 **55.6 s 并行流水线墙钟**相减或计算提速比；后者不含 PDF 解析。当前样本端到端约 **181–287 s**，其中 PDF 解析占主要时间且受缓存影响。

### 5.2 并发调参已到饱和点

| 章节并发 | API 并发 | PDF 解析 | 抽取 | 归纳+问答 | 流水线墙钟（不含解析） | 端到端估算（含解析） |
|---|---|---|---|---|---|---|
| 1（旧行为） | 8 | — | 179.9 s | 25.4 s | 205.3 s（旧脚本，不含解析） | — |
| 4（**默认**） | 8 | — | 53.6 s | 26.4 s | 80.0 s（旧脚本，不含解析） | — |
| 8 | 8 | 223.8 s | 36.5 s | 26.6 s | 63.1 s | **286.8 s** |
| 16 | 8 | 118.6 s | 35.2 s | 26.7 s | 61.9 s | **180.5 s** |
| 8 | **16** | 127.1 s | 30.5 s | 25.1 s | **55.6 s** | **182.7 s** |

\* 解析耗时随文件系统缓存波动（118–224 s），非稳定量；同一配置重复跑会明显不同。

**本组结果的描述性观察**：本次记录中章节并发 4→8 的流水线耗时下降约 21%，8→16 约下降 2%；API 并发 8→16 约下降 12%。每个设置样本有限，解析时间受缓存影响，不能据此作稳定性能结论或断言并发已达普遍饱和点；瓶颈判断需用带运行元数据的重复基准确认。

**注意默认值**：`--section-concurrency` 与 `--concurrency` 的**默认都是 4**。上面 8/16 的效果需要显式传参。

**全局问答 reasoning A/B（探索性，2 次/臂）**：新增 `benchmark_run.py --global-answer-reasoning-effort low|none`，默认仍为 `low`，只影响全局问答首次请求。用同一 PDF/并发（章节 8、API 16）比较：

| 指标 | low（2 轮均值） | none（2 轮均值） |
|---|---:|---:|
| 全流程 API 调用 | 66.5 | 59.5 |
| 总 token | 196,042 | 179,493（−8.4%） |
| 流水线墙钟 | 51.49 s | 46.53 s（−9.6%） |
| 全局问答初次+恢复调用 | 17.5 | 10（无 regen/continuation） |
| 全局问答输入+输出 token | 37,092 | 16,422（−55.7%） |
| 有证据答案 | 9/10、9/10 | 9/10、9/10 |
| unsupported claim rate | 0.35、0.30 | 0.35、0.20 |

none 条件下两轮都未触发全局问答恢复；low 条件每轮分别有 7、8 次恢复调用。当前观测支持把 `none` 当作有前景的节省方案，但每臂只有两轮、事实数/答案采样不同，**不能据此判定质量不退化或把默认切换为 none**。unsupported claim rate 差异仍未判定。两臂 PDF 解析差异巨大（low 约 19 s；none 约 88–94 s），说明端到端比较被解析/缓存波动污染；性能比较应看流水线与 usage ledger。

结果文件（当前 `outputs/` 被 Git 忽略）：`repro_warm_01/02.json` 为 low，`answer_none_01/02.json` 为 none；同名 `_usage.sqlite3` 可查阶段级调用/token。源码开关及测试保留在仓库。

### 5.3 显式证据链建模 MVP

全局问答 JSON 新增 `evidence_chain`：每步包含 `step_index`、简短 `claim`、`source_fact_ids` 和 `relation_to_previous`（`none/supports/combines/contrasts`；最多 3 步）。这是同一次全局问答响应内的附加结构，不增加 API 调用；仅有探索性 QA-only token 记录，尚未分离证据链带来的增量，也未做成本压缩。

程序只确定性验证 fact ID 是否属于当前候选集/事实索引，以及其 `grounded`、原文引句和页码指针；逐步 claim 使用已有词项/数字启发式打标签，不是 NLI 证明。关系字段是模型提出，`relation_validation` 固定为 `not_performed`；整体状态和逐步状态均由程序重算，不接受模型自报“已验证”。旧缓存可保持空链并标 `unavailable_legacy`；矛盾问题继续走现有确定性短路，标 `deterministic_short_circuit`，不伪造通用 LLM 推理链。

2026-10-01 朋友接入反馈（48 页数学论文 + GPT-6 Luna）：报告称 73 次请求、输入/输出 195,790/82,809 tokens、耗时 407 秒；最终 10 问均无链，并指出研究问题混入基金致谢、指标题返回铺砌定义、核心公式缺失。模型/环境与此前 DeepSeek pilot 不同，质量分数不能直接横比。

**已按可复现根因修复，当前提示缓存版本 `v2.21`**：`GlobalAnswer.confidence` 原定义为 0–1 float，但旧 prompt 未明确禁止“高/低”字符串；Pydantic 拒绝会触发结构化重试，耗尽后问答降级为“证据不足”，旧 provenance fallback 又可能用候选靠前事实代答。现兼容 high/medium/low、中文等级及百分数字符串，并要求 prompt 输出 0–1 数值；明确“证据不足”不会再被候选事实覆盖。类别到数字的映射仅为解析兼容，并非校准概率。

**定量问题遗漏公式也已独立复现并修复**：对同篇 Aztec 论文已保存的 226 条事实离线构造旧候选，17 条 `kind=formula` 全被定量问题 kind 规则排除，包括生成多项式事实。现纳入公式并提高生成多项式/配分函数/铺砌计数类公式的召回；离线重放候选数 19→20，含 2 条相关公式，普通结果排序仍优先。增加回归测试；此验证未调用 API。摘要中不同平铺对象的双射归属、局限丢失参数限定，还需标准答案集及逐句证据评估，不能由这些类型/检索修复自动保证。

2026-10-01 QA-only pilot 用缓存事实未重跑解析/抽取/归纳：10 题中模型返回 6 条链共 16 步；原始 step IDs 全为空，离线回填后 16/16 指针通过启发式检查，另 4 题无链。原始 37 次请求、输入/输出/总计 76,319/14,858/91,177 tokens，28.73 秒；候选范围与线上 group-reduced catalog 不完全相同，仅作诊断记录。关系边仍未语义验证。

**Artificial Social Intelligence PDF 分组预算 A/B（2026-10-06；DeepSeek 当前配置）**：17 页、24 sections、26 chunks，输出 291 facts。9,600 冷跑 group reduce 12 calls / 49,350 input / 9,632 output tokens；12,800 暖缓存臂 8 calls / 40,132 input / 6,457 output，stage token 合计 −21%，调用 −33%。该 stage 的请求延迟合计反而 41.49s→72.78s（均值 3.46→9.10s；延迟不含并发排队），不能宣称变快。12,800 臂复用 59 个缓存、总共只发 11 次模型调用，和 9,600 冷跑总 81 次/239,007 tokens **不具备整跑成本或时延可比性**；按 baseline 外推，假设其他 stage 不变，group 节省约占冷跑总 tokens 5.2%，不是 cold 12,800 实测。


固定 12,800 后的 QA-only none 暖跑命中抽取/分组/overview 缓存，仅 10 次 `global_answer` API 请求、无 regen、17,423/2,800 input/output tokens。对照 9,600 low 冷跑 17 次 QA 请求（含 8 regen，32,062/11,013 tokens），none 约少 53% QA tokens；但有 1 个 QA candidate 输入因分组改变，且低臂缓存复用，百分比不是严格同请求配对。两边自动审计均 4/10 answers_with_evidence、citation precision 1.0；unsupported 计数则 1→21，none 在一个“证据不足”问题留下 2 步链，违反拒答空链要求。已在 provenance 最终降级处加确定性空链约束并补回归；reasoning none 仍不应直接默认，且需固定候选做语义复测。朋友先前报告 71,173 输出 tokens 仍无 stage 明细；本次 PDF 的 DeepSeek 冷跑则显示 `chunk_extract` 输出 40,522/73,762（54.9%，26 calls），`global_answer` 首次 9 calls 后又 regen 8 次（共 17 calls），为后续降本的主要实测线索。具体分阶段数据、缓存限制和答案质量解释见 `outputs/perf/artificial_social_intelligence_ab_report.md`。当前完整测试为 224 tests OK。

**chunk_extract-only compact JSON A/B（同一 17 页 PDF）**：新增 `benchmark_run.py --extract-only`，只发 26 次 extraction 请求、跳过后续归纳/QA。原 schema 冷基线 38,375/40,522 input/output（78,897 total）；v2.12 compact 短键臂 40,238/31,629（71,867 total，抽取阶段 −8.9%，输出 −21.9%）。grounded facts 291/318→308/337，比例约91.5%→91.4%；source_quote 平均长度180.5→140.4，但 80.1% 仍超过 prompt 所要求的80字符。按事实字段精确匹配的重合很低，且无 downstream QA/人工语义评估；不能声称语义质量等价。compact 结果 JSON 和 usage ledger 见 `outputs/perf/artificial_social_compact_extract_only.json` / `_usage.sqlite3`，完整说明见 `outputs/perf/artificial_social_intelligence_ab_report.md`。



```
PDF 解析 + 建树           124 s   ← 比并行后的抽取(30–36s)还长 3–4 倍
  └─ pymupdf4llm.to_markdown   96–116 s
       └─ find_tables()         95.5 s   ← 全文只有 4 张表！
逐页 get_text('text')       2.2 s   ← 纯文本其实只要 2.2 秒
```

以下是交接时记录的本地 profiling 结果；对应的 `use_layout` 对照脚本/原始 trace 未随仓库保存，数值与章节差异应在继续优化前用固定 PDF、缓存条件和 commit 重新复测。

**当时定位的原因**：`pymupdf4llm` 1.28 默认启用 layout（交接记录如此），其中表格检测耗时较高。

**但关闭 layout 不能采用**（已实测）：

| | 解析+建树 | 章节数 | 章节总 token | 层级分布 |
|---|---|---|---|---|
| layout=True（当前） | 124.1 s | **18** | 28,068 | {1:1, 2:7, 3:9, 5:1} |
| layout=False | **33.3 s** | **10** | 54,135 | {1:1, 2:9} |

子节标题全部消失（`### 2.1`…），还混入图片文字当标题（`### T 1 T 2 T 3`、`# = ∣ λ`）。章节树退化 → 单节 token 翻倍 → **分块、归纳输入、检索证据全变**。这是真实质量退化。

---

## 6. 本地质量样例（非版本化基线；未必对应当前 HEAD）

下表汇总交接时保存的本地运行结果；`inputs/` 与 `outputs/` 被 Git 忽略，结果未绑定 commit、输入哈希和运行参数，不能单凭仓库内容证明它们均来自当前 HEAD。不同运行样本的事实数也可能不同；比较前应核对具体 JSON 与统计口径。

| 论文 | 事实数 | 有证据 | 无据率 | 备注 |
|---|---|---|---|---|
| GigaGAN | 527 | 9/10 | 0.136 | 数字保真 0 错 |
| InstructIR | 609 | 9/10 | 0.136 | 数字保真 0 错 |
| MambaIRv2 | 1075 | 10/10 | 0.348 | 8 条全为 retrieval_gap |
| NAFNet | 297 | 9/10 | 0.182 | — |
| Attention | 289 | 9/10 | 0.095 | 未来工作题修复后 4/4 轮正确 |
| SegEarth | 258 | 10/10 | 0.098–0.27 | σ 大，见 §4.1 |
| aztec（48页） | 278–296 | 9/10 | 0.15–0.18 | 概述无据数字 0 |

**数字保真度**：交接记录称跨 7 篇共 **162+ 个数字核验、0 错误**；当前未找到逐条数字与原文对照台账，暂无法独立复算。

---

## 7. 已知问题 / 未完成清单

### 7.1 待修

1. **未来工作 `overview.future_work` 字段**（朋友报告的原例）—— v2.15 曾把路线图“将两种模型推广到 k-tilings”误列为未来工作，尽管后续 §2.4 已定义并展示该扩展；还把附录 A 的“北极曲线其他部分可类似处理”误列为未来工作，但同页 Figure 25 已报告模拟与计算的曲线。现有确定性回查分别要求 k-tiling 完成事实，或“其他曲线部分”步骤与同页已计算曲线证据；普通开放问题/无同页完成证据保留。离线应用到 v2.15 后移除上述 2 条，其他 3 条保留；尚未用新代码重跑 LLM/API。
2. **定理/定义公式缺失**—— v2.15 把 Theorem 1.3/1.4 标为正文抽取缺公式；检查原 PDF 物理第 4 页发现，直接 `PyMuPDF Page.get_text("text")` 可读到生成多项式与分段北极曲线公式，而 `pymupdf4llm` 清洁正文流漏掉它们。曾修复两处端到端断点：Markdown 斜体标记导致缺公式触发器漏判；`build_heading_tree()` 重建 `PageMarkdown` 时丢弃页元数据。现已为公式候选页追加压缩 PyMuPDF span 几何提示：相对 bbox、字号比及弱上标标记，每页最多 40 spans/1000 字符；原生 PDF 文本仍是唯一可引用源，布局提示不单独支撑事实。原 PDF 页 1/4/33 的 span 检查显示，几何可区分 Theorem 1.1 的堆叠指数、Theorem 1.3 的乘积上下限，以及 Lemma 6.1 的 binomial 式上下堆叠；布局截断时只允许利用完整可见的局部公式簇，否则维持未核实。抽取缓存版本已升至 v2.20，缓存键包含几何摘要。**本地真实 PDF 路径与 258 项测试通过；没有新 API 全文重跑，未验证新的模型输出/评分；线性化和部分截断公式仍需原页核验，不能宣称准确率提升。图像 OCR 仍单独标记为低置信度。**
3. **概览把数学对象写进 `metrics`**—— v2.15 把生成多项式、北极曲线、耦合 domino 对数、路径步数列为评价指标，但“使用了哪些评价指标？”的答案是证据不足。现已要求文档概览仅列实际评估模型/实验的指标，并在新生成/缓存返回时根据有证据的 metric facts 过滤；离线应用到保存输出，以上 5 项全部移除。尚未用新代码重跑 API。
4. **LLM 产生“元陈述”垃圾事实**，例如：
   ```
   concept='BLEU'  statement='原文使用了评价指标 BLEU，并把数值结果记为 2014 、41.8 、3.5 。'
   ```
   `2014` 是年份被当成数值结果。这类事实无信息量却占预算。
5. **部分模型原始响应中的 `statement` 含裸换行**——若原始 JSON 字符串中的换行没有正确转义，可能导致该响应解析失败；但事实字段内的合法换行经 `json.dumps` 写入最终文件时会被转义，不会单独令输出 JSON 非法。需保留具体原始响应，确认损坏发生在模型响应解析阶段还是下游处理阶段。
6. **AdaIR_ICLR2025.pdf 是损坏文件**（27,013,120 字节、无 `%%EOF`、页树不可读），**需用户重新上传**。已加 `validate_pdf()` 给出可读报错。

### 7.2 已量化的优化候选（未实施）

1. **替换标题检测以关掉 layout** ← **最大性能杠杆**（解析 124s → 约 33s，可把 55.6s 再降到约 35s）。做法：按字号/字体自行判别标题，替代 layout 的标题识别，然后关闭它。**必须先做只读可行性验证**（比对两种方式的标题召回率），确认不丢章节再动。
2. **`chunk_extract` 输出压缩** —— 它是唯一输出>输入的阶段（约 39k out vs 37k in），输出 token 更贵。抽取提示词要求 ≤20 条事实、`source_quote` ≤80 字。
3. **`section_reduce` 的 `source_quote`**（占该 payload 22%）—— 但这一步模型**确实在读**引文做接地，**倾向不动**；若要动，先算功效再测。

### 7.3 有意不做的

- **`citation_precision` 恒为 1.0**：`_recover_answer_fact_ids` 会主动剔除不在白名单的 id，它天然无法下降。定位是**异常探测器**，不是质量指标。
- **`unsupported_claim_rate` 不度量切题性**：答非所问但有原文依据的句子同样得 0 分。
- **`unsupported_undecided` 是幻觉上界不是判定**：同义词（`speedup` vs "加速比"）与凭空编造在词元重叠上都是 0.00，无法区分。

---

## 8. 外部评测（朋友给的）

交接记录称，朋友曾用 V2.19 与另一套系统（“GPT-6 Luna”，普通模式生成剧情帧）做过对照。原始报告 `accuracy-review.md` 当前工作区未找到；下表数据仅为交接时记录，缺少原始报告，**无法独立核验**：

| 模式 | 产物 | 耗时 | 调用 | 输入 token | 输出 token | 知识准确度 A |
|---|---|---|---|---|---|---|
| 普通 | 95帧剧情，4道测验 | 197.9 s | 7 | 338,077 | 23,021 | 100 |
| V2.19 | 18节、297事实、摘要及10问 | 614.7 s | 91 | 182,907 | 100,058 | 92 |

**注意**：
- 这是**两套不同系统、不同产物**的对照，文档自己也写了"不是同一终端任务的严格排名"。
- **614.7 s 与 55.6 s 不可直接比较**：现存 benchmark 的 55.6 s 是不含 PDF 解析的流水线墙钟；朋友原始报告缺失，614.7 s 的计时范围、输入与参数无法确认。复核时需用相同 PDF、代码版本与计时口径重跑。
- 表中 A92 vs A100 是单次外部报告值；原始评分表与评测样本缺失，无法判断差异是否显著。§4 的方差估计针对无据率，不能直接用于判定知识准确度 A 分数的噪声。
- 他报的 4 处错误里，**3 处未复现**（膨胀卷积条件保留、未伪造"道路vs建筑"矛盾、mRNA 与蛋白未混），**1 处（数字-主体错配，BLEU 41.8 配十二小时）确认在概述层**。该例在提示词加约束后**新跑的 9 轮里未再出现**，但单看几轮无法归因。
- 他指出的"假指标 Recall"**归因不对**：不在 `extract_metric_facts`（实测该句返回 0 条），而在 **LLM 抽取层**无门槛标注。已修（`0745010`）。

---

## 9. 常用命令速查

```powershell
# 环境
$env:PYTHONUTF8="1"
cd F:\P2G-V2.19-friend-test

# 测试
powershell -ExecutionPolicy Bypass -File .\test.ps1

# 跑一篇（CLI）
& ".\longtext\Scripts\python.exe" python_scripts\p2g_v2.py "inputs\aztec-k-tilings-v3.pdf" `
  --output outputs\run.json --memory-db outputs\m.sqlite3 `
  --cache-db outputs\c.sqlite3 --usage-db outputs\u.sqlite3 `
  --section-concurrency 8 --concurrency 8

# 只结算已有输出的指标（不需 PDF / 不需 Key）
& ".\longtext\Scripts\python.exe" python_scripts\p2g_v2.py --metrics outputs\run.json

# 多轮评测（复用已抽事实，只重跑归纳+问答）
& ".\longtext\Scripts\python.exe" python_scripts\multiround_eval.py `
  outputs\_e_Attention_1706.03762.json --rounds 3
# 连抽取一起重跑（验证抽取层/续写改动）
& ".\longtext\Scripts\python.exe" python_scripts\multiround_eval.py `
  outputs\_e_Attention_1706.03762.json --rounds 3 --full

# 性能基准（自己测墙钟 + token + 质量）
& ".\longtext\Scripts\python.exe" python_scripts\benchmark_run.py `
  inputs\aztec-k-tilings-v3.pdf --tag myrun --section-concurrency 8 --concurrency 8

# 提交
git add -A; git commit -m "..."; git push origin main
gh run list --limit 1     # CI 约 25-40 秒，应 success
```

**可回退开关**（都经过实测）：
```powershell
--section-concurrency 1     # 回退章节串行
--full-group-payload        # 回退分组 payload 瘦身
--group-batch-tokens 3200   # 回退分组分批（会退化成一节一次调用）
```

---

## 10. 建议的下一步

按价值排序：

1. **让朋友用修正版做一次配对重跑**：固定同一 PDF、模型/endpoint、环境与并发，先保留其“普通模式”产物作质量参照；记录每题原始 confidence 类型、`content_retry_attempts`、API stage/token、证据链是否解析、候选 fact IDs、最终 citations 和是否进入 fallback。确认 10 问不再因 confidence 类型失去链，并逐题复核致谢误入研究问题、指标/公式问答及摘要/局限限定词问题。请求数偏高时先按 stage 用量定位，不先做 token 优化。
2. **建立带标准证据链的诊断集**：为跨章节多跳、数值核验、版本冲突、无答案问题人工标注“问题→必要事实/原文指针→推理链→答案”，报告 chain recall、支持度、关系边质量、题目相关性与拒答质量；覆盖数学论文和实验论文。
3. **评估适配不同模型的输出兼容性**：confidence 文本映射只是容错，不是置信度校准；在 Luna 与 DeepSeek 上各做固定题集的 schema 遵循率/解析重试对照，确认兼容层不掩盖其他类型漂移。
4. **验证 global-answer `none` 的质量非劣性**：将 chain 指标纳入 `low|none` 配对试验；确认之前默认保持 `low`。
5. **最后再做预算受限的证据检索/成本优化**：质量基线通过后才评估动态扩展候选是否减少 tokens/调用；解析 profile 则按 §5.4 固定 PDF/环境复测。

---

## 11. 关键人物与产物

- **用户**：研究生，有导师/课题组，希望这个项目"能发论文、也能继续给 paper2gal 使用"；但后来明确**不与 paper2gal 结合**，只做长文本提取技术本身。**论文的事先搁置**。
- **朋友**：曾提供外部评测信息及三篇论文 PDF（Attention / SegEarth / ANLN）；交接文档提到的 `accuracy-review.md` **当前工作区缺失**，需重新索取后才能核查相关数字。
- **测试语料** `inputs/`（已 gitignore，不入库）：

  | 文件 | 大小 | 说明 |
  |---|---|---|
  | `aztec-k-tilings-v3.pdf` | 41.9 MB | **48 页数学论文**，性能基准与朋友评测都用它 |
  | `Attention_1706.03762.pdf` | 2.1 MB | Transformer 原论文（朋友报错论文之一） |
  | `SegEarth-OV_CVPR2025.pdf` | 2.2 MB | 遥感分割（朋友报错论文之一） |
  | `ANLN_m6A_HCC.pdf` | 9.5 MB | 中英混排医学论文（朋友报错论文之一） |
  | `MambaIRv2_CVPR2025.pdf` | 2.5 MB | 图像恢复 |
  | `NAFNet_ECCV2022.pdf` | 16.0 MB | 图像恢复 |
  | `InstructIR_ECCV2024.pdf` | 25.9 MB | 图像恢复 |
  | `GigaGAN_ICLR2023.pdf` | 46.8 MB | 生成模型（**最大**） |
  | `AdaIR_ICLR2025.pdf` | 25.8 MB | ⚠️ **损坏**（无 `%%EOF`、页树不可读），需重新获取 |

- **产物目录** `outputs/`（已 gitignore）：运行 JSON、SQLite（memory/cache/usage）、`eval*/` 多轮结果、`perf/` 基准结果。

---

## 12. 新会话开场建议

把这份文档贴过去后，可以直接说：

> 读 `HANDOFF.md`，先跑 `.\test.ps1` 确认 224 tests OK，然后我们做 §10 的第 N 项。

如果需要我**先复核某处代码**，说清是哪个 commit / 哪个函数即可（本会话改了很多路由逻辑，`build_question_request` 是重灾区）。
