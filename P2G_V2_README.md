# P2G V2.19 长文理解测试包

这个版本用于测试 PDF 超长文本理解，不负责生成 Galgame。它会把 PDF 转为 Markdown，建立标题树，按约 3200 Token 分块并保留约 180 Token 重叠，再抽取带原文引用和页码的事实，最后执行分层全文归纳与 10 项全局问答。

## 当前能力

- **结构理解**：目录、图注、提示词转储、图片 OCR 文本等噪声在建树前剥离；章节层级归一化；≤500 token 的碎片子节并入父节（子节标题保留为父节正文首行标记，小节号仍能锚定"小节级问题"的证据）。
- **截断不中断**：`finish_reason="length"` 截断时先尝试解析完整内容，否则关闭思考重新生成，或从已产出的部分 JSON 续写（拼接后解析、失败自动接力）。输出加事实上限 + 关思考 + 预算对齐后，抽取与归纳阶段实测截断 0 次。
- **token 高效**：抽取与归纳阶段首次调用 `reasoning_effort="none"`，整个输出预算全部给内容，杜绝"思考烧预算"；reduce 族总 token 实测下降约 34%，无质量回退。恢复机制保留作兜底。
- **可靠答案**：10 项全局问答按句子粒度审计证据（无证据句子移入 `unsupported_claims`，不再整条误杀）；答案强制中文转述、禁止粘贴原文引用，英文泄漏有确定性兜底；"矛盾"问题确定性列出跨章节冲突并附页码出处；证据不足时保守作答。
- **表格净化**：配置表数值行与跨列碎片表头噪声在事实库源头剔除，不再以 `kind="metric"` 进入证据索引，不压过真实指标定义。

## 实测验证

- 多篇不同形态的 PDF（实验论文、英文综述、结构较乱的中译综述）全流程实测：10/10 全局问答零幻觉、零英文泄漏、答案带可定位页码出处、"矛盾"问题可溯源。
- 截断与恢复：抽取与归纳阶段实测 0 截断、0 恢复调用；reduce 族 reasoning token 由约 134K 降至约 0.3K（仅总览保留思考），总 token 降幅约 34%。
- 单元测试 **103 个全部通过**。

## 仍存在的问题

1. 纯图表内文字随图片 OCR 一起剥离，不再入事实库；待用图表密集的论文验证是否削弱"图表内容"类问题的回答。
2. "未来工作"问题偶发仍混入方法实现描述，检索层可再收紧事实 kind 白名单。
3. 高层摘要（`executive_summary`）为事实拼接式归纳，连贯性可提升；概述层文档标题偶发误译（采样波动）。

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

预期结果为 `103 tests ... OK`。

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
