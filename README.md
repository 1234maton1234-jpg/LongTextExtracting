# P2G V2.19 长文理解测试包

本版本用于测试 **PDF 超长文本理解**（不负责生成 Galgame）。流程：PDF 转 Markdown → 建立标题树 → 按约 3200 Token 分块（约 180 Token 重叠）→ 抽取带原文引用和页码的事实 → 分层全文归纳与 10 项全局问答。

> 📄 详细版本说明、已验证基线、已知问题见 [P2G_V2_README.md](P2G_V2_README.md)。

## 环境要求

- Windows 10/11
- 64 位 Python 3.10-3.12
- 自己的 DeepSeek API Key

## 快速开始

```powershell
# 1. 安装依赖（自动创建 .venv）
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 2. 配置自己的 Key
Copy-Item .env.example .env
# 然后编辑 .env，把 DEEPSEEK_API_KEY 改成你自己的（示例文件里已写好注释）

# 3. 跑测试确认环境正常（87 个应全部通过）
powershell -ExecutionPolicy Bypass -File .\test.ps1

# 4. 运行（处理一份 PDF）
.\run.ps1 -PdfPath "你的文件.pdf"
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-PdfPath` | （必填） | 要处理的 PDF 路径 |
| `-Output` | `outputs\p2g_v2_output.json` | 输出文件位置 |
| `-Concurrency` | `4` | 并发数；账号限流时改为 1 或 2 |
| `-FlashModel` | `deepseek-v4-flash` | 块抽取使用的模型 |
| `-ProModel` | `deepseek-v4-pro` | 多块章节归纳使用的模型 |
| `-DryRun` | 关 | 不调 API，只做预处理 |

## 协作

本仓库为**私有仓库**，参与协作需先被添加为 **Collaborator**。完整流程见 [COLLABORATION.md](COLLABORATION.md)。

## 目录结构

```
python_scripts/p2g_v2.py   # 主程序
tests/                     # 单元测试（unittest）
outputs/                   # 运行产物（已 gitignore，各跑各的）
.env                       # 个人 Key（已 gitignore，勿提交）
setup.ps1 / run.ps1 / test.ps1   # 一键安装 / 运行 / 测试
```
