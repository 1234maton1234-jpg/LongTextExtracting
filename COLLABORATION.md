# 协作说明

本仓库为**私有仓库**，只有被添加为 **Collaborator** 的人可见。各自的 GitHub 账号、SSH 密钥、API Key 均**各自保管，不共享**。

## 一、加入项目（第一次）

1. 在 GitHub 上**接受邀请**（邀请会发到你的 GitHub 邮箱）。
2. 克隆仓库：
   ```powershell
   git clone https://github.com/1234maton1234-jpg/LongTextExtracting.git
   cd LongTextExtracting
   ```
3. 安装环境（自动创建 `.venv` 并安装依赖）：
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup.ps1
   ```
4. 配置自己的 Key：
   ```powershell
   Copy-Item .env.example .env
   ```
   编辑 `.env`，把 `DEEPSEEK_API_KEY` 改成**你自己的**（不要用别人的，不要提交）。
5. 跑测试确认环境正常（99 个应全部通过）：
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\test.ps1
   ```

## 二、日常流程（改代码）

铁律：**动手前先 `git pull`，改完就 `git push`**，避免两个人改同一个文件打架。

```powershell
git pull origin main      # 1. 动手前先拉到最新
# ... 修改代码 ...
git add .
git commit -m "描述你改了什么"
git push origin main      # 2. 推送
```

**如果 push 被拒绝**（说明朋友先推了，本地落后了）：
```powershell
git pull origin main      # 拉取并合并朋友的改动
# 若有冲突，打开冲突文件手动解决后：
git add .
git commit -m "合并朋友的改动"
git push origin main
```

## 三、分支约定（可选，改动频繁时建议）

每人一个分支，改完发起 Pull Request（PR）由管理员合并，互不干扰：

```powershell
git checkout -b dev-你的名字     # 从当前分支开新分支
# ... 改完 ...
git push origin dev-你的名字
```
然后在 GitHub 仓库页面点 **Pull requests → New pull request**，选你的分支发起 PR。

## 四、必须遵守的规则

| 规则 | 原因 |
|------|------|
| `.env` 绝不提交 / 绝不分享 | 里面有个人 API Key；`.gitignore` 已保护，但别用 `git add -f` 强制提交 |
| `outputs/` 绝不提交 | 运行产物，每次运行都会变；各跑各的互不干扰 |
| 私钥 `~/.ssh/id_ed25519` 绝不外发 | 等于你的 GitHub 账号钥匙 |
| 改代码前先 `git pull` | 减少冲突、减少返工 |
| 提交信息写清楚 | 方便对方快速看懂你改了什么 |

## 五、常见问题

- **`setup.ps1` 报 "64-bit Python 3.10-3.12 was not found"**：本机需要先安装 64 位 Python 3.10-3.12 并勾选加入 PATH。
- **`run.ps1` 报 "DEEPSEEK_API_KEY is missing"**：还没配 `.env`，见上文第 4 步。
- **API 限流 / 请求报错**：加 `-Concurrency 1`（或 2）降低并发重试。
- **`.venv` 损坏**：删掉 `.venv` 文件夹，重新跑 `setup.ps1`。
