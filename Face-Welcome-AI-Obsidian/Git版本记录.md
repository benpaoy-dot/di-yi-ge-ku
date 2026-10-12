---
tags: [Git, 版本, 交接, Face-Welcome-AI]
status: 本地仓库已建立
---

# Git 版本记录

## 当前快照

本地项目目录已经建立 Git 仓库，用于保存方案一切换后的代码、模型和 Obsidian 文档快照。

- 首次快照提交：`3e30bc5`（切换 InsightFace ArcFace 主识别方案）
- 分支：`main`
- GitHub 远程仓库：`https://github.com/benpaoy-dot/di-yi-ge-ku`
- 上传状态：已推送到 `origin/main`
- 最近上传日期：`2026-10-12`

本次快照包含：

- `face_welcome_app.py`：InsightFace / SCRFD / ArcFace 主路径，YuNet / SFace 备用路径。
- `models/` 下的 ONNX 模型文件。
- `启动人脸迎宾程序.ps1`、`README-第一版.md` 和界面原型文件。
- `Face-Welcome-AI-Obsidian/` 下的项目说明、技术方案、开发日志、风险说明和交接文档。
- `Git版本记录.md`：本页版本说明。

## 不纳入仓库的内容

- `face_welcome.db`：包含人员资料和人脸特征，属于本地隐私数据；通过 `.gitignore` 排除。
- `models/buffalo_sc.zip`：模型压缩包是下载缓存，解压后的模型文件已经保留。
- `__pycache__/`、Python 字节码和 SQLite 临时文件。

## 当前版本说明

- 主识别：InsightFace `buffalo_sc`，SCRFD 检测 + ArcFace 特征。
- 备用识别：OpenCV YuNet + SFace。
- 最终回退：Haar + `legacy` 灰度特征。
- ArcFace 初始匹配阈值：`0.45`，仍需使用重新采集的现场样本校准。
- 当前数据库仍需为每个人重新采集 3-5 份 InsightFace / ArcFace 样本。

## 恢复方式

在项目目录执行 `git log --oneline` 可以查看版本记录；使用 `git show <提交号>` 可以查看某次快照。恢复后需要单独准备本机的 `face_welcome.db`，再通过人员管理重新录入样本。

当前仓库已经配置 GitHub 远程地址并完成首次推送。后续修改可在项目目录执行 `git add -A`、`git commit` 和 `git push`；人员数据库和人脸特征仍被 `.gitignore` 排除，不会自动上传。
