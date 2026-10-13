# Face Welcome AI 第一版

这是可运行的摄像头迎宾程序，当前已实现：

- OpenCV 摄像头预览
- InsightFace / SCRFD 主人脸检测，OpenCV YuNet 和 Haar 备用
- 深色 AI 中控台界面
- 人脸识别框上显示姓名、匹配度和 UNKNOWN 状态
- SQLite 本地人员数据库和“人员管理 / 录入样本”窗口
- 中文、English、中英双语播报，以及自定义欢迎语
- 摄像头断开和重新连接反馈

当前识别优先使用 InsightFace 的 SCRFD 检测和 ArcFace 特征；轻量模型包位于程序目录的 `models/buffalo_sc` 文件夹。OpenCV YuNet + SFace 作为自动备用后端，录入样本时会保存当前活动后端的归一化特征，并要求连续两帧匹配后才确认身份。没有现代模型文件时才回退到旧版 Haar + 灰度特征模式。未登记人员仍然会显示 UNKNOWN，不会被当作已登记人员播报。

采集样本时会先检查检测置信度、人脸大小、画面清晰度、亮度和关键点是否完整；模糊、过暗、过曝、太远或重复样本会被拒绝。人员管理中的“可用样本”只统计当前活动模型真正能够使用的样本；InsightFace、SFace 和旧版灰度特征不会混用。

当前状态：**本地预览识别模式**。程序使用 InsightFace / SCRFD / ArcFace 做主检测和身份特征比对，YuNet + SFace 自动备用，旧版灰度样本仅作为现代模型缺失时的兼容回退。它用于验证迎宾流程和界面，不应当用于门禁、考勤或实际身份认证。

升级识别算法后，旧版灰度样本不能转换成 ArcFace 特征；旧 SFace 样本也不能直接当作 ArcFace 样本。首次使用 InsightFace 时，请在“人员管理 / 录入样本”中为每个人重新采集 3-5 份样本，建议包含正面、轻微转头和不同距离。

## 本地环境配置（2026-10-13）

当前配置使用 **Windows x64 / Python 3.10 独立虚拟环境**，不使用系统默认 Python 或 Anaconda 环境。

固定的直接依赖：

| 依赖 | 版本 |
|---|---|
| InsightFace | 2.1 |
| ONNX Runtime（CPU） | 1.23.2 |
| OpenCV | 4.12.0.88 |
| NumPy | 2.2.6 |

`requirements.txt` 固定直接依赖；`requirements-lock.txt` 保存本次安装的全部依赖版本。
Tkinter、SQLite 使用 Python 自带模块，语音使用 Windows `System.Speech`，不需要另外安装 TTS Python 包。
模型继续使用项目自带的 `models/buffalo_sc/` 和 YuNet / SFace 文件，没有更换模型或识别算法。

### 首次配置或在另一台电脑恢复

先安装 Windows x64 Python 3.10（包含 Tkinter 和 Python 启动器），然后在项目目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\配置项目环境.ps1
```

脚本会创建 `.venv`、安装锁定依赖并检查实际主识别后端。如果 `.venv` 已存在但版本不兼容，脚本会提示错误，不会删除或覆盖原环境。
`-ExecutionPolicy Bypass` 仅作用于这次 PowerShell 进程，不修改系统执行策略。

### 环境自检

```powershell
.\.venv\Scripts\python.exe .\verify_environment.py
```

自检要求 `active_feature_type=insightface`，仅回退到 OpenCV 不算通过；同时检查 SCRFD、ArcFace、YuNet、SFace 的实际推理、Tkinter 界面构建及临时 SQLite 数据库。
自检不打开摄像头、不播放语音，也不创建或修改正式人员数据库。它不能代替真实人脸、识别准确率和扬声器现场验收。

## 启动

1. 连接摄像头。
2. 在项目目录运行：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\启动人脸迎宾程序.ps1
   ```

   启动脚本固定使用 `.venv\Scripts\python.exe`，无需手动激活环境。也可以直接运行：

   ```powershell
   .\.venv\Scripts\python.exe .\face_welcome_app.py
   ```

3. 界面底部应显示 `INSIGHTFACE ENGINE`；如果显示 SFace 或 legacy，说明主识别后端未成功启用，应先运行环境自检。
4. 如果程序启动时摄像头还没有连接，连接后点击右侧“重新连接摄像头”。
5. GitHub 仓库不包含人员数据库。在新目录首次正式启动会创建空的 `face_welcome.db`，请单独恢复自己的数据库，或通过人员管理重新录入 3–5 份 InsightFace / ArcFace 样本。

## 录入人员

1. 点击右侧“人员管理 / 录入样本”。
2. 点击“新建”，填写姓名、员工编号和播报语言，再点击“保存人员”。
3. 回到摄像头前，确保画面出现清晰人脸，在管理窗口选中该人员，点击“采集当前人脸”。
4. 建议同一个人采集 3-5 份样本：正面、轻微转头、不同距离各一份；每次采集时保持脸部清晰、光线均匀。
5. 如果弹出“样本质量不合格”或“样本重复”，按照提示调整距离、光线或角度后再采集。
6. 重新面对摄像头；识别框会显示对应姓名，右侧会显示匹配度，并按该人员的语言播报。

### 从本地图片录入样本

如果不方便到摄像头前采集，可以在人员管理窗口点击“从图片上传样本”，一次选择多张 JPG、PNG、BMP 或 WEBP 图片。每张图片必须是清晰的单人照片；程序会自动检查人脸数量、清晰度、亮度、尺寸、关键点和重复度。不合格图片会在结果窗口中列出原因。

上传时原始图片只在内存中用于提取特征，不会复制到数据库；数据库只保存当前识别后端的特征向量。建议每个人准备 3-5 张不同角度和不同距离的照片。

数据库文件是程序目录下的 `face_welcome.db`。不需要手动编辑 SQLite 文件，优先通过“人员管理”维护；删除人员时会同时删除其人脸样本。

## 语言和欢迎语

- `中文 · zh-CN`：播报“欢迎姓名”。
- `English · en-US`：播报“Welcome, Name”。
- `中英双语 · zh-en`：播报中英双语文本。
- 自定义欢迎语可以使用 `{name}` 占位符，例如 `早上好，{name}`。

Windows 如果没有安装对应的系统语音，程序会使用默认语音作为回退。

如果界面显示“播报失败 · 检查系统语音”，说明 Windows 本地语音组件或音频输出设备没有正常可用；程序会加载 `System.Speech` 程序集、使用默认音频设备并显示失败状态，不再静默吞掉后台错误。
