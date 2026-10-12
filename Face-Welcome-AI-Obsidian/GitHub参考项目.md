---
tags: [GitHub, 调研]
---

# GitHub 参考项目

> 页面核对日期：2026-10-12。以下是代码结构和产品形态参考，不是对准确率的背书。仓库、依赖、预训练模型和权重许可各自独立，复用前逐项核验。

## 1. HelloHuman：最接近轻量迎宾闭环

- 仓库：[jscottvogel/HelloHuman](https://github.com/jscottvogel/HelloHuman)
- README 描述：OpenCV YuNet + SFace ONNX、已知人脸、每人冷却、轮播音频欢迎词；提供 Windows 本地测试说明及树莓派部署路径。
- 可借鉴：轻量模型组合、冷却机制、全屏展示、摄像头到播报的最小闭环。
- 不同点：更偏边缘设备/演示迎宾，不是完整人员管理桌面应用；音频欢迎词与动态中文姓名播报也不是完全相同需求。
- 注意：模型和依赖的来源许可需单独核对；不要把“本地运行”理解成首次安装无需联网。

## 2. Faces Local：最接近 PySide6 实时人脸工作台

- 仓库：[baardove/Faces](https://github.com/baardove/Faces)
- README 描述：OpenCV + MediaPipe + PySide6，本地实时检测/识别/跟踪，SQLite 人员数据，线程、摄像头选择、未知人员管理和实时叠层。
- 可借鉴：摄像头工作线程与 UI 的边界、状态面板、多人追踪、训练多角度样本，以及人脸/人体跟踪标签的组织方式。
- 不同点：面向实验/识别工作台，功能比迎宾首版更重；项目明确说明不是认证、门禁或安全系统，也没有活体检测。
- 注意：首次设置会下载 Python 包和模型；目标系统支持、权限和打包方式要自行验证。

## 3. PhotoAIdent：InsightFace/ArcFace + PySide6 + 本地数据参考

- 仓库：[steinsag/photoaident](https://github.com/steinsag/photoaident)
- README 描述：本地隐私优先的照片人脸识别应用，使用 InsightFace/ArcFace，在 PySide6 桌面 UI 中逐步标注已知人员或匿名人员。
- 可借鉴：InsightFace 特征、人员标注、隐私优先和桌面数据管理思路。
- 不同点：主要处理本地照片库，不是摄像头实时迎宾；实时帧率、摄像头线程和播报队列需要另行设计。

## 4. Windows-Face-Hello：Windows + PySide6 工作线程参考

- 仓库：[everglow01/Windows-Face-Hello](https://github.com/everglow01/Windows-Face-Hello)
- README/文档描述：Windows 人脸登录项目，包含 PySide6 控制台、设置/诊断/录入流程，并把摄像头和推理工作放进 QThread，通过信号更新 UI。
- 可借鉴：Windows 桌面权限、录入流程、诊断页面和 QThread 组织方式。
- 不同点：它是认证/登录项目，不应直接套用到迎宾场景；安全边界和隐私要求更高，代码复用前必须审查。

## 5. OpenCVGUI：PySide6 深色界面参考

- 仓库：[samuellimabraz/OpenCVGUI](https://github.com/samuellimabraz/OpenCVGUI)
- README 描述：包含 PySide6 摄像头界面、深色主题、青绿色点缀、折叠面板和视频渲染；仓库声明 MIT 许可证。
- 可借鉴：桌面摄像头预览、深色主题、控件密度和基本布局。
- 不同点：它主要是 OpenCV 图像处理探索工具，不是人脸身份识别产品。

## 建议的阅读顺序

1. 读 `HelloHuman`：参考识别后播报、冷却和迎宾展示。
2. 读 `Faces Local`：参考 PySide6 摄像头线程、实时叠层和桌面工作台。
3. 读 `PhotoAIdent`：参考本地人员数据和 InsightFace 桌面管理。
4. 看 `Windows-Face-Hello`：参考 Windows 线程、设置和诊断组织。
5. 看 `OpenCVGUI`：参考深色界面和摄像头预览布局。

## 结论

没有必要整套照搬某一个仓库。更稳妥的组合是：`HelloHuman` 的识别/冷却/播报逻辑 + `Faces Local` 的桌面摄像头工作台组织 + `PhotoAIdent` 的本地人员数据思路 + 自己的中文欢迎语、记录和视觉设计。模型选择与代码授权必须分别验证。

## 后续核对清单

- [ ] 实际 clone 后检查最近提交、issue 和可运行性
- [ ] 核对每个仓库是否有 LICENSE；没有明确许可时不复用代码
- [ ] 分别核对 InsightFace、YuNet/SFace、MediaPipe 等模型许可与商业/分发限制
- [ ] 确认首次下载模型是否可提前准备，以满足离线安装要求
- [ ] 在目标摄像头、光照、距离和人员样本下测试误识与拒识
- [ ] 复查是否保存原始照片、特征向量、日志，及删除流程
