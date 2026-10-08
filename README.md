# 金铲铲 OCR Agent

本地 Python Agent：读取商店名称和售价，维护己方持牌账本及对手快照，估计余牌、选择阵容，并逐次确认购买、刷新和买经验。运行时仅使用 PaddleOCR，不接入英雄卡面分类器或视觉大模型。

**首版已有可运行代码和真实录像验证；尚未验证实际游戏点击。** 录屏中的星级只有图标、空位没有文字，因此执行前需核对初始持牌、当前视角和窗口坐标，不能从任意画面直接接管整局。

## 能力与边界

- 录像提帧、单行文字批量 OCR、独立真值评估及 HTML 报告。
- 59 个共享牌库英雄 ID、5 个可编辑阵容模板；余牌区间、同行参考、转型门槛。
- 自动购买、D 牌、买经验、文本偏好选择、详情关闭及分批侦察的控制流程。
- 实际售价、保留金币、回合总支出、目标副本、容量与结果确认。
- 默认只观察；执行模式有版本校验、超时暂停、窗口失焦/外部输入停止、Esc 和停止文件。

规则配置来自原版 TFT 画之灵资料，尚未核实为金铲铲具体返场版本。免费刷新、特殊赠牌/合成需要适配；缺少相应确认时暂停。自动部署、装备和满备战卖牌不属于已实现能力。

## 本机运行

本机 `.venv` 已装好。已测 Windows、Python 3.13.6、PaddleOCR 3.4.1、PaddlePaddle 3.3.1。首次加载包含模型获取/初始化，不能用热态耗时估算。

```powershell
.venv\Scripts\python.exe -m jcc_agent doctor --video "示例视频.mp4"
.venv\Scripts\python.exe -m jcc_agent analyze --video "示例视频.mp4" --timestamps "5,27,35,70,120,140,160,220,260,270,280,289" --out artifacts/my-analysis
.venv\Scripts\python.exe -m jcc_agent demo --out artifacts/my-demo
.venv\Scripts\python.exe -m pytest -q
```

`analyze` 生成日志、图片和 `report.html`，不操作游戏。`demo` 明确使用合成状态，不算实战证据。每次实验使用新的输出目录。

新环境安装：

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[ocr,dev]"
```

复现本机依赖可先安装 `requirements.windows-py313.lock.txt`，再 `pip install --no-deps -e .`。这是 Windows/Python 3.13 版本快照，不是跨系统锁。只保留 `opencv-contrib-python`，避免与提供同一个 `cv2` 模块的 `opencv-python` 混装。

## 接入游戏

1. 用 `windows` 列出窗口句柄。校准实际客户区文字区域和点击点，导出 `configs/local.layout.json`；示例配置仅对应 960×432 录像，`calibrated=false`。
2. 按 [初始持牌说明](docs/initial-inventory.md) 填写 `configs/local.inventory.json`。未确认位置保持未知。
3. 先运行观察模式检查 OCR、拟动作和等待原因。核对完成后再增加 `--execute`；执行期间不要操作键鼠，切换窗口或手动输入会停止。
4. Esc 或创建 `STOP` 文件可停止。`--scout` 另外要求玩家名称映射、玩家入口及格点配置，默认关闭。

```powershell
.venv\Scripts\python.exe -m jcc_agent windows
.venv\Scripts\python.exe -m jcc_agent calibrate --image artifacts/cli-validation/frames/0004_70.00.jpg --out artifacts/calibration.html
# 将 HWND 换成实际窗口句柄；不要原样使用占位符。
.venv\Scripts\python.exe -m jcc_agent live --hwnd HWND --layout configs/local.layout.json --self-state configs/local.inventory.json --duration 60 --out artifacts/live-observe
```

1–2 秒是目标输入间隔，购买确认、未知重试及冷启动可延长。实时 D 牌全流程耗时尚未实测。

## 审阅入口

- [验证结果](docs/validation.md)：真实 OCR、测试及未验证边界。
- [开发架构](docs/architecture.md)：模块、数据流和动作确认。
- [开发与运行说明](docs/development.md)：配置、调试与扩展。
- [规则与策略](docs/rules-and-strategy.md)：来源、算法、预算及特殊机制。
- [OCR 素材发现](docs/perception-findings.md)。
- [未决问题与默认方案](docs/decisions.md)。
- [需求与验收](docs/requirements.md)。

原始录屏保留在根目录；录屏、模型缓存及 `artifacts/` 不纳入源代码版本控制。本次没有自动提交或推送仓库。
