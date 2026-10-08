# 开发与运行说明

## 入口

代码在 `src/jcc_agent/`，配置在 `configs/`，行为测试在 `tests/`，独立人工标签在 `validation/`，真实录像验证脚本在 `scripts/`。执行 `python -m jcc_agent --help` 查看参数。全局 `--rules` 放在子命令之前。

| 命令 | 用途 | 游戏输入 |
|---|---|---|
| `doctor --video ...` | 依赖、规则来源、录像信息 | 无 |
| `extract --video ... --timestamps ...` | 指定时点提帧 | 无 |
| `analyze --video ... --out ...` | 真实 OCR、拟决策及报告 | 无 |
| `replay --observations ... --out ...` | 重新计算结构化策略 | 无 |
| `demo --out ...` | 合成交易确认演示 | 无 |
| `calibrate --image ... --out ...` | 离线区域和坐标编辑 | 无 |
| `windows` | 列出窗口句柄 | 无 |
| `live --hwnd ...` | 观察绑定窗口 | 无 |
| `live --hwnd ... --execute` | 校准后逐次执行和确认 | 有 |

`analyze --start 0 --end 290 --every 15` 可替代指定时间列表。阶段时间标签仅适用于示例视频；使用其他视频时，不应把该时间表作为准确率真值。

## 配置契约

- `reference_size` 为游戏客户区像素尺寸，实时窗口必须匹配；不包括模拟器工具栏和边框。
- `points` 为相对客户区的 `[x,y]`，范围 0–1。输入前映射桌面坐标并核对前台窗口。
- `regions` 为完整文字检测区域；`single_line_regions` 是默认 `recognition` 模式实际读取的单行区域。校准页面分别提供这两种区域和点击点编辑。
- `ocr_options.enable_mkldnn=true`、`cpu_threads=4` 是本机实测配置，其它设备需重新测量。
- `field_regions` 关联多个候选区域，例如回合号早/晚 UI。矛盾结果保留未知。
- `self_player_name` / `player_name_map` 为文字玩家名到逻辑 ID 的映射，不能用固定排行榜行号替代身份。
- `choice_preferences` 可配置 `{"encounter":["一元店"],"augment":["选项标题"]}`，按顺序精确匹配唯一高置信标题；默认无自动偏好。
- `scouting.players`、`slots`、`player_points`、`budget_seconds` 控制侦察。静态棋盘默认仅允许 `preparation`；`gameplay` 不能证明棋子不移动。

## 初始状态与视角

`SelfTracker` 从37个核对位置建立副本和星级分布。只有显式普通合成声明及完整基线，实时执行才通过启动检查。对手星级未知可以估区间，己方追牌和合成需要明确基线。

自己的名字无稳定文字位置时，可在初始 JSON 明确填写 `"ui_context_reviewed": true` 和 `"viewed_player": "self"`。这表示操作者已核对启动时在自己的棋盘，**不是 OCR 结果**。日志来源为 `reviewed_session_context`，保留原核对时刻。未填写绝不默认自己；导航尝试前失效，即使点击失败也不恢复，之后需新鲜身份 OCR。

执行模式检测外部键鼠输入、失焦和窗口变化并停止，避免人机同时操作破坏账本。该上下文依赖独占输入；游戏自动转场和未建模变化仍需实际验证。模板、上一局及历史录像库存均不能替代当前事实。

## 每轮流程

1. 捕获绑定窗口、识别文字、附置信度和时间。
2. 融合允许的视角上下文；在同一有效商店代次补上已确认购买的空槽。
3. 确认上一动作，成功后更新账本和支出。待确认刷新不得借用旧商店空槽。
4. 合并对手最近快照、计算己方汇总、重算阵容和一个动作。
5. 校验画面年龄、阶段、视角、预算和窗口，发送一个动作。
6. 下轮重新读取；待确认时不追加输入。

购买空白槽确认要求精确扣款、同回合、其他槽稳定及两次不同的新画面空白。低置信乱码不能当作空白证据。刷新需要新完整商店。新回合首帧暂停经济动作，第二帧可信向前回合确认后重置预算。超时暂停，不能推断“应该点成功了”。

## 调试

检查 `events.jsonl` 的 `observation.action.reason`、`dispatch.result.reason`、`confirmation`、`stopped`。常见原因是库存不完整、视角未知、详情未关闭、商店某槽未知、价格未知、金币/等级/回合低置信、经验未知、预算不足或容量下界不足。

首次下载/加载模型导致过期帧时不操作，等待之后的新画面。不要为绕过延迟而直接放宽画面年龄。特殊免费刷新或合成导致暂停时，应补机制状态契约，不能关闭确认。`STOP` 文件存在会阻止新运行，重启前由操作者明确移除。

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe scripts/validate_recording.py --out artifacts/new-benchmark --modes recognition --repeats 2
```

识别器不读取人工标签。合成状态只能验证控制流程，不能提供真实准确率或赢率。新增动作需定义前置条件、确认信号、超时行为及账本事件，不是只增加点击坐标。

真实接入仍需核对当前版本与窗口，先观察再验证单次购买、刷新和切换玩家。这次录像开发没有完成这些实际输入测试。
