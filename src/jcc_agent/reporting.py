"""Self-contained local reports with no network or unescaped OCR content."""
from __future__ import annotations

import html
import json
from pathlib import Path


def render_report(output: str | Path, summary: dict, rows: list[dict]) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        state = row["state"]
        names = " / ".join(s.get("hero_id") or s["status"] for s in state["shop"])
        detail = json.dumps(state.get("detail"), ensure_ascii=False)
        action = row.get("action") or {}
        cards.append(f'''<article><a href="{html.escape(row.get('image', ''), quote=True)}"><img loading="lazy" src="{html.escape(row.get('image', ''), quote=True)}"></a>
<div><h2>{row['timestamp']:.2f}s · {html.escape(state['phase'])}</h2>
<p>用户阶段标注：{html.escape(row['reference_phase'])}；OCR 耗时：{row['latency_ms']:.0f} ms</p>
<p>回合 {html.escape(str(state.get('round')))} · 金币 {state.get('gold')} · 等级 {state.get('level')}</p>
<p>商店：{html.escape(names)}</p><p>详情：{html.escape(detail)}</p>
<p>拟执行：{html.escape(action.get('kind', 'none'))} — {html.escape(action.get('reason', ''))}</p>
<details><summary>完整观测与诊断</summary><pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></div></article>''')
    count = summary.get("frames", len(rows))
    median = summary.get("latency_ms_median")
    median_text = f"{median:.0f} ms" if isinstance(median, (int, float)) else "未统计"
    stats = f'<div class="stats"><span>抽检 <b>{count}</b> 帧</span><span>识别中位耗时 <b>{median_text}</b></span><span>实际游戏输入 <b>0</b> 次</span></div>'
    target = output / "report.html"
    target.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>金铲铲 OCR 验证报告</title><style>body{font:15px/1.6 system-ui;background:#101923;color:#e2eaf2;max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:30px}article{display:grid;grid-template-columns:48% minmax(0,1fr);gap:24px;padding:24px 0;border-bottom:1px solid #354354}img{width:100%;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#182634;padding:16px}p{color:#c1d0df;overflow-wrap:anywhere}summary{cursor:pointer;color:#70d7c9}.stats{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}.stats span{padding:16px 24px;background:#182634;border-radius:8px}.stats b{color:#70d7c9}@media(max-width:800px){article{grid-template-columns:1fr}}</style>
<h1>录像验证 · 画之灵 OCR Agent</h1><p>识别结果来自 PaddleOCR。时间标签仅用于独立比较；录屏中原有操作不代表 Agent 已执行这些操作。</p>'''
                      + stats + '<details><summary>素材、指标与验证范围</summary><pre>'
                      + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
                      + "</pre></details>" + "".join(cards) + "</html>", encoding="utf-8")
    return target
