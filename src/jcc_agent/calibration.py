"""Generate an offline calibration editor; no desktop input is sent."""
from __future__ import annotations

import base64
import json
from pathlib import Path


def build_calibration_page(image_path, layout, output) -> Path:
    from PIL import Image
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        size = list(image.size)
        mime = Image.MIME[image.format]
    layout = dict(layout)
    layout["reference_size"] = size
    layout["calibrated"] = False
    source = json.dumps(layout, ensure_ascii=False).replace("<", "\\u003c")
    image_url = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JCC 位置校准</title>
<style>body{background:#14202b;color:#e6eef6;font:15px/1.6 system-ui;margin:24px}h1{font-size:26px}main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:24px}aside{min-width:0;overflow-wrap:anywhere}canvas{display:block;width:100%;box-sizing:border-box;touch-action:none;background:#090e13;border:1px solid #566577}label{display:block;margin:12px 0}input,select,button,textarea{font:inherit;padding:8px;background:#223545;color:#fff;border:1px solid #526878;border-radius:5px;box-sizing:border-box}select,input,textarea{width:100%}button{cursor:pointer;background:#25675b;margin:6px 0;width:100%}textarea{height:160px}#verified{width:auto}code{color:#8ee4c3}@media(max-width:800px){main{grid-template-columns:1fr}}</style>
<h1>棋盘与文字区域校准</h1><p>区域模式：拖动框选文字。点击位置模式：单击按钮或棋盘格中心。导出只生成配置文件，不操作游戏。棋盘皮肤、模拟器工具栏与分辨率变化后需要重新检查。</p>
<main><canvas id="canvas"></canvas><aside><label>编辑类型<select id="kind"><option value="single_line_regions">单行文字区域（默认识别模式）</option><option value="regions">完整文字区域（检测模式）</option><option value="points">点击位置</option></select></label><label>名称<select id="key"></select></label><label>新增名称<input id="newKey" placeholder="例如 board_0 或 detail"></label><button id="add">添加名称</button><label>自己的玩家名称（用于OCR确认视角）<input id="selfName"></label><p id="value"></p><label><input type="checkbox" id="verified">我已核对当前窗口的区域和点击位置</label><button id="save">下载 local.layout.json</button><label>当前配置<textarea id="json" readonly></textarea></label><p>启动前保存到 <code>configs/local.layout.json</code>。录像坐标不能直接当作实际游戏窗口坐标。</p></aside></main>
<script>
const layout=__LAYOUT__; const image=new Image();const c=document.getElementById('canvas'),ctx=c.getContext('2d');
const kind=document.getElementById('kind'),key=document.getElementById('key'),selfName=document.getElementById('selfName');
layout.regions ||= {};layout.single_line_regions ||= {};layout.points ||= {};selfName.value=layout.self_player_name||'';let down=null;
function refreshKeys(){key.replaceChildren();for(const k of Object.keys(layout[kind.value])){const o=document.createElement('option');o.value=o.textContent=k;key.appendChild(o)}draw()}
function draw(){if(!image.complete||!image.naturalWidth)return;ctx.drawImage(image,0,0);const v=layout[kind.value][key.value];ctx.strokeStyle='#27ffb6';ctx.fillStyle='#27ffb6';ctx.lineWidth=Math.max(2,c.width/500);if(v){if(kind.value!=='points'){ctx.strokeRect(v[0]*c.width,v[1]*c.height,(v[2]-v[0])*c.width,(v[3]-v[1])*c.height)}else{ctx.beginPath();ctx.arc(v[0]*c.width,v[1]*c.height,8,0,Math.PI*2);ctx.fill()}}document.getElementById('value').textContent=key.value+': '+JSON.stringify(v);document.getElementById('json').value=JSON.stringify(layout,null,2)}
function point(e){const b=c.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))].map(v=>Math.round(v*1e6)/1e6)}
c.onpointerdown=e=>{down=point(e);c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!down||!key.value)return;const p=point(e);if(kind.value==='points'){layout.points[key.value]=p}else{const r=[Math.min(down[0],p[0]),Math.min(down[1],p[1]),Math.max(down[0],p[0]),Math.max(down[1],p[1])];if(r[2]-r[0]>.001&&r[3]-r[1]>.001)layout[kind.value][key.value]=r}down=null;draw()};
kind.onchange=refreshKeys;key.onchange=draw;document.getElementById('add').onclick=()=>{const k=document.getElementById('newKey').value.trim();if(!/^[a-zA-Z0-9_]+$/.test(k))return alert('名称只用英文、数字和下划线');if(!(k in layout[kind.value]))layout[kind.value][k]=kind.value!=='points'?[.1,.1,.2,.2]:[.5,.5];refreshKeys();key.value=k;draw()};
document.getElementById('save').onclick=()=>{layout.calibrated=document.getElementById('verified').checked;layout.self_player_name=selfName.value.trim();const blob=new Blob([JSON.stringify(layout,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='local.layout.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);draw()};
image.onload=()=>{c.width=image.naturalWidth;c.height=image.naturalHeight;refreshKeys()};image.src='__IMAGE__';
</script></html>'''
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__LAYOUT__", source).replace("__IMAGE__", image_url), encoding="utf-8")
    return target
