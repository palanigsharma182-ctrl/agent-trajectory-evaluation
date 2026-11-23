"""Render an evidence-first, fully offline HTML report for one research run set."""

from __future__ import annotations

import html
from importlib.resources import files
import json
from typing import Any


_BRANCH_NAMES = {
    "original": "正常运行",
    "control": "对照运行",
    "intervention": "屏蔽后运行",
}

_SCRIPT = """(function(){
  const tabs=[...document.querySelectorAll('.run-tab')], panels=[...document.querySelectorAll('.run-panel')];
  const views=panels.map(panel=>{
    const choices=[...panel.querySelectorAll('.graph-choice')];
    const container=panel.querySelector('.graph-frame');
    const payload=panel.querySelector('.graph-data');
    const frames=payload ? JSON.parse(payload.textContent) : [];
    let selected=0, network;
    const draw=()=>{
      if(!container || !frames[selected]) return;
      const frame=frames[selected];
      if(network) network.destroy();
      network=new vis.Network(container,{nodes:frame.nodes,edges:frame.edges},{
        layout:{hierarchical:{direction:'LR',sortMethod:'directed',levelSeparation:150,nodeSpacing:95}},
        physics:false,interaction:{hover:true},edges:{arrows:'to',smooth:false,font:{size:10}},
        nodes:{margin:9,shape:'box',font:{size:13}}
      });
      container.dataset.snapshot=String(selected);
    };
    const select=i=>{
      selected=i;
      choices.forEach((choice,j)=>choice.classList.toggle('selected',i===j));
      draw();
    };
    choices.forEach((c,i)=>c.addEventListener('click',()=>select(i)));
    panel.querySelectorAll('.event').forEach(ev=>ev.addEventListener('toggle',()=>{
      if(!ev.open)return;
      const idx=Number(ev.dataset.eventIndex); let target=-1;
      frames.forEach((f,i)=>{if(f.event_index<=idx)target=i});
      if(target>=0)select(target);
    }));
    return {draw};
  });
  tabs.forEach(t=>t.addEventListener('click',()=>{
    tabs.forEach(x=>x.classList.toggle('active',x===t));
    panels.forEach((p,i)=>p.classList.toggle('active',i===Number(t.dataset.run)));
    views[Number(t.dataset.run)].draw();
  }));
  if(views.length) views[0].draw();
})();"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _text(value: Any, fallback: str = "—") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, (dict, list, tuple)):
        return _json(value)
    return str(value)


def _evidence_link(paper: dict[str, Any], paragraph_id: Any) -> str:
    pid = html.escape(_text(paragraph_id), quote=True)
    if paragraph_id in paper.get("indexed_source_ids", set()):
        return f'<a class="source-link" href="#source-{pid}">{pid}</a>'
    return f'<span class="source-id">{pid}</span>'


def _source_ids(ids: Any, paper: dict[str, Any]) -> str:
    if not ids:
        return '<span class="muted">无关联来源</span>'
    if not isinstance(ids, (list, tuple)):
        ids = [ids]
    return " ".join(_evidence_link(paper, item) for item in ids)


def _graph_frames(run: dict[str, Any]) -> list[dict]:
    """Serialize only graph data for each step and share PyVis's browser library."""
    from pyvis.network import Network

    frames = []
    for i, event in enumerate(run.get("events") or []):
        snapshot = event.get("graph")
        if not isinstance(snapshot, dict) or not snapshot.get("nodes"):
            continue
        net = Network(directed=True)
        for node in snapshot["nodes"]:
            status = node["status"]
            color = {"SUPPORTED": "#d8eee3", "READ": "#e0effb", "UNAVAILABLE": "#f9dad8"}.get(status, "#f4e9d6")
            label = f"{node['id']}\n{status}"
            detail = html.escape(node["label"] + "\n" + node.get("reason", ""))
            net.add_node(node["id"], label=label, title=detail, color=color)
        for edge in snapshot["edges"]:
            net.add_edge(edge["source"], edge["target"], label=edge["kind"])
        frames.append({"event_index": i + 1, "caption": f"第 {event['step']} 步 · {event['event_type']}",
                       "nodes": net.nodes, "edges": net.edges})
    return frames


def _impact_list(items: Any, paper: dict[str, Any], empty: str) -> str:
    if not items:
        return f'<p class="muted">{html.escape(empty)}</p>'
    chunks = ['<ul class="impact-list">']
    for item in items:
        chunks.append(
            '<li><strong>' + html.escape(_text(item.get("label", item.get("id")))) + '</strong>'
            + f'<span class="kind">{html.escape(_text(item.get("kind"), "未分类"))}</span>'
            + '<div class="change"><span>' + html.escape(_text(item.get("before"), "无")) + '</span><b aria-label="变化为">→</b><span>' + html.escape(_text(item.get("after"), "无")) + '</span></div>'
            + '<p>' + html.escape(_text(item.get("reason"), "未记录原因")) + '</p>'
            + '<small>来源链：' + _source_ids(item.get("source_ids", []), paper) + '</small></li>'
        )
    chunks.append('</ul>')
    return "".join(chunks)


def _run_section(run: dict[str, Any], index: int, paper: dict[str, Any]) -> str:
    branch = str(run.get("branch", ""))
    branch_label = _BRANCH_NAMES.get(branch, branch or f"运行 {index + 1}")
    events = run.get("events") or []
    frames = _graph_frames(run)
    controls = ''.join(f'<button type="button" class="graph-choice{" selected" if j == 0 else ""}">{html.escape(frame["caption"])}</button>'
                       for j, frame in enumerate(frames))
    frame_markup = '<div class="empty-graph">此运行没有记录图状态。</div>'
    if frames:
        graph_data = json.dumps(frames, ensure_ascii=False).replace("<", "\\u003c")
        frame_markup = f'<div class="graph-frame active" aria-label="此步骤的证据依赖图"></div><script type="application/json" class="graph-data">{graph_data}</script>'

    timeline = []
    for i, event in enumerate(events):
        step = event.get("step", i + 1)
        event_type = _text(event.get("event_type"), "未命名事件")
        summary = _text(event.get("summary"), "")
        payload = html.escape(_json(event), quote=False)
        timeline.append(
            f'<details class="event" data-event-index="{i + 1}"><summary><span class="step">{html.escape(_text(step))}</span>'
            f'<span class="event-type">{html.escape(event_type)}</span><span class="event-summary">{html.escape(summary)}</span></summary>'
            f'<pre>{payload}</pre></details>'
        )
    if not timeline:
        timeline.append('<p class="muted">没有事件记录。</p>')

    citations = " ".join(_evidence_link(paper, pid) for pid in run.get("citations", [])) or '<span class="muted">无引用段落</span>'
    answer = html.escape(_text(run.get("answer"), "无最终回答"))
    termination = html.escape(_text(run.get("termination_reason"), "未记录"))
    run_id = html.escape(_text(run.get("run_id"), f"run-{index + 1}"))
    additional = run.get("additional_state_changes", [])
    other = ""
    if additional:
        other = ("<section class='graph-area'><h3>不在直接依赖链上的运行差异</h3>"
                 "<p class='muted'>分叉后模型的补查和引用可以改变。这些状态差异单独列出。</p>"
                 + _impact_list(additional, paper, "") + "</section>")
    return f'''<section class="run-panel{" active" if index == 0 else ""}" id="run-{index}" data-branch="{html.escape(branch, quote=True)}">
      <header class="run-head"><div><p class="eyebrow">{html.escape(branch_label)}</p><h2>{run_id}</h2></div><div class="termination"><small>终止状态</small><strong>{termination}</strong></div></header>
      <div class="outcome-grid"><article><h3>最终回答</h3><pre class="answer">{answer}</pre></article><article><h3>引用段落</h3><p>{citations}</p></article></div>
      <div class="compare-grid"><article><h3>直接依赖及上层影响</h3>{_impact_list(run.get("impacts"), paper, "未记录到直接影响。")}</article><article><h3>保持的证据与结果状态</h3>{_impact_list(run.get("preserved"), paper, "未记录保持状态的项。")}</article></div>
      {other}
      <section class="graph-area"><div class="section-title"><div><p class="eyebrow">关系图</p><h3>按步骤检查状态</h3></div><p>图快照来自该分支的执行事件。悬停节点可以查看完整说明。</p></div><nav class="graph-choices" aria-label="关系图状态">{controls}</nav>{frame_markup}</section>
      <section class="events-area"><div class="section-title"><div><p class="eyebrow">执行轨迹</p><h3>完整事件记录</h3></div><p>展开事件可查看模型消息、原生 tool call 参数、工具观测与其他原始字段。</p></div><div class="timeline">{"".join(timeline)}</div></section>
    </section>'''


def render_report(report: dict) -> str:
    """Render three run variants as a standalone offline HTML evidence report."""
    paper = dict(report.get("paper") or {})
    task = report.get("task") or {}
    runs = report.get("runs") or []
    sources = {}
    for run in runs:
        for event in run.get("events", []):
            for paragraph in event.get("observation", {}).get("paragraphs", []):
                sources[paragraph["id"]] = paragraph
    paper["indexed_source_ids"] = set(sources)
    source_index = "".join(f'<article id="source-{html.escape(pid, quote=True)}"><h3>{html.escape(pid)} · '
                           f'{html.escape(row["section"])}</h3><p>{html.escape(row["text"])}</p></article>'
                           for pid, row in sorted(sources.items()))
    library = files("pyvis").joinpath("templates", "lib", "vis-9.1.2", "vis-network.min.js").read_text(encoding="utf-8")
    nav = ''.join(
        f'<button type="button" class="run-tab{" active" if i == 0 else ""}" data-run="{i}"><span>{html.escape(_BRANCH_NAMES.get(str(run.get("branch")), str(run.get("label", f"运行 {i + 1}"))))}</span><small>{html.escape(_text(run.get("termination_reason"), "状态未知"))}</small></button>'
        for i, run in enumerate(runs)
    ) or '<p class="muted">没有运行数据。</p>'
    panels = ''.join(_run_section(run, i, paper) for i, run in enumerate(runs))
    intervention = report.get("intervention") or {}
    paper_url = paper.get("source_url")
    source = f'<a href="{html.escape(str(paper_url), quote=True)}" target="_blank" rel="noopener">打开论文来源 ↗</a>' if paper_url else '<span class="muted">未提供论文链接</span>'
    snapshot = html.escape(_text(report.get("snapshot_id"), "未记录"))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-src 'self' data:; img-src data:;">
<title>{html.escape(_text(paper.get("title"), "Research trajectory report"))} · 轨迹与影响报告</title>
<style>
:root{{--ink:#1d2938;--muted:#667386;--line:#dce3eb;--paper:#fff;--canvas:#f4f6f9;--blue:#2864a3;--blue-soft:#e8f1fb;--green:#18704b;--orange:#a55812;--shadow:0 8px 30px #26384e0d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}a{{color:#145995}}.wrap{{max-width:1250px;margin:0 auto;padding:34px 28px 70px}}.hero{{padding:30px 34px;background:var(--paper);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}}.eyebrow{{margin:0 0 5px;color:var(--blue);font-size:12px;font-weight:750;letter-spacing:.09em;text-transform:uppercase}}h1{{margin:0;font-size:clamp(26px,4vw,39px);line-height:1.18;letter-spacing:-.03em}}.hero-sub{{margin:12px 0 0;color:var(--muted);font-size:16px}}.meta{{display:flex;flex-wrap:wrap;gap:10px 24px;margin-top:20px;color:var(--muted);font-size:13px}}.meta strong{{color:var(--ink)}}.contract{{margin-top:16px;padding:13px 16px;border-left:3px solid #b6cbe0;background:#f7fafd;color:#34465b;border-radius:0 8px 8px 0}}.intervention{{margin-top:20px;padding:16px 18px;background:#fff7e9;border:1px solid #f2dfbd;border-radius:12px}}.intervention h2{{margin:0 0 6px;font-size:14px;color:#794817}}.intervention code{{font-size:12px}}.tabs{{display:flex;gap:10px;margin:24px 0 14px;overflow-x:auto}}button{{font:inherit;cursor:pointer}}.run-tab{{flex:1;min-width:165px;text-align:left;border:1px solid var(--line);border-radius:12px;background:white;padding:13px 16px;color:var(--ink)}}.run-tab.active{{border-color:#76a5d4;background:var(--blue-soft);box-shadow:inset 0 0 0 1px #76a5d4}}.run-tab span,.run-tab small{{display:block}}.run-tab span{{font-weight:750}}.run-tab small{{margin-top:3px;color:var(--muted);font-size:12px}}.run-panel{{display:none;background:white;border:1px solid var(--line);border-radius:16px;padding:25px 28px;box-shadow:var(--shadow)}}.run-panel.active{{display:block}}.run-head,.section-title{{display:flex;justify-content:space-between;align-items:flex-end;gap:20px}}.run-head{{border-bottom:1px solid var(--line);padding-bottom:17px}}h2{{margin:0;font-size:24px}}h3{{margin:0 0 10px;font-size:17px}}.termination{{text-align:right}}.termination small{{display:block;color:var(--muted)}}.termination strong{{color:var(--green)}}.outcome-grid,.compare-grid{{display:grid;grid-template-columns:1.7fr 1fr;gap:18px;margin-top:20px}}.compare-grid{{grid-template-columns:1fr 1fr}}.outcome-grid article,.compare-grid article{{border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}}pre{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}}.answer{{max-height:280px;overflow:auto}}.muted{{color:var(--muted)}}.source-link,.source-id{{display:inline-block;margin:1px 5px 1px 0;font:12px ui-monospace,Consolas,monospace}}.kind{{display:inline-block;margin-left:8px;color:var(--muted);font-size:11px}}.impact-list{{padding:0;list-style:none;margin:0}}.impact-list li{{padding:10px 0;border-top:1px solid #eef1f5}}.impact-list li:first-child{{border-top:0}}.impact-list p{{margin:6px 0;font-size:13px}}.impact-list small{{color:var(--muted)}}.change{{display:flex;align-items:center;gap:8px;margin-top:7px;font-size:13px}}.change span{{padding:4px 7px;background:#f4f6f9;border-radius:5px;overflow-wrap:anywhere}}.change b{{color:var(--blue)}}.graph-area,.events-area{{margin-top:26px;padding-top:22px;border-top:1px solid var(--line)}}.section-title{{align-items:center;margin-bottom:13px}}.section-title p:last-child{{max-width:52%;margin:0;color:var(--muted);font-size:12px;text-align:right}}.graph-choices{{display:flex;gap:7px;overflow-x:auto;margin-bottom:10px}}.graph-choice{{white-space:nowrap;background:white;border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:12px;color:#34465b}}.graph-choice.selected{{background:var(--blue-soft);border-color:#7caddb;color:#164d80}}.graph-frame{{display:none;width:100%;height:470px;border:1px solid var(--line);border-radius:10px;background:white}}.graph-frame.active{{display:block}}.empty-graph{{padding:35px;border:1px dashed var(--line);color:var(--muted);text-align:center}}.timeline{{border-left:2px solid #d5e0ed;margin-left:11px;padding-left:16px}}.event{{position:relative;border-bottom:1px solid #edf0f4}}.event:before{{content:"";position:absolute;left:-22px;top:16px;width:8px;height:8px;background:#6e9bc8;border:2px solid white;border-radius:50%;box-shadow:0 0 0 1px #6e9bc8}}.event summary{{display:flex;align-items:baseline;gap:12px;cursor:pointer;padding:11px 3px;list-style:none}}.event summary::-webkit-details-marker{{display:none}}.step{{min-width:36px;color:var(--muted);font:12px ui-monospace,Consolas,monospace}}.event-type{{font-weight:700;color:#315c82;min-width:120px}}.event-summary{{color:#586779;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.event pre{{margin:0 0 14px 50px;padding:14px;background:#f6f8fa;border:1px solid #e5eaf0;border-radius:9px;max-height:600px;overflow:auto}}.footer{{margin-top:18px;text-align:center;color:#8290a0;font-size:12px}}@media(max-width:760px){{.wrap{{padding:15px 10px 40px}}.hero,.run-panel{{padding:19px}}.run-head,.section-title{{align-items:flex-start;flex-direction:column}}.termination{{text-align:left}}.outcome-grid,.compare-grid{{grid-template-columns:1fr}}.section-title p:last-child{{max-width:none;text-align:left}}.event-summary{{white-space:normal}}}}
</style></head><body><main class="wrap">
<header class="hero"><p class="eyebrow">Research trajectory & impact</p><h1>{html.escape(_text(paper.get("title"), "未命名论文"))}</h1><p class="hero-sub">{html.escape(_text(task.get("question"), "未记录研究问题"))}</p><div class="meta"><span>论文 ID：<strong>{html.escape(_text(paper.get("id")))}</strong></span><span>任务 ID：<strong>{html.escape(_text(task.get("id")))}</strong></span><span>数据快照：<strong>{snapshot}</strong></span><span>{source}</span></div><p class="contract"><strong>阅读提示：</strong>“正常运行 / 对照运行 / 屏蔽后运行”分别展示各分支的回答与终止状态；影响项记录前后变化、原因及可追溯的段落来源。事件详情保留该分支完整原始记录。</p><aside class="intervention"><h2>干预位置</h2><code>paragraph_id={html.escape(_text(intervention.get("paragraph_id")))} · checkpoint_id={html.escape(_text(intervention.get("checkpoint_id")))} · tool_call_id={html.escape(_text(intervention.get("tool_call_id")))}</code><div class="muted" style="margin-top:5px">前缀是否一致：{html.escape(_text(intervention.get("prefix_equal"), "未记录"))}</div></aside></header>
<nav class="tabs" aria-label="选择运行分支">{nav}</nav>{panels}
<section class="hero graph-area"><h2>原文段落索引</h2><p class="muted">收录各分支实际读到的原文，用于跨分支核对。</p>{source_index}</section>
<footer class="footer">本报告为本地独立 HTML；无外部字体、CDN 或网络脚本依赖。</footer></main>
<script>{library}</script>
<script>{_SCRIPT}</script>
</body></html>'''
