"""Build a self-contained Chinese-first research report without web dependencies."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .research import fingerprint


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _num(value, digits=3, signed=False) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _pct(value, digits=2, signed=False) -> str:
    return _num(float(value) * 100, digits, signed) + "%" if value is not None else "—"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _optional(path: Path):
    if not path.exists():
        return None
    try:
        return _load(path)
    except (OSError, ValueError) as exc:
        return {"record_read_error": str(exc), "status": "unverified"}


def _json_details(label: str, value, *, opened=False) -> str:
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    return (f'<details{" open" if opened else ""}><summary>{_e(label)}</summary>'
            f'<pre>{_e(data)}</pre></details>')


def _link(url: str, label: str) -> str:
    parsed = urlparse(str(url))
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        return _e(label)
    return f'<a href="{_e(url)}" target="_blank" rel="noopener noreferrer">{_e(label)} ↗</a>'


def _evaluation_html(evaluation: dict | None) -> str:
    if not evaluation or "groups" not in evaluation:
        return '<p class="muted">真实 API 可靠性评估尚无有效记录。</p>'
    count = evaluation.get("case_count", 0)
    status = evaluation.get("requested_summary", {})
    rows = []
    for name, group in evaluation["groups"].items():
        rows.append(f'<tr><td>{_e(name)}</td><td>{group["completed"]}/{count}</td>'
                    f'<td>{group["passed"]}/{count}</td><td>{group["verdict_correct"]}/{count}</td>'
                    f'<td>{group["valid_citations"]}/{count}</td></tr>')
    failures = []
    for record in evaluation.get("records", []):
        score = record.get("score") or {}
        if record.get("status") == "completed" and score.get("passed"):
            continue
        if record.get("status") == "running":
            continue
        reasons = []
        for key, label in (("verdict_correct", "判定不符"), ("exact_issue_match", "问题分类不符"),
                           ("valid_citations", "引用未通过"), ("schema_valid", "格式未通过")):
            if not score.get(key):
                reasons.append(label)
        if not (score.get("numeric_grounding") or {}).get("passed"):
            reasons.append("数字核验未通过")
        review = (record.get("output") or {}).get("review") or {}
        failures.append(f'<li><strong>{_e(record["key"])}</strong>：{_e("；".join(reasons))}'
                        f'<br>{_e(review.get("summary",record.get("error_type","请求未完成")))}</li>')
    text = (f'<p class="section-desc">目标 {status.get("total",0)} 次；已完成 {status.get("completed",0)}，'
            f'请求失败 {status.get("failed",0)}，待完成 {status.get("pending",0)}。</p>'
            '<div class="table-wrap" style="margin-top:15px"><table><thead><tr><th>模型 / 模式</th>'
            '<th>完成</th><th>全项通过</th><th>判定正确</th><th>有效引用</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>'
            '<p class="footnote">tools 按需读取证据；full_context 一次提供全部证据。分母为每组目标案例数。'
            '通过要求判定、问题分类、引用及规定数字核验全部满足；引用与结构化数字核验不覆盖所有自由文本。'
            '这些是同一研究证据的人工控制变体、小样本单次运行，不能代表独立真实研究上的普遍准确率。</p>')
    if failures:
        text += '<details><summary>查看全部未通过案例</summary><ul class="list">' + ''.join(failures) + '</ul></details>'
    return text + _json_details("展开可靠性评估原始记录", evaluation)


def _short_name(name: str, chosen: str, comparator: str) -> str:
    if name == chosen:
        return "动量 · 逆波动率"
    if name == comparator:
        return "动量 · 等权对照"
    return {"all_industries_equal_weight": "12 行业等权",
            "market_buy_and_hold": "市场买入持有"}.get(name, name)


def _read_curves(root: Path, research: dict, names: list[str]):
    holdout = research["protocol"]["splits"]["holdout"]
    curves = {}
    periods = None
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("Unsafe strategy artifact name")
        filename = name + "_monthly.csv"
        path = root / "outputs" / "research" / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if research["artifacts"].get(filename) != digest:
            raise ValueError(f"Monthly artifact integrity failure: {filename}")
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.PeriodIndex(frame.index, freq="M")
        frame = frame.loc[holdout["start"]:holdout["end"]]
        expected = pd.period_range(holdout["start"], holdout["end"], freq="M")
        if not frame.index.equals(expected):
            raise ValueError("Monthly chart input does not cover the exact holdout")
        returns = frame["net_return"].to_numpy(dtype=float)
        if not np.isfinite(returns).all() or (returns <= -1).any():
            raise ValueError("Invalid chart returns")
        nav = np.r_[1.0, np.cumprod(1 + returns)]
        drawdown = nav / np.maximum.accumulate(nav) - 1
        recorded = research["results"][name]["metrics"]["holdout"]
        if not np.isclose(nav[-1] - 1, recorded["total_return"], atol=1e-8, rtol=1e-8):
            raise ValueError("Chart and recorded holdout return disagree")
        if not np.isclose(drawdown.min(), recorded["max_drawdown"], atol=1e-8, rtol=1e-8):
            raise ValueError("Chart and recorded holdout drawdown disagree")
        curves[name] = {"nav": nav, "drawdown": drawdown}
        periods = [expected[0] - 1, *expected]
    return periods, curves


def _chart(periods, curves, names, labels, colors, metric: str) -> str:
    width, height = 1000, 350
    left, right, top, bottom = 66, 28, 24, 46
    inner_w, inner_h = width - left - right, height - top - bottom
    all_values = np.concatenate([curves[n][metric] for n in names])
    lower, upper = float(all_values.min()), float(all_values.max())
    if metric == "nav":
        lower = max(0., math.floor(lower * 2) / 2)
        upper = math.ceil(upper * 2) / 2
    else:
        lower, upper = math.floor(lower * 20) / 20, 0.
    if upper <= lower:
        upper = lower + 1
    x = lambda i: left + inner_w * i / (len(periods) - 1)
    y = lambda v: top + inner_h * (upper - v) / (upper - lower)
    title = "留出期净值曲线" if metric == "nav" else "留出期回撤曲线"
    items = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}" xmlns="http://www.w3.org/2000/svg">',
             f'<title>{title}</title>',
             '<desc>四种策略的逐月净收益独立累计；期初净值为1，回撤以各自历史最高净值为基准。完整数值见指标表和随附CSV。</desc>']
    for tick in np.linspace(lower, upper, 5):
        label = f"{tick:.1f}" if metric == "nav" else f"{tick * 100:.0f}%"
        items += [f'<line x1="{left}" y1="{y(tick):.2f}" x2="{width-right}" y2="{y(tick):.2f}" stroke="#e5e8e5"/>',
                  f'<text x="{left-14}" y="{y(tick)+5:.2f}" text-anchor="end" class="axis">{_e(label)}</text>']
    ticks = sorted(set([0, len(periods)-1] + [i for i, p in enumerate(periods) if p.month == 1 and p.year % 3 == 0]))
    for i in ticks:
        label = str(periods[i]) if i in (0, len(periods)-1) else str(periods[i].year)
        anchor = "start" if i == 0 else "end" if i == len(periods)-1 else "middle"
        items.append(f'<text x="{x(i):.2f}" y="{height-15}" text-anchor="{anchor}" class="axis">{_e(label)}</text>')
    # Draw controls first and focal strategy last, while the legend stays stable.
    for name in reversed(names):
        points = " ".join(f"{x(i):.2f},{y(v):.2f}" for i, v in enumerate(curves[name][metric]))
        stroke_width = 3.2 if name == names[0] else 2.1
        dash = ' stroke-dasharray="6 4"' if name == names[-1] else ""
        items.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="{stroke_width}"{dash} points="{points}"><title>{_e(labels[name])}</title></polyline>')
    items.append("</svg>")
    return "".join(items)


CSS = r"""
:root{--ink:#172b2c;--muted:#64716f;--paper:#f6f7f3;--white:#fff;--line:#dde4dd;--green:#186b58;--amber:#986125;--amber-bg:#fff7e9;--red:#9a5844}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--green);text-underline-offset:4px}button,a,summary{-webkit-tap-highlight-color:transparent}a:focus-visible,summary:focus-visible{outline:3px solid #9dcbbb;outline-offset:4px}header{max-width:1200px;margin:auto;padding:25px 40px 20px;display:flex;gap:20px;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}.brand{font-size:15px;font-weight:750;letter-spacing:.04em}.brand-mark{display:inline-block;width:22px;height:22px;background:var(--green);border-radius:6px;vertical-align:-5px;margin-right:10px;box-shadow:inset 0 0 0 6px #ffffff40}nav{display:flex;gap:23px;font-size:13px}nav a{text-decoration:none;color:var(--muted)}main{max-width:1200px;margin:auto;padding:52px 40px 70px}.eyebrow{font-size:11px;letter-spacing:.17em;text-transform:uppercase;font-weight:750;color:var(--green)}h1{font-size:42px;line-height:1.35;letter-spacing:-.04em;margin:14px 0 18px;font-weight:750}h2{font-size:25px;line-height:1.4;letter-spacing:-.025em;margin:0 0 9px}h3{font-size:17px;margin:0 0 9px}.intro{font-size:16px;color:var(--muted);max-width:790px;margin:0 0 25px}.badge{display:inline-block;border:1px solid var(--line);padding:4px 11px;border-radius:20px;font-size:11px;background:#fff;color:var(--muted);margin:0 7px 6px 0}.badge.green{background:#e9f3ec;color:var(--green);border-color:#cbded0}.badge.amber{background:var(--amber-bg);color:var(--amber);border-color:#ead9b8}.hero{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(260px,.6fr);gap:40px;align-items:start}.hero-note{border-left:3px solid var(--green);padding:4px 0 4px 22px;margin-top:10px;font-size:13px;color:var(--muted)}.hero-note strong{display:block;font-size:15px;color:var(--ink);margin-bottom:7px}.conclusion{margin-top:30px;padding:25px 28px;background:var(--amber-bg);border:1px solid #e9ddc7;border-radius:12px;display:grid;grid-template-columns:34px 1fr;gap:14px}.conclusion .symbol{font-size:24px;line-height:1.5;color:var(--amber)}.conclusion h2{font-size:20px;margin:0 0 6px}.conclusion p{margin:0;color:#6c5d48;font-size:14px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0 48px}.kpi{background:#fff;border:1px solid var(--line);border-radius:11px;padding:21px}.kpi .label{font-size:12px;color:var(--muted)}.kpi .number{font-size:31px;line-height:1.3;font-weight:650;letter-spacing:-.04em;margin:10px 0 5px;font-variant-numeric:tabular-nums}.kpi .number.negative{color:var(--red)}.kpi .caption{font-size:11px;line-height:1.6;color:var(--muted)}section{scroll-margin-top:25px;margin-top:44px}.section-top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:20px}.section-desc{color:var(--muted);font-size:13px;margin:0;max-width:850px}.section-num{color:#b4c2b8;font-size:36px;line-height:1;font-weight:300}.card{border:1px solid var(--line);background:#fff;border-radius:12px;padding:26px}.chart-card{padding:24px 20px 16px;margin-bottom:18px}.chart-title{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:0 9px}.chart-title h3{margin:0}.chart-unit{font-size:11px;color:var(--muted)}.chart-scroll{width:100%;overflow-x:auto}svg{display:block;width:100%;height:auto;min-width:580px}.axis{fill:#7a8581;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.legend{display:flex;flex-wrap:wrap;gap:10px 24px;font-size:12px;color:var(--muted);margin:14px 9px 5px}.swatch{display:inline-block;width:18px;height:3px;vertical-align:4px;margin-right:7px;border-radius:2px}.footnote{font-size:11px;color:var(--muted);margin:13px 9px 0;line-height:1.7}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:11px;background:#fff}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}th{text-align:right;background:#f0f3ee;color:var(--muted);font-size:11px;font-weight:650;padding:14px 18px}td{padding:15px 18px;border-top:1px solid #edf0eb;text-align:right}th:first-child,td:first-child{text-align:left}tr.focus{background:#edf5ef}tr.matched{background:#f9faf5}.table-name{font-weight:650}.small-sub{display:block;font-size:10px;color:var(--muted);margin-top:3px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}.method-steps{display:grid;grid-template-columns:repeat(3,1fr);gap:0;border:1px solid var(--line);background:#fff;border-radius:12px;overflow:hidden}.method-step{padding:24px;border-right:1px solid var(--line)}.method-step:last-child{border-right:0}.method-step .date{font-size:13px;font-weight:700;color:var(--green);margin-bottom:9px}.method-step p{font-size:12px;color:var(--muted);margin:0}.method-step strong{display:block;margin-bottom:7px}.note{padding:16px 18px;border-left:3px solid #afc5b7;background:#eff3ec;font-size:12px;color:#52685c;margin-top:18px}.formula{font:12px/1.8 Consolas,"Courier New",monospace;background:#f4f6f2;border:1px solid #e4e9df;border-radius:7px;padding:13px 15px;overflow-wrap:anywhere;margin-top:15px}.list{padding-left:19px;margin:12px 0;color:var(--muted);font-size:13px}.list li{margin:7px 0}details{border-top:1px solid var(--line);padding:14px 0;margin-top:9px}summary{cursor:pointer;font-size:12px;font-weight:650;color:var(--green)}pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:11px/1.7 Consolas,"Courier New",monospace;background:#f3f5f1;border:1px solid #e4e8e1;border-radius:7px;padding:16px;max-height:520px;overflow:auto}.hash{font:11px/1.8 Consolas,"Courier New",monospace;overflow-wrap:anywhere;color:#78847c}.record{margin-top:17px;padding-top:14px;border-top:1px solid var(--line)}.record-label{font-size:11px;color:var(--muted);margin-bottom:4px}.money{font-size:34px;font-weight:650;letter-spacing:-.04em}.money small{font-size:12px;color:var(--muted);font-weight:400;letter-spacing:0;margin-left:8px}.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.limitation-list{display:grid;grid-template-columns:1fr 1fr;gap:16px 28px;padding:0;list-style:none;counter-reset:limit}.limitation-list li{padding-left:31px;position:relative;font-size:13px;color:var(--muted)}.limitation-list li:before{counter-increment:limit;content:counter(limit,decimal-leading-zero);position:absolute;left:0;top:0;color:var(--green);font-size:11px;font-weight:700}.limitation-list strong{display:block;color:var(--ink);margin-bottom:5px}.footer{margin-top:45px;border-top:1px solid var(--line);padding-top:20px;display:flex;justify-content:space-between;gap:20px;font-size:11px;color:var(--muted)}.english{font-size:12px;color:var(--muted);margin:14px 0 0}.pill{font-size:10px;color:var(--green);border:1px solid #c9dccc;border-radius:4px;padding:1px 6px;margin-left:7px}.status-title{font-weight:700;font-size:16px;margin:12px 0 9px}.muted{color:var(--muted)}
@media(max-width:800px){header{padding:20px 23px;align-items:flex-start}nav{gap:14px;font-size:12px;flex-wrap:wrap;justify-content:flex-end}main{padding:32px 23px 50px}h1{font-size:33px}.hero{grid-template-columns:1fr;gap:4px}.hero-note{margin-top:4px}.kpis{grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:35px}.kpi{padding:18px}.kpi .number{font-size:28px}.two-col{grid-template-columns:1fr}.method-steps{grid-template-columns:1fr}.method-step{border-right:0;border-bottom:1px solid var(--line);padding:20px}.method-step:last-child{border-bottom:0}.limitation-list{grid-template-columns:1fr}.conclusion{padding:20px;gap:10px;grid-template-columns:24px 1fr}.conclusion h2{font-size:18px}.card{padding:21px}.chart-card{padding:20px 12px 14px}.section-num{font-size:28px}.footer{flex-direction:column;gap:5px}}
@media print{body{background:#fff;font-size:11px}header,main{max-width:none;padding:16px 20px}nav{display:none}.card,.table-wrap,.conclusion,.kpi{break-inside:avoid}section{break-inside:auto;margin-top:26px}h1{font-size:29px}.kpis{margin-bottom:25px}.chart-scroll{overflow:visible}svg{min-width:0}details:not([open]){display:none}.footer{margin-top:25px}a{color:inherit}}
"""


def build_report(root: Path) -> Path:
    root = Path(root).resolve()
    research = _load(root / "outputs" / "research" / "research.json")
    for filename, key in (("protocol.json", "protocol_sha256"), ("selection.json", "selection_sha256")):
        if hashlib.sha256((root / "outputs" / "research" / filename).read_bytes()).hexdigest() != research[key]:
            raise ValueError(f"Frozen {filename} hash mismatch")
    protocol, selection = research["protocol"], research["selection"]
    chosen, comparator = selection["selected_name"], selection["comparator_name"]
    names = [chosen, comparator, "all_industries_equal_weight", "market_buy_and_hold"]
    labels = {n: _short_name(n, chosen, comparator) for n in names}
    colors = dict(zip(names, ["#186b58", "#be8758", "#8ca39c", "#667e9b"]))
    periods, curves = _read_curves(root, research, names)
    comparison = research["comparison"]
    interval = comparison["sharpe_interval_95"]
    holdout = protocol["splits"]["holdout"]
    hmetrics = {n: research["results"][n]["metrics"]["holdout"] for n in names}
    chosen_spec = research["results"][chosen]["spec"]
    review = _optional(root / "outputs" / "ai" / "research_review.json")
    evaluation = _optional(root / "outputs" / "evaluation" / "results.json")
    budget = _optional(root / "outputs" / "budget-summary.json")
    costs = _optional(root / "outputs" / "cost-analysis.json")
    benchmark_notes = _optional(root / "outputs" / "evaluation" / "interpretation.json")
    import_demo = _optional(root / "outputs" / "ai" / "import_demo.json")
    study_hash = fingerprint(research)
    if review and review.get("study_sha256") != study_hash:
        review = {"status":"unverified", "reason":"研究审查与当前研究哈希不匹配；不能据此认证当前研究。"}
    if evaluation and evaluation.get("study_sha256") != study_hash:
        evaluation = {"status":"unverified", "reason":"模型评测与当前研究哈希未建立一致绑定。"}
    if costs and budget and costs.get("budget_snapshot_sha256") != fingerprint(budget):
        costs = {"status":"stale", "reason":"费用分析与最新账本快照不一致；请运行 budget 更新。"}
    if costs and not budget:
        costs = {"status":"unverified", "reason":"缺少对应账本快照，无法确认费用分析覆盖范围。"}
    ai_complete = bool(review and review.get("status") == "completed" and evaluation
                       and evaluation.get("status") == "completed_requested_grid")
    ai_badge = "真实 API 审查与评测已完成 · 详情见证据" if ai_complete else "可选 API 审查或评测尚未完成"
    delta = comparison["delta_sharpe"]
    if delta <= 0:
        verdict = "本次留出结果未支持改进假设"
        result_sentence = (f"逆波动率加权相对同参数等权版的净夏普下降 {_num(abs(delta))}。"
                           "同期的收益、波动及回撤见下表。")
    else:
        verdict = "本次留出期观察到正向差异"
        result_sentence = f"逆波动率加权相对同参数等权版的净夏普提高 {_num(delta)}；这一历史差异不能自动推出未来优势。"
    interval_sentence = ("95% 区间跨越 0，结果不足以确认改善。" if interval["low"] <= 0 <= interval["high"]
                         else "区间属于固定参数下的历史重采样结果，仍受选择不确定性与数据修订限制。")
    table_rows = []
    for name in names:
        m = hmetrics[name]
        cls = "focus" if name == chosen else "matched" if name == comparator else ""
        descriptor = (f"{chosen_spec['lookback']} 个月动量 · 选前 {chosen_spec['top_k']} 个行业"
                      if name in (chosen, comparator) else "统一时期的参照组合")
        table_rows.append(f'<tr class="{cls}"><td><span class="table-name">{_e(labels[name])}</span><span class="small-sub">{_e(descriptor)}</span></td>'
                          + "".join(f"<td>{v}</td>" for v in [_pct(m["cagr"]), _pct(m["ann_vol"]), _num(m["sharpe"]),
                                                             _pct(m["max_drawdown"]), _pct(m["avg_turnover"])]) + "</tr>")
    validation_rows = []
    for name, result in research["results"].items():
        if name not in {s["name"] for s in protocol["candidates"]}:
            continue
        spec, m = result["spec"], result["metrics"]["validation"]
        weighting = "等权" if spec["weighting"] == "equal" else "逆波动率"
        status = "锁定对照" if name == comparator else "配对改进版" if name == chosen else "候选"
        cls = "focus" if name == chosen else "matched" if name == comparator else ""
        validation_rows.append(f'<tr class="{cls}"><td>{spec["lookback"]} 个月 / 前 {spec["top_k"]} 行业</td><td>{weighting}</td>'
                               f'<td>{_num(m["sharpe"])}</td><td>{_pct(m["cagr"])}</td><td>{_pct(m["max_drawdown"])}</td><td>{status}</td></tr>')
    cost_rows = []
    for cost in protocol["cost_sensitivity_bps"]:
        entries = {v["name"]: v["metrics"] for v in research["sensitivity"] if v["cost_bps"] == cost}
        a, b = entries[chosen], entries[comparator]
        cost_rows.append(f'<tr><td>{_num(cost,0)} bp</td><td>{_pct(b["cagr"])}</td><td>{_pct(a["cagr"])}</td>'
                         f'<td>{_num(a["sharpe"]-b["sharpe"],3,True)}</td></tr>')
    legend = '<div class="legend">' + "".join(f'<span><i class="swatch" style="background:{colors[n]}"></i>{_e(labels[n])}</span>' for n in names) + "</div>"
    charts = ""
    for metric, title, unit in (("nav", "留出期净值", "2014-12 期末 = 1"), ("drawdown", "持有过程中的回撤", "相对各自区间内历史高点")):
        charts += f'<div class="card chart-card"><div class="chart-title"><h3>{title}</h3><span class="chart-unit">{unit}</span></div><div class="chart-scroll">{_chart(periods,curves,names,labels,colors,metric)}</div>{legend}</div>'
    manifest = research["manifest"]
    sources = ""
    for kind, source in manifest["sources"].items():
        label = "French 12 行业月度收益" if kind == "industry" else "Fama–French 三因子与 RF"
        sources += (f'<div class="record"><div>{_link(source["url"],label)}</div>'
                    f'<div class="record-label">{_e(source["first_month"])} — {_e(source["last_month"])} · {_e(source["rows"])} 个月</div>'
                    f'<div class="hash">SHA256 {_e(source["sha256"])}</div></div>')
    review_html = ('<p class="muted">真实 API 研究审查待完成。确定性结果已经生成，这不代表模型已验证其结论。</p>'
                   if review is None else '<p style="white-space:pre-line">' + _e((review.get("review") or {}).get("summary",review.get("reason","审查记录不可读")))
                   + '</p>' + _json_details("展开研究审查原始记录", review))
    evaluation_html = _evaluation_html(evaluation)
    if benchmark_notes and evaluation and benchmark_notes.get("evaluation_signature") == evaluation.get("signature"):
        evaluation_html += '<div class="note">' + _e(benchmark_notes["interpretation_zh"]) + '</div>'
    import_demo_html = ''
    if (import_demo and import_demo.get("status") == "completed"
            and import_demo.get("source_artifact_sha256") == research["artifacts"].get("momentum_9m_top6_equal_monthly.csv")):
        import_demo_html = ('<div class="card" style="margin-top:20px"><h3>把自己的回测交给它，先检查，再审查</h3>'
                            '<p class="section-desc">导入一个明确选择的月度 CSV，在本机检查月份、费用、净值和收益。'
                            '确认后，仅将指标摘要与来源声明发送给模型。以下为使用本项目公开月度数据的真实导入实测：</p>'
                            '<div class="note">' + _e(import_demo["review"]["summary"]) + '</div>'
                            '<div class="formula">python run.py review-file --csv "你的月度回测.csv" --dry-run<br>'
                            'python run.py review-file --csv "你的月度回测.csv" --question "这份证据能支持什么结论？"</div>'
                            '<p class="footnote">第一条命令完全离线；第二条会消耗 API 额度。数据格式与可选元数据见 examples/README.md。</p></div>')
    budget_html = ('<div class="note">本地调用账本快照缺失，调用次数和用量待核实；不能按 0 处理。</div>'
                   if budget is None else '<p class="muted">本地账本已生成；其中的估算、预留与实际账户对账应分别阅读。</p>' + _json_details("展开预算账本原始记录", budget))
    if costs and "request_count" in costs:
        budget_html = (f'<div class="record"><div class="record-label">公开单价 × API 返回用量估算 · {_e(costs["request_count"])} 个请求</div>'
                       f'<div class="money">${_num(costs.get("lower_bound_usd"),4)}–${_num(costs.get("upper_bound_usd"),4)}</div>'
                       '<p class="footnote">若 API 未返回缓存写入时长，按短 / 长缓存费率给出区间；这不是平台最终账单。'
                       f'本地预算记账 ${_num(costs.get("conservative_budget_usage_usd"),4)}。</p></div>'
                       + _json_details("展开用量费用分析", costs) + budget_html)
    if costs and "request_count" not in costs:
        budget_html += f'<div class="note">{_e(costs.get("reason","费用分析待核实"))}</div>'
    selected_hashes = {"protocol_sha256": research["protocol_sha256"], "selection_sha256": research["selection_sha256"],
                       "source_hashes": protocol["source_hashes"], "code_hashes": protocol["code_hashes"], "artifact_hashes": research["artifacts"]}
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Quant Evidence Lab · 可复核的 AI 量化研究</title><style>{CSS}</style></head>
<body><header><div class="brand"><span class="brand-mark" aria-hidden="true"></span>QUANT EVIDENCE LAB</div><nav aria-label="报告导航"><a href="#results">研究结果</a><a href="#method">方法</a><a href="#ai">AI 与资金</a><a href="#provenance">证据链</a></nav></header>
<main><div class="hero"><div><div class="eyebrow">Research note 001 · AI × Quantitative Finance</div><h1>让研究结论，<br>经得起复算。</h1><p class="intro">一个可审计的行业动量实验：让 AI 提出与审查研究判断，让确定性代码负责数据、回测与数值证据。</p><span class="badge green">真实公开数据 · 计算已完成</span><span class="badge amber">{_e(ai_badge)}</span><span class="badge">离线报告 · 无在线依赖</span></div><aside class="hero-note"><strong>我们检验的具体问题</strong>在验证期选定行业动量参数后，将等权改为逆波动率加权，能否提高留出期扣费后的风险调整收益？<br><br>数据：美国 12 行业研究组合<br>留出：{_e(holdout['start'])} — {_e(holdout['end'])}<br>主费用：每单位买卖名义金额 {_e(protocol['cost_bps'])} bp</aside></div>
<div class="conclusion"><span class="symbol" aria-hidden="true">↘</span><div><h2>{_e(verdict)}</h2><p>{_e(result_sentence)}{_e(interval_sentence)}</p></div></div>
<div class="kpis"><div class="kpi"><div class="label">主指标 · 净夏普差</div><div class="number negative">{_num(delta,3,True)}</div><div class="caption">逆波动率 − 同参数等权</div></div><div class="kpi"><div class="label">夏普差的 95% 区间</div><div class="number" style="font-size:23px">{_num(interval['low'],3,True)} ～ {_num(interval['high'],3,True)}</div><div class="caption">12 个月区块 · 2,000 次成对重采样</div></div><div class="kpi"><div class="label">净年化收益差</div><div class="number negative">{_num(comparison['delta_cagr']*100,2,True)}<small style="font-size:14px"> pp</small></div><div class="caption">使用几何年化收益 CAGR</div></div><div class="kpi"><div class="label">检验样本长度</div><div class="number">{_e(hmetrics[chosen]['n_months'])}<small style="font-size:14px"> 个月</small></div><div class="caption">历史留出期 · 不能等同前瞻实盘</div></div></div>
<section id="results"><div class="section-top"><div><h2>先看结果，再读解释</h2><p class="section-desc">主表仅展示锁定后的配对实验与两个基线。全部使用同一留出期与费用口径，不按留出收益重新挑选参数。</p></div><span class="section-num">01</span></div><div class="table-wrap"><table><thead><tr><th>策略</th><th>净年化收益</th><th>年化波动</th><th>净夏普</th><th>最大回撤</th><th>月均买卖换手</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div><p class="footnote">净夏普以扣除 RF 的月度超额收益计算；换手为买入与卖出名义金额之和，不是除以 2 的单边换手。市场组合是 FF 市场总收益参照，不是可直接执行的产品。</p><div style="margin-top:23px">{charts}</div><p class="footnote">图线从逐月 CSV 重新累计，并与研究记录核对最终净值及最大回撤。留出期净值重定为 1；实际持仓延续自 1970 年，边界月份保留真实调仓费用。窄屏可横向查看图表。</p></section>
<section id="method"><div class="section-top"><div><h2>把选择与检验分开</h2><p class="section-desc">先冻结协议，再选择参数，最后读取留出期。锁定的是 {_e(chosen_spec['lookback'])} 个月观察窗口、前 {_e(chosen_spec['top_k'])} 个行业；逆波动率版与等权版共用这个选择。</p></div><span class="section-num">02</span></div><div class="method-steps"><div class="method-step"><div class="date">1970 — 1999</div><strong>开发与实现验证</strong><p>构造信号与回测约定，验证时间对齐、漂移换手、费用和净值计算。</p></div><div class="method-step"><div class="date">2000 — 2014</div><strong>仅用验证期选参数</strong><p>在等权候选中按净夏普选择窗口与行业数量；再锁定其逆波动率配对版本。</p></div><div class="method-step"><div class="date">{_e(holdout['start'])} — {_e(holdout['end'])}</div><strong>固定配对后的历史检验</strong><p>比较净夏普差、收益、波动与回撤。保留未改善的结果，不用该区间继续调参。</p></div></div><div class="note">这是本地文件时间戳与哈希形成的事前约定，未经外部公开时间戳认证。历史区间也可能存在于模型训练知识中，不能称为彻底消除知识污染的盲测。</div>
<div class="two-col" style="margin-top:20px"><div class="card"><h3>月份与费用，写成可检查的约定</h3><p class="section-desc">持有月份 t 的动量使用恰好 L 个收益：t−L−1 至 t−2；最近一个月 t−1 被跳过。波动率只使用 t−12 至 t−1。</p><div class="formula">turnover[t] = Σ | target_weight[t] − drifted_weight[t−1] |<br>cost[t] = cost_bps / 10000 × turnover[t]<br>net_return[t] = gross_return[t] − cost[t]</div><p class="footnote">前期持仓先随资产收益漂移，再比较目标权重；从现金开始的首次建仓也收费。费用在月末扣除，使用明确的净收益减费约定。</p></div><div class="card"><h3>费用增加后，结论是否改变？</h3><div class="table-wrap" style="margin-top:15px"><table><thead><tr><th>成本</th><th>等权 CAGR</th><th>逆波动 CAGR</th><th>Δ 夏普</th></tr></thead><tbody>{''.join(cost_rows)}</tbody></table></div><p class="footnote">0、10、25、50 bp 为预先约定的敏感性情景，仅覆盖外层再平衡假设，不包含行业组合底层股票的全部实施成本。</p></div></div>
<details style="margin-top:23px"><summary>查看全部 12 个候选的验证期结果（2000–2014）</summary><p class="section-desc" style="margin:14px 0">选择规则只在等权候选中运行。逆波动率候选全部保留用于透明比较，不因其验证期表现而另选参数。</p><div class="table-wrap"><table><thead><tr><th>观察窗口 / 行业数</th><th>加权</th><th>验证净夏普</th><th>验证净 CAGR</th><th>验证最大回撤</th><th>用途</th></tr></thead><tbody>{''.join(validation_rows)}</tbody></table></div></details></section>
<section id="ai"><div class="section-top"><div><h2>AI 负责判断，证据约束判断</h2><p class="section-desc">实测模型调用与离线程序测试分别记录。每项数字结论应指向确定性证据；缺失、失败和不确定结果均保留。</p></div><span class="section-num">03</span></div><div class="two-col"><div class="card"><span class="badge amber">{_e(ai_badge)}</span><div class="status-title">研究审查与可靠性评估</div>{review_html}{evaluation_html}<p class="footnote">记录文件的存在本身不等于评估全部通过；请查看其中的模式、请求状态、样本数及失败案例。当前页面只呈现已有证据。</p></div><div class="card"><h3>本地 API 用量</h3>{budget_html}</div></div></section>
<section id="provenance"><div class="section-top"><div><h2>留下可以追问的证据链</h2><p class="section-desc">本页无外部脚本、字体或图表库。打开后无需联网；原始来源链接仅在你主动点击时访问。</p></div><span class="section-num">04</span></div><div class="two-col"><div class="card"><h3>公开来源与冻结版本</h3><p class="section-desc">取得时间：{_e(manifest['retrieved_at'])}。收益由百分数转换为小数，0.01 表示 1%；缺月、重复日期、缺失哨兵值和异常输入均不静默填补。</p>{sources}<p class="footnote">{_link(manifest['documentation_url'],'Kenneth French 数据构建与版本说明')}</p></div><div class="card"><h3>协议、选择与产出</h3><div class="record-label">协议冻结</div><div class="hash">{_e(protocol['frozen_at'])}</div><div class="hash">SHA256 {_e(research['protocol_sha256'])}</div><div class="record"><div class="record-label">验证选择锁定</div><div class="hash">{_e(selection['frozen_at'])}</div><div class="hash">SHA256 {_e(research['selection_sha256'])}</div></div>{_json_details('展开完整哈希清单',selected_hashes)}{_json_details('展开原始研究协议',protocol)}</div></div><div class="card" style="margin-top:20px"><h3>这些边界决定结论能走多远</h3><ol class="limitation-list"><li><strong>当前数据版本不等于历史可得数据</strong>French 历史序列会修订；本地哈希固定的是本次下载版本。2025 年起研究收益转用 CRSP CIZ 数据格式。</li><li><strong>行业组合不是可直接购买的基金</strong>实验检验研究组合上的规则。外层费用敏感性不能替代底层交易、容量、滑点和税费评估。</li><li><strong>历史留出不消除模型的历史知识</strong>代码不使用未来数值构造信号，但语言模型可能已了解历史市场事件。真正前瞻验证需要另行冻结并等待未来数据。</li><li><strong>区间与筛选都带有条件</strong>区块重采样描述当前选定参数下的不确定性，不能完全覆盖参数选择、非平稳市场和数据修订风险，也不是盈利保证。</li></ol></div></section>
{import_demo_html}
<footer class="footer"><span>Quant Evidence Lab · 研究工具与方法演示，不构成投资建议</span><span>报告生成：{_e(generated)} · 数据截至 {_e(holdout['end'])}</span></footer></main></body></html>'''
    output = root / "outputs" / "Quant-Evidence-Lab.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output
