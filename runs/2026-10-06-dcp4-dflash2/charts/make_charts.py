#!/usr/bin/env python3
"""make_charts.py DATA.json OUTDIR -- static README charts (light + dark SVG), no dependencies.
Palette: dataviz reference slots 1-2 (validated both modes). Blue = knapcio stack, orange = previous recipe.
Charts: decode (small multiples, lines by concurrency), prefill (grouped bars), suite (single-stream bars), sweep (lines C1-C6)."""
import json, sys, os, html
THEME = {
 "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
 "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926"),
}
FONT = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'
def esc(s): return html.escape(str(s))
def nice_max(v):
    """-> (axis max, tick count) on a round step, 3-6 ticks."""
    for step in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        n = -(-v // step)
        if n <= 6: return int(step * max(n, 3)), int(max(n, 3))
    return int(v), 4
def ticks(o, t, x0, x1, y0, h, ymax, n, fmt=lambda v: f"{v:,}"):
    for g in range(n + 1):
        gy = y0 + h - h * g / n
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="{t["grid"] if g else t["axis"]}" stroke-width="1"/>')
        o.append(text(x0 - 8, gy + 4, fmt(int(ymax * g / n)), t, 10, t["muted"], "end"))
def end_labels(o, t, x, items, top_y, bot_y):
    """items: [(y, label, color)] -> right-of-endpoint labels, value-ordered, >= 14 px apart, each led by a
    series-colored dot so identity never rests on position alone (text stays in ink)."""
    items = sorted(items, key=lambda it: it[0]); ys = []
    for it in items:
        y = max(it[0] + 4, (ys[-1] + 14) if ys else top_y); ys.append(min(y, bot_y))
    for (y, lab, color), ly in zip(items, ys):
        o.append(f'<circle cx="{x+3:.1f}" cy="{ly-4:.1f}" r="3.5" fill="{color}"/>')
        o.append(text(x + 10, ly, lab, t, 11, t["ink"], "start", "600"))
def svg_open(w, h, t, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">',
            f'<rect width="{w}" height="{h}" fill="{t["surface"]}"/>']
def text(x, y, s, t, size=12, fill=None, anchor="start", weight="normal"):
    return f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" fill="{fill or t["ink2"]}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'
def legend(x, y, t, items):
    out = []; cx = x
    for label, color in items:
        out.append(f'<rect x="{cx}" y="{y-9}" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(text(cx + 17, y + 1, label, t, 12, t["ink2"])); cx += 17 + 7.2 * len(label) + 22
    return out

def decode_multiples(d, t):
    types = d["types"]; levels = d["levels"]; W, H = 900, 356
    pw, ph, top, left, gap = 216, 200, 78, 52, 80
    o = svg_open(W, H, t, d["title"])
    o.append(text(24, 30, d["title"], t, 16, t["ink"], weight="600"))
    o += legend(24, 56, t, [(d["new_label"], t["s1"]), (d["old_label"], t["s2"])])
    for i, ty in enumerate(types):
        x0 = left + i * (pw + gap); y0 = top + 18
        new = d["new"][ty]; old = d["old"][ty]
        ymax, nt = nice_max(max(new + old) * 1.05)
        o.append(text(x0, top + 6, ty, t, 13, t["ink"], weight="600"))
        ticks(o, t, x0, x0 + pw, y0, ph, ymax, nt)
        xs = [x0 + pw * j / (len(levels) - 1) for j in range(len(levels))]
        for j, lv in enumerate(levels):
            o.append(text(xs[j], y0 + ph + 16, f"c{lv}", t, 10, t["muted"], "middle"))
        for series, color in ((old, t["s2"]), (new, t["s1"])):
            pts = [(xs[j], y0 + ph - ph * v / ymax) for j, v in enumerate(series)]
            o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for x, y in pts:
                o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        end_labels(o, t, xs[-1] + 9, [(y0 + ph - ph * s_[-1] / ymax, f"{s_[-1]:.0f}", c_) for s_, c_ in ((new, t["s1"]), (old, t["s2"]))], y0, y0 + ph)
        o.append(text(x0 + pw / 2, y0 + ph + 32, "concurrent streams", t, 10, t["muted"], "middle"))
    o.append(text(24, H - 8, d["note"], t, 10, t["muted"]))
    o.append("</svg>"); return "\n".join(o)

def grouped_bars(d, t):
    cats = d["cats"]; W = 900; H = 318; top, left, ch, cw = 78, 64, 170, 800
    o = svg_open(W, H, t, d["title"])
    o.append(text(24, 30, d["title"], t, 16, t["ink"], weight="600"))
    o += legend(24, 56, t, [(d["new_label"], t["s1"]), (d["old_label"], t["s2"])])
    ymax, nt = nice_max(max(d["new"] + d["old"]) * 1.12); y0 = top + 12
    ticks(o, t, left, left + cw, y0, ch, ymax, nt)
    slot = cw / len(cats); bw = min(56, slot / 3.2)
    for i, c in enumerate(cats):
        cx = left + slot * (i + 0.5)
        for k, (v, color) in enumerate(((d["old"][i], t["s2"]), (d["new"][i], t["s1"]))):
            x = cx - bw - 1 + k * (bw + 2); h = ch * v / ymax; y = y0 + ch - h
            o.append(f'<path d="M{x:.1f},{y0+ch:.1f} V{y+4:.1f} Q{x:.1f},{y:.1f} {x+4:.1f},{y:.1f} H{x+bw-4:.1f} Q{x+bw:.1f},{y:.1f} {x+bw:.1f},{y+4:.1f} V{y0+ch:.1f} Z" fill="{color}"/>')
            o.append(text(x + bw / 2, y - 6, f"{v:,.0f}", t, 11, t["ink"], "middle", "600" if k else "normal"))
        o.append(text(cx, y0 + ch + 18, c, t, 11, t["ink2"], "middle"))
        if d.get("delta"): o.append(text(cx, y0 + ch + 34, d["delta"][i], t, 11, t["ink"], "middle", "600"))
    o.append(text(24, H - 8, d["note"], t, 10, t["muted"]))
    o.append("</svg>"); return "\n".join(o)

def hbars(d, t):
    cats = d["cats"]; vals = d["vals"]; W = 900; rowh = 26; top = 64; left = 150; cw = 660
    H = top + rowh * len(cats) + 52
    o = svg_open(W, H, t, d["title"])
    o.append(text(24, 30, d["title"], t, 16, t["ink"], weight="600"))
    o.append(text(24, 50, d["sub"], t, 12, t["ink2"]))
    xmax, nt = nice_max(max(vals) * 1.1)
    for g in range(nt + 1):
        gx = left + cw * g / nt
        o.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top}" y2="{top+rowh*len(cats)}" stroke="{t["grid"] if g else t["axis"]}" stroke-width="1"/>')
        o.append(text(gx, top + rowh * len(cats) + 16, int(xmax * g / nt), t, 10, t["muted"], "middle"))
    for i, (c, v) in enumerate(zip(cats, vals)):
        y = top + rowh * i + 5; bh = rowh - 10; w = cw * v / xmax
        o.append(text(left - 10, y + bh - 3, c, t, 12, t["ink2"], "end"))
        o.append(f'<path d="M{left},{y:.1f} H{left+w-4:.1f} Q{left+w:.1f},{y:.1f} {left+w:.1f},{y+4:.1f} V{y+bh-4:.1f} Q{left+w:.1f},{y+bh:.1f} {left+w-4:.1f},{y+bh:.1f} H{left} Z" fill="{t["s1"]}"/>')
        o.append(text(left + w + 6, y + bh - 3, f"{v:.1f}", t, 11, t["ink"], "start", "600"))
    o.append(text(24, H - 8, d["note"], t, 10, t["muted"]))
    o.append("</svg>"); return "\n".join(o)

def lines(d, t):
    xs_l = d["x"]; W, H = 900, 330; top, left, cw, ch = 78, 64, 760, 180
    o = svg_open(W, H, t, d["title"])
    o.append(text(24, 30, d["title"], t, 16, t["ink"], weight="600"))
    if len(d["series"]) > 1: o += legend(24, 56, t, [(lab, t["s1" if i == 0 else "s2"]) for i, (lab, _) in enumerate(d["series"])])
    else: o.append(text(24, 52, d.get("sub", ""), t, 12, t["ink2"]))
    ymax, nt = nice_max(max(max(v) for _, v in d["series"]) * 1.08); y0 = top + 12
    ticks(o, t, left, left + cw, y0, ch, ymax, nt)
    xs = [left + cw * j / (len(xs_l) - 1) for j in range(len(xs_l))]
    for j, lv in enumerate(xs_l): o.append(text(xs[j], y0 + ch + 18, lv, t, 11, t["muted"], "middle"))
    for i, (lab, v) in enumerate(d["series"]):
        color = t["s1" if i == 0 else "s2"]
        pts = [(xs[j], y0 + ch - ch * val / ymax) for j, val in enumerate(v)]
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts: o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        pass
    end_labels(o, t, xs[-1] + 9, [(y0 + ch - ch * v[-1] / ymax, f"{v[-1]:.0f}", t["s1" if i == 0 else "s2"]) for i, (_, v) in enumerate(d["series"])], y0, y0 + ch)
    o.append(text(24, H - 8, d["note"], t, 10, t["muted"]))
    o.append("</svg>"); return "\n".join(o)


def hbars2(d, t):
    """Per-category horizontal bars, two series (s1 = first, s2 = second), value labels on each bar."""
    cats = d["cats"]; ser = d["series"]; W = 900; rowh = 40; top = 84; left = 150; cw = 640
    H = top + rowh * len(cats) + 52
    o = svg_open(W, H, t, d["title"])
    o.append(text(24, 30, d["title"], t, 16, t["ink"], weight="600"))
    o.append(text(24, 50, d["sub"], t, 12, t["ink2"]))
    o += legend(24, 70, t, [(lab, t["s1" if i == 0 else "s2"]) for i, (lab, _) in enumerate(ser)])
    xmax, nt = nice_max(max(max(v for v in vals if v is not None) for _, vals in ser) * 1.12)
    for g in range(nt + 1):
        gx = left + cw * g / nt
        o.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top}" y2="{top+rowh*len(cats)}" stroke="{t["grid"] if g else t["axis"]}" stroke-width="1"/>')
        o.append(text(gx, top + rowh * len(cats) + 16, int(xmax * g / nt), t, 10, t["muted"], "middle"))
    bh = (rowh - 12) / 2
    for i, c in enumerate(cats):
        y0 = top + rowh * i + 5
        o.append(text(left - 10, y0 + bh + 4, c, t, 12, t["ink2"], "end"))
        for k, (lab, vals) in enumerate(ser):
            v = vals[i]
            if v is None: continue
            y = y0 + k * (bh + 2); w = max(cw * v / xmax, 2); col = t["s1" if k == 0 else "s2"]
            o.append(f'<path d="M{left},{y:.1f} H{left+w-3:.1f} Q{left+w:.1f},{y:.1f} {left+w:.1f},{y+3:.1f} V{y+bh-3:.1f} Q{left+w:.1f},{y+bh:.1f} {left+w-3:.1f},{y+bh:.1f} H{left} Z" fill="{col}"/>')
            o.append(text(left + w + 6, y + bh - 2, f"{v:.1f}", t, 10, t["ink"], "start", "600"))
    o.append(text(24, H - 8, d["note"], t, 10, t["muted"]))
    o.append("</svg>"); return "\n".join(o)

KINDS = {"decode": decode_multiples, "prefill": grouped_bars, "suite": hbars, "suite2": hbars2, "sweep": lines}
if __name__ == "__main__":
    data = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    for name, d in data.items():
        for mode in ("light", "dark"):
            open(os.path.join(out, f"{name}-{mode}.svg"), "w").write(KINDS[d["kind"]](d, THEME[mode]))
            print("wrote", f"{name}-{mode}.svg")
