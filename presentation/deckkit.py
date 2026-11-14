"""Small helper over python-pptx: every element is also recorded so we can
(1) draw an approximate PNG preview with PIL and (2) check text overflow,
because LibreOffice is not available in this environment."""
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from PIL import Image, ImageDraw, ImageFont

FONT = "Malgun Gothic"
PREVIEW_FONT = {False: "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                True: "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"}
W, H = 13.333, 7.5
LINE = 1.2  # line height factor used for both pptx spacing and the fit check


def rgb(h):
    return RGBColor.from_string(h)


def _set_font(run, size, bold, color):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = rgb(color)
    f.name = FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def norm_paras(text, size, color, bold, align):
    """text: str (\\n = new paragraph) | list of paragraphs; a paragraph is a str,
    or dict(runs=[(text, {size,bold,color})...] or text=..., size, color, bold, align, after)."""
    if isinstance(text, str):
        text = text.split("\n")
    out = []
    for p in text:
        if isinstance(p, str):
            p = {"text": p}
        runs = p.get("runs") or [(p.get("text", ""), {})]
        psize = p.get("size", size)
        rr = []
        for t, o in runs:
            rr.append((t, o.get("size", psize), o.get("bold", p.get("bold", bold)), o.get("color", p.get("color", color))))
        out.append({"runs": rr, "align": p.get("align", align), "after": p.get("after", 0),
                    "bullet": p.get("bullet", False)})
    return out


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(W)
        self.prs.slide_height = Inches(H)
        self.blank = self.prs.slide_layouts[6]
        self.pages = []  # (slide, [elements], bg)

    def slide(self, bg="FFFFFF", notes=""):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = rgb(bg)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        self.pages.append((s, [], bg))
        return len(self.pages) - 1

    def box(self, i, x, y, w, h, text=None, fill=None, line=None, shape="rect", radius=0.12,
            size=16, color="1B2B34", bold=False, align="l", anchor="t", margin=0.1, name=None, line_w=1.0):
        s, els, _ = self.pages[i]
        kind = {"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE, "oval": MSO_SHAPE.OVAL,
                "chev": MSO_SHAPE.CHEVRON, "arrow": MSO_SHAPE.RIGHT_ARROW}[shape]
        if fill is None and line is None and shape == "rect":
            sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        else:
            sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
            if shape == "round":
                sh.adjustments[0] = min(0.5, radius / min(w, h))
            if fill:
                sh.fill.solid()
                sh.fill.fore_color.rgb = rgb(fill)
            else:
                sh.fill.background()
            if line:
                sh.line.color.rgb = rgb(line)
                sh.line.width = Pt(line_w)
            else:
                sh.line.fill.background()
            sh.shadow.inherit = False
        if name:
            sh.name = name
        paras = None
        if text is not None:
            tf = sh.text_frame
            tf.word_wrap = True
            m = Inches(margin)
            tf.margin_left = tf.margin_right = m
            tf.margin_top = tf.margin_bottom = Inches(margin * 0.6)
            tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
            paras = norm_paras(text, size, color, bold, align)
            for k, p in enumerate(paras):
                para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p["align"]]
                para.line_spacing = LINE
                para.space_after = Pt(p["after"])
                if p["bullet"]:
                    pPr = para._p.get_or_add_pPr()
                    pPr.set("marL", str(Emu(Inches(0.25))))
                    pPr.set("indent", str(-Emu(Inches(0.22))))
                    bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
                    pPr.append(bu)
                for t, sz, b, c in p["runs"]:
                    r = para.add_run()
                    r.text = t
                    _set_font(r, sz, b, c)
        els.append(dict(kind=shape, x=x, y=y, w=w, h=h, fill=fill, line=line, paras=paras,
                        anchor=anchor, margin=margin, name=name or ""))
        return sh

    def line_chart(self, i, x, y, w, h, cats, vals, color, label_color="1B2B34"):
        s, els, _ = self.pages[i]
        cd = CategoryChartData()
        cd.categories = cats
        cd.add_series("실제 점수", vals)
        gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
        ch = gf.chart
        ch.has_legend = False
        ch.font.size = Pt(12)
        ch.font.name = FONT
        ch.font.color.rgb = rgb(label_color)
        ser = ch.plots[0].series[0]
        ser.format.line.color.rgb = rgb(color)
        ser.format.line.width = Pt(3)
        ser.smooth = False
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(color)
        ser.marker.format.line.color.rgb = rgb(color)
        pl = ch.plots[0]
        pl.has_data_labels = True
        dl = pl.data_labels
        dl.number_format = "0.0"
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.ABOVE
        dl.font.size = Pt(12)
        dl.font.bold = True
        va = ch.value_axis
        va.minimum_scale, va.maximum_scale = 75, 95
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = rgb("E3E8EA")
        va.format.line.fill.background()
        va.tick_labels.font.size = Pt(11)
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(11)
        ca.format.line.color.rgb = rgb("B8C2C6")
        els.append(dict(kind="chart", x=x, y=y, w=w, h=h, cats=cats, vals=vals, color=color, paras=None))

    def save(self, path):
        self.prs.save(path)

    # ---------- checks and preview ----------
    @staticmethod
    def _wrap(text, font, width_px):
        lines, cur = [], ""
        for word in text.split(" "):
            cand = (cur + " " + word) if cur else word
            if font.getlength(cand) <= width_px:
                cur = cand
                continue
            if cur:
                lines.append(cur)
            cur = ""
            for ch in word:  # very long word: break by character
                if font.getlength(cur + ch) <= width_px:
                    cur += ch
                else:
                    lines.append(cur)
                    cur = ch
        lines.append(cur)
        return lines

    def _layout(self, el, scale):
        """Returns [(line_text_runs, size, bold, color, align, y_offset)] and total height (pt)."""
        out, ypt = [], 0.0
        inner_w = (el["w"] - 2 * el["margin"]) * 72
        for p in el["paras"]:
            # treat paragraph as one style per run sequence; wrap on concatenated text with the first run's font
            sz = max(r[1] for r in p["runs"])
            bold = p["runs"][0][2]
            font = ImageFont.truetype(PREVIEW_FONT[bold], int(round(sz * 1.06)))  # +6% ~ Malgun is wider
            text = "".join(r[0] for r in p["runs"])
            w = inner_w - (0.25 * 72 if p["bullet"] else 0)
            lines = self._wrap(text, font, w) if text else [""]
            for ln in lines:
                out.append((ln, sz, bold, p["runs"][0][3], p["align"], ypt, p["bullet"] and ln is lines[0]))
                ypt += sz * LINE
            ypt += p["after"]
        return out, ypt

    def check(self):
        problems = []
        for n, (_, els, _) in enumerate(self.pages, 1):
            for el in els:
                if el["x"] < 0 or el["y"] < 0 or el["x"] + el["w"] > W + 1e-6 or el["y"] + el["h"] > H + 1e-6:
                    problems.append(f"slide {n}: '{el['name']}' off slide")
                if not el["paras"]:
                    continue
                _, hpt = self._layout(el, 1)
                avail = (el["h"] - 1.2 * el["margin"]) * 72
                if hpt > avail * 0.97:
                    first = "".join(r[0] for r in el["paras"][0]["runs"])[:30]
                    problems.append(f"slide {n}: overflow {hpt:.0f}pt > {avail:.0f}pt in '{el['name'] or first}'")
        return problems

    def preview(self, prefix, dpi=90):
        paths = []
        k = dpi / 72
        for n, (_, els, bg) in enumerate(self.pages, 1):
            img = Image.new("RGB", (int(W * dpi), int(H * dpi)), "#" + bg)
            d = ImageDraw.Draw(img)
            for el in els:
                x0, y0 = el["x"] * dpi, el["y"] * dpi
                x1, y1 = x0 + el["w"] * dpi, y0 + el["h"] * dpi
                fill = ("#" + el["fill"]) if el.get("fill") else None
                outline = ("#" + el["line"]) if el.get("line") else None
                if el["kind"] == "oval":
                    d.ellipse([x0, y0, x1, y1], fill=fill, outline=outline)
                elif el["kind"] == "round":
                    d.rounded_rectangle([x0, y0, x1, y1], radius=int(0.12 * dpi), fill=fill, outline=outline)
                elif el["kind"] in ("chev", "arrow"):
                    d.polygon([(x0, y0), (x1 - (y1 - y0) / 2, y0), (x1, (y0 + y1) / 2), (x1 - (y1 - y0) / 2, y1), (x0, y1)],
                              fill=fill, outline=outline)
                elif el["kind"] == "chart":
                    vals, cats = el["vals"], el["cats"]
                    lo, hi = 75, 95
                    pts = []
                    for j, v in enumerate(vals):
                        px = x0 + 40 + j * (x1 - x0 - 60) / max(1, len(vals) - 1)
                        py = y1 - 40 - (v - lo) / (hi - lo) * (y1 - y0 - 70)
                        pts.append((px, py))
                    d.line(pts, fill="#" + el["color"], width=4)
                    f = ImageFont.truetype(PREVIEW_FONT[True], int(12 * k))
                    fs = ImageFont.truetype(PREVIEW_FONT[False], int(10 * k))
                    for (px, py), v, c in zip(pts, vals, cats):
                        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill="#" + el["color"])
                        d.text((px - 14, py - 24), f"{v:.1f}", font=f, fill="#1B2B34")
                        d.text((px - 18, y1 - 30), c, font=fs, fill="#5B6B73")
                    continue
                elif fill or outline:
                    d.rectangle([x0, y0, x1, y1], fill=fill, outline=outline)
                if el["paras"]:
                    lines, hpt = self._layout(el, k)
                    top = y0 + el["margin"] * 0.6 * dpi
                    avail = (el["h"] - 1.2 * el["margin"]) * dpi
                    if el["anchor"] == "m":
                        top += max(0, (avail - hpt * k) / 2)
                    elif el["anchor"] == "b":
                        top += max(0, avail - hpt * k)
                    for ln, sz, bold, color, align, yoff, bul in lines:
                        font = ImageFont.truetype(PREVIEW_FONT[bold], int(round(sz * k)))
                        lx = x0 + el["margin"] * dpi
                        iw = (el["w"] - 2 * el["margin"]) * dpi
                        tw = font.getlength(ln)
                        if align == "c":
                            lx += (iw - tw) / 2
                        elif align == "r":
                            lx += iw - tw
                        d.text((lx, top + yoff * k), ln, font=font, fill="#" + color)
            p = f"{prefix}-{n:02d}.png"
            img.save(p)
            paths.append(p)
        return paths
