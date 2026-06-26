"""Minimal Markdown -> PDF for the company hand-off docs (headings, bullets, tables, bold).
Pure-python (fpdf2), no system libraries.  Usage: python md2pdf.py in.md out.pdf [Title]
"""
import sys, re
from fpdf import FPDF

INK = (30, 30, 30); GREY = (110, 110, 110); RULE = (200, 200, 200)
ACC = (20, 60, 110)

_MAP = {"—": "-", "–": "-", "•": "-", "≈": "~", "≠": "!=", "×": "x", "∪": "U",
        "→": "->", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "²": "2", "°": "deg"}
def _ascii(s):
    for a, b in _MAP.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def _cell_md(pdf, text, size=10.5, style="", color=INK, h=5.6):
    """Write a line honouring **bold** spans."""
    pdf.set_font("Helvetica", style, size); pdf.set_text_color(*color)
    parts = re.split(r"(\*\*.+?\*\*)", _ascii(text))
    for p in parts:
        if p.startswith("**") and p.endswith("**"):
            pdf.set_font("Helvetica", (style + "B").replace("BB", "B"), size)
            pdf.write(h, p[2:-2])
            pdf.set_font("Helvetica", style, size)
        else:
            pdf.write(h, p)
    pdf.ln(h)


def render(md_path, pdf_path, title=None):
    lines = open(md_path).read().splitlines()
    pdf = FPDF(format="A4"); pdf.set_auto_page_break(True, margin=16)
    pdf.add_page(); pdf.set_margins(18, 16, 18)
    W = pdf.w - 36

    i = 0
    while i < len(lines):
        ln = _ascii(lines[i].rstrip())
        if not ln.strip():
            pdf.ln(2.5); i += 1; continue

        # table block
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [_ascii(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[-:\s|]+$", lines[i].strip().strip("|")):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = len(rows[0]); cw = W / ncol
                for ri, r in enumerate(rows):
                    pdf.set_font("Helvetica", "B" if ri == 0 else "", 9.5)
                    pdf.set_text_color(*(ACC if ri == 0 else INK))
                    y0 = pdf.get_y()
                    heights = []
                    for c in r:
                        heights.append(len(pdf.multi_cell(cw, 5, re.sub(r"\*\*", "", c),
                                          border=0, split_only=True)))
                    rh = max(heights) * 5
                    x0 = pdf.get_x()
                    for c in r:
                        x = pdf.get_x()
                        pdf.multi_cell(cw, 5, re.sub(r"\*\*", "", c), border=0, align="L",
                                       max_line_height=5)
                        pdf.set_xy(x + cw, y0)
                    pdf.set_xy(x0, y0 + rh)
                    pdf.set_draw_color(*RULE); pdf.line(18, pdf.get_y(), 18 + W, pdf.get_y())
                pdf.ln(3)
            continue

        if ln.startswith("# "):
            pdf.set_font("Helvetica", "B", 20); pdf.set_text_color(*INK)
            pdf.multi_cell(W, 9, ln[2:]); pdf.ln(1)
            pdf.set_draw_color(*ACC); pdf.set_line_width(0.6)
            pdf.line(18, pdf.get_y(), 18 + W, pdf.get_y()); pdf.set_line_width(0.2); pdf.ln(3)
        elif ln.startswith("## "):
            pdf.ln(1); pdf.set_font("Helvetica", "B", 13.5); pdf.set_text_color(*ACC)
            pdf.multi_cell(W, 7, ln[3:]); pdf.ln(1)
        elif ln.startswith("- ") or ln.startswith("* "):
            pdf.set_x(22)
            pdf.set_font("Helvetica", "", 10.5); pdf.set_text_color(*INK)
            pdf.write(5.4, "-  ")
            _cell_md(pdf, ln[2:])
        elif re.match(r"^\d+\.\s", ln):
            pdf.set_x(22); _cell_md(pdf, ln)
        else:
            _cell_md(pdf, ln)
        i += 1

    pdf.output(pdf_path)
    print("wrote", pdf_path)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
