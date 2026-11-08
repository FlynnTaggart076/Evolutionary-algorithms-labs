"""Build restrained academic PDF reports from the laboratory Markdown sources.

This module is a small, explicit Markdown reader, not a browser or a TeX engine.
It supports the constructs used in the supplied reports and fails on missing
images rather than silently omitting part of a submitted report.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2 * cm
TEXT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def register_fonts(font_dir: Path | None = None) -> None:
    """Embed the Windows Times New Roman family, including Cyrillic glyphs."""
    directory = font_dir or Path("C:/Windows/Fonts")
    for name, filename in (
        ("Academic", "times.ttf"),
        ("Academic-Bold", "timesbd.ttf"),
        ("Academic-Italic", "timesi.ttf"),
        ("Academic-BoldItalic", "timesbi.ttf"),
    ):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Times New Roman font not found: {path}. "
                "Pass --font-dir with the four times*.ttf files."
            )
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "Academic", normal="Academic", bold="Academic-Bold",
        italic="Academic-Italic", boldItalic="Academic-BoldItalic",
    )
    import matplotlib
    symbols = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "STIXGeneral.ttf"
    pdfmetrics.registerFont(TTFont("Academic-Math", str(symbols)))


def escaped_text(value: str) -> str:
    """Use the matching serif math font for symbols absent in Windows Times."""
    base = pdfmetrics.getFont("Academic").face.charToGlyph
    math = pdfmetrics.getFont("Academic-Math").face.charToGlyph
    parts = []
    for char in value:
        escaped = escape(char)
        if ord(char) > 32 and ord(char) not in base:
            if ord(char) not in math:
                raise ValueError(f"No report font has glyph U+{ord(char):04X}")
            escaped = f'<font name="Academic-Math">{escaped}</font>'
        parts.append(escaped)
    return "".join(parts)


def readable_math(value: str) -> str:
    """Keep inline mathematical expressions legible without a TeX dependency."""
    value = value.strip().replace(r"\left", "").replace(r"\right", "")
    commands = {
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
        "epsilon": "ε", "sigma": "σ", "lambda": "λ", "mu": "μ",
        "pi": "π", "rho": "ρ", "theta": "θ", "eta": "η", "Delta": "Δ",
        "Sigma": "Σ", "Omega": "Ω", "times": "×", "cdot": "·",
        "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
        "infty": "∞", "in": "∈", "notin": "∉", "approx": "≈",
        "forall": "∀", "exists": "∃", "to": "→", "rightarrow": "→",
        "sum": "Σ", "prod": "Π", "min": "min", "max": "max",
        "exp": "exp", "log": "log", "ln": "ln", "sin": "sin",
        "cos": "cos", "sqrt": "sqrt", "bmod": "mod", "mod": "mod",
    }
    value = re.sub(
        r"\\([A-Za-z]+)", lambda match: commands.get(match[1], match[1]), value
    )
    # Simple fraction/formatting constructs are unambiguous in short inline math.
    value = re.sub(r"frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
    value = re.sub(r"(?:mathbf|mathrm|text|operatorname)\{([^{}]+)\}", r"\1", value)
    value = value.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    return value.replace("{", "(").replace("}", ")").replace("\\", "")


def inline(value: str) -> str:
    """Translate a bounded inline Markdown subset into ReportLab XML."""
    pieces: list[str] = []
    token = re.compile(
        r"(`[^`\n]+`|\*\*.+?\*\*|__.+?__|\[[^\]]+\]\([^)]+\)|"
        r"\$[^$\n]+\$|\\\([^\n]+?\\\)|(?<![\w*])\*(?=\S)[^*\n]+(?<=\S)\*(?![\w*]))"
    )
    cursor = 0
    for match in token.finditer(value):
        pieces.append(escaped_text(value[cursor:match.start()]))
        item = match[0]
        if item.startswith("`"):
            # The embedded serif face includes Cyrillic and mathematical symbols.
            pieces.append(f'<font name="Academic">{escaped_text(item[1:-1])}</font>')
        elif item.startswith(("**", "__")):
            pieces.append(f"<b>{inline(item[2:-2])}</b>")
        elif item.startswith("["):
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", item)
            assert link is not None
            label, target = link.groups()
            if target.startswith(("https://", "http://")):
                pieces.append(f'<link href="{escape(target, {chr(34): "&quot;"})}">{escaped_text(label)}</link>')
            else:
                pieces.append(escaped_text(label))
        elif item.startswith("$"):
            pieces.append(escaped_text(readable_math(item[1:-1])))
        elif item.startswith(r"\("):
            pieces.append(escaped_text(readable_math(item[2:-2])))
        else:
            pieces.append(f"<i>{inline(item[1:-1])}</i>")
        cursor = match.end()
    pieces.append(escaped_text(value[cursor:]))
    return "".join(pieces)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    normal = ParagraphStyle(
        "AcademicBody", parent=base["BodyText"], fontName="Academic",
        fontSize=11.5, leading=14.5, alignment=TA_JUSTIFY,
        spaceBefore=0, spaceAfter=5, firstLineIndent=0,
        allowWidows=0, allowOrphans=0, splitLongWords=True,
    )
    styles = {"body": normal}
    for level, size in ((1, 16), (2, 13), (3, 12), (4, 11.5)):
        styles[f"h{level}"] = ParagraphStyle(
            f"AcademicH{level}", parent=normal, fontName="Academic-Bold",
            fontSize=size, leading=size * 1.2,
            alignment=TA_CENTER if level == 1 else TA_LEFT,
            spaceBefore=12 if level > 1 else 0,
            spaceAfter=7 if level > 1 else 13, keepWithNext=True,
        )
    styles["table"] = ParagraphStyle(
        "AcademicCell", parent=normal, fontSize=9.5, leading=12,
        alignment=TA_LEFT, spaceAfter=0, allowWidows=1, allowOrphans=1,
    )
    styles["table_header"] = ParagraphStyle(
        "AcademicHeaderCell", parent=styles["table"], fontName="Academic-Bold",
    )
    styles["list"] = ParagraphStyle(
        "AcademicList", parent=normal, alignment=TA_LEFT,
        leftIndent=13, firstLineIndent=-11, spaceAfter=3,
    )
    styles["caption"] = ParagraphStyle(
        "AcademicCaption", parent=normal, fontSize=10.5, leading=13,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=8,
    )
    styles["code"] = ParagraphStyle(
        "AcademicCode", parent=normal, fontSize=9.5, leading=12,
        alignment=TA_LEFT, leftIndent=8, rightIndent=8,
        spaceBefore=4, spaceAfter=6, allowWidows=1, allowOrphans=1,
    )
    return styles


def split_table_row(line: str) -> list[str]:
    """Split unescaped table separators, including the optional edge pipes."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith(r"\|"):
        line = line[:-1]
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


def is_table_rule(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def make_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> LongTable:
    rows = [split_table_row(line) for line in lines if not is_table_rule(line)]
    columns = max(len(row) for row in rows)
    if columns > 9:
        raise ValueError("A4 report table has more than 9 columns; split it in the source.")
    for row in rows:
        row.extend([""] * (columns - len(row)))
    # Moderate width weighting avoids a single long cell starving every other column.
    weights = []
    for column in range(columns):
        lengths = [len(re.sub(r"[*`_]", "", row[column])) for row in rows]
        weights.append(max(4, min(max(lengths), 75)) ** 0.55)
    minimums = []
    for column in range(columns):
        numeric = [
            pdfmetrics.stringWidth(row[column], "Academic", 9.5) + 11
            for row in rows[1:]
            if re.fullmatch(r"[-+0-9.eE%/]+", row[column])
        ]
        minimums.append(max([30.0, *numeric]))
    if sum(minimums) < TEXT_WIDTH:
        remaining = TEXT_WIDTH - sum(minimums)
        widths = [minimum + remaining * weight / sum(weights)
                  for minimum, weight in zip(minimums, weights)]
    else:
        widths = [TEXT_WIDTH * weight / sum(weights) for weight in weights]
    cells = [
        [Paragraph(inline(cell), styles["table_header" if i == 0 else "table"])
         for cell in row]
        for i, row in enumerate(rows)
    ]
    table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
        ("LINEBELOW", (0, 1), (-1, -2), 0.2, colors.Color(.75, .75, .75)),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    table.spaceBefore = 3
    table.spaceAfter = 8
    return table


def bounded_image(
    source: Path | BytesIO, max_height: float = 245,
    *, dpi: int = 72, allow_upscale: bool = True,
) -> Image:
    image = Image(str(source) if isinstance(source, Path) else source)
    width = image.imageWidth * 72 / dpi
    height = image.imageHeight * 72 / dpi
    ratio = min(TEXT_WIDTH / width, max_height / height)
    if not allow_upscale:
        ratio = min(1.0, ratio)
    image.drawWidth = width * ratio
    image.drawHeight = height * ratio
    image.hAlign = "CENTER"
    image.spaceBefore = 3
    image.spaceAfter = 3
    return image


def equation(value: str, styles: dict[str, ParagraphStyle]):
    """Render supported display TeX, with a wrapped readable fallback."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import mathtext, rc_context
        stream = BytesIO()
        with rc_context({"mathtext.fontset": "stix", "font.family": "STIXGeneral"}):
            mathtext.math_to_image(
                "$" + " ".join(value.splitlines()).strip() + "$",
                stream, dpi=180, format="png", color="black",
            )
        stream.seek(0)
        return bounded_image(stream, max_height=90, dpi=180, allow_upscale=False)
    except (ValueError, RuntimeError, ImportError):
        return Paragraph(escape(readable_math(value)), styles["code"])


def markdown_story(source: Path, styles: dict[str, ParagraphStyle]) -> list:
    text = source.read_text(encoding="utf-8-sig")
    text = text.translate(str.maketrans({"\u2011": "-", "\u2013": "-", "\u2014": "-"}))
    lines = text.splitlines()
    story: list = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            code: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError(f"Unclosed code fence in {source}")
            if stripped[3:].strip().lower() == "mermaid":
                raise ValueError(
                    f"Mermaid needs a rendered PNG before PDF export: {source}. "
                    "Use ![caption](relative/path.png)."
                )
            for line in code:
                # Paragraph wrapping also prevents long command lines being clipped.
                visible = escape(line.expandtabs(4)).replace(" ", "&#160;")
                story.append(Paragraph(visible or "&#160;", styles["code"]))
            i += 1
            continue
        if stripped.startswith("$$") or stripped.startswith(r"\["):
            delimiter = "$$" if stripped.startswith("$$") else r"\]"
            opening = "$$" if delimiter == "$$" else r"\["
            remaining = stripped[len(opening):]
            math_lines: list[str] = []
            if remaining.endswith(delimiter) and remaining:
                math_lines.append(remaining[:-len(delimiter)])
            else:
                if remaining:
                    math_lines.append(remaining)
                i += 1
                while i < len(lines) and delimiter not in lines[i]:
                    math_lines.append(lines[i])
                    i += 1
                if i == len(lines):
                    raise ValueError(f"Unclosed mathematical display in {source}")
                math_lines.append(lines[i].split(delimiter, 1)[0])
            story.append(equation("\n".join(math_lines), styles))
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", stripped)
        if heading:
            story.append(Paragraph(inline(heading[2]), styles[f"h{min(len(heading[1]), 4)}"]))
            i += 1
            continue
        picture = re.fullmatch(r"!\[([^\]]*)\]\((.+?)\)", stripped)
        if picture:
            caption, target = picture.groups()
            target = target.strip().strip("<>")
            path = (source.parent / target).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image from {source} does not exist: {path}")
            graphic = [bounded_image(path)]
            if caption:
                graphic.append(Paragraph(inline(caption), styles["caption"]))
            if story and isinstance(story[-1], Paragraph) and getattr(story[-1].style, "keepWithNext", False):
                graphic.insert(0, story.pop())
            story.append(KeepTogether(graphic))
            i += 1
            continue
        if "|" in stripped and i + 1 < len(lines) and is_table_rule(lines[i + 1]):
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            story.append(make_table(table_lines, styles))
            continue
        item = re.match(r"^([-+*]|\d+[.)])\s+(.+)$", stripped)
        if item:
            marker = "•" if item[1] in "-+*" else item[1]
            story.append(Paragraph(escape(marker) + " " + inline(item[2]), styles["list"]))
            i += 1
            continue
        if re.fullmatch(r"[-*_]{3,}", stripped):
            story.append(Spacer(1, 6))
            i += 1
            continue
        paragraph = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip():
            next_line = lines[i].strip()
            if (re.match(r"^(#{1,6}\s|[-+*]\s|\d+[.)]\s|!\[|```|~~~|\$\$|\\\[)", next_line)
                    or ("|" in next_line and i + 1 < len(lines) and is_table_rule(lines[i + 1]))):
                break
            paragraph.append(next_line)
            i += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
    return story


def page_frame(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Academic", 9)
    canvas.drawString(MARGIN, PAGE_HEIGHT - 1.3 * cm, document.subject)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 1.1 * cm, str(document.page))
    canvas.restoreState()


def build_report(source: Path) -> Path:
    output = source.with_suffix(".pdf")
    first_heading = next(
        (line.lstrip("# ").strip() for line in source.read_text(encoding="utf-8-sig").splitlines()
         if line.startswith("# ")), source.parent.name,
    )
    lab = re.search(r"(\d+)", source.parent.name)
    subject = "Эволюционное программирование и генетические алгоритмы"
    if lab:
        subject = f"Лабораторная работа № {lab[1]}"
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=first_heading, author="", subject=subject,
        keywords="source-sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
        pageCompression=1, allowSplitting=1,
    )
    story = markdown_story(source, make_styles())
    if not story:
        raise ValueError(f"The report is empty: {source}")
    document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    return output


def render_report(pdf: Path, qa_dir: Path, dpi: int) -> dict:
    import pymupdf
    from PIL import Image as PILImage, ImageDraw

    directory = qa_dir / pdf.parent.name
    directory.mkdir(parents=True, exist_ok=True)
    for stale_page in directory.glob("page-*.png"):
        stale_page.unlink()
    paths: list[Path] = []
    with pymupdf.open(pdf) as document:
        for index, page in enumerate(document):
            path = directory / f"page-{index + 1:02d}.png"
            page.get_pixmap(dpi=dpi, alpha=False).save(path)
            paths.append(path)
        page_count = len(document)
        text = "\n".join(page.get_text() for page in document)
    thumb_width = 420
    thumb_height = int(thumb_width * PAGE_HEIGHT / PAGE_WIDTH)
    columns = min(3, len(paths))
    rows = (len(paths) + columns - 1) // columns
    sheet = PILImage.new("RGB", (columns * (thumb_width + 16), rows * (thumb_height + 34)), "#dddddd")
    drawer = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with PILImage.open(path) as original:
            thumbnail = original.convert("RGB")
            thumbnail.thumbnail((thumb_width, thumb_height))
            x = (index % columns) * (thumb_width + 16) + 8
            y = (index // columns) * (thumb_height + 34) + 8
            sheet.paste(thumbnail, (x, y))
            drawer.text((x, y + thumb_height + 4), f"{pdf.parent.name}, page {index + 1}", fill="black")
    contact = directory / "contact-sheet.png"
    sheet.save(contact)
    return {"pdf": str(pdf), "pages": page_count, "text_characters": len(text), "contact_sheet": str(contact)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="*", help="Markdown reports; default: Lab*/report.md")
    parser.add_argument("--font-dir", type=Path, help="Directory containing times.ttf/timesbd.ttf/timesi.ttf/timesbi.ttf")
    parser.add_argument("--render", action="store_true", help="Render all PDF pages and a contact sheet for visual QA")
    parser.add_argument("--qa-dir", type=Path, default=ROOT / "tools" / "_qa")
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()
    sources = args.reports or sorted(ROOT.glob("Lab*/report.md"))
    if not sources:
        parser.error("No report.md sources were found.")
    if not 72 <= args.dpi <= 300:
        parser.error("--dpi must be between 72 and 300.")
    register_fonts(args.font_dir)
    for source in sources:
        source = source.resolve()
        if not source.is_file():
            parser.error(f"Report source does not exist: {source}")
        output = build_report(source)
        if args.render:
            print(render_report(output, args.qa_dir, args.dpi))
        else:
            print(output)


if __name__ == "__main__":
    main()
