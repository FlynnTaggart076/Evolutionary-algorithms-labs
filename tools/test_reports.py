"""Check preservation of mathematical notation through the PDF exporter."""
import hashlib

from pypdf import PdfReader

from build_reports import build_report, register_fonts


def test_math_optimum_stars_are_not_markdown_emphasis(tmp_path):
    source = tmp_path / "report.md"
    source.write_text(
        "# Проверка обозначений\n\n"
        "Точный оптимум F* = 1404. Разрыв: (F* - F)/F*. Значение f* = 0.\n\n"
        "*Курсив* и **полужирный** сохраняют обычный текст.\n",
        encoding="utf-8",
    )
    register_fonts()
    pdf = build_report(source)
    reader = PdfReader(pdf)
    text = " ".join(page.extract_text() for page in reader.pages)
    assert text.count("F*") == 3
    assert text.count("f*") == 1
    assert "*Курсив*" not in text and "Курсив" in text
    assert "**полужирный**" not in text and "полужирный" in text
    assert reader.metadata["/Keywords"] == "source-sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
