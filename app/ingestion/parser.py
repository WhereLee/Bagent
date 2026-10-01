"""文档解析：把各格式文件统一抽成纯文本。

M1 覆盖常见格式；表格/图片等复杂结构在 M2 增强（此处保证流程完整，
解析能力可逐步加深，但不作为链路缺口）。
"""
from __future__ import annotations

from pathlib import Path


class ParseError(Exception):
    pass


def parse_file(path: str | Path) -> tuple[str, str]:
    """返回 (纯文本, media_type)。"""
    p = Path(path)
    if not p.exists():
        raise ParseError(f"文件不存在: {p}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(p), "pdf"
    if ext == ".docx":
        return _parse_docx(p), "docx"
    if ext in (".html", ".htm"):
        return _parse_html(p), "html"
    if ext in (".md", ".markdown", ".txt"):
        return p.read_text(encoding="utf-8", errors="ignore"), ext.lstrip(".")
    raise ParseError(f"暂不支持的格式: {ext}")


def _parse_pdf(p: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _parse_docx(p: Path) -> str:
    import docx

    doc = docx.Document(str(p))
    return "\n\n".join(par.text for par in doc.paragraphs if par.text.strip())


def _parse_html(p: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")
