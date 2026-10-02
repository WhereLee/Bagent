"""引用校验单元测试。"""
from app.generation.citation import (
    extract_citations,
    has_citation,
    split_valid_invalid,
    strip_invalid_citations,
)


def test_extract_citations_in_order():
    assert extract_citations("端口是8443[1]，质保3年[2][1]。") == [1, 2, 1]


def test_split_valid_invalid_by_num_sources():
    valid, invalid = split_valid_invalid("见[1]和[2]还有[7]", num_sources=3)
    assert valid == [1, 2]
    assert invalid == [7]


def test_strip_removes_only_out_of_range():
    out = strip_invalid_citations("结论[1]正确，[9]是乱引。", num_sources=2)
    assert "[1]" in out
    assert "[9]" not in out


def test_has_citation():
    assert has_citation("答案[1]", 1) is True
    assert has_citation("答案[5]", 1) is False
    assert has_citation("无引用", 3) is False
