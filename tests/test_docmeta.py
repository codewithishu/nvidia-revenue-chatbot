from pathlib import Path

from nvrag.docmeta import infer_meta


def test_from_filename():
    m = infer_meta(Path("nvidia_proxy_2024.pdf"), "")
    assert (m.doc_type, m.year, m.year_source) == ("proxy", 2024, "filename")
    m = infer_meta(Path("NVIDIA-2019-Annual-Report.pdf"), "")
    assert (m.doc_type, m.year) == ("annual_report", 2019)


def test_from_text_when_filename_is_opaque():
    text = "FORM 10-K For the fiscal year ended January 28, 2024"
    m = infer_meta(Path("doc123.pdf"), text)
    assert (m.doc_type, m.year, m.year_source) == ("annual_report", 2024, "text")


def test_manifest_overrides():
    man = {"a.pdf": {"file": "a.pdf", "doc_type": "proxy", "year": "2003"}}
    m = infer_meta(Path("a.pdf"), "", man)
    assert (m.doc_type, m.year, m.year_source) == ("proxy", 2003, "manifest")
