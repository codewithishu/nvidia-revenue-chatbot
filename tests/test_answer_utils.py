from nvrag.pipeline import extract_citations, format_context, unverified_numbers


def test_extract_citations_dedupes_in_order():
    assert extract_citations("A [S2]. B [S1][S2]. C [S10]") == [2, 1, 10]


def test_unverified_numbers_flags_only_missing():
    ctx = "Revenue | 60,922 | 26,974 ... margin 72.7"
    ans = "Revenue was $60,922 million in fiscal 2024 [S1], up from 26,974; margin 72.7%, growth 126%."
    assert unverified_numbers(ans, ctx) == ["126"]


def test_unverified_ignores_years_and_small_ints():
    assert unverified_numbers("In fiscal 2024 there were 3 segments [S1]", "nothing") == []


def test_format_context_contains_ids_and_units():
    s = [{"id": 1, "doc_type": "annual_report", "period": "FY2024", "file": "f.pdf", "page": 3,
          "kind": "table", "units": "in millions", "section": "", "text": "x"}]
    out = format_context(s)
    assert "[S1]" in out and "PDF page 3" in out and "in millions" in out
