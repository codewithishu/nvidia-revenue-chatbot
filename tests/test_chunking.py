from nvrag.chunking import detect_heading, make_chunks, split_long
from nvrag.config import load_config
from nvrag.docmeta import DocMeta
from nvrag.parsing import PageContent, Table, rows_to_markdown

CFG = load_config()
META = DocMeta("nvidia_annual_report_2024", "x.pdf", "annual_report", 2024, "filename")


def test_headings():
    assert detect_heading("Item 7. Management's Discussion and Analysis") is not None
    assert detect_heading("RISK FACTORS") == "RISK FACTORS"
    assert detect_heading("This is a normal sentence about revenue.") is None
    assert detect_heading("42") is None


def test_split_long_respects_limit():
    text = " ".join(["This is a sentence."] * 400)
    parts = split_long(text, 500)
    assert all(len(p) <= 520 for p in parts) and len(parts) > 5


def test_chunks_stay_on_their_page_and_carry_metadata():
    pages = [
        PageContent(1, ["RISK FACTORS", "Para one. " * 50, "Para two. " * 50]),
        PageContent(2, ["Revenue grew strongly."], tables=[Table((0, 0, 1, 1), "| a | b |\n|---|---|\n| 1 | 2 |", "Revenue")], units="in millions"),
    ]
    chunks = make_chunks(META, pages, CFG)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.year == 2024 and c.doc_type == "annual_report" for c in chunks)
    t = [c for c in chunks if c.kind == "table"]
    assert len(t) == 1 and "Revenue" in t[0].text and t[0].units == "in millions"
    assert "FY2024" in t[0].embed_text() and "page 2" in t[0].embed_text()
    assert chunks[0].section == "RISK FACTORS"


def test_big_table_split_repeats_header():
    rows = [["Metric", "FY24", "FY23"]] + [[f"Row {i}", str(i), str(i + 1)] for i in range(600)]
    md = rows_to_markdown(rows)
    pc = PageContent(5, [], tables=[Table((0, 0, 1, 1), md, "Big table")])
    chunks = make_chunks(META, [pc], CFG)
    assert len(chunks) > 1
    assert all("| Metric | FY24 | FY23 |" in c.text for c in chunks)


def test_rows_to_markdown_folds_symbol_columns():
    rows = [["", "FY24", "", "FY23", ""],
            ["Revenue", "60,922", "", "26,974", ""],
            ["Change", "(5", ")", "(12", ")"],
            ["Margin", "75", "%", "70", "%"]]
    md = rows_to_markdown(rows)
    assert "60,922" in md and "26,974" in md
    assert "(5)" in md and "(12)" in md
    assert "75%" in md and "70%" in md


def test_rows_to_markdown_drops_dollar_only_column():
    md = rows_to_markdown([["Metric", "", "FY24"], ["Revenue", "$", "60,922"]])
    assert "$" not in md
    assert "| Metric | FY24 |" in md
