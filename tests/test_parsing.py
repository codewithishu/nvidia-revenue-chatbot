import fitz

from nvrag.parsing import detect_units, parse_pdf


def test_detect_units():
    assert detect_units("Revenue (in millions, except per share data)") == "in millions"
    assert detect_units("amounts in thousands") == "in thousands"
    assert detect_units("no units here") is None


def test_parse_pdf_prose_and_table(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "RISK FACTORS", fontsize=14)
    page.insert_text((72, 100), "Our business depends on demand for GPUs.", fontsize=10)
    page.insert_text((72, 140), "(in millions)", fontsize=9)
    x0, y0, cw, rh = 72, 160, 100, 20
    data = [["Metric", "FY24", "FY23"], ["Revenue", "60,922", "26,974"], ["Net income", "29,760", "4,368"]]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), val, fontsize=9)
    p = tmp_path / "t.pdf"
    doc.save(p)

    pages = parse_pdf(p)
    assert len(pages) == 1 and pages[0].page == 1
    assert any("GPUs" in b for b in pages[0].blocks)
    assert pages[0].units == "in millions"
    assert pages[0].tables, "table should be detected"
    md = pages[0].tables[0].markdown
    assert "60,922" in md and "26,974" in md
    # table text must not be duplicated in prose blocks
    assert not any("60,922" in b for b in pages[0].blocks)
