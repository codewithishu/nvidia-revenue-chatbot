from nvrag.query import analyze_query


def test_fiscal_year_forms():
    assert analyze_query("revenue in fiscal 2024").years == [2024]
    assert analyze_query("FY23 gross margin").years == [2023]
    assert analyze_query("fiscal year 2019 net income").years == [2019]
    assert analyze_query("FY99 revenue").years == [1999]


def test_plain_year_is_marked_plain():
    q = analyze_query("What happened in 2022?")
    assert q.years == [2022] and q.plain == {2022}
    assert analyze_query("fiscal 2022").plain == set()


def test_multiple_and_ranges():
    assert analyze_query("compare fiscal 2023 and fiscal 2024").years == [2023, 2024]
    assert analyze_query("revenue 2019-2022").years == [2019, 2020, 2021, 2022]
    assert analyze_query("revenue from 2019 to 2021").years == [2019, 2020, 2021]


def test_numbers_are_not_years():
    assert analyze_query("revenue was $2,023 million").years == []
    assert analyze_query("grew 2024% ").years == []


def test_no_year_and_doc_type():
    q = analyze_query("Who are the director nominees?")
    assert q.years == [] and q.doc_type == "proxy"
    assert analyze_query("What are the main risk factors?").doc_type is None
