from experiments.compare_results import build_comparison, format_table


def test_build_comparison_has_four_rows():
    a = {"mean_f1": 0.62, "mean_ceiling_f1": 0.81}
    c = {"mean_f1": 0.68}
    rows = build_comparison(a, c, baseline_f1=0.40)
    methods = [r["method"] for r in rows]
    assert len(rows) == 4
    assert any("Baseline" in m for m in methods)
    assert any("Ceiling" in m for m in methods)
    assert any("Experiment A" in m for m in methods)
    assert any("Experiment C" in m for m in methods)


def test_format_table_contains_numbers():
    rows = [{"method": "Experiment C", "mean_f1": 0.68}]
    text = format_table(rows)
    assert "0.68" in text
    assert "Experiment C" in text
