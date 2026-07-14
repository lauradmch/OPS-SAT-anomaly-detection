from ops_sat_ad.serving.report import (
    infer_type_hint, build_facts, faithfulness_check,
    TemplateNarrator, render_report,
)

def _pred(ae, hp, is_anom, channel="CADC0872", n=60, version="1.0"):
    """Build a prediction dict matching predict_segment's contract."""
    return {"channel": channel, "score": 0.5 * (ae + hp), "is_anomaly": is_anom,
            "threshold": 0.50, "ae_pct": ae, "hp_pct": hp,
            "n_points": n, "model_version": version}

# --- type-hint rule: every branch ---
def test_type_hint_branches():
    assert infer_type_hint(0.10, 0.10, False) == "nominal"
    assert infer_type_hint(0.95, 0.30, True)  == "shape"      # ae dominates
    assert infer_type_hint(0.30, 0.95, True)  == "transient"  # hp dominates
    assert infer_type_hint(0.92, 0.90, True)  == "mixed"      # within margin

# --- faithfulness checker: pass and fail ---
def test_grounded_text_passes():
    facts = build_facts(_pred(0.95, 0.30, True))       # score renders 0.62
    ok, offending = faithfulness_check("score 0.62 vs 0.50 on CADC0872", facts)
    assert ok and offending == []

def test_hallucinated_number_flagged():
    facts = build_facts(_pred(0.95, 0.30, True))
    ok, offending = faithfulness_check("rose 47 percent over 3-sigma", facts)
    assert not ok and "47" in offending

# --- template narrator must be number-free ---
def test_template_narrator_number_free():
    txt = TemplateNarrator().narrate(_pred(0.95, 0.30, True), "shape")
    assert not any(c.isdigit() for c in txt)

# --- composer, happy path ---
def test_render_report_template_path():
    out = render_report(_pred(0.95, 0.30, True), narrator=TemplateNarrator())
    assert out["faithful"] and out["generator"] == "template"
    assert out["type_hint"] == "shape"

# --- composer, fallback path (the branch that hid your bug) ---
class _LeakyNarrator:
    """Stub narrator that always leaks an ungrounded number."""
    name = "llm"                                  # name != 'template' -> fallback eligible
    def narrate(self, pred, type_hint):
        return "The anomaly rose 999 units."      # 999 is not a grounded token

def test_render_report_falls_back_on_leak():
    out = render_report(_pred(0.95, 0.30, True), narrator=_LeakyNarrator())
    assert out["generator"] == "template"   # composer swapped it out
    assert out["faithful"] is True          # final report is clean
    assert "999" not in out["report"]       # the leaked number is gone