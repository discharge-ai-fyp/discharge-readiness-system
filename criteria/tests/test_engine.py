import copy

import pytest

from criteria import engine

CFG = engine.load_criteria()


def crit(cid):
    return next(c for c in CFG["criteria"] if c["id"] == cid)


def test_config_valid_and_every_item_has_source():
    assert CFG["criteria"]
    assert all(c["source"].strip() for c in CFG["criteria"])


def test_duplicate_id_rejected():
    bad = copy.deepcopy(CFG)
    bad["criteria"].append(copy.deepcopy(bad["criteria"][0]))
    with pytest.raises(ValueError):
        engine.validate(bad)


@pytest.mark.parametrize("cid,field,value,expected", [
    ("C1", "temperature_c", 37.8, "pass"),
    ("C1", "temperature_c", 37.9, "fail"),
    ("C2", "heart_rate", 100, "pass"),
    ("C2", "heart_rate", 101, "fail"),
    ("C3", "resp_rate", 24, "pass"),
    ("C3", "resp_rate", 25, "fail"),
    ("C4", "systolic_bp", 90, "pass"),
    ("C4", "systolic_bp", 89, "fail"),
    ("C5", "spo2", 90, "pass"),
    ("C5", "spo2", 89, "fail"),
])
def test_halm_thresholds_at_boundaries(cid, field, value, expected):
    assert engine.evaluate_one(crit(cid), {field: value}) == expected


@pytest.mark.parametrize("value", [None, "", "  ", float("nan"), "abc"])
def test_missing_or_bad_numeric_is_unknown(value):
    assert engine.evaluate_one(crit("C1"), {"temperature_c": value}) == "unknown"


def test_bool_criteria():
    c = crit("C10")
    assert engine.evaluate_one(c, {"med_reconciliation_done": True}) == "pass"
    assert engine.evaluate_one(c, {"med_reconciliation_done": False}) == "fail"
    assert engine.evaluate_one(c, {}) == "unknown"
    assert engine.evaluate_one(c, {"med_reconciliation_done": "yes"}) == "unknown"


@pytest.mark.parametrize("value,expected", [
    (">7", "pass"), (">8", "pass"), ("Norm", "pass"),
    ("None", "unknown"), ("Not_measured", "unknown"), (None, "unknown"),
])
def test_a1c_never_fails_on_missing(value, expected):
    assert engine.evaluate_one(crit("C8"), {"a1c_result": value}) == expected


def test_assess_has_no_ready_score_and_lists_priority_unknown():
    result = engine.assess({})
    priority_ids = {c["id"] for c in CFG["criteria"] if c["severity"] == "priority"}
    assert set(result["summary"]["priority_unknown"]) == priority_ids
    assert result["summary"]["priority_not_met"] == []
    assert result["criteria_version"] == CFG["version"]
    assert not any("ready" in k or "score" in k for k in result)


def test_unknown_group_rejected():
    bad = copy.deepcopy(CFG)
    bad["criteria"][0]["group"] = "nope"
    with pytest.raises(ValueError):
        engine.validate(bad)


def test_old_blocking_label_rejected():
    bad = copy.deepcopy(CFG)
    bad["criteria"][0]["severity"] = "blocking"
    with pytest.raises(ValueError):
        engine.validate(bad)


def test_stability_group_counts_instabilities():
    g = engine.assess({"temperature_c": 38.2, "heart_rate": 88})["groups"]["stability"]
    assert (g["total"], g["fail"], g["pass"], g["unknown"]) == (7, 1, 1, 5)


def test_two_instabilities_counted():
    g = engine.assess({"temperature_c": 38.2, "heart_rate": 110, "spo2": 95})["groups"]["stability"]
    assert g["fail"] == 2 and g["pass"] == 1 and g["unknown"] == 4
