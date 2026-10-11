from pathlib import Path

import yaml

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"
CONFIG = Path(__file__).with_name("criteria.yaml")
REQUIRED_KEYS = {"id", "group", "label", "source", "input", "field", "rule", "severity"}
RULE_TYPES = {"max", "min", "bool", "in_set"}
SEVERITIES = {"priority", "advisory"}
INPUTS = {"clinician", "data"}


def validate(cfg):
    groups = cfg.get("groups") or {}
    seen = set()
    for c in cfg["criteria"]:
        missing = REQUIRED_KEYS - c.keys()
        if missing:
            raise ValueError(f"{c.get('id', '?')}: missing keys {sorted(missing)}")
        if c["id"] in seen:
            raise ValueError(f"duplicate id {c['id']}")
        seen.add(c["id"])
        if not str(c["source"]).strip():
            raise ValueError(f"{c['id']}: empty source")
        if c["severity"] not in SEVERITIES:
            raise ValueError(f"{c['id']}: bad severity")
        if c["input"] not in INPUTS:
            raise ValueError(f"{c['id']}: bad input")
        if c["rule"].get("type") not in RULE_TYPES:
            raise ValueError(f"{c['id']}: bad rule type")
        if c["group"] not in groups:
            raise ValueError(f"{c['id']}: unknown group {c['group']}")
    return cfg


def load_criteria(path=CONFIG):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate(cfg)


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return isinstance(v, float) and v != v


def evaluate_one(criterion, patient):
    v = patient.get(criterion["field"])
    if _is_missing(v):
        return UNKNOWN
    rule = criterion["rule"]
    kind = rule["type"]
    if kind in ("max", "min"):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return UNKNOWN
        if x != x:
            return UNKNOWN
        ok = x <= rule["value"] if kind == "max" else x >= rule["value"]
        return PASS if ok else FAIL
    if kind == "bool":
        if isinstance(v, bool):
            return PASS if v else FAIL
        return UNKNOWN
    s = str(v)
    if s in rule.get("pass_values", []):
        return PASS
    if s in rule.get("fail_values", []):
        return FAIL
    return UNKNOWN


def _group_summary(items, groups):
    out = {}
    for gid, meta in groups.items():
        members = [i for i in items if i["group"] == gid]
        out[gid] = {
            "label": meta["label"],
            "note": meta.get("note", ""),
            "total": len(members),
            **{s: sum(1 for i in members if i["status"] == s) for s in (PASS, FAIL, UNKNOWN)},
        }
    return out


def assess(patient, cfg=None):
    cfg = cfg or load_criteria()
    items = []
    for c in cfg["criteria"]:
        items.append({
            "id": c["id"],
            "group": c["group"],
            "label": c["label"],
            "source": c["source"],
            "severity": c["severity"],
            "input": c["input"],
            "note": c.get("note", ""),
            "status": evaluate_one(c, patient),
        })
    counts = {s: sum(1 for i in items if i["status"] == s) for s in (PASS, FAIL, UNKNOWN)}
    priority = [i for i in items if i["severity"] == "priority"]
    return {
        "criteria_version": cfg["version"],
        "disclaimer": cfg["disclaimer"],
        "summary": {
            **counts,
            "priority_not_met": [i["id"] for i in priority if i["status"] == FAIL],
            "priority_unknown": [i["id"] for i in priority if i["status"] == UNKNOWN],
        },
        "groups": _group_summary(items, cfg["groups"]),
        "items": items,
    }
