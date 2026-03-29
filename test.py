"""
GreenChain Smoke Test Suite
===========================
Tests every endpoint, action type, grader, and edge case.
Run against a live server: python smoke_test.py [BASE_URL]

Exit code 0 = all passed. Exit code 1 = failures found.
"""

import sys
import math
import requests

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:7860"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
HEAD = "\033[94m[----]\033[0m"

failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)

def approx(a, b, tol=1e-6):
    """Float-safe equality check."""
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)

def get(path):
    return requests.get(f"{BASE_URL}{path}", timeout=10)

def post(path, body=None):
    return requests.post(f"{BASE_URL}{path}", json=body, timeout=10)

def reset():
    """Helper: reset env and return observation dict."""
    return post("/reset").json()["observation"]

def section(title):
    print(f"\n{HEAD} {title}")

# ─────────────────────────────────────────────
# 1. CONNECTIVITY
# ─────────────────────────────────────────────
section("1. Connectivity")
try:
    r = get("/state")
    check("Server is reachable", r.status_code == 200)
except requests.exceptions.ConnectionError:
    print(f"  {FAIL} Cannot connect to {BASE_URL}. Is the server running?")
    sys.exit(1)

# ─────────────────────────────────────────────
# 2. /reset
# ─────────────────────────────────────────────
section("2. POST /reset")
r = post("/reset")
check("Returns 200", r.status_code == 200)
data = r.json()
check("Has 'observation' key", "observation" in data)
obs = data.get("observation", {})
check("budget_remaining == 1000000.0", approx(obs.get("budget_remaining"), 1000000.0))
check("current_carbon_footprint == 100000.0", approx(obs.get("current_carbon_footprint"), 100000.0))
check("step_count == 0", obs.get("step_count") == 0)
check("active_suppliers is a list", isinstance(obs.get("active_suppliers"), list))
check("active_shipments is a list", isinstance(obs.get("active_shipments"), list))
check("available_suppliers is a list", isinstance(obs.get("available_suppliers"), list))
check("available_suppliers has 2 entries", len(obs.get("available_suppliers", [])) == 2)
check("identified_flags is empty list", obs.get("identified_flags") == [])
check("total_guesses == 0", obs.get("total_guesses") == 0)
check("greenwashing_risk_score ~= 0.85", approx(obs.get("greenwashing_risk_score"), 0.85))
check("sustainability_score present", "sustainability_score" in obs)
check("audit_document is a non-empty string", isinstance(obs.get("audit_document"), str) and len(obs.get("audit_document", "")) > 0)

# ─────────────────────────────────────────────
# 3. /state
# ─────────────────────────────────────────────
section("3. GET /state")
r = get("/state")
check("Returns 200", r.status_code == 200)
data = r.json()
check("Has 'observation' key", "observation" in data)
obs = data["observation"]
check("All required fields present", all(k in obs for k in [
    "step_count", "budget_remaining", "current_carbon_footprint",
    "sustainability_score", "greenwashing_risk_score",
    "active_suppliers", "available_suppliers", "active_shipments",
    "audit_document", "identified_flags", "total_guesses"
]))

# ─────────────────────────────────────────────
# 4. /tasks
# ─────────────────────────────────────────────
section("4. GET /tasks")
r = get("/tasks")
check("Returns 200", r.status_code == 200)
data = r.json()
tasks = data.get("tasks", [])
check("Returns 3+ tasks", len(tasks) >= 3)
task_ids = [t["id"] for t in tasks]
check("task_1_swap present", "task_1_swap" in task_ids)
check("task_2_route present", "task_2_route" in task_ids)
check("task_3_audit present", "task_3_audit" in task_ids)
check("action_schema present", "action_schema" in data)
check("current_observation present", "current_observation" in data)
for t in tasks:
    check(f"Task '{t['id']}' has difficulty", "difficulty" in t)
    check(f"Task '{t['id']}' has description", "description" in t)

# ─────────────────────────────────────────────
# 5. /step — SwitchSupplier (valid)
# ─────────────────────────────────────────────
section("5. POST /step — SwitchSupplier (valid)")
reset()
r = post("/step", {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": "SUP-2"}})
check("Returns 200", r.status_code == 200)
data = r.json()
check("Has 'observation' key", "observation" in data)
check("Has 'reward' key", "reward" in data)
check("Has 'done' key", "done" in data)
check("Has 'info' key", "info" in data)
obs = data["observation"]
check("step_count incremented to 1", obs.get("step_count") == 1)
check("Carbon reduced after swap", obs.get("current_carbon_footprint", 100000) < 100000.0)
check("Budget reduced after swap", obs.get("budget_remaining", 1000000) < 1000000.0)
reward_val = data["reward"].get("value")
check("Reward value is a number", isinstance(reward_val, (int, float)))
check("Reward in [-1.0, 1.0]", reward_val is not None and -1.0 <= reward_val <= 1.0)
check("No error in info for valid action", "error" not in data["info"])

# ─────────────────────────────────────────────
# 6. /step — SwitchSupplier (invalid ID)
# ─────────────────────────────────────────────
section("6. POST /step — SwitchSupplier (invalid ID)")
reset()
r = post("/step", {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": "SUP-FAKE"}})
check("Returns 200 (graceful)", r.status_code == 200)
data = r.json()
info = data.get("info", {})
check("Info contains 'error' key", "error" in info, f"info keys: {list(info.keys())}")
check("Error mentions the bad ID", "SUP-FAKE" in info.get("error", ""), f"error: {info.get('error')}")

# ─────────────────────────────────────────────
# 7. /step — RerouteShipment (valid, short route)
# ─────────────────────────────────────────────
section("7. POST /step — RerouteShipment (valid, short route)")
reset()
r = post("/step", {"action_type": "RerouteShipment", "parameters": {"shipment_id": "SHIP-1", "route_hubs": ["HUB-1", "HUB-3"]}})
check("Returns 200", r.status_code == 200)
obs = r.json()["observation"]
shipment = obs["active_shipments"][0]
check("Route updated to 2 hubs", len(shipment["route"]) == 2)
check("Perishability decreased slightly", shipment["perishability"] < 1.0)
check("Carbon reduced by ~15%", obs["current_carbon_footprint"] <= 85000.0 + 1e-6)

# ─────────────────────────────────────────────
# 8. /step — RerouteShipment (invalid shipment ID)
# ─────────────────────────────────────────────
section("8. POST /step — RerouteShipment (invalid ID)")
reset()
r = post("/step", {"action_type": "RerouteShipment", "parameters": {"shipment_id": "SHIP-FAKE", "route_hubs": ["HUB-1"]}})
check("Returns 200 (graceful)", r.status_code == 200)
info = r.json().get("info", {})
check("Info contains 'error' key", "error" in info, f"info keys: {list(info.keys())}")

# ─────────────────────────────────────────────
# 9. /step — FlagForAudit (all 3 correct)
# ─────────────────────────────────────────────
section("9. POST /step — FlagForAudit (all 3 correct flags)")
reset()
r = post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": [
    "duplicate carbon credit ID: 8841",
    "offset volume exceeds capacity",
    "ghost ship manifest detected"
]}})
check("Returns 200", r.status_code == 200)
obs = r.json()["observation"]
check("identified_flags has 3 entries", len(obs.get("identified_flags", [])) == 3,
      f"got {len(obs.get('identified_flags', []))}")
check("total_guesses == 3", obs.get("total_guesses") == 3,
      f"got {obs.get('total_guesses')}")
check("greenwashing_risk_score ~= 0.0", approx(obs.get("greenwashing_risk_score"), 0.0),
      f"got {obs.get('greenwashing_risk_score')}")

# ─────────────────────────────────────────────
# 10. /step — FlagForAudit (hallucinated flags)
# ─────────────────────────────────────────────
section("10. POST /step — FlagForAudit (hallucinated flags)")
reset()
r = post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": ["totally made up flag"]}})
check("Returns 200 (graceful)", r.status_code == 200)
data = r.json()
info = data.get("info", {})
check("Info contains 'error' key", "error" in info, f"info keys: {list(info.keys())}")
check("identified_flags still empty", data["observation"].get("identified_flags") == [],
      f"got {data['observation'].get('identified_flags')}")

# ─────────────────────────────────────────────
# 11. /step — DoNothing
# ─────────────────────────────────────────────
section("11. POST /step — DoNothing")
reset()
r = post("/step", {"action_type": "DoNothing"})
check("Returns 200", r.status_code == 200)
data = r.json()
info = data.get("info", {})
check("Info contains 'error' key for DoNothing", "error" in info, f"info keys: {list(info.keys())}")
check("step_count incremented to 1", data["observation"].get("step_count") == 1,
      f"got {data['observation'].get('step_count')}")

# ─────────────────────────────────────────────
# 12. /grader — task_1_swap baseline
# ─────────────────────────────────────────────
section("12. POST /grader — task_1_swap baseline")
reset()
post("/step", {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": "SUP-2"}})
r = post("/grader?task_id=task_1_swap")
check("Returns 200", r.status_code == 200)
score = r.json().get("score")
check("Score is a number", isinstance(score, (int, float)), f"got type {type(score)}")
check("Score in [0.0, 1.0]", score is not None and 0.0 <= score <= 1.0, f"got {score}")
check("Baseline achieves 1.0", approx(score, 1.0), f"got {score}")

# ─────────────────────────────────────────────
# 13. /grader — task_2_route baseline
# ─────────────────────────────────────────────
section("13. POST /grader — task_2_route baseline")
reset()
post("/step", {"action_type": "RerouteShipment", "parameters": {"shipment_id": "SHIP-1", "route_hubs": ["HUB-1", "HUB-3"]}})
r = post("/grader?task_id=task_2_route")
check("Returns 200", r.status_code == 200)
score = r.json().get("score")
check("Score is a number", isinstance(score, (int, float)))
check("Score in [0.0, 1.0]", score is not None and 0.0 <= score <= 1.0, f"got {score}")
check("Baseline achieves 1.0", approx(score, 1.0), f"got {score}")

# ─────────────────────────────────────────────
# 14. /grader — task_3_audit baseline
# ─────────────────────────────────────────────
section("14. POST /grader — task_3_audit baseline")
reset()
post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": [
    "duplicate carbon credit ID: 8841",
    "offset volume exceeds capacity",
    "ghost ship manifest detected"
]}})
r = post("/grader?task_id=task_3_audit")
check("Returns 200", r.status_code == 200)
score = r.json().get("score")
check("Score is a number", isinstance(score, (int, float)))
check("Score in [0.0, 1.0]", score is not None and 0.0 <= score <= 1.0, f"got {score}")
check("Baseline achieves 1.0", approx(score, 1.0), f"got {score}")

# ─────────────────────────────────────────────
# 15. Partial credit checks
# ─────────────────────────────────────────────
section("15. Partial credit — graders return intermediate scores")
reset()
r = post("/grader?task_id=task_1_swap")
score_zero = r.json().get("score")
check("task_1_swap returns 0.0 before any action", approx(score_zero, 0.0), f"got {score_zero}")

reset()
post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": [
    "duplicate carbon credit ID: 8841"
]}})
r = post("/grader?task_id=task_3_audit")
partial = r.json().get("score")
check("task_3_audit partial (1/3 flags) > 0.0 and < 1.0",
      partial is not None and 0.0 < partial < 1.0, f"got {partial}")

# ─────────────────────────────────────────────
# 16. /grader — unknown task_id
# ─────────────────────────────────────────────
section("16. POST /grader — unknown task_id")
reset()
r = post("/grader?task_id=task_fake")
check("Returns 200 (graceful)", r.status_code == 200)
score = r.json().get("score")
check("Returns 0.0 for unknown task", approx(score, 0.0), f"got {score}")

# ─────────────────────────────────────────────
# 17. /baseline endpoint
# ─────────────────────────────────────────────
section("17. GET /baseline")
r = get("/baseline")
check("Returns 200", r.status_code == 200, f"status: {r.status_code}, body: {r.text[:200]}")
data = r.json()
check("Has 'scores' key", "scores" in data, f"keys: {list(data.keys())}")
check("Has 'total_score' key", "total_score" in data, f"keys: {list(data.keys())}")
scores = data.get("scores", {})
check("task_1_swap score present", "task_1_swap" in scores)
check("task_2_route score present", "task_2_route" in scores)
check("task_3_audit score present", "task_3_audit" in scores)
for k, v in scores.items():
    check(f"{k} score in [0.0, 1.0]", isinstance(v, (int, float)) and 0.0 <= v <= 1.0, f"got {v}")
check("total_score ~= 3.0", approx(data.get("total_score"), 3.0), f"got {data.get('total_score')}")

# ─────────────────────────────────────────────
# 18. /baseline cleans up state
# ─────────────────────────────────────────────
section("18. /baseline leaves clean state")
get("/baseline")
r = get("/state")
obs = r.json()["observation"]
check("step_count reset to 0 after /baseline", obs.get("step_count") == 0,
      f"got {obs.get('step_count')}")
check("budget reset to 1000000.0 after /baseline",
      approx(obs.get("budget_remaining"), 1000000.0))
check("identified_flags empty after /baseline", obs.get("identified_flags") == [],
      f"got {obs.get('identified_flags')}")

# ─────────────────────────────────────────────
# 19. /render
# ─────────────────────────────────────────────
section("19. GET /render")
reset()
r = get("/render")
check("Returns 200", r.status_code == 200, f"status: {r.status_code}")
content_type = r.headers.get("content-type", "")
check("Content-Type is HTML", "text/html" in content_type, f"got: {content_type}")
html = r.text
check("Contains GreenChain title", "GreenChain" in html)
check("Shows Budget", "Budget" in html)
check("Shows Carbon", "Carbon" in html)
check("Shows Shipments section", "Shipment" in html)
check("Shows Fraud Flags section", "Fraud Flag" in html)
check("Shows Available Suppliers", "Available" in html)

# ─────────────────────────────────────────────
# 20. Episode termination (done flag)
# ─────────────────────────────────────────────
section("20. Episode terminates after 10 steps")
reset()
done = False
for i in range(10):
    r = post("/step", {"action_type": "DoNothing"})
    done = r.json().get("done", False)
check("done == True after 10 steps", done == True, f"done={done}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
if failures:
    print(f"\033[91m✗ {len(failures)} test(s) FAILED:\033[0m")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
else:
    print(f"\033[92m✓ All tests passed!\033[0m")
    sys.exit(0)