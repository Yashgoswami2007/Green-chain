"""
GreenChain Smoke Test Suite
===========================
Tests every endpoint, action type, grader, and edge case.
Run against a live server: python smoke_test.py [BASE_URL]

Exit code 0 = all passed. Exit code 1 = failures found.
"""

import sys
import json
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

def get(path):
    return requests.get(f"{BASE_URL}{path}", timeout=10)

def post(path, body=None):
    return requests.post(f"{BASE_URL}{path}", json=body, timeout=10)

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
check("budget_remaining == 1000000.0", obs.get("budget_remaining") == 1000000.0)
check("current_carbon_footprint == 100000.0", obs.get("current_carbon_footprint") == 100000.0)
check("step_count == 0", obs.get("step_count") == 0)
check("active_suppliers is a list", isinstance(obs.get("active_suppliers"), list))
check("active_shipments is a list", isinstance(obs.get("active_shipments"), list))
check("available_suppliers is a list", isinstance(obs.get("available_suppliers"), list))
check("identified_flags is empty list", obs.get("identified_flags") == [])
check("total_guesses == 0", obs.get("total_guesses") == 0)
check("greenwashing_risk_score == 0.85", obs.get("greenwashing_risk_score") == 0.85)

# ─────────────────────────────────────────────
# 3. /state
# ─────────────────────────────────────────────
section("3. GET /state")
r = get("/state")
check("Returns 200", r.status_code == 200)
check("Has 'observation' key", "observation" in r.json())

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

# ─────────────────────────────────────────────
# 5. /step — SwitchSupplier (valid)
# ─────────────────────────────────────────────
section("5. POST /step — SwitchSupplier (valid)")
post("/reset")
r = post("/step", {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": "SUP-2"}})
check("Returns 200", r.status_code == 200)
data = r.json()
check("Has observation", "observation" in data)
check("Has reward", "reward" in data)
check("Has done flag", "done" in data)
check("Has info dict", "info" in data)
obs = data["observation"]
check("step_count incremented to 1", obs.get("step_count") == 1)
check("Carbon reduced after swap", obs.get("current_carbon_footprint") < 100000.0)
check("Budget reduced after swap", obs.get("budget_remaining") < 1000000.0)
reward_val = data["reward"].get("value", None)
check("Reward value is float", isinstance(reward_val, float))
check("Reward in [-1.0, 1.0]", reward_val is not None and -1.0 <= reward_val <= 1.0)

# ─────────────────────────────────────────────
# 6. /step — SwitchSupplier (invalid ID)
# ─────────────────────────────────────────────
section("6. POST /step — SwitchSupplier (invalid ID)")
post("/reset")
r = post("/step", {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": "SUP-FAKE"}})
check("Returns 200 (graceful)", r.status_code == 200)
data = r.json()
check("Info contains error key", "error" in data.get("info", {}))
check("Error mentions supplier", "SUP-FAKE" in data["info"].get("error", ""))

# ─────────────────────────────────────────────
# 7. /step — RerouteShipment (valid, short route)
# ─────────────────────────────────────────────
section("7. POST /step — RerouteShipment (valid, short route)")
post("/reset")
r = post("/step", {"action_type": "RerouteShipment", "parameters": {"shipment_id": "SHIP-1", "route_hubs": ["HUB-1", "HUB-3"]}})
check("Returns 200", r.status_code == 200)
obs = r.json()["observation"]
shipment = obs["active_shipments"][0]
check("Route updated to 2 hubs", len(shipment["route"]) == 2)
check("Perishability decreased slightly", shipment["perishability"] < 1.0)
check("Carbon reduced by ~15%", obs["current_carbon_footprint"] <= 85000.0)

# ─────────────────────────────────────────────
# 8. /step — RerouteShipment (invalid shipment ID)
# ─────────────────────────────────────────────
section("8. POST /step — RerouteShipment (invalid ID)")
post("/reset")
r = post("/step", {"action_type": "RerouteShipment", "parameters": {"shipment_id": "SHIP-FAKE", "route_hubs": ["HUB-1"]}})
check("Returns 200 (graceful)", r.status_code == 200)
check("Info contains error key", "error" in r.json().get("info", {}))

# ─────────────────────────────────────────────
# 9. /step — FlagForAudit (all 3 correct)
# ─────────────────────────────────────────────
section("9. POST /step — FlagForAudit (all 3 correct flags)")
post("/reset")
# Fetch randomized ground truth from /debug — flags are random.sample() each reset
debug_r = get("/debug")
check("GET /debug returns 200", debug_r.status_code == 200)
truth_flags = debug_r.json().get("ground_truth_fraud_flags", [])
check("Ground truth has 3 flags", len(truth_flags) == 3, f"got {truth_flags}")

r = post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": truth_flags}})
check("Returns 200", r.status_code == 200)
obs = r.json()["observation"]
check("identified_flags has 3 entries", len(obs.get("identified_flags", [])) == 3,
      f"got {len(obs.get('identified_flags', []))}")
check("total_guesses == 3", obs.get("total_guesses") == 3,
      f"got {obs.get('total_guesses')}")
check("greenwashing_risk_score == 0.0", obs.get("greenwashing_risk_score") == 0.0,
      f"got {obs.get('greenwashing_risk_score')}")

# ─────────────────────────────────────────────
# 10. /step — FlagForAudit (hallucinated flags)
# ─────────────────────────────────────────────
section("10. POST /step — FlagForAudit (hallucinated flags)")
post("/reset")
r = post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": ["totally made up flag"]}})
check("Returns 200 (graceful)", r.status_code == 200)
data = r.json()
check("Info contains error key", "error" in data.get("info", {}))
check("identified_flags still empty", data["observation"].get("identified_flags") == [])

# ─────────────────────────────────────────────
# 11. /step — DoNothing
# ─────────────────────────────────────────────
section("11. POST /step — DoNothing")
post("/reset")
r = post("/step", {"action_type": "DoNothing"})
check("Returns 200", r.status_code == 200)
data = r.json()
check("Info contains error/message", "error" in data.get("info", {}))
check("step_count incremented", data["observation"].get("step_count") == 1)

# ─────────────────────────────────────────────
# 12. /grader — task_1_swap
# ─────────────────────────────────────────────
section("12. POST /grader — task_1_swap")
post("/reset")
post("/step", {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": "SUP-2"}})
r = post("/grader?task_id=task_1_swap")
check("Returns 200", r.status_code == 200)
score = r.json().get("score")
check("Score is float", isinstance(score, float))
check("Score in [0.0, 1.0]", 0.0 <= score <= 1.0)
check("Baseline achieves 1.0", score == 1.0, f"got {score}")

# ─────────────────────────────────────────────
# 13. /grader — task_2_route
# ─────────────────────────────────────────────
section("13. POST /grader — task_2_route")
post("/reset")
post("/step", {"action_type": "RerouteShipment", "parameters": {"shipment_id": "SHIP-1", "route_hubs": ["HUB-1", "HUB-3"]}})
r = post("/grader?task_id=task_2_route")
check("Returns 200", r.status_code == 200)
score = r.json().get("score")
check("Score is float", isinstance(score, float))
check("Score in [0.0, 1.0]", 0.0 <= score <= 1.0)
check("Baseline achieves 1.0", score == 1.0, f"got {score}")

# ─────────────────────────────────────────────
# 14. /grader — task_3_audit
# ─────────────────────────────────────────────
section("14. POST /grader — task_3_audit")
post("/reset")
# Use /debug to get dynamically randomized truth flags
truth_flags_14 = get("/debug").json().get("ground_truth_fraud_flags", [])
post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": truth_flags_14}})
r = post("/grader?task_id=task_3_audit")
check("Returns 200", r.status_code == 200)
score = r.json().get("score")
check("Score is float", isinstance(score, float))
check("Score in [0.0, 1.0]", 0.0 <= score <= 1.0)
check("Baseline achieves 1.0", score == 1.0, f"got {score}")

# ─────────────────────────────────────────────
# 15. /grader — partial credit checks
# ─────────────────────────────────────────────
section("15. Partial credit — graders return intermediate scores")
post("/reset")
r = post("/grader?task_id=task_1_swap")
score_zero = r.json().get("score")
check("task_1_swap returns 0.0 before any action", score_zero == 0.0, f"got {score_zero}")

post("/reset")
# Use /debug to get just 1 flag (partial submission)
partial_truth = get("/debug").json().get("ground_truth_fraud_flags", [])
post("/step", {"action_type": "FlagForAudit", "parameters": {"fraud_flags": partial_truth[:1]}})
r = post("/grader?task_id=task_3_audit")
partial = r.json().get("score")
check("task_3_audit partial (1/3 flags) > 0.0 and < 1.0", 0.0 < partial < 1.0, f"got {partial}")

# ─────────────────────────────────────────────
# 16. /grader — unknown task_id
# ─────────────────────────────────────────────
section("16. POST /grader — unknown task_id")
post("/reset")
r = post("/grader?task_id=task_fake")
check("Returns 200 (graceful)", r.status_code == 200)
score = r.json().get("score")
check("Returns score of 0.0 for unknown task", score == 0.0, f"got {score}")

# ─────────────────────────────────────────────
# 17. /baseline endpoint
# ─────────────────────────────────────────────
section("17. GET /baseline")
r = get("/baseline")
check("Returns 200", r.status_code == 200)
data = r.json()
check("Has 'scores' key", "scores" in data)
check("Has 'total_score' key", "total_score" in data)
scores = data.get("scores", {})
check("task_1_swap score present", "task_1_swap" in scores)
check("task_2_route score present", "task_2_route" in scores)
check("task_3_audit score present", "task_3_audit" in scores)
for k, v in scores.items():
    check(f"{k} score in [0.0, 1.0]", 0.0 <= v <= 1.0, f"got {v}")
check("total_score == 3.0", data.get("total_score") == 3.0, f"got {data.get('total_score')}")

# ─────────────────────────────────────────────
# 18. /baseline cleans up state
# ─────────────────────────────────────────────
section("18. /baseline leaves clean state")
get("/baseline")
r = get("/state")
obs = r.json()["observation"]
check("step_count reset to 0 after /baseline", obs.get("step_count") == 0)
check("budget reset to 1000000.0 after /baseline", obs.get("budget_remaining") == 1000000.0)
check("identified_flags empty after /baseline", obs.get("identified_flags") == [])

# ─────────────────────────────────────────────
# 19. /render
# ─────────────────────────────────────────────
section("19. GET /render")
r = get("/render")
check("Returns 200", r.status_code == 200)
check("Content-Type is HTML", "text/html" in r.headers.get("content-type", ""))
html = r.text
check("Contains GreenChain title", "GreenChain" in html)
check("Shows budget", "Budget" in html)
check("Shows carbon footprint", "Carbon" in html)
check("Shows shipments section", "Shipments" in html)
check("Shows fraud flags section", "Fraud Flags" in html)

# ─────────────────────────────────────────────
# 20. Episode termination (done flag)
# ─────────────────────────────────────────────
section("20. Episode terminates after 10 steps")
post("/reset")
done = False
for i in range(10):
    r = post("/step", {"action_type": "DoNothing"})
    done = r.json().get("done", False)
check("done == True after 10 steps", done == True, f"done={done}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = 0
# count all check() calls by re-reading failures vs passes — approximate via output
print(f"\n{'='*50}")
if failures:
    print(f"\033[91m✗ {len(failures)} test(s) FAILED:\033[0m")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
else:
    print(f"\033[92m✓ All tests passed!\033[0m")
    sys.exit(0)