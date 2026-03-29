import requests
import time
import sys

BASE_URL = "http://localhost:7860"

def run_baseline():
    print("Testing API connectivity...")
    try:
        requests.get(f"{BASE_URL}/state")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {BASE_URL}. Is uvicorn running?")
        sys.exit(1)
        
    # Task 1: Swap Supplier
    print("\n=== Executing Level 1: The Basic Swap ===")
    requests.post(f"{BASE_URL}/reset")
    action_1 = {
        "action_type": "SwitchSupplier",
        "parameters": {"new_supplier_id": "SUP-2"}
    }
    requests.post(f"{BASE_URL}/step", json=action_1)
    
    score_1 = requests.post(f"{BASE_URL}/grader?task_id=task_1_swap").json()
    print(f"Score: {score_1['score']}")
    
    # Task 2: Reroute Shipment
    print("\n=== Executing Level 2: The Logistics Puzzle ===")
    requests.post(f"{BASE_URL}/reset")
    action_2 = {
        "action_type": "RerouteShipment",
        "parameters": {"shipment_id": "SHIP-1", "route_hubs": ["HUB-1", "HUB-3"]}
    }
    requests.post(f"{BASE_URL}/step", json=action_2)
    
    score_2 = requests.post(f"{BASE_URL}/grader?task_id=task_2_route").json()
    print(f"Score: {score_2['score']}")
    
    # Task 3: Audit Document
    print("\n=== Executing Level 3: The Deep Audit ===")
    requests.post(f"{BASE_URL}/reset")
    action_3 = {
        "action_type": "FlagForAudit",
        "parameters": {"fraud_flags": [
            "duplicate carbon credit ID: 8841",
            "offset volume exceeds capacity",
            "ghost ship manifest detected"
        ]}
    }
    requests.post(f"{BASE_URL}/step", json=action_3)
    
    score_3 = requests.post(f"{BASE_URL}/grader?task_id=task_3_audit").json()
    print(f"Score: {score_3['score']}")
    
    print("\n--- Baseline Agent Run Complete ---")
    if score_1['score'] == 1.0 and score_2['score'] == 1.0 and score_3['score'] == 1.0:
        print("Success: Baseline achieved perfect execution!")

if __name__ == "__main__":
    run_baseline()
