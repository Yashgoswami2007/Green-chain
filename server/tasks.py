from .models import Observation
from typing import Dict

def grade_task_1_swap(state: Observation) -> float:
    # Use sustainability_score (already computed correctly in state()) rather than hardcoded initial
    carbon_drop = state.sustainability_score
    budget_drop = (1000000.0 - state.budget_remaining) / 1000000.0
    
    if carbon_drop >= 0.30:
        if budget_drop <= 0.10:
            return 1.0 # Perfect
        return 0.5 # Partial: Swapped, but blew budget
    elif carbon_drop > 0.10:
        return 0.2 # Partial effort
    return 0.0

def grade_task_2_route(state: Observation) -> float:
    # Objective: Optimize any shipment route to minimize transportation miles.
    # Find the shipment with the shortest (most-optimized) route — not just index 0
    if not state.active_shipments:
        return 0.0

    best = min(state.active_shipments, key=lambda s: len(s.route))
    score = 0.0
    
    # Partial credit for reducing route below 3 hubs
    if len(best.route) < 3:
        score += 0.5
        
    # Partial credit for keeping perishability up
    if best.perishability >= 0.8:
        score += 0.5
        
    # Penalty if carbon didn't drop by 15%
    if state.current_carbon_footprint > 85001.0:
        score -= 0.5
        
    return max(0.0, min(1.0, score))

def grade_task_3_audit(state: Observation) -> float:
    # Objective: Identify inconsistencies in a Logistics Manifest.
    # Success Criteria: F1 Precision/Recall score on the 3 identified fraudulent records
    correct = len(state.identified_flags)
    truth_total = 3
    if state.total_guesses == 0 or correct == 0:
        return 0.0
        
    precision = correct / state.total_guesses
    recall = correct / truth_total
    
    if precision + recall == 0: 
        return 0.0
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    return max(0.0, min(1.0, f1_score))

def evaluate_task(task_id: str, current_state: Observation) -> float:
    score = 0.0
    if task_id == "task_1_swap":
        score = grade_task_1_swap(current_state)
    elif task_id == "task_2_route":
        score = grade_task_2_route(current_state)
    elif task_id == "task_3_audit":
        score = grade_task_3_audit(current_state)
    return score
