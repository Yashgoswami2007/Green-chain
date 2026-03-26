from models import Observation, Action, Reward, Supplier, Shipment
from typing import Tuple, Dict, Any

class GreenChainEnv:
    def __init__(self):
        self.reset()
        
    def reset(self) -> Observation:
        self.initial_budget = 1000000.0
        self.initial_carbon = 100000.0
        
        self.budget = self.initial_budget
        self.carbon = self.initial_carbon
        self.step_count = 0
        self.total_guesses = 0
        
        self.suppliers = [
            Supplier(id="SUP-1", cost=100.0, carbon_index=50.0, location_id="LOC-A"),
            Supplier(id="SUP-2", cost=105.0, carbon_index=20.0, location_id="LOC-B") # Target swap
        ]
        self.active_supplier_id = "SUP-1"
        
        self.shipments = [
            Shipment(id="SHIP-1", route=["HUB-1", "HUB-2", "HUB-3"], perishability=1.0)
        ]
        
        self.manifest = """GlobX Logistics Annual Audit Report - 2026

Summary of Operations:
This year GlobX processed over 150,000 TEU across major Pacific routes. Our commitment to sustainability remains a core pillar. We are proud to announce our partnership with EcoOffsets Inc.

Carbon Accounting Details:
We purchased 50,000 tons of offsets for the Q2 transit. However, internal tracking indicates a duplicate carbon credit ID: 8841 was submitted for both the Atlantic and Pacific routes. Furthermore, concerning the bulk carriers, the offset volume exceeds capacity of our registered vessel class by 20%.

Routing Anomalies:
While 90% of our routes followed standard protocols, the external auditor noted a ghost ship manifest detected on the South China route which did not correspond to any active GPS beacon.

Conclusion:
We remain dedicated to ESG standards and aim to resolve these tracking anomalies by Q4."""
        
        self.ground_truth_fraud_flags = [
            "duplicate carbon credit ID: 8841",
            "offset volume exceeds capacity",
            "ghost ship manifest detected"
        ]
        self.identified_flags = set()
        self.all_submitted_flags = set()
        
        return self.state()
        
    def state(self) -> Observation:
        active_sups = [s for s in self.suppliers if s.id == self.active_supplier_id]
        carbon_red = max(0, (self.initial_carbon - self.carbon) / self.initial_carbon)
        risk = max(0.0, 0.85 - (len(self.identified_flags) / 3) * 0.85)
        return Observation(
            step_count=self.step_count,
            budget_remaining=self.budget,
            current_carbon_footprint=self.carbon,
            sustainability_score=carbon_red,
            greenwashing_risk_score=risk,
            active_suppliers=active_sups,
            available_suppliers=self.suppliers,
            active_shipments=self.shipments,
            audit_document=self.manifest,
            identified_flags=list(self.identified_flags),
            total_guesses=len(self.all_submitted_flags)
        )
        
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        penalty = 0.0
        bonus = 0.0
        error_msg = None
        
        if action.action_type == "SwitchSupplier":
            new_id = action.parameters.get("new_supplier_id") if action.parameters else None
            if new_id and any(s.id == new_id for s in self.suppliers):
                old_sup = next(s for s in self.suppliers if s.id == self.active_supplier_id)
                self.active_supplier_id = new_id
                self.budget -= 50000.0 # Transition cost
                new_sup = next(s for s in self.suppliers if s.id == new_id)
                self.carbon -= (old_sup.carbon_index - new_sup.carbon_index) * 1000
                bonus += 0.1 # Directional shaping for taking an action
            else:
                error_msg = f"Supplier '{new_id}' not found."
                penalty += 0.1
                
        elif action.action_type == "RerouteShipment":
            shipment_id = action.parameters.get("shipment_id") if action.parameters else None
            new_route = action.parameters.get("route_hubs", []) if action.parameters else []
            found = False
            for s in self.shipments:
                if s.id == shipment_id:
                    found = True
                    s.route = new_route
                    if len(new_route) < 3: # Reduced route
                        self.carbon *= 0.85 # Cut emissions
                        s.perishability -= 0.1 # Slight age delay
                        bonus += 0.1
                    else:
                        penalty += 0.1
            if not found:
                error_msg = f"Shipment '{shipment_id}' not found."
                penalty += 0.1
                        
        elif action.action_type == "FlagForAudit":
            flags = action.parameters.get("fraud_flags", []) if action.parameters else []
            if flags:
                self.all_submitted_flags.update(flags)
                correct_flags = set(flags).intersection(set(self.ground_truth_fraud_flags))
                self.identified_flags.update(correct_flags)
                if len(correct_flags) > 0:
                    bonus += (len(correct_flags) / 3) * 0.2
                else:
                    error_msg = "No correct flags identified. Hallucination penalty applied."
                    penalty += 0.1
            else:
                error_msg = "No flags provided in parameter payload."
                penalty += 0.1
                
        elif action.action_type == "DoNothing":
            error_msg = "Agent chose to do nothing. Applied time penalty."
            penalty += 0.05
                
        # Calculate Reward
        budget_eff = max(0, self.budget / self.initial_budget)
        carbon_red = max(0, (self.initial_carbon - self.carbon) / self.initial_carbon)
        
        step_penalty = self.step_count * 0.01
        perish_pen = 0.0
        if self.shipments and self.shipments[0].perishability < 0.8:
            perish_pen = 0.8 - self.shipments[0].perishability
            
        reward_val = (budget_eff * carbon_red) - step_penalty - perish_pen - penalty + bonus
            
        reward = Reward(value=max(-1.0, min(1.0, reward_val)))
        done = self.step_count >= 10
        info = {
            "identified_flags": list(self.identified_flags),
            "budget_efficiency": budget_eff,
            "carbon_reduction": carbon_red,
            "error": error_msg
        }
        # remove None errors to keep JSON clean
        info = {k: v for k, v in info.items() if v is not None}
        
        return self.state(), reward, done, info
