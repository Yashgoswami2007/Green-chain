from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any

class Supplier(BaseModel):
    id: str
    cost: float
    carbon_index: float
    location_id: str

class Shipment(BaseModel):
    id: str
    route: List[str]
    perishability: float

class Observation(BaseModel):
    step_count: int = 0
    budget_remaining: float
    current_carbon_footprint: float
    sustainability_score: float = 0.0
    greenwashing_risk_score: float = 0.0
    active_suppliers: List[Supplier]
    available_suppliers: List[Supplier] = []
    active_shipments: List[Shipment]
    audit_document: Optional[str] = None
    identified_flags: List[str] = []
    total_guesses: int = 0

class Action(BaseModel):
    action_type: Literal["SwitchSupplier", "RerouteShipment", "FlagForAudit", "DoNothing"]
    target_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Reward(BaseModel):
    value: float
