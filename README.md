---
title: Project GreenChain
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# Project GreenChain - Autonomous Supply Chain Auditing Environment

Project GreenChain is an OpenEnv specification compliant RL environment that simulates global procurement and logistics. AI agents must minimize carbon footprint and balance budgets, while also serving as auditors to flag greenwashing fraud in logistics texts.

## Observation Space
- `budget_remaining` (float): Current available capital.
- `current_carbon_footprint` (float): Total environmental impact.
- `active_suppliers` (list): Current supplying vendors, detailing cost and carbon index.
- `active_shipments` (list): Current active routes and perishability status.
- `audit_document` (str): Logistics manifest containing text for OCR testing.
- `greenwashing_risk_score` (float): Statistical risk variable.
- `identified_flags` (list): Text strings accurately flagged.

## Action Space
1. `SwitchSupplier` -> target `new_supplier_id`
2. `RerouteShipment` -> target `shipment_id`, `route_hubs`
3. `FlagForAudit` -> target `fraud_flags` (list of text extracted)

## Task Descriptions
**Task 1: The Basic Swap (Easy)**
Replace a Tier-1 supplier with a 40% lower carbon alternative without destroying the budget. 

**Task 2: The Logistics Puzzle (Medium)**
Optimize a shipment route to minimize emissions. Perishability must remain >0.8 while emissions drop. 

**Task 3: The Deep Audit (Hard)**
Identify all three inconsistencies (fraud flags) in a logistics manifest. Precision/Recall on the identified fraudulent records determines the partial score credit.

## Setup Instructions

Build the local API:
```bash
pip install -r requirements.txt
uvicorn main:app --port 7860
```

Run test suite:
```bash
python baseline.py
```

## Baseline Scores
- Task 1: 1.0 (Direct swap matching >30% carbon drop math)
- Task 2: 1.0 (Route reduction keeping perishability high)
- Task 3: 1.0 (100% precision and recall on the 3 seeded strings)
