from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .models import Observation, Action, Reward
from .environment import GreenChainEnv
from typing import Dict, Any
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn
import os

app = FastAPI(title="Project GreenChain OpenEnv")

# Enable CORS for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Redirect root to the dashboard for better UX on HF Spaces."""
    return RedirectResponse(url="/render")

@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    return HTMLResponse(
        status_code=404,
        content=f"""
        <html>
          <body style="font-family: sans-serif; padding: 50px; text-align: center;">
            <h1 style="color: #e74c3c;">🌍 404 - Not Found</h1>
            <p>The path <code>{request.url.path}</code> does not exist in GreenChain.</p>
            <p><a href="/render" style="color: #2ecc71; text-decoration: none; font-weight: bold;">Go to Dashboard →</a></p>
          </body>
        </html>
        """
    )

# Multi-tenant session state — LRU-capped to prevent OOM on HF Spaces
from collections import OrderedDict

MAX_SESSIONS = 100
sessions: OrderedDict[str, GreenChainEnv] = OrderedDict()

def get_env(session_id: str = "default") -> GreenChainEnv:
    if session_id in sessions:
        sessions.move_to_end(session_id)  # Mark as recently used
        return sessions[session_id]
    if len(sessions) >= MAX_SESSIONS:
        sessions.popitem(last=False)  # Evict least-recently-used
    new_env = GreenChainEnv()
    sessions[session_id] = new_env
    return new_env

@app.post("/step")
async def step(action: Action, session_id: str = "default") -> Dict[str, Any]:
    env = get_env(session_id)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.post("/reset")
async def reset(session_id: str = "default") -> Dict[str, Any]:
    env = get_env(session_id)
    obs = env.reset()
    return {"observation": obs.dict()}

@app.get("/state")
async def state(session_id: str = "default") -> Dict[str, Any]:
    env = get_env(session_id)
    obs = env.state()
    return {"observation": obs.dict()}

@app.get("/tasks")
async def get_tasks(session_id: str = "default") -> Dict[str, Any]:
    env = get_env(session_id)
    return {
        "tasks": [
            {
                "id": "task_1_swap",
                "difficulty": "easy",
                "description": "Replace the active Tier-1 supplier with a lower-carbon alternative without exceeding 10% budget increase.",
                "has_grader": True,
                "grader": "/grader?task_id=task_1_swap"
            },
            {
                "id": "task_2_route",
                "difficulty": "medium",
                "description": "Reroute an active shipment to reduce emissions by 15% while keeping perishability above 0.8.",
                "has_grader": True,
                "grader": "/grader?task_id=task_2_route"
            },
            {
                "id": "task_3_audit",
                "difficulty": "hard",
                "description": "Identify all fraudulent anomalies in the logistics manifest using precision/recall scoring.",
                "has_grader": True,
                "grader": "/grader?task_id=task_3_audit"
            }
        ],
        "action_schema": Action.schema(),
        "current_observation": env.state().dict()
    }

@app.api_route("/grader", methods=["GET", "POST"])
async def grader(task_id: str, session_id: str = "default") -> Dict[str, Any]:
    from .tasks import evaluate_task
    env = get_env(session_id)
    current_state = env.state()
    score = evaluate_task(task_id, current_state)
    return {"score": score, "task_id": task_id}

@app.get("/baseline")
async def baseline(session_id: str = "default") -> Dict[str, Any]:
    from .tasks import evaluate_task
    env = get_env(session_id)
    
    # Task 1
    env.reset()
    act1 = Action(action_type="SwitchSupplier", parameters={"new_supplier_id": "SUP-2"})
    env.step(act1)
    task_1 = evaluate_task("task_1_swap", env.state())

    # Task 2
    env.reset()
    act2 = Action(action_type="RerouteShipment", parameters={"shipment_id": "SHIP-1", "route_hubs": ["HUB-1", "HUB-3"]})
    env.step(act2)
    task_2 = evaluate_task("task_2_route", env.state())

    # Task 3
    env.reset()
    # Pull flags from ground truth dynamically since they are randomized
    truth_flags = list(env.ground_truth_fraud_flags)
    act3 = Action(action_type="FlagForAudit", parameters={"fraud_flags": truth_flags})
    env.step(act3)
    task_3 = evaluate_task("task_3_audit", env.state())

    env.reset() # Cleanup dirty state after baseline test

    return {
        "scores": {
            "task_1_swap": task_1,
            "task_2_route": task_2,
            "task_3_audit": task_3
        },
        "total_score": task_1 + task_2 + task_3
    }

@app.get("/debug")
async def debug(session_id: str = "default") -> Dict[str, Any]:
    """Exposes ground truth for external smoke tests when randomness makes tests difficult."""
    env = get_env(session_id)
    return {"ground_truth_fraud_flags": list(env.ground_truth_fraud_flags)}

@app.get("/render", response_class=HTMLResponse)
async def render(session_id: str = "default"):
    env = get_env(session_id)
    obs = env.state()
    active_html = "".join(f"<li>{s.id} (Cost: ${s.cost}, Carbon: {s.carbon_index})</li>" for s in obs.active_suppliers)
    avail_html = "".join(f"<li>{s.id} (Cost: ${s.cost}, Carbon: {s.carbon_index})</li>" for s in obs.available_suppliers)
    ship_html = "".join(f"<li>{s.id} (Route: {' -> '.join(s.route)}, Perishability: {s.perishability:.2f})</li>" for s in obs.active_shipments)
    flags_html = "".join(f"<li>✅ {f}</li>" for f in obs.identified_flags)
    html = f"""
    <html>
      <head><title>GreenChain State ({session_id})</title></head>
      <body style="font-family: sans-serif; padding: 20px;">
        <h1>🌍 Project GreenChain Dashboard</h1>
        <h3>Session: {session_id} | Step: {obs.step_count}</h3>
        <p><b>💰 Budget:</b> ${obs.budget_remaining:,.2f}</p>
        <p><b>☁️ Carbon Footprint:</b> {obs.current_carbon_footprint:,.2f} tons</p>
        <p><b>🌱 Sustainability Score:</b> {obs.sustainability_score:.2f}</p>
        <p><b>🚨 Greenwashing Risk:</b> {obs.greenwashing_risk_score}</p>
        <p><b>🧠 Total Audit Guesses:</b> {obs.total_guesses}</p>
        <hr/>
        <h3>Active Suppliers</h3>
        <ul>{active_html}</ul>
        <h3>Available Suppliers Directory</h3>
        <ul>{avail_html}</ul>
        <h3>Active Shipments</h3>
        <ul>{ship_html}</ul>
        <h3>Identified Fraud Flags ({len(obs.identified_flags)}/3)</h3>
        <ul>{flags_html}</ul>
      </body>
    </html>
    """
    return html

def main():
    """Entry point for the OpenEnv server command."""
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()
