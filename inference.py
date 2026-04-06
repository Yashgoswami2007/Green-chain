"""
Inference Script for GreenChain
===================================
MANDATORY
- Strictly follows hackathon constraints: Uses OpenAI python client explicitly.
- Uses API_BASE_URL, MODEL_NAME, and HF_TOKEN/API_KEY from environment variables.
- Named inference.py and placed in root directory.
"""

import os
import re
import sys
import json
import textwrap
import requests
from typing import List, Optional, Dict

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "default-model"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # Optional: used if using from_docker_image()
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "greenchain_audit")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "greenchain")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS = 10
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "DoNothing()"

ACTION_PREFIX_RE = re.compile(
    r"^(action|next action)\s*[:\-]\s*",
    re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"[A-Za-z_]+\s*\(.*\)", re.DOTALL)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a GreenChain supply chain auditing environment.
    Reply with exactly one action string.
    The action must be a valid GreenChain command such as:
    - SwitchSupplier('supplier_id')
    - RerouteShipment('shipment_id', ['HUB-X', 'HUB-Y'])
    - FlagForAudit('fraud flag 1', 'fraud flag 2', ...)
    - DoNothing()
    
    Use single quotes around string arguments.
    Evaluate the current budget, carbon footprint, and audit manifest to optimize operations.
    If you are unsure, respond with DoNothing().
    Do not include explanations or additional text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])


def build_user_prompt(step: int, observation: dict, history: List[str]) -> str:
    # Summarize state
    obs_text = json.dumps(observation, indent=2)

    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Observation context:
        {obs_text}

        Previous steps:
        {build_history_lines(history)}

        Review the 'audit_document' for anomalies and check 'available_suppliers' for lower-carbon options.
        Reply with exactly one Python-like action string.
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    lines = response_text.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = match.group(0).strip()
            action = re.sub(r"\s+", " ", action)
            return action

    match = ACTION_PATTERN.search(response_text)
    if match:
        action = match.group(0).strip()
        action = re.sub(r"\s+", " ", action)
        return action

    return FALLBACK_ACTION

# Helper parsers used to inject into an eval context for safe and perfect LLM string unpacking
def SwitchSupplier(new_id):
    return {"action_type": "SwitchSupplier", "parameters": {"new_supplier_id": new_id}}
    
def RerouteShipment(shipment_id, route_list):
    return {"action_type": "RerouteShipment", "parameters": {"shipment_id": shipment_id, "route_hubs": route_list}}
    
def FlagForAudit(*flags):
    return {"action_type": "FlagForAudit", "parameters": {"fraud_flags": list(flags)}}
    
def DoNothing():
    return {"action_type": "DoNothing"}


def execute_action_in_env(action_str: str) -> dict:
    # Safely evaluate the action string to get the JSON payload
    action_payload = None
    safe_locals = {
        "SwitchSupplier": SwitchSupplier,
        "RerouteShipment": RerouteShipment,
        "FlagForAudit": FlagForAudit,
        "DoNothing": DoNothing,
        "noop": DoNothing # fallback alias
    }
    
    try:
        # evaluate string natively mapping to the API format
        action_payload = eval(action_str, {"__builtins__": {}}, safe_locals)
    except Exception as e:
        action_payload = {"action_type": "DoNothing"}
        
    try:
        response = requests.post(f"{ENV_URL}/step", json=action_payload, timeout=10)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"done": True, "reward": {"value": 0.0}, "observation": {}, "info": {"error": "Connection failed"}}


def main() -> None:
    try:
        requests.get(f"{ENV_URL}/state", timeout=5)
    except requests.exceptions.ConnectionError:
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_res = requests.post(f"{ENV_URL}/reset", timeout=10).json()
        observation = reset_res.get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            user_prompt = build_user_prompt(step, observation, history)
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc: 
                response_text = FALLBACK_ACTION

            action_str = parse_model_action(response_text)

            step_res = execute_action_in_env(action_str)
            observation = step_res.get("observation", {})
            reward_val = step_res.get("reward", {}).get("value", 0.0)
            done = step_res.get("done", False)
            info = step_res.get("info", {})
            error_val = info.get("error")
            
            rewards.append(reward_val)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error_val)

            history_line = f"Step {step}: {action_str} -> reward {reward_val:+.2f}"
            if error_val:
                history_line += f" ERROR: {error_val}"
            history.append(history_line)

            if done:
                break

        total_reward = sum(rewards)
        # Normalize score assuming total reward might be around 10.0 for max items
        score = total_reward / 10.0 if total_reward > 0.0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
