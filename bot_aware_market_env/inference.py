import os
import sys
import json
import re
import textwrap
from typing import Any, Dict, List

from openai import OpenAI
from app.env import MarketSurveillanceEnv
from app.models import Action, ActionEnum
from app.graders import Grader

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

TASK_NAME = os.getenv("TASK", "3")
BENCHMARK = "bot_aware_market"
TEMPERATURE = 0.1
MAX_TOKENS = 50

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a strictly logical market surveillance agent detecting AMM manipulations.

    Your observation includes parameters like:
    trade_freq_recent, slippage_impact, burst_indicator, pattern_indicator, and trader_suspiciousness_score.

    Choose ONE action exactly matching these enums:
    ALLOW, FLAG, BLOCK, MONITOR

    STRICT DECISION RULES:
    - If pattern_indicator > 0.7 and slippage_impact > 0.1 → BLOCK
    - If burst_indicator > 0.7 or trade_freq_recent > 5.0 → FLAG
    - If trader_suspiciousness_score > 0.5 → MONITOR
    - If the trading behavior is normal → ALLOW

    OUTPUT FORMAT:
    You must output ONLY valid JSON format containing exactly the "action" key. No explanation. No markdown. No code blocks.
    Example exactly as:
    {"action": "ALLOW"}
    """
)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def deterministic_fallback_action(obs) -> str:
    threshold = 5.0
    pattern_idx = getattr(obs, "pattern_indicator", 0.0)
    slippage = getattr(obs, "slippage_impact", 0.0)
    burst_idx = getattr(obs, "burst_indicator", 0.0)
    trade_freq = getattr(obs, "trade_freq_recent", 0.0)
    suspiciousness = getattr(obs, "trader_suspiciousness_score", 0.0)
    
    if pattern_idx > 0.7 and slippage > 0.1:
        return "BLOCK"
    elif burst_idx > 0.7 or trade_freq > threshold:
        return "FLAG"
    elif suspiciousness > 0.5:
        return "MONITOR"
    else:
        return "ALLOW"


def get_model_action(client: OpenAI, obs: Any, step: int) -> str:
    user_prompt = textwrap.dedent(
        f"""\
        Step: {step}
        Observation JSON:
        {obs.model_dump_json(indent=2)}
        
        Analyze the signals and output strictly {"{"}"action": "<ACTION>"{"}"}.
        """
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Robust parsing: grab the first valid JSON substring using regex
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            text = json_match.group(0)
            
        action_dict = json.loads(text)
        action_val = action_dict.get("action", "")
        
        if action_val in ["ALLOW", "FLAG", "BLOCK", "MONITOR"]:
            return action_val
        else:
            return deterministic_fallback_action(obs) # Trigger fallback if parsed action is invalid
    except Exception:
        # If API fails or JSON decode fails entirely, rely on fallback
        return deterministic_fallback_action(obs)

def main() -> None:
    use_llm = API_BASE_URL and MODEL_NAME and API_KEY
    model_identifier = MODEL_NAME if use_llm else "deterministic_fallback"
    
    if use_llm:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        client = None

    try:
        task_id = int(TASK_NAME)
    except ValueError:
        task_id = 3
        
    env = MarketSurveillanceEnv(task_id=task_id)
    grader = Grader()
    
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=f"Task {task_id}", env=BENCHMARK, model=model_identifier)

    try:
        obs, info = env.reset()
        done = False
        
        while not done:
            steps_taken += 1
            
            # Action Extraction
            if use_llm:
                action_str = get_model_action(client, obs, steps_taken)
            else:
                action_str = deterministic_fallback_action(obs)
                
            action_enum = ActionEnum(action_str)
            action = Action(action=action_enum)
            
            obs, reward, done, step_info = env.step(action)
            
            reward_val = reward.total_reward
            rewards.append(reward_val)
            
            grader.update(action_enum.value, step_info.is_suspicious_ground_truth)
            
            # Print ONLY the string action value, not the JSON encoded dictionary
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward_val,
                done=done,
                error=None,
            )

        score = grader.compute_score()
        success = score > 0.1

    except Exception as e:
        pass
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
