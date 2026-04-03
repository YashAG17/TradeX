from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ActionEnum(str, Enum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"
    MONITOR = "MONITOR"

class Observation(BaseModel):
    current_price: float
    liquidity_k: float 
    trade_freq_recent: float
    avg_trade_size: float
    max_trade_size: float
    slippage_impact: float
    time_gaps_mean: float
    burst_indicator: float
    pattern_indicator: float
    recent_short_term_profit: float
    trader_suspiciousness_score: float

class Action(BaseModel):
    action: ActionEnum

class Reward(BaseModel):
    total_reward: float
    components: Dict[str, float] = Field(default_factory=dict)

class Info(BaseModel):
    step_metrics: Dict[str, Any] = Field(default_factory=dict)
    is_suspicious_ground_truth: bool
