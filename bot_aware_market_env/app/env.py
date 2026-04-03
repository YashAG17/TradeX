from typing import Optional, Dict, Any, Tuple
import math

from .models import Observation, Action, Reward, Info, ActionEnum
from .market import AMM
from .trade_generator import TradeGenerator
from .reward import RewardEngine

class MarketSurveillanceEnv:
    def __init__(self, task_id: int = 3, **reward_kwargs):
        self.task_id = task_id
        self.amm = AMM()
        self.trade_gen = TradeGenerator(task_id)
        self.reward_engine = RewardEngine(**reward_kwargs)
        
        self.max_steps = 100
        self.current_step = 0
        
        # State tracking for observation
        self.recent_trades = []
        self.current_trade = None
        
    def reset(self) -> Tuple[Observation, Info]:
        self.amm = AMM()
        self.trade_gen = TradeGenerator(self.task_id)
        self.current_step = 0
        self.recent_trades = []
        self.current_trade = self.trade_gen.next_trade()
        
        return self._get_obs(), self._get_info()
        
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        if self.current_step >= self.max_steps:
            return self._get_obs(), Reward(total_reward=0.0, components={}), True, self._get_info()
            
        action_val = action.action.value
        trade = self.current_trade
        
        # Calculate derived severity (e.g. slippage impact)
        _, slippage_pct = self.amm.estimate_out(trade.amount_in, trade.is_x_to_y)
        severity = 1.0 + (slippage_pct / 10.0) # slightly increase severity based on size
        
        # Calculate reward based on agent action and ground truth
        rew_val, rew_comps = self.reward_engine.calculate(
            action_val, 
            trade.is_suspicious,
            severity=severity
        )
        
        reward = Reward(total_reward=rew_val, components=rew_comps)
        
        # Execute trade if allowed
        if action_val in ["ALLOW", "MONITOR", "FLAG"]: # FLAG still allows it to go through, just marks it. BLOCK stops it
            self.amm.swap(trade.amount_in, trade.is_x_to_y)
            self.recent_trades.append(trade)
            if len(self.recent_trades) > 10:
                self.recent_trades.pop(0)
                
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Prepare next step
        if not done:
            self.current_trade = self.trade_gen.next_trade()
            
        return self._get_obs(), reward, done, self._get_info()
        
    def state(self) -> Observation:
        return self._get_obs()
        
    def _get_obs(self) -> Observation:
        current_price = self.amm.get_price()
        liquidity_k = self.amm.k
        
        if not self.recent_trades or not self.current_trade:
            return Observation(
                current_price=current_price,
                liquidity_k=liquidity_k,
                trade_freq_recent=0.0,
                avg_trade_size=0.0,
                max_trade_size=0.0,
                slippage_impact=0.0,
                time_gaps_mean=0.0,
                burst_indicator=0.0,
                pattern_indicator=0.0,
                recent_short_term_profit=0.0,
                trader_suspiciousness_score=0.0
            )
            
        # Calculate indicators based on recent trades & current trade proposed
        trade_freq = 10.0 / sum(t.time_delta for t in self.recent_trades[-10:]) if len(self.recent_trades) > 0 and sum(t.time_delta for t in self.recent_trades[-10:]) > 0 else 0.0
        avg_size = sum(t.amount_in for t in self.recent_trades) / len(self.recent_trades) if self.recent_trades else 0.0
        max_size = max(t.amount_in for t in self.recent_trades) if self.recent_trades else 0.0
        mean_gaps = sum(t.time_delta for t in self.recent_trades) / len(self.recent_trades) if self.recent_trades else 0.0
        
        _, slippage_impact = self.amm.estimate_out(self.current_trade.amount_in, self.current_trade.is_x_to_y)
        
        # Simple derived signals
        burst_idx = 1.0 if (self.current_trade.time_delta < 1.0 and trade_freq > 2.0) else 0.0
        pattern_idx = 1.0 if (slippage_impact > 5.0 and self.current_trade.time_delta < 2.0) else 0.0
        susp_score = (burst_idx + pattern_idx) / 2.0
        
        return Observation(
            current_price=current_price,
            liquidity_k=liquidity_k,
            trade_freq_recent=trade_freq,
            avg_trade_size=avg_size,
            max_trade_size=max(max_size, self.current_trade.amount_in),
            slippage_impact=slippage_impact,
            time_gaps_mean=mean_gaps,
            burst_indicator=burst_idx,
            pattern_indicator=pattern_idx,
            recent_short_term_profit=0.0,
            trader_suspiciousness_score=susp_score
        )
        
    def _get_info(self) -> Info:
        return Info(
            step_metrics={"current_step": self.current_step},
            is_suspicious_ground_truth=self.current_trade.is_suspicious if self.current_trade else False
        )
