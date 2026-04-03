import random
from dataclasses import dataclass

@dataclass
class TradeEvent:
    amount_in: float
    is_x_to_y: bool
    time_delta: float  # seconds since last trade
    is_suspicious: bool
    suspicious_type: str = "none" # "burst", "pattern"

class TradeGenerator:
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.time = 0.0
        self.step_idx = 0
        
    def next_trade(self) -> TradeEvent:
        # returns the next trade to process.
        # This will be generated dynamically based on the task type.
        
        # Decide if this step should inject suspicious behavior based on random chance
        is_suspicious = False
        susp_type = "none"
        
        # Difficulty configuration
        if self.task_id == 1:
            # Easy: High probability of very obvious bursts
            prob_suspicious = 0.3
            normal_gap = (5.0, 30.0)
            normal_size = (10.0, 100.0)
        elif self.task_id == 2:
            # Medium: Lower probability, patterns mixed with larger normal trades
            prob_suspicious = 0.15
            normal_gap = (2.0, 20.0)
            normal_size = (50.0, 600.0)
        else:
            # Hard: Very noisy normal trades, low signal-to-noise ratio
            prob_suspicious = 0.1
            normal_gap = (0.5, 30.0)
            normal_size = (10.0, 1000.0)
            
        if random.random() < prob_suspicious:
            is_suspicious = True
            if self.task_id == 1:
                susp_type = "burst"
            elif self.task_id == 2:
                susp_type = "pattern"
            else: # task 3
                susp_type = random.choice(["burst", "pattern"])
                
        # Normal generation
        if not is_suspicious:
            return TradeEvent(
                amount_in=random.uniform(*normal_size),
                is_x_to_y=random.choice([True, False]),
                time_delta=random.uniform(*normal_gap),
                is_suspicious=False
            )
            
        # Suspicious generation
        if susp_type == "burst":
            # high frequency, short time delta, repeated sizes
            return TradeEvent(
                amount_in=random.choice([10.0, 50.0, 100.0]), # Repetitive sizes
                is_x_to_y=random.choice([True, False]),
                time_delta=random.uniform(0.01, 0.2), # Extreme short gap
                is_suspicious=True,
                suspicious_type="burst"
            )
        else: # pattern (sandwich/front-run style)
            # High amount, causing slippage, short time delta
            return TradeEvent(
                amount_in=random.uniform(1000.0, 5000.0), # Large size for impact
                is_x_to_y=random.choice([True, False]),
                time_delta=random.uniform(0.1, 1.5),
                is_suspicious=True,
                suspicious_type="pattern"
            )
