import sys
import os

# Add parent directory to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.env import MarketSurveillanceEnv
from app.models import Action, ActionEnum
from app.graders import Grader

def run_baseline(task_id: int):
    # Initializes environment and grader
    env = MarketSurveillanceEnv(task_id=task_id)
    grader = Grader()
    
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    
    while not done:
        # Rule-based policy
        action_val = ActionEnum.ALLOW
        
        if obs.burst_indicator > 0.5 or obs.trade_freq_recent > 5.0:
            action_val = ActionEnum.FLAG
            
        if obs.pattern_indicator > 0.5:
            action_val = ActionEnum.BLOCK
            
        action = Action(action=action_val)
        
        obs, reward, done, step_info = env.step(action)
        episode_reward += reward.total_reward
        grader.update(action_val.value, step_info.is_suspicious_ground_truth)
        
    score = grader.compute_score()
    return score, episode_reward

def main():
    print("=== Running Baseline Evaluator ===")
    
    total_score = 0
    for task_id in [1, 2, 3]:
        score, ep_rew = run_baseline(task_id)
        print(f"Task {task_id} | Score: {score:.3f} | Total Reward: {ep_rew:.2f}")
        total_score += score
        
    print(f"Average Score: {total_score / 3.0:.3f}")
    
if __name__ == "__main__":
    main()
