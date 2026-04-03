import sys
import os

# Add parent directory to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.env import MarketSurveillanceEnv
from app.models import Action, ActionEnum

def main():
    print("Testing Environment Initialization...")
    env = MarketSurveillanceEnv(task_id=3)
    
    obs, info = env.reset()
    assert obs is not None
    assert "current_price" in obs.model_dump()
    
    print("Testing Environment Step...")
    obs, reward, done, info = env.step(Action(action=ActionEnum.ALLOW))
    
    assert obs is not None
    assert reward is not None
    assert isinstance(done, bool)
    assert info is not None
    
    print("Validations passed!")

if __name__ == "__main__":
    main()
