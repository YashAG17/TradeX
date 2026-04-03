class RewardEngine:
    def __init__(self, 
                 correct_suspicious_detection: float = 8.0,
                 correct_allow_normal: float = 2.0,
                 false_positive: float = -6.0,
                 false_negative: float = -10.0,
                 overblocking_penalty: float = -2.0,
                 severity_multiplier: float = 1.5):
        self.cfg = {
            "correct_suspicious_detection": correct_suspicious_detection,
            "correct_allow_normal": correct_allow_normal,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "overblocking_penalty": overblocking_penalty,
            "severity_multiplier": severity_multiplier
        }
    
    def calculate(self, action_enum: str, is_suspicious: bool, severity: float = 1.0):
        reward = 0.0
        components = {}
        
        # We classify BLOCK, FLAG as restrictive actions
        # ALLOW, MONITOR as non-restrictive
        is_restrictive_action = action_enum in ["BLOCK", "FLAG"]
        
        if is_suspicious:
            if is_restrictive_action:
                # Correctly identified suspicious activity
                base = self.cfg["correct_suspicious_detection"]
                reward += base * severity
                components["correct_block"] = base * severity
            else:
                # Missed suspicious activity
                base = self.cfg["false_negative"]
                reward += base * severity
                components["missed_bot"] = base * severity
        else:
            if is_restrictive_action:
                # False positive / penalized normal user
                reward += self.cfg["false_positive"]
                components["false_positive"] = self.cfg["false_positive"]
                if action_enum == "BLOCK":
                    reward += self.cfg["overblocking_penalty"]
                    components["overblocking"] = self.cfg["overblocking_penalty"]
            else:
                # Correctly allowed normal behavior
                reward += self.cfg["correct_allow_normal"]
                components["correct_allow"] = self.cfg["correct_allow_normal"]
                
        return reward, components
