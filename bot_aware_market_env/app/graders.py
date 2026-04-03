class Grader:
    def __init__(self):
        # We track metrics over the episode
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
        
    def update(self, action_enum: str, is_suspicious: bool):
        is_restrictive = action_enum in ["BLOCK", "FLAG"]
        
        if is_suspicious and is_restrictive:
            self.true_positive += 1
        elif is_suspicious and not is_restrictive:
            self.false_negative += 1
        elif not is_suspicious and is_restrictive:
            self.false_positive += 1
        else:
            self.true_negative += 1
            
    def compute_score(self) -> float:
        total_suspicious = self.true_positive + self.false_negative
        total_normal = self.true_negative + self.false_positive
        
        recall = self.true_positive / total_suspicious if total_suspicious > 0 else 1.0
        fp_rate = self.false_positive / total_normal if total_normal > 0 else 0.0
        
        # Penalize false positives heavily, reward recall
        score = (recall * 0.7) + ((1.0 - fp_rate) * 0.3)
        return max(0.0, min(1.0, score))
