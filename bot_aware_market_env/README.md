# Bot-aware Market Surveillance in Simulated AMM Trading

## Overview
This is an OpenEnv-compatible reinforcement learning environment where an agent conducts market surveillance over a simulated Automated Market Maker (AMM). The goal is to detect and respond to suspicious bot-like behavior (such as high-frequency bursts and sandwich/front-run patterns) while maintaining normal trading conditions.

## Motivation & Real-world Relevance
With the continued rise of decentralized finance (DeFi), AMMs are regularly targeted by automated bots doing MEV (Maximal Extractable Value) extraction. Traditional static thresholds eventually fail against evolving bots. RL provides an adaptive approach to learn and classify sophisticated manipulation strategies on the fly without halting legitimate activity. 

## Environment Design

### State (Observation)
Includes current AMM parameters (price, liquidity) and calculated temporal features like trade frequency, slippage impact, and moving averages of trade dimensions.

### Action
The agent outputs one of four discrete actions:
* `ALLOW`: Process the trade normally.
* `FLAG`: Process the trade but mark it as suspicious.
* `BLOCK`: Prevent the trade from executing.
* `MONITOR`: Same effect as `ALLOW` but signals intention to collect more data.

### Reward Function
The environment evaluates decisions against ground truth (generated synthetically). 
Hyper-parameters allow config of reward components, standard defaults are:
- Correct Suspicious Detection: +8
- Correct Allow Normal: +2
- False Positive (blocking normal): -6
- False Negative (missing bot): -10
- Overblocking Penalty: -2 

## Tasks 
Task 1: Burst Anomaly Detection
Task 2: Pattern-based Manipulation Detection
Task 3: Full Market Surveillance

## Graders
Metrics emphasize detection accuracy and lack of harm to normal users. Uses a geometric synthesis of Recall and Inverse False Positive Rate. Returns a continuous score between `0.0` and `1.0`.

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run baseline:
```bash
python -m scripts.run_baseline
```
