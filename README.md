# 🧠 AI Algorithms for the Mancala Game

This project explores the design and development of intelligent agents for the game of **Mancala (Kalah variant)** using both classical search-based algorithms and modern deep reinforcement learning techniques. The goal is to build a flexible, modular AI framework that enables strategic decision-making through various approaches.

## 📁 Project Structure

```
AI-algorithm-for-Mancala-game/
├── ai/
│   ├── __init__.py
│   ├── rules.py                 # Game rules and logic (legal moves, termination)
│   ├── minimax.py               # Basic Minimax agent
│   ├── alpha_beta.py            # Minimax with alpha-beta pruning
│   ├── advanced_heuristic.py    # Minimax with domain-specific heuristics
│   ├── MCTS.py                  # Monte Carlo Tree Search agent
├── dqn.py                       # Deep Q-Network agent
├── save/
│   └── policy_final.pt          # Trained DQN model weights
├── runs/                        # TensorBoard training logs
├── dqn_wrapper.py               # Flask-compatible wrapper for DQN inference
├── server.py                    # RESTful API to interact with AI agents
├── requirements.txt             # Python dependency file
├── simulations.py               # Match simulations between AI strategies
├── evaluations/                        # Simulations Result Logs by simulations.py
├── analyze_results/                    # Analyze and Visualize the simulated results in evaluations/
└── README.md                    # Project overview and usage guide
```

## 🧠 AI Strategy Selection

This project includes multiple AI agents to enable comparative evaluation and hybrid strategies:

| Agent Type           | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Minimax`            | Classic adversarial search assuming perfect play.                           |
| `Alpha-Beta`         | Optimized Minimax with pruning for deeper search trees.                     |
| `Advanced Heuristic` | Alpha-Beta enhanced with domain-specific heuristics (e.g., extra turn, store control). |
| `MCTS`               | Stochastic search that balances exploration and exploitation.               |
| `DQN`                | Model-free Deep Reinforcement Learning agent trained via self-play.         |

## 🧪 Running Simulations

To simulate matches between any two agents:

```bash
python simulations.py
```

Simulations will produce performance statistics (e.g., win rates, draw rates) across agent matchups.

---

## 🌐 Mancala Web Interface

### 1. Run the Flask API backend:
```bash
python server.py
```

### 2. Serve the frontend (requires `mancala.html`):
```bash
python -m http.server 8000
```

Then open in your browser:
```
http://localhost:8000/mancala.html
```

You can choose agents like `alpha_beta`, `advanced`, `dqn`, `MCTS`, etc., in the UI.

---

## 🧠 Training the DQN Agent

To train the Deep Q-Network agent from scratch:

```bash
python dqn.py
```

This will store training logs in `runs/` and save the best policy to `save/policy_final.pt`.

### 📈 Monitor Training Progress

You can visualize training metrics (e.g., episode rewards, losses) using TensorBoard:

```bash
python -m tensorboard.main --logdir=runs
```

Then visit:
```
http://localhost:6006/
```

---

## 📦 Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda activate mancala-ai
```

## 🤝 Credits

Developed by:
- **Dang Quang Nguyen**
Trent University – AMOD 5310H: Artificial Intelligence