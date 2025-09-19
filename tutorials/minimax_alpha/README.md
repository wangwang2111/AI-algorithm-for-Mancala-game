# Mancala Search Tutorial — Minimax & Alpha–Beta (State-Based)

Hands-on notebooks and scripts that teach **Minimax** and **Alpha–Beta pruning** on your Mancala engine.  

## What you’ll learn
- Representing Mancala with a **state dict**:
```python
  state = {
    "pits": [[int]*6, [int]*6],   # row 0 = player_1, row 1 = player_2
    "stores": [int, int],         # stores[0] for P0, stores[1] for P1
    "current_player": 0 | 1
  }
````

* Implementing **Minimax** and **Alpha–Beta** with your engine API:

  * `new_game()`, `legal_actions(state)`, `step(state, action)`, `evaluate(state)`
* Handling Mancala’s **extra turn**: when the same player moves again, **do not decrement depth**.
* Measuring node expansions; seeing how **move ordering** reduces work.

## Folder structure

```
tutorials/
  minimax_alpha/
    README.md                # ← this file
    01_minimax.ipynb         # guided notebook: plain Minimax
    02_alpha_beta.ipynb      # guided notebook: Alpha–Beta + ordering
    tutorial_minimax.py      # script version of 01
    tutorial_alphabeta.py    # script version of 02
```

## Prerequisites

* Python 3.9+
* Your engine at `src/mancala_ai/engine/core.py` exporting:

  * `new_game`, `legal_actions`, `step`, `evaluate`

> The notebooks/scripts only rely on the **standard library** plus your engine.
> If you use a virtualenv: `python -m venv .venv && . .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows).

## Quickstart (notebooks)

From the **repo root**:

```bash
pip install jupyterlab  # if needed
jupyter lab
```

Open:

* `tutorials/minimax_alpha/01_minimax.ipynb`
* `tutorials/minimax_alpha/02_alpha_beta.ipynb`

Each notebook:

* sets `sys.path` to include `src/`
* imports your engine
* runs examples you can modify

## Quickstart (scripts)

From the **repo root**:

```bash
python tutorials/minimax_alpha/tutorial_minimax.py
python tutorials/minimax_alpha/tutorial_alphabeta.py
```

They print the chosen move and node counts, and include optional experiments in comments.

## Key ideas covered

* **Root-perspective scoring:** when evaluating internal nodes, temporarily set
  `state["current_player"] = root_player_idx` before calling `evaluate(state)` to keep scores consistent.
* **Extra turn rule:** after `step(state, action) → next_state`, if the next state’s `current_player`
  equals the previous, depth **does not** decrease (still the same ply).
* **Move ordering:** try one-ply lookahead or simple heuristics (e.g., moves that land in the store first)
  to boost Alpha–Beta pruning.

## Suggested exercises

* Add a simple **move ordering** to Minimax; compare node counts.
* Add a **transposition table** keyed by:

  ```python
  key = (
    tuple(state["pits"][0]), tuple(state["pits"][1]),
    state["stores"][0], state["stores"][1],
    state["current_player"]
  )
  ```