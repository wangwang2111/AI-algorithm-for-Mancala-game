# Mancala AI — End-to-End Game AI (Flask API + Animated UI)

Play Mancala against multiple AI agents (DQN, Minimax, Alpha-Beta, MCTS, Advanced Heuristic) with a smooth, animated web UI. The project demonstrates an end-to-end ML product: REST API, realtime UI with stone animations, and a simple model registry for hot-swapping policies.

## ✨ Features

* **Agents:** `dqn`, `minimax`, `alpha_beta`, `mcts`, `advanced` (aliases: `advanced_heuristic`, `adv`, `ah`)
* **Animated sowing:** real stones move and persist; counts update; hover tooltips show values
* **REST API:** `/api/health`, `/api/newgame`, `/api/apply`, `/api/move`
* **Model registry:** drop `policy.pt` + `meta.json` into `model_registry/latest/`
* **Dockerized:** `docker compose up` runs API + UI; UI proxies `/api/*` to the API
* **Local dev friendly:** run API and UI separately; optional `env.js` to point to API

---

## 📦 Project Structure

```
mancala-ai/
├─ src/
│  └─ mancala_ai/
│     ├─ engine/
│     │  └─ core.py              # game rules (initialize_board, make_move, etc.)
│     ├─ agents/
│     │  ├─ dqn.py               # DQN wrapper (loads training.dqn.DQNAgent lazily)
│     │  ├─ minimax.py           # simple_minimax(...) or choose_move(...)
│     │  ├─ alpha_beta.py        # minimax_alpha_beta(...) or choose_move(...)
│     │  ├─ MCTS.py              # mcts_decide(...)
│     │  └─ advanced_heuristic.py# advanced_heuristic_minimax(...)
│     ├─ api/
│     │  ├─ app.py               # Flask app factory (create_app)
│     │  └─ routes.py            # /api endpoints
│     ├─ io/
│     │  └─ registry.py          # pick_action(), current_meta()
│     ├─ training/
│     │  └─ dqn.py               # DQNAgent (used by agents/dqn.py)
│     ├─ utils/
│     │  └─ features.py          # state encoders, etc.
│     └─ ...
├─ ui_static/
│  ├─ index.html
│  ├─ styles.css
│  └─ js/
│     ├─ app.js                  # game loop + animations
│     ├─ api.js                  # calls /api/*
│     └─ components/
│        ├─ Board.js
│        ├─ Controls.js
│        └─ Sound.js
├─ model_registry/
│  └─ latest/
│     ├─ policy.pt
│     └─ meta.json               # {"version":"v0.1","win_rate":0.83,"trained_at":"..."}
├─ docker/
│  ├─ api.Dockerfile             # Flask API (Gunicorn)
│  └─ ui.Dockerfile              # Nginx static UI on port 5173 with /api proxy
├─ docker-compose.yml
├─ requirements.txt
├─ src/wsgi.py                   # wsgi:app shim for Gunicorn
└─ README.md
```

---

## 🚀 Quickstart (Docker)

**Prereqs:** Docker Desktop (Windows/macOS) or Docker Engine (Linux).

```bash
docker compose up --build
```

* UI: [http://localhost:5173](http://localhost:5173)
* API (direct): [http://localhost:8000/api/health](http://localhost:8000/api/health)

**How it’s wired:** the Nginx UI container serves static files on **5173** and **proxies `/api/*` to the API** service on **8000**, so the browser uses same-origin URLs like `/api/move`.

### Hot-swap model

Drop new weights and metadata into `./model_registry/latest/` (mounted read-only into the container). If your DQN wrapper caches the model, restart the API service to reload:

```bash
docker compose restart api
```

---

## 🧰 Local Development (without Docker)

### 1) Backend (Flask API)

```bash
python -m venv .venv
. .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
# If you don't need GPU, use CPU torch in requirements.txt: torch==2.3.1
pip install -r requirements.txt

# Run the API (Gunicorn)
python -m gunicorn -w 2 -k gthread -b 0.0.0.0:8000 wsgi:app
# Health check
curl http://localhost:8000/api/health
```

> If you see cuDNN/CUDA issues locally, stick to CPU Torch (`torch==2.3.1`) or let the code’s lazy import fall back to non-DQN agents.

### 2) Frontend (static UI)

Serve the `ui_static/` folder on **5173**. Two simple options:

**Python:**

```bash
python -m http.server 5173 -d ui_static
```

**Node (http-server):**

```bash
npx http-server ui_static -p 5173 -c-1
```

## 🔌 API Reference

Base URL (Docker UI via proxy): `/api`
Base URL (direct API): `http://localhost:8000/api`

### `GET /api/health`

Health info and model meta.

**Response**

```json
{
  "status": "ok",
  "model": { "version": "v0.1", "win_rate": 0.83, "trained_at": "..." }
}
```

### `POST /api/newgame`

Starts a new game.

**Response**

```json
{
  "state": {
    "pits": [[4,4,4,4,4,4],[4,4,4,4,4,4]],
    "stores": [0,0],
    "current_player": 0
  }
}
```

### `POST /api/apply`

Apply a **human** move (no AI).
**Body**

```json
{ "state": { ... }, "action": 3 }
```

**Response**

```json
{
  "next_state": { ... },
  "reward": 0.0,
  "done": false
}
```

### `POST /api/move`

Ask an **agent** to move.
**Body**

```json
{ "state": { ... }, "agent": "advanced" }
```

**Accepted agents**

* `dqn`, `minimax`, `alpha_beta`, `mcts`, `advanced`
* Aliases: `advanced_heuristic`, `adv`, `ah`, `alpha-beta`, `alphabeta`

**Response**

```json
{
  "action": 0,
  "next_state": { ... },
  "reward": 0.0,
  "done": false
}
```

## 🧠 Model Registry

* Folder: `model_registry/latest/`
* Files:

  * `policy.pt` — DQN weights
  * `meta.json` — arbitrary metadata used by `/api/health`, e.g.:

    ```json
    {"version": "v0.2", "win_rate": 0.67, "trained_at": "2025-08-29 01:55"}
    ```

The DQN service wrapper (`mancala_ai/agents/dqn.py`) lazily loads `DQNAgent` from `mancala_ai/training/dqn.py` with `state_shape=(29,)` by default and guards against missing Torch/CUDA. If weights change, restart the API to reload.


## ⚙️ Configuration

* `MODEL_REGISTRY` (env var): override model path (default: `model_registry/latest`)
* **CORS:** When running UI and API on different origins without the Nginx proxy, enable CORS in Flask:

  ```python
  from flask_cors import CORS
  app = create_app()
  CORS(app, resources={r"/api/*": {"origins": FRONTWEB_ORIGIN}})
  ```


## 🛠️ Troubleshooting

* **UI can’t reach API (404 on `/api/...`)**
  Use Docker (UI proxy on 5173), or create `env.js` that sets `window.API_BASE="http://localhost:8000/api"` and ensure it loads **before** `js/api.js`.

* **Gunicorn error `--factory` not recognized**
  We use a WSGI shim: `src/wsgi.py` and run `gunicorn wsgi:app`.

* **Torch/CUDA errors**
  Use CPU Torch (`torch==2.3.1`) or let the app fall back to non-DQN agents. In Docker, default images install CPU wheels.

* **Stones duplicate during animation**
  The UI never re-renders stone DOM after a move; it only animates real stones and updates counts. If you tweak UI code, avoid re-adding stones after sowing.

* **Swagger/OpenAPI**
  If you wire `flask-smorest`/`apispec`, mount docs under your preferred route; the current setup focuses on a compact REST surface.

## 📝 Roadmap Ideas

* `/api/reload` to hot-reload DQN weights without restart
* Self-play training job + MLflow tracking
* ELO evaluator across agents
* Cloud deploy (Render/Fly/EC2) + HTTPS + CDN for UI assets

## 🙌 Credits

Developed by:

- **Dylan (Quang) Nguyen*