# Mancala AI — End-to-End Game AI (Flask API + React Animated UI)

Play Mancala against multiple AI agents (DQN, Minimax, Alpha-Beta, MCTS, Advanced Heuristic) with a smooth **React** UI: real stone animations, capture/row sweep effects, turn highlighting, tutorial, and synth SFX. The project demonstrates an end-to-end ML product: **REST API**, **animated UI**, and a lightweight **model registry** for swapping policies.

## ✨ Features

* **Agents:** `dqn`, `minimax`, `alpha_beta`, `mcts`, `advanced`
  (aliases: `advanced_heuristic`, `adv`, `ah`, `alpha-beta`, `alphabeta`)
* **Animated sowing:** actual DOM stones fly pit→pit, persist position; counts reconcile from server
* **Tutorial:** sowing, extra turn, capture, endgame sweep (with replay)
* **SFX:** WebAudio plops/captures/row sweep + win fanfare; unlocks on first gesture; global mute/volume
* **REST API:** `/api/health`, `/api/newgame`, `/api/apply`, `/api/move`
* **Model registry:** drop `policy.pt` + `meta.json` into `model_registry/latest/`
* **Dockerized:** `docker compose up` runs API + built UI; UI proxies `/api/*` to API
* **Local dev friendly:** run API and **React** UI separately; Vite proxy or `env.js`

## 📦 Project Structure

```
mancala-ai/
├─ src/
│  └─ mancala_ai/
│     ├─ engine/
│     │  └─ core.py              # rules: initialize_board, make_move, capture, sweep
│     ├─ agents/
│     │  ├─ dqn.py               # lazily loads training.dqn.DQNAgent
│     │  ├─ minimax.py           # simple_minimax(...)
│     │  ├─ alpha_beta.py        # minimax_alpha_beta(...)
│     │  ├─ MCTS.py              # mcts_decide(...)
│     │  └─ advanced_heuristic.py# advanced_heuristic_minimax(...)
│     ├─ api/
│     │  ├─ app.py               # Flask factory (create_app)
│     │  └─ routes.py            # /api endpoints
│     ├─ io/
│     │  └─ registry.py          # pick_action(), current_meta()
│     ├─ training/
│     │  └─ dqn.py               # DQNAgent
│     └─ utils/
│        └─ features.py          # encoders, helpers
├─ web/                           # React UI (Vite)  ← (folder name may be "ui" in your repo)
│  ├─ index.html
│  ├─ vite.config.ts/js
│  ├─ package.json
│  ├─ public/
│  │  └─ assets/                 # music.mp3, icons, etc.
│  └─ src/
│     ├─ App.jsx
│     ├─ styles.css
│     ├─ api.js                  # calls /api/*
│     ├─ simulateMoveLite.js
│     ├─ components/
│     │  ├─ Board.jsx
│     │  ├─ Pit.jsx
│     │  ├─ Store.jsx
│     │  ├─ ControlPanel.jsx
│     │  ├─ MoveLog.jsx
│     │  ├─ Menu.jsx
│     │  ├─ DomTutorial.jsx
│     │  └─ ErrorBoundary.jsx
│     └─ ui/
│        ├─ sowing.js            # animations (seed-flight, capture, sweep)
│        ├─ scatterSeedsController.js
│        ├─ fireworks.js
│        └─ sfx.js               # WebAudio synth SFX
├─ model_registry/
│  └─ latest/
│     ├─ policy.pt
│     └─ meta.json               # {"version":"v0.1","win_rate":0.83,"trained_at":"..."}
├─ docker/
│  ├─ api.Dockerfile             # Flask API (Gunicorn)
│  └─ ui.Dockerfile              # Nginx serves React build on 8080; proxies /api → 8000
├─ docker-compose.yml
├─ requirements.txt
├─ src/wsgi.py                   # wsgi:app shim for Gunicorn
└─ README.md
```

## 🚀 Quickstart (Docker)

**Prereqs:** Docker Desktop (Win/macOS) or Docker Engine (Linux).

```bash
docker compose up --build
```

* UI: **[http://localhost:8080](http://localhost:8080)**
* API (direct): **[http://localhost:8000/api/health](http://localhost:8000/api/health)**

**How it’s wired:** Nginx serves the React **build** on **8080** and **proxies `/api/*` to 8000**, so the browser can use same-origin URLs like `/api/move`.

### Hot-swap model

Drop new weights/metadata into `./model_registry/latest/` (mounted read-only). If your DQN wrapper caches the model, restart the API to reload:

```bash
docker compose restart api
```

## 🧰 Local Development (without Docker)

### 1) Backend (Flask API)

```bash
export PYTHONPATH=/src \
       MODEL_REGISTRY="$(pwd)/model_registry/latest"
export FLASK_APP=wsgi:app

python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
# If you don't need GPU: torch==2.3.1 in requirements.txt is CPU-only
pip install -r requirements.txt

# Gunicorn dev run
python -m gunicorn -w 2 --threads 8 -k gthread \
  --chdir src -b 0.0.0.0:8000 \
  wsgi:app

# Health
curl http://localhost:8000/api/health
```

### 2) Frontend (React + Vite)

```bash
cd web
npm i
```

**Option A: Vite proxy (recommended)**
Add this to `vite.config.js` so `/api` goes to Flask in dev:

```js
export default defineConfig({
  server: {
    port: 8080,
    proxy: { '/api': 'http://localhost:8000' }
  }
});
```

Run dev server:

```bash
npm run dev
# UI on http://localhost:5173, proxying /api → 8000
```

**Option B: explicit API base**
Create `public/env.js` (loaded by `index.html` **before** your bundle):

```html
<script src="/env.js"></script>
```

…and in `web/src/api.js` read `import.meta.env?.VITE_API_BASE || '/api'`.

**Production build preview**

```bash
npm run build
npm run preview   # serves dist/ locally (no /api proxy)
```

## 🔌 API Reference (UI-used shapes)

Base URL: `/api` (Docker UI via proxy)
Direct API: `http://localhost:8000/api`

### `GET /api/health`

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

**Response** (UI expects `state`)

```json
{ "state": { ... } }
```

### `POST /api/move`

Ask an **agent** to move.

**Body**

```json
{ "state": { ... }, "agent": "advanced" }
```

**Response** (UI expects `state` and optionally `action`)

```json
{
  "action": 0,
  "state": { ... }
}
```

> If your current API returns `next_state`, either:
>
> 1. keep UI compatible by also returning `state`, **or**
> 2. adjust `web/src/api.js` to map `next_state → state`.

## 🧠 Model Registry

* Folder: `model_registry/latest/`
* Files:

  * `policy.pt` — DQN weights
  * `meta.json` — metadata used by `/api/health`, e.g.:

    ```json
    {"version": "v0.2", "win_rate": 0.67, "trained_at": "2025-08-29 01:55"}
    ```
* DQN wrapper (`mancala_ai/agents/dqn.py`) lazily loads `training/DQNAgent` and guards CUDA.
  Restart API after swapping weights (or add `/api/reload`, see roadmap).

## 🖥️ React UI Notes

* **Board state & animations**

  * Stones are real DOM nodes. Sowing animates along the ring; counts are reconciled from server **by delta** (no reshuffle).
  * Tutorial uses `DomTutorial` with `setDemoState` for initial step layout and `hydrate` for post-commit reconciliation.
  * Turn highlighting works without React re-render using root classes `.turn-0 / .turn-1`.

* **SFX**

  * `ui/sfx.js` uses WebAudio (synth), unlocked on first user gesture.
  * Global mute/volume via `setMuted()` / `setVolume()`; optional ducking if background music is attached.
  * Seed landing events (`mancala:seed-landed`) trigger `sfx.seedPit()` / `sfx.seedStore()`.

* **Error boundaries**

  * The app wraps `<Board/>` and `<DomTutorial/>` with `ErrorBoundary` for graceful fallback if a step fails.

## 🛠️ Troubleshooting

* **UI can’t reach API in dev**

  * Use Vite proxy in `vite.config` (see above), or set `window.API_BASE` via `public/env.js`.
* **Autoplay blocked / silent SFX**

  * Ensure `unlockAudio()` is called on first user interaction (the UI does this on `pointerdown`).
* **Counts correct but stones “jump” after commit**

  * Ensure tutorial uses `boardApi.hydrate(next)` (not `setDemoState(next)`) after server commits.
* **React console: error occurred in `<DomTutorial>`**

  * Guard `cur = steps[step]`, clamp `step` to valid range, and keep the step runner wrapped in try/catch. Wrap the component in `ErrorBoundary`.

## 🧪 Useful NPM Scripts (example)

In `web/package.json`:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 8080"
  }
}
```

## 📝 Roadmap Ideas

* `/api/reload` to hot-reload DQN weights without restart
* Self-play training + MLflow/Weights & Biases tracking
* Cross-agent ELO evaluator
* Cloud deploy (Render/Fly/EC2) + HTTPS + CDN
* Sample-based SFX library with the same `sfx.*()` API


## 🙌 Credits

Developed by **Dylan (Quang) Nguyen**.