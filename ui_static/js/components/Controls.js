// ui-static/js/components/Controls.js
function Controls(container, { onNewGame, onHumanMove, onAIMove, getState }) {
  container.innerHTML = `
    <div class="controls">
      <div class="row">
        <label>Agent:&nbsp;
          <select id="agent">
            <option value="dqn">DQN</option>
            <option value="minimax">Minimax</option>
            <option value="alpha_beta">Alpha-Beta</option>
            <option value="mcts">MCTS</option>
            <option value="advanced">Advanced Heuristic</option>
            <option value="random">Random</option>
          </select>
        </label>
      </div>
      <div class="row">
        <label>Your pit:&nbsp;<select id="pit"></select></label>
      </div>
      <div class="row">
        <button id="btn-human" class="warn">Apply Human Move</button>
        <button id="btn-ai" class="cta">AI Move</button>
        <button id="btn-new" class="danger">New Game</button>
      </div>
    </div>
  `;

  const selAgent = container.querySelector("#agent");
  const selPit   = container.querySelector("#pit");
  const btnHuman = container.querySelector("#btn-human");
  const btnAI    = container.querySelector("#btn-ai");
  const btnNew   = container.querySelector("#btn-new");

  function refreshPits(state) {
    // PURE: only changes the dropdown; does not touch the board or global state.
    selPit.innerHTML = "";

    if (!state || !state.pits || state.current_player == null) {
      for (let i = 0; i < 6; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `Pit ${i}`;
        opt.disabled = true;
        selPit.appendChild(opt);
      }
      btnHuman.disabled = true;
      return;
    }

    btnHuman.disabled = false;
    const row = state.current_player;
    const pits = state.pits[row];

    for (let i = 0; i < 6; i++) {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `Pit ${i} (${pits[i]})`;
      selPit.appendChild(opt);
    }
  }

  btnNew.addEventListener("click", onNewGame);
  btnAI.addEventListener("click", () => onAIMove(selAgent.value));
  btnHuman.addEventListener("click", () => onHumanMove(Number(selPit.value)));

  // IMPORTANT: do NOT call refreshPits(getState()) here; let app.js call it after setState.

  return {
    refreshPits,
    getAgent: () => selAgent.value,
    setBusy: (v) => { btnHuman.disabled = btnAI.disabled = btnNew.disabled = !!v; }
  };
}
window.Controls = Controls;
