// ui-static/js/components/Board.js
function Board(container, { onPitClick } = {}) {
  container.innerHTML = `
    <div class="board">
      <div class="store store-left"><div class="count" id="store-1">0</div></div>
      <div class="center">
        <div class="row top" id="row-1"></div>
        <div class="row bottom" id="row-0"></div>
      </div>
      <div class="store store-right"><div class="count" id="store-0">0</div></div>
    </div>
  `;

  const row0 = container.querySelector("#row-0");
  const row1 = container.querySelector("#row-1");
  const store0 = container.querySelector("#store-0");
  const store1 = container.querySelector("#store-1");

  function render(state) {
    // stores text
    store0.textContent = state.stores[0];
    store1.textContent = state.stores[1];

    // rebuild pits
    row0.innerHTML = "";
    row1.innerHTML = "";

    // Player 0 (bottom) left→right
    state.pits[0].forEach((n, i) => {
      const pit = document.createElement("button");
      pit.className = "pit";
      pit.dataset.row = "0";
      pit.dataset.idx = String(i);
      pit.innerHTML = `<div class="count">${n}</div>`;
      pit.addEventListener("click", () => onPitClick?.(0, i));
      row0.appendChild(pit);
    });

    // Player 1 (top) right→left (CSS does the rtl)
    state.pits[1].forEach((n, i) => {
      const pit = document.createElement("button");
      pit.className = "pit";
      pit.dataset.row = "1";
      pit.dataset.idx = String(i);
      pit.innerHTML = `<div class="count">${n}</div>`;
      pit.addEventListener("click", () => onPitClick?.(1, i));
      row1.appendChild(pit);
    });

    // highlight legal
    const moves = legalMoves(state);
    container.querySelectorAll(".pit").forEach(el => {
      const row = Number(el.dataset.row), idx = Number(el.dataset.idx);
      el.classList.toggle("legal", row === state.current_player && moves.includes(idx));
    });
  }

  function legalMoves(s) {
    const row = s.current_player;
    const list = [];
    for (let i = 0; i < 6; i++) if (s.pits[row][i] > 0) list.push(i);
    return list;
  }

  return { render };
}
window.Board = Board;
