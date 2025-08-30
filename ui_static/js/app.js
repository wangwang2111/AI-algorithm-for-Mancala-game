// ui-static/js/app.js
(function () {
  // --------- DOM refs ----------
  const $board  = document.getElementById("board");
  const $turn   = document.getElementById("turn");
  const $legal  = document.getElementById("legal");
  const $log    = document.getElementById("log");
  const $health = document.getElementById("health");
  const $mute   = document.getElementById("mute-btn"); // optional

  let state = null;
  let busy = false;
  let ended = false;

  // ---- optional sounds ----
  try {
    if (window.Sound?.init) {
      Sound.init();
      if ($mute) {
        const muted = localStorage.getItem("mancala-muted") === "1";
        $mute.textContent = muted ? "🔇" : "🔈";
        $mute.onclick = () => { const m = Sound.toggleMute(); $mute.textContent = m ? "🔇" : "🔈"; };
      }
      setTimeout(() => Sound.startBackground?.(), 250);
    }
  } catch {}

  // ---------- Board + Controls ----------
  const board = Board($board, {
    onPitClick: (row, idx) => {
      if (!state || ended || busy) return;
      if (row !== state.current_player) return;
      onHumanMove(idx);
    }
  });

  const controls = Controls(document.getElementById("controls"), {
    getState: () => state,
    onNewGame: newGame,
    onHumanMove: (idx) => onHumanMove(idx),
    onAIMove: (agent) => onAIMove(agent),
  });

  // ---------- bootstrap ----------
  (async function init() {
    try {
      const meta = await API.health();
      $health.textContent = `model ${meta.model.version} • win_rate ${meta.model.win_rate ?? "?"}`;
    } catch { $health.textContent = "health: unavailable"; }

    const res = await API.newGame();
    // Initial build: render pits + stones once
    setState(res.state, { rebuild: true, withStones: true });
  })();

  // ---------- helpers ----------
  function legalMoves(s) {
    const row = s.current_player, res = [];
    for (let i = 0; i < 6; i++) if (s.pits[row][i] > 0) res.push(i);
    return res;
  }
  function updateStatus() {
    if (!state) return;
    $turn.innerHTML = `<span class="badge"><span class="dot"></span> Turn: P${state.current_player}</span>`;
    $legal.textContent = `Legal: [${legalMoves(state).join(", ")}]`;
  }
  function appendLog(text) { const li = document.createElement("li"); li.textContent = text; $log.appendChild(li); }

  // ----- DOM lookup helpers -----
  function pitEl(row, idx) {
    return $board.querySelector(`.pit[data-row="${row}"][data-idx="${idx}"]`);
  }
  function storeEl(playerIdx) {
    return $board.querySelector(playerIdx === 0 ? "#store-0" : "#store-1").closest(".store");
  }
  function ensureStonesLayer(el) {
    let layer = el.querySelector('.stones');
    if (!layer) { layer = document.createElement('div'); layer.className = 'stones'; el.appendChild(layer); }
    return layer;
  }
  function layerRect(el) { return ensureStonesLayer(el).getBoundingClientRect(); }
  function randomPos(rect) {
    const pad = 8;
    const x = pad + Math.random() * Math.max(1, rect.width  - 2*pad);
    const y = pad + Math.random() * Math.max(1, rect.height - 2*pad);
    return { x, y };
  }
  function setCount(el, n) {
    const val = String(n);
    // Update hidden count node only if it exists (harmless if removed)
    const c = el.querySelector('.count');
    if (c) c.textContent = val;

    // Tooltip content + a11y
    el.dataset.count = val;           // tooltip text
    el.title = val;                   // long-press fallback on mobile
    el.setAttribute('aria-label', val);
  }

  // ---------- ONLY update numbers & legal highlights ----------
  function updateCountsAndLegal(s) {
    // stores
    const st0 = storeEl(0), st1 = storeEl(1);
    setCount(st0, s.stores[0]);
    setCount(st1, s.stores[1]);

    // pits
    for (let i=0;i<6;i++) setCount(pitEl(0,i), s.pits[0][i]);
    for (let i=0;i<6;i++) setCount(pitEl(1,i), s.pits[1][i]);

    // legal highlight
    const moves = legalMoves(s);
    $board.querySelectorAll(".pit").forEach(el => {
      const row = Number(el.dataset.row), idx = Number(el.dataset.idx);
      el.classList.toggle("legal", row === s.current_player && moves.includes(idx));
    });
  }

  // ---------- render stones from state (ONLY used at build/new game) ----------
  function fillContainerWithStones(container, count, isStore=false) {
    const layer = ensureStonesLayer(container);
    placeStonesInLayer(layer, count, { isStore });
  }
  function renderStonesFromState(s) {
    for (let i=0;i<6;i++) fillContainerWithStones(pitEl(0,i), s.pits[0][i], false);
    for (let i=0;i<6;i++) fillContainerWithStones(pitEl(1,i), s.pits[1][i], false);
    fillContainerWithStones(storeEl(0), s.stores[0], true);
    fillContainerWithStones(storeEl(1), s.stores[1], true);
  }
  function placeStonesInLayer(layer, count, { isStore=false } = {}) {
    layer.innerHTML = '';
    const W = layer.clientWidth, H = layer.clientHeight;
    const base = isStore ? 18 : 16;
    const size = count > 12 ? 12 : count > 8 ? 14 : base;
    const minDist = Math.max(10, size * 0.9);
    const cx = W/2, cy = H/2, centerBlock = isStore ? 22 : 18;
    const pts = []; const pad = 8;
    function ok(x,y){
      if (Math.hypot(x-cx,y-cy) < centerBlock) return false;
      for (const p of pts){ const dx=x-p.x, dy=y-p.y; if (dx*dx+dy*dy < minDist*minDist) return false; }
      return true;
    }
    for (let i=0;i<count;i++){
      let x=0,y=0, tries=0;
      do { x=pad+Math.random()*Math.max(1,W-2*pad); y=pad+Math.random()*Math.max(1,H-2*pad); tries++; }
      while(!ok(x,y) && tries<60);
      pts.push({x,y});
      const s = document.createElement('div');
      s.className = 'stone';
      s.style.width = `${size}px`; s.style.height = `${size}px`;
      s.style.left = `${x - size/2}px`; s.style.top = `${y - size/2}px`;
      s.style.filter = `hue-rotate(${(Math.random()*8-4)}deg) brightness(1.04)`;
      layer.appendChild(s);
    }
  }

  // ---------- setState ----------
  function setState(next, { rebuild=false, withStones=false, numbersOnly=false } = {}) {
    state = next;

    if (rebuild) {
      // Build/refresh the pit buttons once (this will NOT wipe stones later)
      board.render(state);
      if (withStones) renderStonesFromState(state); // only on first load or new game
      updateCountsAndLegal(state);
    } else if (numbersOnly) {
      updateCountsAndLegal(state); // keep stones as-is, update numbers + legal
    } else {
      // default safe path (rarely used)
      board.render(state);
      updateCountsAndLegal(state);
    }

    controls.refreshPits(state);  // PURE: dropdown only
    updateStatus();
  }

  // ---------- flat mapping & path (14 slots) ----------
  // 0..5 = P0 pits, 6 = P0 store, 7..12 = P1 pits, 13 = P1 store
  function elForFlatIndex(idx) {
    if (idx === 6) return storeEl(0);
    if (idx === 13) return storeEl(1);
    if (idx >= 0 && idx <= 5) return pitEl(0, idx);
    if (idx >= 7 && idx <= 12) return pitEl(1, idx - 7);
    return null;
  }
  function sowTargets(startFlat, seeds, playerIdx) {
    const targets = []; let pos = startFlat;
    for (let k=0;k<seeds;k++) {
      pos = (pos + 1) % 14;
      if (playerIdx === 0 && pos === 13) { pos = (pos + 1) % 14; } // skip P1 store
      if (playerIdx === 1 && pos === 6)  { pos = (pos + 1) % 14; } // skip P0 store
      targets.push(pos);
    }
    return targets;
  }

  // ---------- local rules simulator (for animation plan) ----------
  function simulateMove(s, action) {
    const next = structuredClone(s);
    const player = next.current_player;
    const pits = next.pits[player];
    let stones = pits[action];
    if (!stones) return null;

    pits[action] = 0;
    let flat = (player === 0) ? action : (7 + action);
    const targets = [];

    while (stones > 0) {
      flat = (flat + 1) % 14;
      if (player === 0 && flat === 13) continue;
      if (player === 1 && flat === 6)  continue;

      targets.push(flat);
      if (flat <= 5)       next.pits[0][flat] += 1;
      else if (flat === 6) next.stores[0] += 1;
      else if (flat <= 12) next.pits[1][flat - 7] += 1;
      else                 next.stores[1] += 1;
      stones--;
    }

    const last = targets[targets.length - 1];
    const extraTurn = (player === 0 && last === 6) || (player === 1 && last === 13);
    let capture = null;

    if (!extraTurn) {
      // capture for P0
      if (player === 0 && last >= 0 && last <= 5 && next.pits[0][last] === 1) {
        const oppFlat = 12 - last;
        const oppIdx  = oppFlat - 7;
        const cap = next.pits[1][oppIdx];
        if (cap > 0) {
          next.pits[1][oppIdx] = 0;
          next.pits[0][last] = 0;
          next.stores[0] += cap + 1;
          capture = { fromFlat: last, oppFlat, toStoreFlat: 6, count: cap + 1 };
        }
      }
      // capture for P1
      if (player === 1 && last >= 7 && last <= 12 && next.pits[1][last - 7] === 1) {
        const oppFlat = 12 - last;
        const oppIdx  = oppFlat; // 0..5
        const cap = next.pits[0][oppIdx];
        if (cap > 0) {
          next.pits[0][oppIdx] = 0;
          next.pits[1][last - 7] = 0;
          next.stores[1] += cap + 1;
          capture = { fromFlat: last, oppFlat, toStoreFlat: 13, count: cap + 1 };
        }
      }
    }

    // terminal sweep
    const sum0 = next.pits[0].reduce((a,b)=>a+b,0);
    const sum1 = next.pits[1].reduce((a,b)=>a+b,0);
    let sweep = null;
    let done = false;
    if (sum0 === 0 || sum1 === 0) {
      done = true;
      sweep = { p0: [...next.pits[0]], p1: [...next.pits[1]] };
      next.stores[0] += sum0;
      next.stores[1] += sum1;
      next.pits[0] = [0,0,0,0,0,0];
      next.pits[1] = [0,0,0,0,0,0];
    }

    next.current_player = extraTurn ? player : (1 - player);
    return { next, plan: { startFlat: (player===0? action : 7+action), targets, capture, sweep } };
  }

  // ---------- animate REAL stones (no stone re-render later) ----------
  async function animateRealMove(sim) {
    const { startFlat, targets, capture, sweep } = sim.plan;

    // pick up all stones from the source visually
    const srcEl = elForFlatIndex(startFlat);
    const srcLayer = ensureStonesLayer(srcEl);
    const seedCount = srcLayer.querySelectorAll('.stone').length;
    srcLayer.innerHTML = "";

    const HOP_MS = 200, STAGGER_MS = 70;
    try { Sound?.play?.("stone"); } catch {}

    // throw distinct stones from source -> each target, and increment counts live
    for (let i=0;i<targets.length;i++) {
      const toEl = elForFlatIndex(targets[i]);
      const toLayer = ensureStonesLayer(toEl);
      const sRect = layerRect(srcEl);
      const dRect = layerRect(toEl);
      const sp = randomPos(sRect);
      const tp = randomPos(dRect);

      const fly = document.createElement("div");
      fly.className = "stone sowing";
      fly.style.position = "fixed";
      fly.style.left = `${sRect.left}px`;
      fly.style.top  = `${sRect.top}px`;
      fly.style.transform = `translate(${sp.x}px, ${sp.y}px)`;
      document.body.appendChild(fly);

      await delay(i * STAGGER_MS);
      await nextFrame();
      fly.style.transition = `transform ${HOP_MS}ms ease-in-out`;
      fly.style.transform  = `translate(${dRect.left - sRect.left + tp.x}px, ${dRect.top - sRect.top + tp.y}px)`;
      await delay(HOP_MS);

      const landed = document.createElement("div");
      landed.className = "stone";
      landed.style.left = `${tp.x - 9}px`;
      landed.style.top  = `${tp.y - 9}px`;
      toLayer.appendChild(landed);

      // update count label (+1) as we go
      if (targets[i] === 6 || targets[i] === 13) {
        const storeIdx = (targets[i] === 6 ? 0 : 1);
        const storeBox = storeEl(storeIdx);
        setCount(storeBox, Number(storeBox.querySelector('.count')?.textContent || 0) + 1);
      } else {
        const row = (targets[i] <= 5) ? 0 : 1;
        const pitIdx = (row === 0) ? targets[i] : (targets[i] - 7);
        const pitBox = pitEl(row, pitIdx);
        setCount(pitBox, Number(pitBox.querySelector('.count')?.textContent || 0) + 1);
      }

      fly.remove();
    }

    // capture (move real stones from last + opposite to store)
    if (capture) {
      const lastEl = elForFlatIndex(capture.fromFlat);
      const oppEl  = elForFlatIndex(capture.oppFlat);
      const store  = elForFlatIndex(capture.toStoreFlat);
      const lastLayer = ensureStonesLayer(lastEl);
      const oppLayer  = ensureStonesLayer(oppEl);
      const storeLayer= ensureStonesLayer(store);

      const lastRect = layerRect(lastEl);
      const oppRect  = layerRect(oppEl);
      const storeRect= layerRect(store);

      const moveOne = async (srcLayer, sRect) => {
        const sp = randomPos(sRect);
        const tp = randomPos(storeRect);
        const fly = document.createElement("div");
        fly.className = "stone sowing";
        fly.style.position = "fixed";
        fly.style.left = `${sRect.left}px`;
        fly.style.top  = `${sRect.top}px`;
        fly.style.transform = `translate(${sp.x}px, ${sp.y}px)`;
        document.body.appendChild(fly);
        await nextFrame();
        fly.style.transition = "transform 230ms ease-in-out";
        fly.style.transform  = `translate(${storeRect.left - sRect.left + tp.x}px, ${storeRect.top - sRect.top + tp.y}px)`;
        await delay(230);
        const rem = srcLayer.querySelector('.stone'); if (rem) rem.remove();
        const landed = document.createElement("div");
        landed.className = "stone";
        landed.style.left = `${tp.x - 9}px`;
        landed.style.top  = `${tp.y - 9}px`;
        storeLayer.appendChild(landed);
        fly.remove();
      };

      // last pit has 1, opposite has (capture.count - 1)
      const oppCount = Math.max(0, capture.count - 1);
      if (ensureStonesLayer(lastEl).querySelector('.stone')) {
        await moveOne(lastLayer, lastRect);
      }
      for (let k=0;k<oppCount;k++) {
        if (!ensureStonesLayer(oppEl).querySelector('.stone')) break;
        await moveOne(oppLayer, oppRect);
      }

      // update labels
      const sIdx = (capture.toStoreFlat === 6 ? 0 : 1);
      const sBox = storeEl(sIdx);
      setCount(sBox, Number(sBox.querySelector('.count')?.textContent || 0) + capture.count);
      setCount(lastEl, 0);
      setCount(oppEl, 0);
    }

    // terminal sweep (move remaining stones to stores quickly)
    if (sweep) {
      const sweepSide = async (row) => {
        for (let i=0;i<6;i++) {
          let n = sweep[row === 0 ? 'p0' : 'p1'][i];
          if (!n) continue;
          const from = pitEl(row, i);
          const toStore = storeEl(row);
          const fromLayer = ensureStonesLayer(from);
          const toLayer   = ensureStonesLayer(toStore);
          const fRect = layerRect(from);
          const tRect = layerRect(toStore);
          for (let k=0;k<n;k++) {
            const sp = randomPos(fRect), tp = randomPos(tRect);
            const fly = document.createElement("div");
            fly.className = "stone sowing";
            fly.style.position = "fixed";
            fly.style.left = `${fRect.left}px`;
            fly.style.top  = `${fRect.top}px`;
            fly.style.transform = `translate(${sp.x}px, ${sp.y}px)`;
            document.body.appendChild(fly);
            await nextFrame();
            fly.style.transition = "transform 160ms ease-in-out";
            fly.style.transform  = `translate(${tRect.left - fRect.left + tp.x}px, ${tRect.top - fRect.top + tp.y}px)`;
            await delay(160);
            const rem = fromLayer.querySelector('.stone'); if (rem) rem.remove();
            const landed = document.createElement("div");
            landed.className = "stone";
            landed.style.left = `${tp.x - 9}px`;
            landed.style.top  = `${tp.y - 9}px`;
            toLayer.appendChild(landed);
            fly.remove();
          }
          setCount(from, 0);
        }
        const storeBox = storeEl(row);
        const add = sweep[row === 0 ? 'p0' : 'p1'].reduce((a,b)=>a+b,0);
        setCount(storeBox, Number(storeBox.querySelector('.count')?.textContent || 0) + add);
      };
      await sweepSide(0);
      await sweepSide(1);
    }
  }

  // ---------- actions ----------
  async function newGame() {
    if (busy) return;
    busy = true;
    try {
      const res = await API.newGame();
      // Rebuild + re-seed stones on new game
      setState(res.state, { rebuild: true, withStones: true });
      $log.innerHTML = "";
      ended = false;
      try { Sound?.play?.("restart"); } catch {}
    } finally { busy = false; }
  }

  async function onHumanMove(idx) {
    if (!state || ended || busy) return;
    const legal = legalMoves(state);
    if (!legal.includes(idx)) { alert(`Pit ${idx} is not legal. Legal: ${legal.join(", ")}`); return; }
    busy = true;

    const prev = structuredClone(state);
    const sim  = simulateMove(prev, idx);
    if (!sim) { busy = false; return; }

    try {
      // Animate real stones based on local plan (no stone re-render later)
      await animateRealMove(sim);

      // Update numbers to server truth (stones remain as animated)
      const res  = await API.apply(prev, idx);
      setState(res.next_state, { numbersOnly: true });
      if (res.done) onGameOver(res);
    } catch (e) {
      console.error(e); alert("Move failed.");
      // (optional) You could revert by forcing a full rebuild from server:
      // const back = await API.newGame(); setState(back.state, { rebuild:true, withStones:true });
    } finally { busy = false; }
  }

  async function onAIMove(agent) {
    if (!state || ended || busy) return;
    busy = true;
    try {
      const prev = structuredClone(state);
      const res  = await API.move(prev, agent);   // get AI’s chosen action + next_state
      appendLog(`AI(${agent}) for P${prev.current_player} ➜ pit ${res.action}`);

      const sim = simulateMove(prev, res.action);
      await animateRealMove(sim);                 // move real stones
      setState(res.next_state, { numbersOnly: true }); // update numbers only
      if (res.done) onGameOver(res);
    } catch (e) {
      console.error(e); alert("AI move failed.");
    } finally { busy = false; }
  }

  function onGameOver(res) {
    ended = true;
    const [s0, s1] = res.next_state.stores;
    const draw = s0 === s1;
    const msg = `Game Over • P0=${s0}, P1=${s1} • ` + (draw ? "Draw" : (s0>s1 ? "P0 wins!" : "P1 wins!"));
    appendLog(msg);
    try { Sound?.play?.(draw ? "gameover" : "victory"); } catch {}
    alert(msg);
  }

  // ---------- tiny utils ----------
  function delay(ms){ return new Promise(r=>setTimeout(r,ms)); }
  function nextFrame(){ return new Promise(r=>requestAnimationFrame(()=>r())); }
})();
