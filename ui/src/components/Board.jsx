// Board.jsx
import Pit from './Pit'
import Store from './Store'
import React, { useRef, useState, forwardRef, useImperativeHandle } from "react";
import { animateSow as animateFanout, animateCapture as doCapture } from "../ui/sowing";
import { animateCollectRow as doCollectRow } from "../ui/sowing";
import { getScatterController } from "../ui/scatterSeedsController";

// --- helpers (put near hydrateFromState) ---------------------------------

function clearAllScatters(root){
  root.querySelectorAll('.pit__scatter, .store__scatter').forEach(sc => {
    // fast drop: remove all children (no animations)
    while (sc.firstChild) sc.removeChild(sc.firstChild);
  });
}

function updateCountsText(root, s){
  // pits: .pit__face[data-side][data-index] -> .pit__count
  for (let side = 0; side <= 1; side++){
    const arr = s?.pits?.[side] || [];
    for (let i = 0; i < 6; i++){
      const face = root.querySelector(`.pit__face[data-side="${side}"][data-index="${i}"]`)
              || (side === 0
                    ? root.querySelectorAll('.board__pits--bottom .pit__face')[i]
                    : root.querySelectorAll('.board__pits--top .pit__face')[5 - i]);
      if (!face) continue;
      const label = face.querySelector('.pit__count');
      if (label) label.textContent = String(arr[i] || 0);
    }
  }

  // stores: prefer data-owner, fallback to position
  const st0Face = root.querySelector(`.board__store[data-owner="0"] .store__face`)
                || root.querySelector('.board__store--right .store__face')
                || root.querySelector('.board__store--right .store');
  const st1Face = root.querySelector(`.board__store[data-owner="1"] .store__face`)
                || root.querySelector('.board__store--left .store__face')
                || root.querySelector('.board__store--left .store');

  const st0Val = st0Face?.querySelector('.store__value');
  const st1Val = st1Face?.querySelector('.store__value');
  if (st0Val) st0Val.textContent = String(s?.stores?.[0] ?? 0);
  if (st1Val) st1Val.textContent = String(s?.stores?.[1] ?? 0);
}

function hydrateFromState(boardEl, state){
  if (!boardEl || !state) return;

  // P0 pits: bottom DOM order 0..5
  const bottomFaces = [...boardEl.querySelectorAll(".board__pits--bottom .pit__face")];
  bottomFaces.forEach((face, i) => {
    const scatter = face.querySelector(".pit__scatter");
    if (!scatter) return;
    const ctl = getScatterController(scatter);
    ctl.clear();
    ctl.add(state.pits?.[0]?.[i] ?? 0);
  });

  // P1 pits: top DOM is reversed => DOM idx d maps to pit idx (5 - d)
  const topFaces = [...boardEl.querySelectorAll(".board__pits--top .pit__face")];
  topFaces.forEach((face, d) => {
    const i = 5 - d;
    const scatter = face.querySelector(".pit__scatter");
    if (!scatter) return;
    const ctl = getScatterController(scatter);
    ctl.clear();
    ctl.add(state.pits?.[1]?.[i] ?? 0);
  });

  // Stores
  const rightStoreFace = boardEl.querySelector(".board__store--right .store__face");
  const leftStoreFace  = boardEl.querySelector(".board__store--left .store__face");

  if (rightStoreFace){
    const sc = rightStoreFace.querySelector(".store__scatter");
    if (sc){
      const cx = sc.clientWidth/2, cy = sc.clientHeight/2, r = 36;
      const ctl = getScatterController(sc, { centerExclusion: { x: cx, y: cy, r } });
      ctl.clear();
      ctl.add(state.stores?.[0] ?? 0);
    }
  }
  if (leftStoreFace){
    const sc = leftStoreFace.querySelector(".store__scatter");
    if (sc){
      const cx = sc.clientWidth/2, cy = sc.clientHeight/2, r = 36;
      const ctl = getScatterController(sc, { centerExclusion: { x: cx, y: cy, r } });
      ctl.clear();
      ctl.add(state.stores?.[1] ?? 0);
    }
  }
}


const Board = forwardRef(function Board({ state, canPlay, onPlay }, apiRef) {
  if (!state) return null
  const [p0, p1] = state.pits
  const [st0, st1] = state.stores
  const turn = state.current_player

  const boardRef = useRef(null)
  const [busy, setBusy] = useState(false)

  // Full ring in P0 (bottom-left→right) visual order:
  // bottom pits 0..5, right store, top pits 5..0, left store
  function getFullRing() {
    if (!boardRef.current) return []
    const bottom = [...boardRef.current.querySelectorAll(".board__pits--bottom .pit__face")]
    const rightStore = boardRef.current.querySelector(".board__store--right .store__face") || boardRef.current.querySelector(".board__store--right .store")
    const top = [...boardRef.current.querySelectorAll(".board__pits--top .pit__face")].reverse()
    const leftStore = boardRef.current.querySelector(".board__store--left .store__face") || boardRef.current.querySelector(".board__store--left .store")
    return [...bottom, rightStore, ...top, leftStore].filter(Boolean)
  }

  // Rotate from the *ring* index that corresponds to (player, startPitIdx),
  // then skip opponent's store.
  function buildPathFrom(player, startPitIdx) {
    const ring = getFullRing()
    if (ring.length === 0) return []

    // Map (player, pit) to index in the ring
    // bottom pits occupy ring[0..5]; top pits occupy ring[7..12] (since ring[6] is right store)
    // top row is reversed in DOM → ring index = 7 + (startPitIdx)
    const startIndexInRing = (player === 0)
      ? startPitIdx
      : 7 + (startPitIdx)

    const rotated = ring.slice(startIndexInRing).concat(ring.slice(0, startIndexInRing))

    // Skip opponent store
    return rotated.filter(el => {
      const isRightStore = el.closest('.board__store--right') != null
      const isLeftStore  = el.closest('.board__store--left')  != null
      if (player === 0 && isLeftStore)  return false
      if (player === 1 && isRightStore) return false
      return true
    })
  }

  useImperativeHandle(apiRef, () => ({
    async animateSowFromPit(player, pitIndex, seedCount, opts = {}) {
      if (!boardRef.current || seedCount <= 0) return
      let startEl
      if (player === 0) {
        const bottom = [...boardRef.current.querySelectorAll(".board__pits--bottom .pit__face")]
        startEl = bottom[pitIndex]
      } else {
        const topFaces = [...boardRef.current.querySelectorAll(".board__pits--top .pit__face")]
        const domIndex = 5 - pitIndex // top is reversed in DOM
        startEl = topFaces[domIndex]
      }
      if (!startEl) return

      const path    = buildPathFrom(player, pitIndex)
      await animateFanout(boardRef.current, startEl, path, seedCount, opts)
    },
    async animateCapture(player, landingPitIndex, capturedOppCount, opts = {}) {
      if (!boardRef.current) return;
      await doCapture(boardRef.current, player, landingPitIndex, capturedOppCount, opts);
    }, 
    async animateCollectRow(player, counts, opts) {
      if (!boardRef.current) return;
      await doCollectRow(boardRef.current, player, counts, opts);
    },
    hydrate: (s) => {
      const root = boardRef.current;
      if (!root) return;

      const deadline = Date.now() + 1500; // 1.5s safety cap
      let rafId = 0;

      const ready = () => {
        const any = root.querySelector('.pit__scatter, .store__scatter');
        return any && any.clientWidth > 0 && any.clientHeight > 0;
      };

      const tick = () => {
        if (ready() || Date.now() > deadline) {
          hydrateFromState(root, s);
          return;
        }
        rafId = requestAnimationFrame(tick);
      };

      tick();

      // (optional) return a cancel function
      return () => cancelAnimationFrame(rafId);
    },
    setDemoState: (demo) => {
      const root = boardRef.current;
      if (!root || !demo) return;
      // 1) wipe any existing seeds (no React re-render)
      clearAllScatters(root);
      // 2) lay down seeds according to the demo state
      hydrateFromState(root, demo);
      // 3) sync the numeric labels so they match the scattered seeds
      updateCountsText(root, demo);
    },
    setBusy
  }), [state])

  async function handleSowAndApply(player, startPitIdx, stagger = 300, hopMs = 650) {
    if (!boardRef.current || busy) return
    if (!canPlay || turn !== player) return
    const seedCount = state?.pits?.[player]?.[startPitIdx] ?? 0
    if (seedCount <= 0) return

    setBusy(true)
    try {
      await apiRef.current?.animateSowFromPit?.(player, startPitIdx, seedCount, { stagger: stagger, hopMs: hopMs } )
      await onPlay(startPitIdx) // backend applies for current_player
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="board card board--grid" ref={boardRef}>
      <div className="side-label side-label--top" aria-hidden>
        Player 1’s side
      </div>

      <div className="board__store board__store--left"  data-owner="1">
        <Store id={p1} value={st1} title="Store P1" />
      </div>

      <div className="board__pits board__pits--top">
        {[...p1].reverse().map((c, displayIdx) => {
          const realIdx = 5 - displayIdx
          return (
            <Pit
              key={displayIdx}
              id={`P1:${realIdx}`}
              count={c}
              side={1}
              pitIndex={realIdx}
              active={turn === 1 && c !== 0}
              disabled={busy || !canPlay || turn !== 1 || c === 0}
              onClick={() => handleSowAndApply(1, realIdx)}
            />
          )
        })}
      </div>

      <div className="board__pits board__pits--bottom">
        {p0.map((c, i) => (
          <Pit
            key={i}
            id={`P0:${i}`}
            count={c}
            side={0}
            pitIndex={i}
            active={turn === 0 && c !== 0}
            disabled={busy || !canPlay || turn !== 0 || c === 0}
            onClick={() => handleSowAndApply(0, i)}
          />
        ))}
      </div>

      <div className="board__store board__store--right"  data-owner="0">
        <Store id={p0} value={st0} title="Store P0" />
      </div>
      {/* under the lower row (Player 0) */}
      <div className="side-label side-label--bottom" aria-hidden>
        Player 0’s side
      </div>
    </div>
  )
})

export default Board
