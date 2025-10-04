// components/DomTutorial.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { applyMove } from "../api"; // fix path: DomTutorial -> components/, api is one level up
// after await commitServerMove(5) in the endgame step
import { sfx, unlockAudio } from "../ui/sfx";

// util used by drag code
function clamp(v, min, max) {
  return Math.min(Math.max(v, min), max);
}
/** Minimal feature detection so we can show helpful tips instead of breaking. */
function hasApi(api) {
  if (!api) return {};
  return {
    setDemoState: typeof api.setDemoState === "function",
    animateSowFromPit: typeof api.animateSowFromPit === "function",
    animateCapture: typeof api.animateCapture === "function",
    animateCollectRow: typeof api.animateCollectRow === "function",
    hydrate: typeof api.hydrate === "function",
  };
}

function buildSteps({ setDemo, commitServerMove }) {
  const wait = (ms) => new Promise((res) => setTimeout(res, ms));

  return [
    {
      key: "welcome",
      title: "Welcome to Mancala",
      text: "This tutorial uses the real board and animations. Tap Next to see sowing, extra turn, capturing, and endgame sweep.",
      run: async () => {},
    },
    {
      key: "sowing",
      title: "Sowing",
      text: "Pick up all stones in a pit and distribute them counter-clockwise (skip opponent store).",
      run: async (api) => {
        const initial = {
          pits: [
            [1, 1, 1, 1, 4, 1],
            [1, 1, 1, 1, 1, 1],
          ],
          stores: [1, 1],
          current_player: 0,
        };
        await setDemo(initial);
        await wait(1500);
        await api?.animateSowFromPit?.(0, 4, 4, { stagger: 900, hopMs: 800 });
        await commitServerMove(4); // update counts from server
      },
    },
    {
      key: "extra",
      title: "Extra Turn",
      text: "If your last stone lands in your own store, you get another turn immediately.",
      run: async (api) => {
        // Setup: Player 0 to move; pit 4 has exactly 2 stones, so the last one lands in P0 store.
        const setup = {
          pits: [
            /* P0 (bottom) */ [1, 1, 4, 1, 1, 1],
            /* P1 (top)    */ [1, 1, 1, 1, 1, 1],
          ],
          stores: [0, 0],
          current_player: 0,
        };
        await setDemo(setup);
        await wait(1500);

        // First move: P0 sows from pit 4 with 2 stones → last lands in P0 store → extra turn.
        await api?.animateSowFromPit?.(0, 2, 4, { stagger: 650, hopMs: 700 });
        await commitServerMove(2); // server applies and should leave current_player = 0 (extra turn)

        await wait(1000);

        // Bonus move: still P0; demonstrate taking another pit (pit 5 has 2).
        await api?.animateSowFromPit?.(0, 4, 2, { stagger: 700, hopMs: 700 });
        await commitServerMove(4);

        await wait(1000);

        await api?.animateSowFromPit?.(0, 0, 1, { stagger: 700, hopMs: 700 });
        await commitServerMove(0);
      },
    },
    {
      key: "capture",
      title: "Capturing",
      text: "If your last stone lands in an empty pit on your side, you capture opposite stones plus the landing stone into your store.",
      run: async (api) => {
        const setup = {
          pits: [
            [1, 1, 1, 0, 1, 1],
            [1, 1, 3, 1, 1, 1],
          ],
          stores: [0, 0],
          current_player: 0,
        };
        await setDemo(setup);
        await wait(1500);
        await api?.animateSowFromPit?.(0, 2, 1, { stagger: 900, hopMs: 800 });
        await wait(1000);
        await api?.animateCapture?.(0, 3, 3, { stagger: 700, hopMs: 800 });
        await commitServerMove(2, { forceP1: true });
      },
    },
    {
      key: "endgame",
      title: "Endgame Sweep",
      text: "When one side becomes empty, the other player collects all remaining stones into their store.",
      run: async (api) => {
        const endSetup = {
          pits: [
            [0, 0, 0, 0, 0, 1],
            [3, 4, 2, 0, 0, 0],
          ],
          stores: [10, 6],
          current_player: 0,
        };
        await setDemo(endSetup);
        await wait(1500);
        await api?.animateSowFromPit?.(0, 5, 1, { stagger: 900, hopMs: 800 });
        await wait(1000);
        await api?.animateCollectRow?.(1, [3, 4, 2, 0, 0, 0], {
          pitStagger: 900,
          stoneStagger: 520,
          hopMs: 600,
        });
        await commitServerMove(5);
        // ensure audio is unlocked at some earlier user gesture; in case of autoplay edge cases:
        unlockAudio();
        sfx.win();
      },
    },
    {
      key: "modes",
      title: "Modes & Play",
      text: "Human vs AI: AI moves automatically on its turn (you can choose who starts). Playground: both sides manual; press “AI Move” to invoke the agent.",
      run: async () => {},
    },
    {
      key: "done",
      title: "You’re ready!",
      text: "Exit the tutorial to start a new game. Have fun.",
      run: async () => {},
    },
  ];
}

export default function DomTutorial({ boardApi, onClose, onFinish }) {
  const [step, setStep] = useState(0);
  const finishBtnRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [nonce, setNonce] = useState(0);
  const caps = useMemo(() => hasApi(boardApi), [boardApi]);

  // Keep a local “tutorial state” in sync with the server after each demo move
  const demoRef = useRef(null);

  
  // Set both the board visuals and our local demo state
  async function setDemo(state) {
    demoRef.current = state;
    if (boardApi?.setDemoState) {
      await boardApi.setDemoState(state);
    } else if (boardApi?.hydrate) {
      boardApi.hydrate(state);
    }
  }
  
  // Call backend to apply a move to the current demo state; then reconcile (no reset)
  async function commitServerMove(action) {
    const prev = demoRef.current;
    if (!prev) return null;
    const res = await applyMove(prev, action); // server returns { state, ... }
    const next = res?.state;
    console.log("previous demo state", prev);
    console.log("Demo move", action, "gives", next);
    
    if (next) {
      demoRef.current = next;
      // reflect definitive counts from server
      // IMPORTANT: reconcile counts without re-shuffling stone positions
      if (boardApi?.hydrate) boardApi.hydrate(next);
      else if (boardApi?.setDemoState) await boardApi.setDemoState(next); // fallback only
    }
    return next;
  }
  
  const steps = useMemo(() => buildSteps({ setDemo, commitServerMove }));
  
  const cur = steps[step];
  
  // at top of component:
  const lastTokenRef = useRef(null);
  const runningRef = useRef(false);
  
  // When we arrive at the last step, focus the Back to Menu button
  useEffect(() => {
    if (step === steps.length - 1) {
      requestAnimationFrame(() => finishBtnRef.current?.focus());
    }
  }, [step, steps.length]);
  
  // inside component, replace your step-run effect with:
  useEffect(() => {
    const token = `${step}|${nonce}`;
    if (lastTokenRef.current === token) return; // already ran this exact step+replay
    lastTokenRef.current = token;

    if (runningRef.current) return;
    runningRef.current = true;
    setRunning(true);

    let alive = true;
    (async () => {
      try {
        await cur?.run?.(boardApi || {});
      } catch (e) {
        // optional console.warn(e);
      } finally {
        if (alive) setRunning(false);
        runningRef.current = false;
      }
    })();

    return () => {
      alive = false;
    };
  }, [step, nonce]); // 👈 ONLY these deps

  // --- DRAG STATE ---
  const cardRef = useRef(null);
  const draggingRef = useRef(false);
  const offsetRef = useRef({ x: 0, y: 0 });
  const posRef = useRef({ x: 0, y: 0 }); // current committed position
  const [pos, setPos] = useState({ x: 0, y: 64 }); // initial top-right-ish in CSS we place via right:20px; top:64px. We'll convert to left/top as soon as drag starts.

  // On first render, try to restore previous position
  useEffect(() => {
    const raw = localStorage.getItem("tutorialCardPos");
    if (raw) {
      try {
        const saved = JSON.parse(raw);
        if (typeof saved.x === "number" && typeof saved.y === "number") {
          posRef.current = saved;
          setPos(saved);
        }
      } catch {}
    }
  }, []);

  // Save position on change (debounced via requestAnimationFrame)
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      localStorage.setItem("tutorialCardPos", JSON.stringify(pos));
    });
    return () => cancelAnimationFrame(id);
  }, [pos]);

  // Convert card to absolute w/ left/top so we can move it freely
  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    el.style.left = `${pos.x}px`;
    el.style.top = `${pos.y}px`;
    el.style.right = "auto"; // override "right: 20px" from CSS when moved
  }, [pos]);

  // Global pointer handlers
  useEffect(() => {
    function onMove(e) {
      if (!draggingRef.current || !cardRef.current) return;
      const pointerX =
        "touches" in e && e.touches.length ? e.touches[0].clientX : e.clientX;
      const pointerY =
        "touches" in e && e.touches.length ? e.touches[0].clientY : e.clientY;

      const x = pointerX - offsetRef.current.x;
      const y = pointerY - offsetRef.current.y;

      // clamp to viewport
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const rect = cardRef.current.getBoundingClientRect();
      const clamped = {
        x: clamp(x, 8, vw - rect.width - 8),
        y: clamp(y, 8, vh - rect.height - 8),
      };
      posRef.current = clamped;
      setPos(clamped);
      e.preventDefault();
    }
    function onUp() {
      draggingRef.current = false;
      document.body.classList.remove("dragging-tutorial");
    }
    window.addEventListener("pointermove", onMove, { passive: false });
    window.addEventListener("pointerup", onUp);
    window.addEventListener("touchmove", onMove, { passive: false });
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, []);

  function onDragStart(e) {
    if (!cardRef.current) return;
    const header = e.currentTarget; // the drag handle element
    const rect = cardRef.current.getBoundingClientRect();
    const pointerX =
      "touches" in e && e.touches.length ? e.touches[0].clientX : e.clientX;
    const pointerY =
      "touches" in e && e.touches.length ? e.touches[0].clientY : e.clientY;

    offsetRef.current = {
      x: pointerX - rect.left,
      y: pointerY - rect.top,
    };
    draggingRef.current = true;
    document.body.classList.add("dragging-tutorial");
    // Improve dragging on touch
    if (header.setPointerCapture && e.pointerId !== undefined) {
      try {
        header.setPointerCapture(e.pointerId);
      } catch {}
    }
    e.preventDefault();
  }

  function onKeyDownHandle(e) {
    if (!cardRef.current) return;
    const step = e.shiftKey ? 20 : 8;
    let { x, y } = posRef.current;
    if (e.key === "ArrowLeft") x -= step;
    if (e.key === "ArrowRight") x += step;
    if (e.key === "ArrowUp") y -= step;
    if (e.key === "ArrowDown") y += step;
    if (x !== posRef.current.x || y !== posRef.current.y) {
      const rect = cardRef.current.getBoundingClientRect();
      x = clamp(x, 8, window.innerWidth - rect.width - 8);
      y = clamp(y, 8, window.innerHeight - rect.height - 8);
      posRef.current = { x, y };
      setPos({ x, y });
      e.preventDefault();
    }
  }

  return (
    <div
      className="modal-overlay tutorial"
      onClick={() => !running && onClose?.()}
    >
      <div
        className="modal-card tutorial-card"
        ref={cardRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="modal-head tutorial-drag-handle"
          tabIndex={0} // 👈 add this
          onPointerDown={onDragStart}
          onTouchStart={onDragStart}
          onKeyDown={onKeyDownHandle}
          aria-label="Drag tutorial"
          aria-grabbed={draggingRef.current ? "true" : "false"}
        >
          <div className="modal-title">{cur.title}</div>
          <div className="modal-sub">
            {step + 1} / {steps.length}
          </div>
        </div>

        <div className="tutorial-body">
          <p className="tutorial-text">{cur.text}</p>

          {/* Capability hints if Board API is incomplete */}
          {!caps.setDemoState && (
            <div className="tutorial-hint">
              Tip: expose <code>setDemoState(state)</code> on{" "}
              <code>boardApi</code> to stage demo boards.
            </div>
          )}
          {!caps.animateSowFromPit && (
            <div className="tutorial-hint">
              Tip: expose{" "}
              <code>animateSowFromPit(player, index, count, opts)</code> for
              sowing demo.
            </div>
          )}
          {!caps.animateCapture && (
            <div className="tutorial-hint">
              Tip: expose{" "}
              <code>
                animateCapture(player, landingIndex, capturedOppCount, opts)
              </code>{" "}
              for capture demo.
            </div>
          )}
          {!caps.animateCollectRow && (
            <div className="tutorial-hint">
              Tip: expose <code>animateCollectRow(player, counts, opts)</code>{" "}
              for endgame sweep demo.
            </div>
          )}
        </div>

        <div
          className="modal-actions"
          style={{ justifyContent: "space-between" }}
        >
          <div className="tutorial-controls-left">
            <button
              className="btn btn--small"
              onClick={() => setStep((s) => Math.max(0, s - 1))}
              disabled={step === 0}
            >
              ‹ Prev
            </button>
            <button
              className="btn btn--small"
              onClick={() => setNonce((n) => n + 1)} // 👈 replay trigger
              // disabled={running}
              title="Replay this step"
            >
              ↺ Replay
            </button>
          </div>

          <div className="tutorial-controls-right">
            <button
              className="btn btn--small"
              onClick={() => onClose?.()}
              // disabled={running}
            >
              Exit
            </button>
            {step === steps.length - 1 ? (
              <button
                className="btn btn--accent"
                ref={finishBtnRef}
                onClick={() => {
                  // Prefer explicit finish callback; fall back to onClose.
                  if (onFinish) onFinish();
                  else onClose?.();
                }}
              >
                Back to Menu
              </button>
            ) : (
              <button
                className="btn btn--primary"
                onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))}
                disabled={step === steps.length - 1}
              >
                Next ›
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
