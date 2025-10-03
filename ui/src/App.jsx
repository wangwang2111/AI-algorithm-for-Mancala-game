import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import { newGame, applyMove, aiMove } from "./api";
import { legalPits } from "./utils";
import { simulateMoveLite } from "./simulateMoveLite";
import Board from "./components/Board";
import ControlPanel from "./components/ControlPanel";
import MoveLog from "./components/MoveLog";
import { launchFireworks } from "./ui/fireworks";
import GameMenu from "./components/Menu";
import DomTutorial from "./components/DomTutorial";
import "./styles.css";

function useAudio() {
  const musicRef = useRef(null);
  const [musicMuted, setMusicMuted] = useState(false);
  const [sfxMuted, setSfxMuted] = useState(false);

  useEffect(() => {
    // lazily create a looping music track (replace src with your file)
    if (!musicRef.current) {
      const a = new Audio("/assets/music.mp3"); // ← replace path
      a.loop = true;
      a.volume = 0.25;
      musicRef.current = a;
    }
    musicRef.current.muted = musicMuted;
  }, [musicMuted]);

  const ensureMusic = useCallback(() => {
    // start music on first user gesture (browsers require it)
    if (musicRef.current && musicRef.current.paused) {
      musicRef.current.play().catch(() => {
        /* ignore autoplay block */
      });
    }
  }, []);

  const playSfx = useCallback(
    (name) => {
      if (sfxMuted) return;
      // very lightweight: one-shot HTMLAudio (replace with real assets)
      const m = {
        click: "/assets/click.mp3",
        capture: "/assets/capture.mp3",
        sweep: "/assets/sweep.mp3",
      };
      const src = m[name];
      if (!src) return;
      const a = new Audio(src);
      a.volume = 0.6;
      a.play().catch(() => {});
    },
    [sfxMuted]
  );

  return {
    musicMuted,
    setMusicMuted,
    sfxMuted,
    setSfxMuted,
    ensureMusic,
    playSfx,
  };
}

export default function App() {
  const [state, setState] = useState(null);
  const [agent, setAgent] = useState("alpha_beta"); // valid: dqn|minimax|alpha_beta|mcts|random|advanced
  const [pit, setPit] = useState(-1);
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // --- Mode / Menu state ---
  const [menuOpen, setMenuOpen] = useState(true);
  const [mode, setMode] = useState(null); // 'hva' | 'playground' | null
  const [firstTurn, setFirstTurn] = useState("human"); // 'human' | 'ai'
  // 0 = Player 0 (bottom), 1 = Player 1 (top)
  const [humanSide, setHumanSide] = useState(0); 
  const aiSide = 1 - humanSide;

  // tutorial + sidebar
  const [showTutorial, setShowTutorial] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // for winner fireworks (you likely already have these)
  const [winner, setWinner] = useState(null);
  const [endedOnce, setEndedOnce] = useState(false);
  const boardShellRef = useRef(null); // wrap around the Board + stores layout

  // audio
  const {
    musicMuted,
    setMusicMuted,
    sfxMuted,
    setSfxMuted,
    ensureMusic,
    playSfx,
  } = useAudio();

  // derived
  const legal = useMemo(() => legalPits(state), [state]);
  const turnText = state ? (state.current_player === 0 ? "P0" : "P1") : "—";
  const boardApi = useRef(null);

  const overlayActive = menuOpen || winner !== null;
  const tutorialActive = showTutorial;

  const isTerminal = useMemo(() => {
    if (!state) return false;
    // Backend may also return {done:true}; prefer that if available
    const pits0 = state.pits?.[0] || [];
    const pits1 = state.pits?.[1] || [];
    const sideEmpty = (arr) => arr.reduce((a, b) => a + b, 0) === 0;
    return sideEmpty(pits0) || sideEmpty(pits1);
  }, [state]);

  // initial new game
  useEffect(() => {
    handleNew();
  }, []); // eslint-disable-line

  // auto-pick first legal pit when state or turn changes
  useEffect(() => {
    setPit(legal.length ? legal[0] : -1);
  }, [state?.current_player, legal.length]); // enough to re-run on turn/availability changes

  useEffect(() => {
    if (!showTutorial) return;
    const firstBtn = document.querySelector(".modal-card .btn");
    firstBtn?.focus();
  }, [showTutorial]);

  useEffect(() => {
    document.documentElement.style.overflow = overlayActive ? "hidden" : "";
  }, [overlayActive]);

  // Helper
  // helper to compute winner once the API state is swept
  function computeWinner(s) {
    if (!s) return null;
    const p0 = s.stores?.[0] ?? 0;
    const p1 = s.stores?.[1] ?? 0;
    if (
      (s.pits?.[0] || []).every((v) => v === 0) &&
      (s.pits?.[1] || []).every((v) => v === 0)
    ) {
      if (p0 > p1) return 0;
      if (p1 > p0) return 1;
      return "draw";
    }
    return null;
  }

  const handleNew = useCallback(async () => {
    setError("");
    setBusy(true);
    try {
      const ng = await newGame();
      if (!ng?.state) throw new Error("Bad response: no state");
      setState(ng.state);
      setLog([]);
      setWinner(null);
      setEndedOnce(false);

      // Hydrate the dots on the next frame so the layers are present
      requestAnimationFrame(() => {
        // wipe demo/fake dots and lay down the new server state
        boardApi.current?.setDemoState?.(ng.state) ||
          boardApi.current?.hydrate?.(ng.state);
      });
      // kick off bg music after first user action
      ensureMusic();

      // If we chose Human vs AI and AI goes first, auto-trigger AI after mount
      if (mode === "hva" && firstTurn === "ai") {
        // Wait a tick so Board renders
        setTimeout(() => {
          handleAI(ng.state);
        }, 500);
      }
    } catch (e) {
      setError(e?.message || "Failed to start new game");
    } finally {
      setBusy(false);
    }
  }, [mode, firstTurn, ensureMusic]);

  const handleAI = useCallback(async (forcedPrevState = null) => {
    if (!state || busy || isTerminal) return;
    setError("");
    setBusy(true);
    try {
      const prev = forcedPrevState || state;
      const r = await aiMove(prev, agent);
      if (!r?.state) throw new Error("Bad response: no state");
      const next = r.state;
      const player = prev.current_player;

      const action =
        r.action ?? r.move ?? r.playIdx;
      console.log("AI chose action", action); // --- IGNORE ---

      if (action >= 0) {
        const sim = simulateMoveLite(prev, player, action);
        const { capture, end } = await sim;
        console.log("sim", sim); // --- IGNORE ---
        console.log("capture", capture, "end", end); // --- IGNORE ---
        // sow
        const seeds = prev.pits[player][action];
        await boardApi.current?.animateSowFromPit(player, action, seeds, {
          stagger: 200,
          hopMs: 550,
        });

        // capture
        if (capture) {
          await boardApi.current?.animateCapture?.(
            player,
            capture.landingIndex,
            capture.capturedOpp,
            { stagger: 200, hopMs: 500 }
          );
        }

        // Animate end sweep using **sim.end** (pre-sweep counts!)
        if (end) {
          await boardApi.current?.animateCollectRow?.(
            end.collector,
            end.counts,
            { pitStagger: 200, stoneStagger: 200, hopMs: 500 }
          );
        }
      }

      // commit server state (already swept)
      setState(next);
      setLog((L) => [
        `P${player} (${agent}) AI moved${action >= 0 ? ` pit ${action}` : ""}`,
        ...L,
      ]);

      // Winner check AFTER commit (server has already swept if terminal)
      const w = computeWinner(next);
      if (w !== null && !endedOnce) {
        setWinner(w);
        setEndedOnce(true);

        // Optional: launch fireworks over the board container
        const cleanup = launchFireworks(boardShellRef.current, {
          durationMs: 5000,
        });
        setTimeout(() => cleanup && cleanup(), 6000);
        setLog((L) => [w === "draw" ? `Draw` : `Player ${w} wins`, ...L]);
      }

      // HvA: if it's AI's turn now, auto-play AI
      if (mode === "hva" && next.current_player === player) {
        // Give the render a frame
        setTimeout(() => {
          handleAI(next);
        }, 500);
      }
    } catch (e) {
      setError(e?.message || "AI move failed");
    } finally {
      setBusy(false);
    }
  }, [state, agent, busy, isTerminal]);

  const handleHuman = useCallback(
    async (clickedIdx) => {
      if (!state || busy || isTerminal) return;
      // HvA: ignore clicks if not human turn
      if (mode === 'hva' && state.current_player !== humanSide) return;

      const player = state.current_player; // now supports P0 or P1
      const playIdx = typeof clickedIdx === "number" ? clickedIdx : pit;

      const legal = (state.pits?.[player] || [])
        .map((v, i) => (v > 0 ? i : -1))
        .filter((i) => i >= 0);
      if (!legal.includes(playIdx)) {
        setError(`Illegal move: Pit ${playIdx}. Legal: [${legal.join(", ")}]`);
        return;
      }

      setError("");
      setBusy(true);
      try {
        const prev = state;

        // use sim for capture/sweep animation pre-visualization
        const sim = simulateMoveLite(prev, player, playIdx);
        const { capture, end } = sim || {};

        console.log("sim", sim); // --- IGNORE ---
        console.log("capture", capture, "end", end); // --- IGNORE ---
        // sow/capture/collect animations here if you wired them
        if (capture)
          await boardApi.current?.animateCapture?.(
            player,
            capture.landingIndex,
            capture.capturedOpp,
            { stagger: 180, hopMs: 460 }
          );
        if (end)
          await boardApi.current?.animateCollectRow?.(
            end.collector,
            end.counts,
            { pitStagger: 160, stoneStagger: 60, hopMs: 460 }
          );

        // Apply move to server
        const r = await applyMove(prev, playIdx);
        const next = r.state;
        if (!next) throw new Error("Bad response: no state");

        setState(next);
        setPit(playIdx);
        setLog((L) => [`P${player} plays Pit ${playIdx}`, ...L]);

        // Winner check (server will have swept if terminal)
        const w = computeWinner(next);
        if (w !== null && !endedOnce) {
          setWinner(w);
          setEndedOnce(true);
          const cleanup = launchFireworks(boardShellRef.current, {
            durationMs: 4000,
          });
          setTimeout(() => cleanup && cleanup(), 6000);
          setLog((L) => [w === "draw" ? `Draw` : `Player ${w} wins`, ...L]);
          return; // stop here; game is over
        }

        // HvA: if it's AI's turn now, auto-play AI
        if (mode === 'hva' && next.current_player === aiSide) {
          // Give the render a frame
          setTimeout(() => {
            handleAI(next);
          }, 500);
        }
      } catch (e) {
        setError(e?.message || "Move failed");
      } finally {
        setBusy(false);
      }
    },
    [state, pit, busy, isTerminal, mode, endedOnce]
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="title">Mancala AI</div>
        <div className="badge">
          <span>model v1.1 • win_rate 0.83</span>
        </div>

        {/* topbar actions */}
        <div className="topbar-actions">
          <button
            className="btn btn--small"
            onClick={() => {
              setMenuOpen(true);
            }}
          >
            {"←"} Menu
          </button>
          <button
            className="btn btn--small"
            onClick={() => setShowTutorial(true)}
          >
            Tutorial
          </button>
          <button
            className="btn btn--small"
            onClick={() => setMusicMuted((m) => !m)}
          >
            {musicMuted ? "Unmute Music" : "Mute Music"}
          </button>
          <button
            className="btn btn--small"
            onClick={() => setSfxMuted((s) => !s)}
          >
            {sfxMuted ? "Unmute SFX" : "Mute SFX"}
          </button>
          <button
            className="btn btn--small only-mobile"
            onClick={() => setSidebarOpen((v) => !v)}
          >
            {sidebarOpen ? "Close Panel" : "Panel"}
          </button>
        </div>
      </header>

      <div
        className={`content ${tutorialActive ? 'tutorial-bg' : ''}`}
        inert={!!overlayActive}
      >
        <div className="left">
          <div
            className="board-shell"
            ref={boardShellRef}
            style={{ position: "relative" }}
          >
            <Board
              ref={boardApi}
              state={state}
              canPlay={
                !!state && 
                (mode === 'playground' ? true : state.current_player === humanSide)
              }
              onPlay={(idx) => {
                // toggle panel on small screens after a move
                if (sidebarOpen) setSidebarOpen(false);
                // your handler:
                // handleHuman(idx);
                handleHuman(idx);
              }}
              mode={mode}
            />
          </div>
          <div className="status">
            <span className={`dot ${busy ? "animate-pulse" : ""}`} />
            <span>Turn: {turnText}</span>
            <span className="legal">Legal: [{legal.join(", ")}]</span>
            {isTerminal && (
              <span style={{ marginLeft: 8, opacity: 0.8 }}>(game over)</span>
            )}
          </div>

          {!!error && (
            <div
              className="card"
              style={{ color: "#ffb4a2", borderColor: "rgba(255,110,64,.35)" }}
            >
              {error}
            </div>
          )}

          <MoveLog log={log} />
        </div>

        <div className="right desktop-only">
          <ControlPanel
            agent={agent}
            setAgent={setAgent}
            pit={pit}
            setPit={setPit}
            legal={legal}
            onHuman={handleHuman}
            onAI={handleAI}
            onNew={() => {
              setMenuOpen(false);
              handleNew();
            }}
            mode={mode}
            onBackToMenu={() => setMenuOpen(true)}
          />
          <div style={{ marginTop: 10, fontSize: 12, opacity: 0.7 }}>
            {busy ? "thinking…" : "ready"}
          </div>
        </div>

        {/* OFF-CANVAS SIDEBAR (iPad & smaller) */}
        <aside className={`offcanvas ${sidebarOpen ? "open" : ""}`}>
          <ControlPanel
            agent={agent}
            setAgent={setAgent}
            pit={pit}
            setPit={setPit}
            legal={legal}
            onHuman={handleHuman}
            onAI={handleAI}
            onNew={() => setMenuOpen(true)}
            mode={mode}
            onBackToMenu={() => setMenuOpen(true)}
          />
        </aside>
      </div>

      {/* Overlays OUTSIDE of .content */}
      {menuOpen && (
        <GameMenu
          mode={mode}
          setMode={setMode}
          firstTurn={firstTurn}
          setFirstTurn={setFirstTurn}
          onStart={() => {
            setMenuOpen(false);
            handleNew();
          }}
          onTutorial={() => {
            setMenuOpen(false);
            setShowTutorial(true);
          }} // close menu before tutorial
          onQuit={() => {
            setMenuOpen(true);
            setMode(null);
            setWinner(null);
            setEndedOnce(false);
          }}
        />
      )}

      {showTutorial && (
        <DomTutorial
          boardApi={boardApi.current}
          liveState={state} // pass the current real game state
          onClose={() => {
            // re-hydrate the *real* state so demo scatters don’t linger
            if (boardApi.current && state) {
              boardApi.current.hydrate?.(state);
            }
            setShowTutorial(false);
          }}
        />
      )}

      {winner !== null && (
        <div className="winner-overlay">
          <div className="winner-card">
            {winner === "draw" ? "It’s a draw!" : `Player ${winner} wins! 🎉`}
            <button
              className="btn btn--accent"
              onClick={() => {
                setWinner(null);
                setEndedOnce(false);
                handleNew(); // start a new game
              }}
            >
              New Game
            </button>
            <button
              className="btn btn--accent"
              onClick={() => {
                setWinner(null);
                setEndedOnce(false);
                setMenuOpen(true);
              }}
            >
              Back to Menu
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
