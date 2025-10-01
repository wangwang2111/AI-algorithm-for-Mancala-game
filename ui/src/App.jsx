import { useEffect, useMemo, useState, useCallback, useRef } from 'react'
import { newGame, applyMove, aiMove } from './api'
import { legalPits } from './utils'
import Board from './components/Board'
import ControlPanel from './components/ControlPanel'
import MoveLog from './components/MoveLog'
import './styles.css'


export default function App() {
  const [state, setState] = useState(null)
  const [agent, setAgent] = useState('alpha_beta') // valid: dqn|minimax|alpha_beta|mcts|random|advanced
  const [pit, setPit] = useState(-1)
  const [log, setLog] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  
  // derived
  const legal = useMemo(() => legalPits(state), [state])
  const turnText = state ? (state.current_player === 0 ? 'P0' : 'P1') : '—'
  const boardApi = useRef(null)
  
  // initial new game
  useEffect(() => { handleNew() }, []) // eslint-disable-line

  // auto-pick first legal pit when state or turn changes
  useEffect(() => {
    setPit(legal.length ? legal[0] : -1)
  }, [state?.current_player, legal.length]) // enough to re-run on turn/availability changes
  
  const isTerminal = useMemo(() => {
    if (!state) return false
    // Backend may also return {done:true}; prefer that if available
    const pits0 = state.pits?.[0] || []
    const pits1 = state.pits?.[1] || []
    const sideEmpty = (arr) => arr.reduce((a,b)=>a+b,0) === 0
    return sideEmpty(pits0) || sideEmpty(pits1)
  }, [state])
  

  // Helper
  // Robust capture detector: works even if the opposite pit was 0 in prev,
  // got +1 earlier in the sow, and is 0 again in next.
  function detectCapture(prev, next, player) {
    if (!prev || !next) return null;
    const opp = 1 - player;

    const prevOwn = prev.pits?.[player] || [];
    const nextOwn = next.pits?.[player] || [];
    const prevOpp = prev.pits?.[opp]    || [];
    const nextOpp = next.pits?.[opp]    || [];

    const storeGain = (next.stores?.[player] ?? 0) - (prev.stores?.[player] ?? 0);
    console.log('detectCapture: player', player, 'storeGain', storeGain); // --- IGNORE ---
    if (storeGain <= 0) return null;

    // Build candidate landing pits: own i with 0 -> 0 and opposite j with next==0
    const candidates = [];
    for (let i = 0; i < 6; i++) {
      const ownWas = prevOwn[i] ?? 0;
      const ownNow = nextOwn[i] ?? 0;
      if (!(ownWas === 0 && ownNow === 0)) continue; // capture landing pit must stay 0

      const j = 5 - i; // opposite index on opponent side
      const oppNow = nextOpp[j] ?? 0;
      if (oppNow !== 0) continue; // after capture, opposite pit must be empty

      // How many did we capture from opposite?
      // If prev had stones, that's the captured amount.
      // If prev was 0 (transient deposit during sow), we infer captured=1.
      const oppWas = prevOpp[j] ?? 0;
      const capturedOpp = (oppWas > 0) ? oppWas : 1; // at least 1 from opposite, plus 1 from landing

      // Store must have gained at least (capturedOpp + 1 last stone)
      if (storeGain >= capturedOpp + 1) {
        candidates.push({ landingIndex: i, capturedOpp, score: storeGain - (capturedOpp + 1) });
      }
    }

    if (candidates.length === 0) return null;
    if (candidates.length === 1) return candidates[0];

    // If multiple fit, prefer the one that used real prevOpp>0 (larger capture),
    // else the one whose (capturedOpp+1) best explains the storeGain (smallest non-negative score).
    candidates.sort((a, b) => {
      if (a.capturedOpp !== b.capturedOpp) return b.capturedOpp - a.capturedOpp;
      return a.score - b.score;
    });
    return candidates[0];
  }

  // Helper
  function sum(arr){ return (arr || []).reduce((a,b)=> a + (b||0), 0); }
  function allZero(arr){ return sum(arr) === 0; }

  /**
   * Decide which side collects and which counts to animate FROM.
   * Returns: { collector: 0|1, counts: number[6] } or null if no end-game sweep.
   */
  function detectEndCollection(prev, next){
    if (!prev || !next) return null

    const p0n = next.pits?.[0] || []
    const p1n = next.pits?.[1] || []
    const p0Empty = allZero(p0n)
    const p1Empty = allZero(p1n)

    // both sides empty
    if (p0Empty && !p1Empty){
      // P1 collects remaining (prefer next counts if server hasn't swept yet)
      const counts = allZero(p1n) ? (prev.pits?.[1] || [0,0,0,0,0,0]) : p1n
      return { collector: 1, counts }
    }
    if (p1Empty && !p0Empty){
      const counts = allZero(p0n) ? (prev.pits?.[0] || [0,0,0,0,0,0]) : p0n
      return { collector: 0, counts }
    }
    return null
  }

  const handleNew = useCallback(async () => {
    setError('')
    setBusy(true)
    try {
      const ng = await newGame()
      if (!ng?.state) throw new Error('Bad response: no state')
      setState(ng.state)
      // Hydrate the dots on the next frame so the layers are present
      requestAnimationFrame(() => {
        boardApi.current?.hydrate?.(ng.state);
      });
      setLog([])
    } catch (e) {
      setError(e?.message || 'Failed to start new game')
    } finally {
      setBusy(false)
    }
  }, [])

  const handleHuman = useCallback(async (clickedIdx) => {
    if (!state || busy || isTerminal) return;

    const player = state.current_player;           // 0 or 1 (now supports P1 too)
    const playIdx = typeof clickedIdx === 'number' ? clickedIdx : pit;
    const legal = (state.pits?.[player] || []).map((v,i)=>v>0?i:-1).filter(i=>i>=0);
    if (!legal.includes(playIdx)) { setError(`Illegal move: Pit ${playIdx}. Legal: [${legal.join(', ')}]`); return; }

    setError('');
    setBusy(true);
    try {
      const prev = state;

      // // 1) sow animation
      // const seeds = prev.pits[player][playIdx];
      // await boardApi.current?.animateSowFromPit(player, playIdx, seeds, { stagger:150, hopMs:460 });

      // 2) server apply
      const r = await applyMove(prev, playIdx);
      if (!r?.state) throw new Error('Bad response: no state');
      const next = r.state;

      // 3) capture
      const cap = detectCapture(prev, next, player);
      if (cap) {
        await boardApi.current?.animateCapture?.(player, cap.landingIndex, cap.capturedOpp, { stagger:90, hopMs:420 });
      }

      // 4) end-game sweep (collector may be the *other* player!)
      const end = detectEndCollection(prev, next);
      if (end) {
        await boardApi.current?.animateCollectRow?.(end.collector, end.counts, { pitStagger:120, stoneStagger:45, hopMs:420 });
      }

      // 5) commit
      setState(next);
      setPit(playIdx);
      setLog(L => [`P${player} plays Pit ${playIdx}`, ...L]);
    } catch (e) {
      setError(e?.message || 'Move failed');
    } finally {
      setBusy(false);
    }
  }, [state, pit, busy, isTerminal]);


  const handleAI = useCallback(async () => {
    if (!state || busy || isTerminal) return
    setError('')
    setBusy(true)
    try {
      const mover = state.current_player === 0 ? 'P0' : 'P1'

      const prev = state
      const prevTurn = prev.current_player

      const r = await aiMove(prev, agent) // ideally returns { state, action }
      if (!r?.state) throw new Error('Bad response: no state')
      const next = r.state

      // Determine which pit AI used
      const action = (r.action ?? r.move ?? r.playIdx)
      if (action >= 0) {
        const seedCount = prev.pits[prevTurn][action]
        // Animate for the AI side
        await boardApi.current?.animateSowFromPit(prevTurn, action, seedCount, { stagger: 150, hopMs: 460 })
      }

      // capture animation (if any), still using prev DOM
      const cap = detectCapture(prev, next, prevTurn);
      if (cap) {
        await boardApi.current?.animateCapture?.(prevTurn, cap.landingIndex, cap.capturedOpp, { stagger: 90, hopMs: 420 });
      }

      // terminal sweep
      const end = detectEndCollection(prev, next);
      if (end) {
        await boardApi.current?.animateCollectRow?.(end.collector, end.counts, {
          pitStagger: 120, stoneStagger: 45, hopMs: 420
        });
      }
      
      // Now commit the new state
      setState(next)
      setLog(L => [`${mover} (${agent}) AI moved${action>=0?` pit ${action}`:''}`, ...L])

    } catch (e) {
      setError(e?.message || 'AI move failed')
    } finally {
      setBusy(false)
    }
  }, [state, agent, busy, isTerminal])


  return (
    <div className="app">
      <header className="topbar">
        <div className="title">Mancala AI</div>
        <div className="badge"><span>model v1.1 • win_rate 0.83</span></div>
      </header>

      <div className="content">
        <div className="left">
          <Board
            ref={boardApi}
            state={state}
            canPlay={!!state && !busy && !isTerminal}
            onPlay={(idx) => handleHuman(idx)}
          />

          <div className="status">
            <span className={`dot ${busy ? 'animate-pulse' : ''}`} />
            <span>Turn: {turnText}</span>
            <span className="legal">Legal: [{legal.join(', ')}]</span>
            {isTerminal && <span style={{marginLeft:8, opacity:.8}}>(game over)</span>}
          </div>

          {!!error && (
            <div className="card" style={{color:'#ffb4a2', borderColor:'rgba(255,110,64,.35)'}}>
              {error}
            </div>
          )}

          <MoveLog log={log} />
        </div>

        <div className="right">
          <ControlPanel
            agent={agent}
            setAgent={setAgent}
            pit={pit}
            setPit={setPit}
            legal={legal}
            onHuman={handleHuman}
            onAI={handleAI}
            onNew={handleNew}
          />
          <div style={{marginTop:10, fontSize:12, opacity:.7}}>
            {busy ? 'thinking…' : 'ready'}
          </div>
        </div>
      </div>
    </div>
  )
}
