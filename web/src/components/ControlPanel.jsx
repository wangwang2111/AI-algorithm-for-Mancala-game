// ControlPanel.jsx (only the bits to add)

export default function ControlPanel({
  agent, setAgent,
  pit, setPit, legal,
  onHuman, onAI, onNew,
  mode,
  onBackToMenu,
}) {
  return (
    <div className="card panel">
      <div className="panel__row">
        <label>Mode</label>
        <div>{mode === 'hva' ? 'Human vs AI' : 'Playground'}</div>
      </div>

      <div className="panel__row">
        <label>Agent</label>
        <select value={agent} onChange={e=>setAgent(e.target.value)}>
          <option value="dqn">DQN</option>
          <option value="minimax">Minimax</option>
          <option value="alpha_beta">Alpha-Beta</option>
          <option value="mcts">MCTS</option>
          <option value="random">Random</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>

      <div className="panel__row">
        <label>Choose Pit</label>
        <select value={pit} onChange={e=>setPit(+e.target.value)}>
          {legal.map(i => <option key={i} value={i}>{i}</option>)}
        </select>
        <div className="panel__buttons">
          <button className="btn btn--primary" onClick={()=> onHuman(pit)}>Play</button>
          {mode === 'playground' && (
            <button className="btn btn--accent" onClick={onAI}>AI Move</button>
          )}
          <button className="btn" onClick={onNew}>New Game</button>
          <button className="btn btn--ghost" onClick={onBackToMenu}>Back to Menu</button>
        </div>
      </div>
    </div>
  );
}
