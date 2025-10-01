import { agentLabels } from '../utils'

export default function ControlPanel({
  agent, setAgent, pit, setPit,
  legal, onHuman, onAI, onNew, modelBadge='model v1.1 • win_rate 0.83'
}) {
  return (
    <aside className="panel card">
      <div className="panel__row">
        <label>Agent:</label>
        <select value={agent} onChange={e => setAgent(e.target.value)}>
          {agentLabels.map(a => (
            <option key={a.value} value={a.value}>{a.label}</option>
          ))}
        </select>
      </div>

      <div className="panel__row">
        <label>Your pit:</label>
        <select value={pit} onChange={e => setPit(parseInt(e.target.value))}>
          {legal.length ? legal.map(i => <option key={i} value={i}>Pit {i}</option>)
                        : <option value={-1}>No legal pits</option>}
        </select>
      </div>

      <div className="panel__buttons">
        <button className="btn btn--primary" onClick={onHuman} disabled={pit<0}>Apply Human Move</button>
        <button className="btn btn--accent" onClick={onAI}>AI Move</button>
        <button className="btn btn--ghost" onClick={onNew}>New Game</button>
      </div>

      <div className="panel__badge">
        <span className="badge__icon">🔊</span>
        <span className="badge__text">{modelBadge}</span>
      </div>
    </aside>
  )
}
