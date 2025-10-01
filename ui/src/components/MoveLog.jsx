export default function MoveLog({ log }) {
  return (
    <div className="log card">
      <div className="log__title">Move Log</div>
      <div className="log__scroll">
        {log.length === 0 && <div className="log__empty">—</div>}
        {log.map((line, i) => (
          <div className="log__line" key={i}>{line}</div>
        ))}
      </div>
    </div>
  )
}
