function GameMenu({ mode, setMode, firstTurn, setFirstTurn, onStart, onTutorial, onQuit }) {
  return (
    <div className="menu-overlay">
      <div className="menu-card">
        <div className="menu-head">
          <div className="menu-title">Mancala AI</div>
          <div className="menu-sub">Choose a mode to begin</div>
        </div>

        <div className="menu-grid">
          <label className={`menu-tile ${mode==='hva' ? 'is-active' : ''}`} onClick={() => setMode('hva')}>
            <div className="tile-title">Human vs AI</div>
            <div className="tile-desc">AI auto-moves after your turn</div>
            <div className="tile-options">
              <span>First turn:</span>
              <div className="seg">
                <button
                  className={`seg-btn ${firstTurn==='human' ? 'on' : ''}`}
                  onClick={(e)=>{e.stopPropagation(); setFirstTurn('human');}}
                >Human</button>
                <button
                  className={`seg-btn ${firstTurn==='ai' ? 'on' : ''}`}
                  onClick={(e)=>{e.stopPropagation(); setFirstTurn('ai');}}
                >AI</button>
              </div>
            </div>
          </label>

          <label className={`menu-tile ${mode==='playground' ? 'is-active' : ''}`} onClick={() => setMode('playground')}>
            <div className="tile-title">Playground</div>
            <div className="tile-desc">Both sides manual; use “AI Move” button</div>
          </label>
        </div>

        <div className="menu-actions">
          <button className="btn btn--ghost" onClick={onTutorial}>Tutorial</button>
          <div style={{flex:1}} />
          <button
            className="btn btn--primary"
            disabled={!mode}
            onClick={onStart}
            title={mode ? '' : 'Pick a mode'}
          >Start</button>
          <button className="btn" onClick={onQuit}>Quit</button>
        </div>
      </div>
    </div>
  );
}

export default GameMenu;