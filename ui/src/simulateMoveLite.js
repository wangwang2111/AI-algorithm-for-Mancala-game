// simulateMoveLite.js
export function simulateMoveLite(prev, player, action) {
  // Deep clone pits & stores
  const pits = [ [...(prev.pits?.[0]||[])], [...(prev.pits?.[1]||[])] ];
  const stores = [ prev.stores?.[0] || 0, prev.stores?.[1] || 0 ];

  // Build ring as [ {type:'pit'|'store', side, pitIndex?}, ... ]
  // Order (P0 perspective): bottom pits 0..5, P0 store, top pits 5..0, P1 store
  const ring = [];
  for (let i=0;i<6;i++) ring.push({type:'pit', side:0, pitIndex:i});
  ring.push({type:'store', side:0});
  for (let i=5;i>=0;i--) ring.push({type:'pit', side:1, pitIndex:i});
  ring.push({type:'store', side:1});

  // find starting ring index for (player, action)
  let startIdx = -1;
  for (let r=0; r<ring.length; r++) {
    const n = ring[r];
    if (n.type==='pit' && n.side===player && n.pitIndex===action) { startIdx = r; break; }
  }
  if (startIdx === -1) return null;

  // pick up stones
  // helper: advance to next placeable cell (skip opponent store)
  const nextPlace = (curIdx) => {
    let k = curIdx;
    while (true) {
      k = (k + 1) % ring.length;
      const cell = ring[k];
      if (cell.type === 'store' && cell.side !== player) continue; // skip opp store
      return k;
    }
  };

  let seeds = pits[player][action] || 0;
  pits[player][action] = 0;

  let idx = startIdx;
  let last = null;
  let capture = null;

  while (seeds > 0) {
    // choose the actual next destination (skipping opp store)
    idx = nextPlace(idx);
    const cell = ring[idx];

    // Special case: this is the LAST stone we're about to drop.
    if (seeds === 1 && cell.type === 'pit' && cell.side === player) {
      const i = cell.pitIndex;

      // Was that pit empty right before dropping?
      const wasEmpty = (pits[player][i] || 0) === 0;

      // Place the last stone
      pits[player][i] = (pits[player][i] || 0) + 1;
      last = cell;
      seeds = 0;

      if (wasEmpty) {
        const opp = 1 - player;
        const j = 5 - i;

        // Opposite pit stones *right now*, may include transient sowed ones
        const capturedOpp = pits[opp][j] || 0;

        // ✅ Capture if there are any stones there now (even if it was 0 in prev)
        if (capturedOpp >= 0) {
          pits[player][i] = 0;               // remove the landing stone
          pits[opp][j] = 0;                  // remove opponent stones
          stores[player] += capturedOpp + 1; // captured + landing
          capture = { landingIndex: i, capturedOpp: capturedOpp + 1 };
        }
      }
      break;
    }

    // Normal sow for non-final stones (or final stone into a store/opponent pit)
    if (cell.type === 'pit') {
      pits[cell.side][cell.pitIndex] += 1;
    } else {
      // own store
      stores[cell.side] += 1;
    }
    last = cell;
    seeds -= 1;
  }

  // Terminal check (pre-sweep)
  const sum = a => (a||[]).reduce((x,y)=>x+(y||0),0);
  const empty0 = sum(pits[0]) === 0;
  const empty1 = sum(pits[1]) === 0;
  let end = null;
  if (empty0 ^ empty1) {
    const collector = empty0 ? 1 : 0;
    end = { collector, counts: pits[collector].slice() }; // pre-sweep counts to animate
  }

  return { pits, stores, capture, end };
}
