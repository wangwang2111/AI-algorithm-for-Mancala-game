// simulateMoveLite.js
// Lightweight, rule-accurate simulator for Mancala/Kalah.
// Arrays are left->right for BOTH sides: pits[0][0..5], pits[1][0..5].
// Returns a PRE-SWEEP board so you can animate capture/sweep correctly.

/**
 * @typedef {{ pits:number[][], stores:number[], capture?:{landingIndex:number, capturedOpp:number}|null, end?:{collector:0|1, counts:number[]}|null, extraTurn:boolean }} SimResult
 */

/**
 * Simulate one move locally (pre-sweep).
 * @param {{pits:number[][], stores:number[]}} prev - previous state (numbers only)
 * @param {0|1} player - mover (0 = bottom, 1 = top)
 * @param {number} action - pit index on that player's row (0..5), left->right
 * @returns {SimResult|null}
 */
export function simulateMoveLite(prev, player, action) {
  // Defensive copies (numbers only)
  const pits = [ [...(prev.pits?.[0] || [])], [...(prev.pits?.[1] || [])] ];
  const stores = [ prev.stores?.[0] || 0, prev.stores?.[1] || 0 ];

  // Sanity checks
  if (player !== 0 && player !== 1) return null;
  if (action < 0 || action > 5) return null;
  if (!Array.isArray(pits[0]) || !Array.isArray(pits[1])) return null;

  // Position model
  const posPit   = (side, i) => ({ type: 'pit',   side, i });
  const posStore = (side)    => ({ type: 'store', side     });

  // Counter-clockwise stepping where BOTH rows are left->right (0..5).
  // Sequence: bottom 0..5 -> store0 -> top 0..5 -> store1 -> bottom 0..5 ...
  // We must SKIP the opponent's store for the current mover.
  function nextDrop(pos, mover) {
    let n;
    if (pos.type === 'pit') {
      // Move forward along the same side
      if (pos.i < 5) {
        n = posPit(pos.side, pos.i + 1);
      } else {
        // Reached the edge pit → go to that side's store
        n = posStore(pos.side);
      }
    } else {
      // From a store, jump to the other side's LEFTMOST pit (index 0)
      n = posPit(pos.side === 0 ? 1 : 0, 0);
    }
    // Skip opponent store
    if (n.type === 'store' && n.side !== mover) {
      return nextDrop(n, mover);
    }
    return n;
  }

  // Pick up all stones from the chosen pit
  let seeds = pits[player][action] || 0;
  pits[player][action] = 0;

  // Start position is "on" that pit; the next drop will move to its successor
  let pos = posPit(player, action);

  let capture = null;       // { landingIndex, capturedOpp } (opponent stones only)
  let extraTurn = false;    // last stone lands in own store

  // Sow all stones
  while (seeds > 0) {
    pos = nextDrop(pos, player);
    const isLast = (seeds === 1);

    if (pos.type === 'store') {
      // Opponent store is already skipped above; this is mover's store
      stores[pos.side] += 1;
      if (isLast && pos.side === player) {
        // Extra turn if the LAST stone lands in own store
        extraTurn = true;
      }
    } else {
      // Pit deposit
      if (isLast && pos.side === player) {
        // Capture check: pit must be EMPTY *right before* last drop
        const wasEmpty = (pits[player][pos.i] || 0) === 0;
        pits[player][pos.i] = (pits[player][pos.i] || 0) + 1;

        if (wasEmpty) {
          const opp = 1 - player;
          const j = 5 - pos.i;                 // opposite index (mirrored)
          const capturedOpp = pits[opp][j] || 0;  // includes transient sow stones
          if (capturedOpp > 0) {
            // Remove landing + opposite, credit to mover's store
            pits[player][pos.i] = 0;
            pits[opp][j] = 0;
            stores[player] += capturedOpp + 1;
            capture = { landingIndex: pos.i, capturedOpp }; // opponent stones only
          }
        }
      } else {
        // Normal non-final drop (or final into opponent pit)
        pits[pos.side][pos.i] += 1;
      }
    }

    seeds -= 1;
  }

  // Terminal detection (PRE-SWEEP): exactly one side empty
  const sum = a => (a || []).reduce((s, v) => s + (v || 0), 0);
  const empty0 = sum(pits[0]) === 0;
  const empty1 = sum(pits[1]) === 0;
  let end = null;
  if (empty0 ^ empty1) {
    const collector = empty0 ? 1 : 0;
    end = { collector, counts: pits[collector].slice() };
  }

  return { pits, stores, capture, end, extraTurn };
}

/* ------------------------------------------------------------------ */
/* Optional helpers if your API indexes P1 actions from the RIGHT side
   (comment out if not needed):

// Convert incoming action to left->right index for arrays
export function normalizeActionForArrays(player, actionFromUIorAPI) {
  return player === 1 ? (5 - actionFromUIorAPI) : actionFromUIorAPI;
}

// Convert internal left->right index back to UI (if your UI shows P1 right->left)
export function toUiIndex(player, arrayIndex) {
  return player === 1 ? (5 - arrayIndex) : arrayIndex;
}
*/
