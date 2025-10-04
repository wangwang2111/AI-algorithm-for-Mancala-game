// utils.js
export const uiToModel = (side, uiIdx) => (side === 0 ? uiIdx : 5 - uiIdx);
export const modelToUi = (side, modelIdx) => (side === 0 ? modelIdx : 5 - modelIdx);

export function legalPits(state) {
  if (!state) return []
  const cp = state.current_player
  const pits = state.pits[cp] || []
  const legal = []

  for (let i = 0; i < pits.length; i++) {
    if (pits[i] > 0) legal.push(i)
  }
  return legal
}

export const agentLabels = [
  { label: 'Random', value: 'random' },
  { label: 'Minimax', value: 'minimax' },
  { label: 'AlphaBeta', value: 'alpha_beta' },
  { label: 'DQN', value: 'dqn' },
  { label: 'MCTS', value: 'mcts' },
  { label: 'Advanced Heuristic Minimax', value: 'advanced' },
]
