import { create } from 'zustand'
import { newGame, applyMove, aiMove } from './api'

export const useGame = create((set, get) => ({
  state: null,
  loading: false,
  p0: { side: 'human', engine: 'alphabeta_ordered' },
  p1: { side: 'ai',    engine: 'minimax' },

  async init(p0, p1) {
    set(s => ({ ...s, loading: true, p0: { ...s.p0, ...p0 }, p1: { ...s.p1, ...p1 } }))
    const ng = await newGame()
    set({ state: ng.state, loading: false })
  },

  async playPit(pitIdx) {
    const s = get().state
    if (!s) return
    set({ loading: true })
    const r = await applyMove(s, pitIdx)
    set({ state: r.state, loading: false })
    await get().stepAI()
  },

  async stepAI() {
    const st = get().state
    if (!st) return
    const { p0, p1 } = get()
    const now = st.current_player
    const side = now === 0 ? p0 : p1
    if (side.side !== 'ai' || !side.engine) return
    set({ loading: true })
    const r = await aiMove(st, side.engine)
    set({ state: r.state, loading: false })
    // Extra turn? Let AI move again.
    if (r.state.current_player === now) await get().stepAI()
  },
}))
