// src/api.js
import axios from 'axios';

const BASE = import.meta.env?.VITE_API_BASE || `${window.location.origin}/api`;

const http = axios.create({
  baseURL: BASE,
  headers: { 'Content-Type': 'application/json' },
});

// Map API responses to always include .state
function normalize(data) {
  if (!data) return data;
  const st = data.state ?? data.next_state; // backend sends next_state for /apply and /move

  return st ? { ...data, state: st } : data;
}

const unwrap = p =>
  p.then(r => normalize(r.data)).catch(err => {
    console.log('API error', err);
    const msg =
      err?.response?.data?.message ||
      err?.response?.data?.error ||
      (typeof err?.response?.data === 'string' ? err.response.data : err.message);
    const e = new Error(msg);
    e.status = err?.response?.status;
    e.data = err?.response?.data;
    throw e;
  });

export const health    = () => unwrap(http.get('/health'));
export const newGame   = () => unwrap(http.post('/newgame', {}));
// /apply expects { state, action }
export const apply     = (state, action) => unwrap(http.post('/apply', { state, action }));
export const applyMove = (state, pitIdx) => apply(state, pitIdx);
// /move expects { state, agent } where agent ∈ {"dqn","minimax","alpha_beta","mcts","random","advanced"}
export const aiMove    = (state, agent) => unwrap(http.post('/move', { state, agent }));

const API = { health, newGame, apply, applyMove, aiMove, http, BASE };
export default API;
