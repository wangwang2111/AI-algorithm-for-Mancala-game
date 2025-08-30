// ui-static/js/api.js
(() => {
  const BASE =
    (typeof window.API_BASE === "string" && window.API_BASE) ||
    `${location.origin}/api`; // fallback to same-origin /api

  const API = {
    async health() {
      const r = await fetch(`${BASE}/health`);
      if (!r.ok) throw new Error("health failed");
      return r.json();
    },
    async newGame() {
      const r = await fetch(`${BASE}/newgame`, { method: "POST" });
      if (!r.ok) throw new Error("newgame failed");
      return r.json();
    },
    async apply(state, action) {
      const r = await fetch(`${BASE}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, action }),
      });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    },
    async move(state, agent) {
      const r = await fetch(`${BASE}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, agent }),
      });
      if (!r.ok) throw new Error(await r.text());
      return r.json();
    },
  };

  window.API = API;
})();
