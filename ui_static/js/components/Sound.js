// ui-static/js/components/Sound.js
// lightweight sound manager (no libs). handles autoplay policies, pooling, mute, bg loop.

const Sound = (() => {
  const ROOT = "./assets/sounds/";

  // config per sound
  const cfg = {
    bg:       { file: "background-sound.mp3", loop: true, volume: 0.15 },
    stone:    { file: "stone-sound.mp3",      pool: 6,   volume: 0.55 },
    restart:  { file: "restart-sound.mp3",    volume: 0.7 },
    victory:  { file: "victory-sound.mp3",    volume: 0.9 },
    gameover: { file: "game-over-sound.mp3",  volume: 0.8 },
  };

  const state = {
    ready: false,
    muted: false,
    pools: new Map(),     // name -> { list: Audio[], idx: number }
    bg: null,             // background Audio
    unlocked: false,      // user gesture passed autoplay policy
  };

  function _createAudio(src, { loop=false, volume=1 } = {}) {
    const a = new Audio(src);
    a.loop = loop;
    a.volume = volume;
    a.preload = "auto";
    return a;
  }

  function _makePool(name, file, count, volume) {
    const list = [];
    for (let i = 0; i < count; i++) {
      const a = _createAudio(ROOT + file, { volume });
      list.push(a);
    }
    state.pools.set(name, { list, idx: 0, volume });
  }

  function init() {
    if (state.ready) return;
    // build pools
    if (cfg.stone.pool) _makePool("stone", cfg.stone.file, cfg.stone.pool, cfg.stone.volume);
    // build bg but don't play yet (autoplay restrictions)
    state.bg = _createAudio(ROOT + cfg.bg.file, { loop: cfg.bg.loop, volume: cfg.bg.volume });
    // single-shot sounds (no pool)
    for (const n of ["restart","victory","gameover"]) {
      const c = cfg[n];
      state.pools.set(n, { list: [_createAudio(ROOT + c.file, { volume: c.volume })], idx: 0, volume: c.volume });
    }
    // remember mute across reloads
    state.muted = localStorage.getItem("mancala-muted") === "1";
    _applyMute();
    // unlock audio on first user gesture
    const unlock = () => {
      if (state.unlocked) return;
      state.unlocked = true;
      // try to start background softly
      try { if (!state.muted) state.bg.play().catch(()=>{}); } catch {}
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
    window.addEventListener("pointerdown", unlock, { once: true });
    window.addEventListener("keydown", unlock, { once: true });

    state.ready = true;
  }

  function _applyMute() {
    const all = [];
    if (state.bg) all.push(state.bg);
    for (const { list } of state.pools.values()) all.push(...list);
    for (const a of all) a.muted = state.muted;
    localStorage.setItem("mancala-muted", state.muted ? "1" : "0");
  }

  function toggleMute() { state.muted = !state.muted; _applyMute(); return state.muted; }
  function mute(v=true) { state.muted = !!v; _applyMute(); }

  function play(name) {
    if (!state.ready) init();
    const pool = state.pools.get(name);
    if (!pool) return;

    // rotate through pool for overlapping plays
    const a = pool.list[pool.idx];
    pool.idx = (pool.idx + 1) % pool.list.length;
    a.currentTime = 0;
    a.play().catch(()=>{});
  }

  function startBackground() {
    if (!state.ready) init();
    try { if (!state.muted) state.bg.play().catch(()=>{}); } catch {}
  }
  function stopBackground() {
    try { state.bg.pause(); state.bg.currentTime = 0; } catch {}
  }

  return { init, play, toggleMute, mute, startBackground, stopBackground };
})();

window.Sound = Sound;
