// src/ui/fireworks.js
let raf = null;

export function launchFireworks(rootEl, { durationMs = 4000 } = {}) {
  if (!rootEl) return () => {};
  // Canvas overlay
  const c = document.createElement('canvas');
  c.className = 'fx-fireworks';
  rootEl.appendChild(c);

  const ctx = c.getContext('2d');
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  const resize = () => {
    const r = rootEl.getBoundingClientRect();
    c.style.position = 'absolute';
    c.style.inset = '0';
    c.style.pointerEvents = 'none';
    c.style.zIndex = '50';
    c.width = Math.max(1, Math.floor(r.width * dpr));
    c.height = Math.max(1, Math.floor(r.height * dpr));
    c.style.width = r.width + 'px';
    c.style.height = r.height + 'px';
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(rootEl);

  // simple particle system
  const parts = [];
  const now = () => performance.now();
  const T0 = now();

  const spawnBurst = (x, y) => {
    const N = 80 + (Math.random()*40|0);
    for (let i=0;i<N;i++){
      const a = Math.random() * Math.PI * 2;
      const sp = 1.5 + Math.random() * 3.5;
      parts.push({
        x, y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        life: 900 + Math.random()*900,
        born: now(),
        r: 1.5 + Math.random()*2.2,
        hue: (Math.random()*360)|0,
      });
    }
  };

  // spawn a few bursts across the board
  const spawnLoop = () => {
    const W = c.width, H = c.height;
    const bursts = 3 + (Math.random()*2|0);
    for (let i=0;i<bursts;i++) {
      const x = (0.15 + Math.random()*0.7) * W;
      const y = (0.15 + Math.random()*0.4) * H; // upper half
      spawnBurst(x, y);
    }
  };

  let lastSpawn = T0;
  const SPANW_EVERY = 700; // ms

  function step(t) {
    // stop condition
    if (t - T0 > durationMs && parts.length === 0) {
      cleanup();
      return;
    }

    // timed spawns only during active window
    if (t - T0 < durationMs && t - lastSpawn > SPANW_EVERY) {
      spawnLoop();
      lastSpawn = t;
    }

    // draw
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(0,0,0,0.20)';
    ctx.fillRect(0,0,c.width,c.height);

    ctx.globalCompositeOperation = 'lighter';

    const g = 0.015 * (window.devicePixelRatio || 1); // gravity
    const air = 0.992;

    for (let i=parts.length-1;i>=0;i--){
      const p = parts[i];
      const age = t - p.born;
      if (age > p.life) { parts.splice(i,1); continue; }
      p.vx *= air; p.vy = p.vy*air + g;
      p.x += p.vx * dpr; p.y += p.vy * dpr;

      const alpha = 1 - age / p.life;
      ctx.beginPath();
      ctx.fillStyle = `hsla(${p.hue}, 90%, 60%, ${alpha})`;
      ctx.arc(p.x, p.y, p.r * dpr, 0, Math.PI*2);
      ctx.fill();
    }

    raf = requestAnimationFrame(step);
  }

  raf = requestAnimationFrame(step);

  const cleanup = () => {
    if (raf) cancelAnimationFrame(raf);
    try { ro.disconnect(); } catch {}
    c.remove();
  };

  // external cancel function
  return cleanup;
}
