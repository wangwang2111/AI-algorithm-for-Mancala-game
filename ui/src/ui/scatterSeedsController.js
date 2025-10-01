// Generic scatter controller for pits & stores (no React re-rendering)
const controllers = new WeakMap(); // scatterEl -> controller

export function getScatterController(scatterEl, opts = {}) {
  if (controllers.has(scatterEl)) return controllers.get(scatterEl);
  const c = createController(scatterEl, opts);
  controllers.set(scatterEl, c);
  return c;
}

function createController(scatterEl, { minDistFactor = 1.05, centerExclusion = null } = {}) {
  const seeds = []; // { el, x, y }
  let w=0, h=0, seedSize=10;

  const readVars = () => {
    const cs = getComputedStyle(scatterEl);
    seedSize = parseFloat(cs.getPropertyValue('--seed-size')) || 10;
  };
  const measure = () => {
    const r = scatterEl.getBoundingClientRect();
    w = r.width; h = r.height;
  };
  const ro = new ResizeObserver(() => { measure(); });
  ro.observe(scatterEl);
  readVars(); measure();

  const minD = () => seedSize * minDistFactor;

  function overlaps(x, y){
    const d2 = Math.pow(minD(), 2);
    for (const s of seeds){
      const dx = x - s.x, dy = y - s.y;
      if (dx*dx + dy*dy < d2) return true;
    }
    return false;
  }
  function randomPoint(){
    const pad = seedSize * 0.3; // keep a tiny padding so seeds don't clip sides
    const X = Math.max(1, w - 2*pad);
    const Y = Math.max(1, h - 2*pad);
    return { x: pad + Math.random() * X, y: pad + Math.random() * Y };
  }

  // Ensure final coords are inside box
  function clampToBox(x, y){
    const half = seedSize / 2;
    const Xmin = half, Xmax = Math.max(half, w - half);
    const Ymin = half, Ymax = Math.max(half, h - half);
    return {
      x: Math.min(Xmax, Math.max(Xmin, x)),
      y: Math.min(Ymax, Math.max(Ymin, y)),
    };
  }

  function placeOne() {
    readVars(); measure();
    const MAX = 80;
    for (let i=0;i<MAX;i++){
      let p = randomPoint();
      if (overlaps(p.x, p.y)) continue;
      // Clamp to container to be 100% safe
      p = clampToBox(p.x, p.y);
      const el = document.createElement('span');
      el.className = scatterEl.classList.contains('pit__scatter') ? 'pit__seed' : 'store__seed';
      el.style.setProperty('--x', `${p.x - seedSize/2}px`);
      el.style.setProperty('--y', `${p.y - seedSize/2}px`);
      scatterEl.appendChild(el);
      seeds.push({ el, x:p.x, y:p.y });
      return el;
    }
    // crowded fallback
    let p = randomPoint();
    // Clamp to container to be 100% safe
    p = clampToBox(p.x, p.y);
    const el = document.createElement('span');
    el.className = scatterEl.classList.contains('pit__scatter') ? 'pit__seed' : 'store__seed';
    el.style.setProperty('--x', `${p.x - seedSize/2}px`);
    el.style.setProperty('--y', `${p.y - seedSize/2}px`);
    scatterEl.appendChild(el);
    seeds.push({ el, x:p.x, y:p.y });
    return el;
  }

  function add(n=1){ for(let i=0;i<n;i++) placeOne(); }
  function remove(n=1){
    for(let i=0;i<n && seeds.length;i++){
      const s = seeds.pop();
      s.el.remove();
    }
  }
  function clear(){ remove(seeds.length); }

  return { add, remove, clear, destroy: () => ro.disconnect() };
}
