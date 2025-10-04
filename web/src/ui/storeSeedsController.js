// Manage stones DOM inside a .store__scatter without React re-rendering.

const controllers = new Map(); // storeId -> controller

export function getStoreController(scatterEl){
  const key = scatterEl.dataset.storeId || Symbol();
  if (controllers.has(key)) return controllers.get(key);
  const c = createController(scatterEl);
  controllers.set(key, c);
  return c;
}

function createController(scatterEl){
  const seeds = []; // {el, x, y}
  let w=0, h=0, seedSize=10, exclusion=50;

  const readVars = () => {
    const cs = getComputedStyle(scatterEl);
    seedSize = parseFloat(cs.getPropertyValue('--seed-size')) || 10;
    exclusion = parseFloat(cs.getPropertyValue('--store-center-exclusion')) || 36;
  };

  const measure = () => {
    const cr = scatterEl.getBoundingClientRect();
    w = cr.width; h = cr.height;
  };

  const ro = new ResizeObserver(() => { measure(); });
  ro.observe(scatterEl);
  readVars(); measure();

  function withinCenterExclusion(x, y){
    // center of scatter area in local coords
    const cx = w/2, cy = h/2;
    const dx = x - cx, dy = y - cy;
    return (dx*dx + dy*dy) < (exclusion*exclusion);
  }
  function overlaps(x,y){
    const d = seedSize * 1.05;
    const d2 = d*d;
    for (const s of seeds){
      const dx = x - s.x, dy = y - s.y;
      if (dx*dx + dy*dy < d2) return true;
    }
    return false;
  }

  function randomPoint(){
    // keep a tiny padding so seeds don't clip sides
    const pad = seedSize * 0.6;
    const X = (w - 2*pad); const Y = (h - 2*pad);
    const x = pad + Math.random() * Math.max(1, X);
    const y = pad + Math.random() * Math.max(1, Y);
    return {x,y};
  }

  function placeSeed(){
    readVars(); measure();
    const MAX = 100;
    for (let i=0;i<MAX;i++){
      const p = randomPoint();
      if (withinCenterExclusion(p.x, p.y)) continue;
      if (overlaps(p.x, p.y)) continue;
      const el = document.createElement('span');
      el.className = 'store__seed';
      el.style.setProperty('--x', `${p.x - seedSize/2}px`);
      el.style.setProperty('--y', `${p.y - seedSize/2}px`);
      scatterEl.appendChild(el);
      seeds.push({ el, x:p.x, y:p.y });
      return el;
    }
    // fallback: if crowded, still append near a random existing one
    const el = document.createElement('span');
    el.className = 'store__seed';
    const p = randomPoint();
    el.style.setProperty('--x', `${p.x - seedSize/2}px`);
    el.style.setProperty('--y', `${p.y - seedSize/2}px`);
    scatterEl.appendChild(el);
    seeds.push({ el, x:p.x, y:p.y });
    return el;
  }

  function add(n=1){ for(let i=0;i<n;i++) placeSeed(); }
  function remove(n=1){
    for(let i=0;i<n && seeds.length;i++){
      const s = seeds.pop();
      s.el.remove();
    }
  }

  return { add, remove, destroy: () => { ro.disconnect(); } };
}
