// src/ui/sowing.js
import { getStoreController } from './storeSeedsController';
import { getScatterController } from './scatterSeedsController';

function addOneTo(toEl){
  // store?
  const storeFace = toEl.closest('.store__face') || toEl.closest('.store');
  if (storeFace) {
    let scatter = storeFace.querySelector('.store__scatter');
    if (scatter) {
      // center exclusion for store: compute once from box
      const r = scatter.getBoundingClientRect();
      const cx = r.width/2, cy = r.height/2;
      const ctl = getScatterController(scatter, { centerExclusion: { x: cx, y: cy, r: 36 } });
      ctl.add(1);
      return;
    }
  }
  // pit?
  const pitFace = toEl.closest('.pit__face');
  if (pitFace) {
    let scatter = pitFace.querySelector('.pit__scatter');
    if (scatter) {
      const ctl = getScatterController(scatter); // no center exclusion for pits
      ctl.add(1);
    }
  }
}

// Before animating from a PIT, remove all stones from its scatter (picked up)
export function prepareSourcePit(startEl, seedCount){
  const pitFace = startEl.closest('.pit__face');
  if (!pitFace) return;
  const scatter = pitFace.querySelector('.pit__scatter');
  if (!scatter) return;
  const ctl = getScatterController(scatter);
  ctl.remove(seedCount); // or ctl.clear() if you don't trust the count
}

export function centerIn(el, container){
  const a = el.getBoundingClientRect();
  const b = container.getBoundingClientRect();
  return { x: a.left - b.left + a.width/2, y: a.top - b.top + a.height/2 };
}

export function ensureSowLayer(boardEl){
  let layer = boardEl.querySelector('.sow-layer');
  if(!layer){
    layer = document.createElement('div');
    layer.className = 'sow-layer';
    boardEl.appendChild(layer);
  }
  return layer;
}

export function animateSeedHop(boardEl, fromEl, toEl, delayMs=0, hopMs=260){
  const layer = ensureSowLayer(boardEl);
  const from = centerIn(fromEl, layer);
  const to   = centerIn(toEl, layer);

  const dot = document.createElement('div');
  dot.className = 'fly-seed';
  dot.style.setProperty('--from-x', `${from.x}px`);
  dot.style.setProperty('--from-y', `${from.y}px`);
  dot.style.setProperty('--to-x',   `${to.x}px`);
  dot.style.setProperty('--to-y',   `${to.y}px`);
  dot.style.setProperty('--hop-ms', `${hopMs}ms`);
  dot.style.animationDelay = `${delayMs}ms`;
  layer.appendChild(dot);

  return new Promise(res => {
    const done = () => {
      dot.remove();
      
      // If landing element is a store face (has a nearby .store__scatter), add one persistent seed
      addOneTo(toEl);

      toEl.classList.add('landed');
      setTimeout(() => toEl.classList.remove('landed'), 200);
      const sp = document.createElement('div');
      sp.className = 'sparkle';
      sp.style.left = `${to.x - 4}px`;
      sp.style.top  = `${to.y - 4}px`;
      layer.appendChild(sp);
      setTimeout(() => sp.remove(), 280);
      res();
    };
    dot.addEventListener('animationend', done, { once:true });
  });
}

// Fan-out sow: create one flying seed per deposit.
// startEl: the clicked pit face
// cycle:   array of faces in visit order starting at startEl (index 0)
// seedCount: number of stones picked up
export async function animateSow(boardEl, startEl, cycle, seedCount, stagger=200, hopMs=420){
  if (!boardEl || !startEl || !cycle?.length || seedCount <= 0) return;

  // highlight on the source pit
  startEl.classList.add('sowing');

  // remove stones visually from the source pit
  prepareSourcePit(startEl, seedCount);

  // We deposit to cycle[1], cycle[2], ..., wrapping around and skipping opponent store
  const L = cycle.length;
  const promises = [];
  for (let i = 0; i < seedCount; i++) {
    const dest = cycle[(i + 1) % L];
    // each stone leaves startEl → lands dest, staggered
    promises.push(animateSeedHop(boardEl, startEl, dest, i * stagger, hopMs));
  }

  await Promise.all(promises);
  startEl.classList.remove('sowing');
}


function pitFaceFor(boardEl, player, pitIndex){
  if (player === 0) {
    const bottom = [...boardEl.querySelectorAll(".board__pits--bottom .pit__face")];
    return bottom[pitIndex] || null;
  } else {
    const tops = [...boardEl.querySelectorAll(".board__pits--top .pit__face")];
    const domIdx = 5 - pitIndex; // top row is reversed in DOM
    return tops[domIdx] || null;
  }
}
function storeFaceFor(boardEl, player){
  // Select the store by logical owner, not by visual side
  const ownerRoot = boardEl.querySelector(`.board__store[data-owner="${player}"]`);
  if (!ownerRoot) {
    console.warn('storeFaceFor: no root for player', player);
    return null;
  }
  return ownerRoot.querySelector('.store__face, .store');
}


/**
 * Animate capture sweep: (landingPit + oppositePit) -> player's store.
 * Removes stones visually from pits first, then flies fake seeds, then adds to store.
 */
export async function animateCapture(boardEl, player, landingPitIndex, capturedOppCount, { stagger=200, hopMs=420 } = {}){
  const landingFace = pitFaceFor(boardEl, player, landingPitIndex);
  const oppIndex = 5 - landingPitIndex;
  const oppFace = pitFaceFor(boardEl, 1 - player, oppIndex);
  const storeFace = storeFaceFor(boardEl, player);
  if (!landingFace || !oppFace || !storeFace) return;

  // Remove the single landing stone from landing pit's scatter
  const landingScatter = landingFace.querySelector('.pit__scatter');
  if (landingScatter) {
    const ctl = getScatterController(landingScatter);
    ctl.remove(1);
  }
  // Remove all captured stones from opposite pit's scatter
  const oppScatter = oppFace.querySelector('.pit__scatter');
  if (oppScatter && capturedOppCount > 0) {
    const ctl = getScatterController(oppScatter);
    ctl.remove(capturedOppCount);
  }

  // Fly (capturedOppCount + 1) seeds from pits to store
  const total = (capturedOppCount || 0) + 1;
  const flights = [];
  // Send the landing stone from landing pit
  flights.push(animateSeedHop(boardEl, landingFace, storeFace, 0, hopMs));
  // Send the opponent stones from opposite pit (fan out with stagger)
  for (let i = 0; i < (capturedOppCount || 0); i++){
    flights.push(animateSeedHop(boardEl, oppFace, storeFace, (i+1)*stagger, hopMs));
  }

  await Promise.all(flights);

  // Add to the player's store scatter (persistent dots)
  const storeScatter = storeFace.querySelector('.store__scatter');
  if (storeScatter) {
    // If you use a center exclusion in stores, pass it here
    const cx = storeScatter.clientWidth / 2, cy = storeScatter.clientHeight / 2, r = 36;
    const ctl = getScatterController(storeScatter, { centerExclusion: { x: cx, y: cy, r } });
    ctl.add(total);
  }
}

/**
 * Animate collecting all stones from `player`'s row into their store.
 * `counts` is the 6-length pits array for that player *before* the server clears them.
 */
export async function animateCollectRow(boardEl, player, counts, { pitStagger=140, stoneStagger=50, hopMs=420 } = {}){
  if (!boardEl || !counts) return;
  const storeFace = storeFaceFor(boardEl, player);
  if (!storeFace) return;

  const flights = [];

  for (let i = 0; i < 6; i++){
    const n = counts[i] || 0;
    if (n <= 0) continue;

    const pitFace = pitFaceFor(boardEl, player, i);
    if (!pitFace) continue;

    // remove all existing dots from that pit's scatter first
    const scatter = pitFace.querySelector('.pit__scatter');
    if (scatter) {
      const ctl = getScatterController(scatter);
      ctl.remove(n); // visual drop to 0
    }

    // fan out flights from this pit to the store
    for (let k = 0; k < n; k++){
      const delay = i * pitStagger + k * stoneStagger;
      flights.push(animateSeedHop(boardEl, pitFace, storeFace, delay, hopMs));
    }
  }

  await Promise.all(flights);
}
