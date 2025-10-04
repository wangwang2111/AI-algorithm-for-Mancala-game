// src/ui/sowing.js
import { getStoreController } from "./storeSeedsController";
import { getScatterController } from "./scatterSeedsController";

function addOneTo(toEl) {
  // store?
  const storeFace = toEl.closest(".store__face") || toEl.closest(".store");
  if (storeFace) {
    let scatter = storeFace.querySelector(".store__scatter");
    if (scatter) {
      // center exclusion for store: compute once from box
      const r = scatter.getBoundingClientRect();
      const cx = r.width / 2,
        cy = r.height / 2;
      const ctl = getScatterController(scatter, {
        centerExclusion: { x: cx, y: cy, r: 36 },
      });
      ctl.add(1);
      return;
    }
  }
  // pit?
  const pitFace = toEl.closest(".pit__face");
  if (pitFace) {
    let scatter = pitFace.querySelector(".pit__scatter");
    if (scatter) {
      const ctl = getScatterController(scatter); // no center exclusion for pits
      ctl.add(1);
    }
  }
}

// Before animating from a PIT, remove all stones from its scatter (picked up)
// export function prepareSourcePit(startEl, seedCount){
//   const pitFace = startEl.closest('.pit__face');
//   if (!pitFace) return;
//   const scatter = pitFace.querySelector('.pit__scatter');
//   if (!scatter) return;
//   const ctl = getScatterController(scatter);
//   ctl.remove(seedCount); // or ctl.clear() if you don't trust the count
// }

export function centerIn(el, container) {
  const a = el.getBoundingClientRect();
  const b = container.getBoundingClientRect();
  return { x: a.left - b.left + a.width / 2, y: a.top - b.top + a.height / 2 };
}

export function ensureSowLayer(boardEl) {
  let layer = boardEl.querySelector(".sow-layer");
  if (!layer) {
    layer = document.createElement("div");
    layer.className = "sow-layer";
    boardEl.appendChild(layer);
  }
  return layer;
}

export function animateSeedHop(
  boardEl,
  fromEl,
  toEl,
  delayMs = 0,
  hopMs = 260
) {
  const layer = ensureSowLayer(boardEl);
  const from = centerIn(fromEl, layer);
  const to = centerIn(toEl, layer);

  const dot = document.createElement("div");
  dot.className = "fly-seed";
  dot.style.setProperty("--from-x", `${from.x}px`);
  dot.style.setProperty("--from-y", `${from.y}px`);
  dot.style.setProperty("--to-x", `${to.x}px`);
  dot.style.setProperty("--to-y", `${to.y}px`);
  dot.style.setProperty("--hop-ms", `${hopMs}ms`);
  dot.style.animationDelay = `${delayMs}ms`;
  layer.appendChild(dot);

  return new Promise((res) => {
    const done = () => {
      dot.remove();

      // If landing element is a store face (has a nearby .store__scatter), add one persistent seed
      addOneTo(toEl);

      toEl.classList.add("landed");
      setTimeout(() => toEl.classList.remove("landed"), 200);
      const sp = document.createElement("div");
      sp.className = "sparkle";
      sp.style.left = `${to.x - 4}px`;
      sp.style.top = `${to.y - 4}px`;
      layer.appendChild(sp);
      setTimeout(() => sp.remove(), 280);
      res();
    };
    dot.addEventListener("animationend", done, { once: true });
  });
}

// ✨ Fly using the actual DOM seed element as pickup origin.
// We hide that specific seed in the source pit, spawn a fly-dot at its spot,
// then remove the original seed node after flight completes.
function animateFromSeedEl(boardEl, seedEl, toEl, delayMs = 0, hopMs = 420) {
  const layer = ensureSowLayer(boardEl);
  const from = centerIn(seedEl, layer);
  const to = centerIn(toEl, layer);

  // visually "pick up" this seed
  seedEl.style.visibility = "hidden";

  const dot = document.createElement("div");
  dot.className = "fly-seed";
  dot.style.setProperty("--from-x", `${from.x}px`);
  dot.style.setProperty("--from-y", `${from.y}px`);
  dot.style.setProperty("--to-x", `${to.x}px`);
  dot.style.setProperty("--to-y", `${to.y}px`);
  dot.style.setProperty("--hop-ms", `${hopMs}ms`);
  dot.style.animationDelay = `${delayMs}ms`;
  layer.appendChild(dot);
  // after you compute `from` in animateFromSeedEl(...)
  dot.style.transform = `translate(${from.x}px, ${from.y}px)`;
  dot.style.opacity = "0"; // hide during any delay

  return new Promise((res) => {
    const done = () => {
      dot.remove();
      // remove the original, now-used seed node from DOM
      seedEl.remove();

      layer.dispatchEvent(
        new CustomEvent("mancala:seed-landed", {
          bubbles: true,
          detail: { toEl },
        })
      );
      
      const isStore = !!(
        toEl.closest(".store__face") || toEl.closest(".store")
      );
      layer.dispatchEvent(
        new CustomEvent("mancala:seed-landed", {
          bubbles: true,
          detail: { toEl, isStore },
        })
      );

      // make a persistent landing seed (pit/store)
      addOneTo(toEl);

      toEl.classList.add("landed");
      setTimeout(() => toEl.classList.remove("landed"), hopMs);
      res();
    };
    dot.addEventListener(
      "animationend",
      (e) => {
        if (e.animationName !== "seed-flight") return; // ignore seed-pop’s end
        done();
      },
      { once: false }
    );
  });
}

/* ---------------------------- Path construction --------------------------- */

function buildFullRing(boardEl) {
  // P0 perspective: bottom pits 0..5, right store, top pits 5..0, left store
  const bottom = [
    ...boardEl.querySelectorAll(".board__pits--bottom .pit__face"),
  ]; // 0..5
  const rightS = boardEl.querySelector(
    ".board__store--right .store__face, .board__store--right .store"
  ); // P0 store
  const top = [
    ...boardEl.querySelectorAll(".board__pits--top .pit__face"),
  ].reverse(); // 5..0
  const leftS = boardEl.querySelector(
    ".board__store--left .store__face, .board__store--left .store"
  ); // P1 store
  return [...bottom, rightS, ...top, leftS].filter(Boolean);
}

function pathFrom(boardEl, player, pitIndex) {
  const ring = buildFullRing(boardEl);
  if (!ring.length) return [];

  // start index in ring
  const startRingIdx = player === 0 ? pitIndex : 7 + pitIndex; // top was reversed

  // rotate so index 0 is the clicked pit
  const rotated = ring.slice(startRingIdx).concat(ring.slice(0, startRingIdx));

  // filter out opponent store
  return rotated.filter((el) => {
    const isRightStore = !!el.closest(".board__store--right");
    const isLeftStore = !!el.closest(".board__store--left");
    if (player === 0 && isLeftStore) return false;
    if (player === 1 && isRightStore) return false;
    return true;
  });
}

/* --------------------------------- Public -------------------------------- */

/**
 * Animate sow using the *real seeds* from the selected pit.
 * Signature matches your Board: (boardEl, startEl, cycle, seedCount, stagger, hopMs)
 */
export async function animateSow(
  boardEl,
  startEl,
  cycle,
  seedCount,
  { stagger = 250, hopMs = 560 }
) {
  if (!boardEl || !startEl || !cycle?.length || seedCount <= 0) return;

  const scatter = startEl.querySelector(".pit__scatter");
  // All current seeds in the pit (their positions are already random from the scatter controller)
  let bag = scatter
    ? Array.from(scatter.querySelectorAll(".pit__seed, .store__seed"))
    : [];
  // Randomize pickup order for more natural look
  bag.sort(() => Math.random() - 0.5);

  // We’ll animate exactly `seedCount` stones. If the DOM has fewer seeds than that,
  // we’ll fallback to center-based fake dots for the remainder.
  const flights = [];
  const L = cycle.length;

  // visual cue
  startEl.classList.add("sowing");

  for (let i = 0; i < seedCount; i++) {
    const dest = cycle[(i + 1) % L];
    const seedEl = bag[i]; // may be undefined if DOM had fewer nodes

    if (seedEl) {
      flights.push(
        animateFromSeedEl(boardEl, seedEl, dest, i * stagger, hopMs)
      );
    } else {
      // fallback: no real seed node available; use a fake dot from pit center
      flights.push(animateSeedHop(boardEl, startEl, dest, i * stagger, hopMs));
    }
  }

  await Promise.all(flights);
  startEl.classList.remove("sowing");
}

// helpers used by both capture and sweep
function pitFaceFor(boardEl, player, pitIndex) {
  if (player === 0) {
    const bottoms = [
      ...boardEl.querySelectorAll(".board__pits--bottom .pit__face"),
    ];
    return bottoms[pitIndex] || null;
  } else {
    const tops = [...boardEl.querySelectorAll(".board__pits--top .pit__face")];
    const domIdx = 5 - pitIndex; // top row reversed in DOM
    return tops[domIdx] || null;
  }
}
function storeFaceFor(boardEl, player) {
  const ownerRoot =
    boardEl.querySelector(`.board__store[data-owner="${player}"]`) ||
    (player === 0
      ? boardEl.querySelector(".board__store--right")
      : boardEl.querySelector(".board__store--left"));
  return ownerRoot?.querySelector(".store__face, .store") || null;
}

// Pull up to n seed elements from a scatter (oldest last so layout looks natural)
function takeSeeds(scatter, n) {
  if (!scatter || !n) return [];
  const list = [...scatter.querySelectorAll(".pit__seed, .store__seed")];
  if (!list.length) return [];
  // take from the end (visually top-most if appended order)
  const out = [];
  for (let i = 0; i < n && list.length; i++) {
    const el = list.pop();
    out.push(el);
  }
  return out;
}

// If a real seed el is missing, create a temporary ghost at `fromEl` center
function makeGhostSeed(fromEl) {
  const host = fromEl.closest(".board, .board--grid") || fromEl.parentElement;
  const layer =
    host?.querySelector(".sow-layer") ||
    (() => {
      const l = document.createElement("div");
      l.className = "sow-layer";
      host.appendChild(l);
      return l;
    })();

  // position at the visual center of fromEl
  const a = fromEl.getBoundingClientRect();
  const b = layer.getBoundingClientRect();
  const cx = a.left - b.left + a.width / 2;
  const cy = a.top - b.top + a.height / 2;

  const ghost = document.createElement("span");
  ghost.className = "pit__seed"; // reuse same size via CSS
  ghost.style.position = "absolute";
  ghost.style.transform = `translate(${cx}px, ${cy}px) translate(-50%, -50%)`;
  ghost.style.pointerEvents = "none";
  layer.appendChild(ghost);
  return ghost;
}

/**
 * Capture animation using real seed elements.
 * - Moves 1 landing stone from `player`'s landing pit
 * - Moves `capturedOppCount` stones from the opposite pit
 * - All into `player`'s store via animateFromSeedEl
 */
export async function animateCapture(
  boardEl,
  player,
  landingPitIndex,
  capturedOppCount,
  { stagger = 200, hopMs = 420 } = {}
) {
  if (!boardEl) return;

  const oppIndex = 5 - landingPitIndex;
  const landing = pitFaceFor(boardEl, player, landingPitIndex);
  const opposite = pitFaceFor(boardEl, 1 - player, oppIndex);
  const storeFace = storeFaceFor(boardEl, player);
  if (!landing || !opposite || !storeFace) return;

  // Take exactly 1 landing stone from landing pit (or ghost if empty)
  const landingScatter = landing.querySelector(".pit__scatter");
  let landingSeeds = takeSeeds(landingScatter, 1);
  if (!landingSeeds.length) {
    landingSeeds = [makeGhostSeed(landing)];
  }

  // Take captured stones from opposite pit (or ghosts if short)
  const k = Math.max(0, capturedOppCount | 0);
  const oppScatter = opposite.querySelector(".pit__scatter");
  let oppSeeds = takeSeeds(oppScatter, k);
  while (oppSeeds.length < k) {
    oppSeeds.push(makeGhostSeed(opposite));
  }

  // Launch flights: landing stone first, then opponent stones with stagger
  const flights = [];
  // move the landing one immediately
  flights.push(
    animateFromSeedEl(boardEl, landingSeeds[0], storeFace, 0, hopMs)
  );

  // move opponent stones with fan-out delays
  for (let i = 0; i < oppSeeds.length; i++) {
    flights.push(
      animateFromSeedEl(
        boardEl,
        oppSeeds[i],
        storeFace,
        (i + 1) * stagger,
        hopMs
      )
    );
  }

  // Optional emphasis on the store while receiving
  storeFace.classList.add("collecting");
  try {
    await Promise.all(flights);
  } finally {
    storeFace.classList.remove("collecting");
  }
}

/**
 * Sweep all stones from `player`'s row into their store using real elements.
 * `counts` should be that row’s pit counts *before* the server clears them.
 */
export async function animateCollectRow(
  boardEl,
  player,
  counts,
  { pitStagger = 140, stoneStagger = 50, hopMs = 420 } = {}
) {
  if (!boardEl || !counts) return;

  const storeFace = storeFaceFor(boardEl, player);
  if (!storeFace) return;

  const flights = [];

  for (let pitIdx = 0; pitIdx < 6; pitIdx++) {
    const n = counts[pitIdx] | 0;
    if (n <= 0) continue;

    const pitFace = pitFaceFor(boardEl, player, pitIdx);
    if (!pitFace) continue;

    const scatter = pitFace.querySelector(".pit__scatter");

    // take real stones (or create ghosts if short)
    let seeds = takeSeeds(scatter, n);
    while (seeds.length < n) {
      seeds.push(makeGhostSeed(pitFace));
    }

    // fan out stones for this pit, with pit-level and stone-level stagger
    for (let s = 0; s < seeds.length; s++) {
      const delay = pitIdx * pitStagger + s * stoneStagger;
      flights.push(
        animateFromSeedEl(boardEl, seeds[s], storeFace, delay, hopMs)
      );
    }
  }

  storeFace.classList.add("collecting");
  try {
    await Promise.all(flights);
  } finally {
    storeFace.classList.remove("collecting");
  }
}
