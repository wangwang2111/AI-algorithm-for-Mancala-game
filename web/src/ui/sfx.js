// ui/sfx.js
// Lightweight Web Audio SFX engine: louder, controlled, and mobile-safe.

let ctx = null;
let unlocked = false;

let master = null;          // GainNode
let comp = null;            // DynamicsCompressorNode
let music = null;           // Optional: external music <audio> element
let musicDuck = { enabled: true, base: 1.0 }; // music ducking

// Global controls
let muted = false;          // SFX mute
let masterGain = 0.9;       // Overall SFX volume (0..1)

const now = () => (ctx ? ctx.currentTime : 0);

// ---- init / unlock --------------------------------------------------------
function ensureCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (!master) {
    comp = ctx.createDynamicsCompressor();
    // Gentle glue: tame spikes, keep loud but not harsh
    comp.threshold.setValueAtTime(-18, now());
    comp.knee.setValueAtTime(24, now());
    comp.ratio.setValueAtTime(3, now());
    comp.attack.setValueAtTime(0.003, now());
    comp.release.setValueAtTime(0.08, now());

    master = ctx.createGain();
    master.gain.setValueAtTime(muted ? 0 : masterGain, now());

    comp.connect(master).connect(ctx.destination);
  }
  return ctx;
}

export function unlockAudio() {
  const c = ensureCtx();
  if (c.state === "suspended") c.resume();
  unlocked = true;
}

// ---- global controls ------------------------------------------------------
export function setMuted(v) {
  muted = !!v;
  ensureCtx();
  master.gain.cancelScheduledValues(now());
  master.gain.setValueAtTime(muted ? 0 : masterGain, now());
}

export function setVolume(v = 0.9) {
  masterGain = Math.max(0, Math.min(1, v));
  ensureCtx();
  if (!muted) {
    master.gain.cancelScheduledValues(now());
    master.gain.setValueAtTime(masterGain, now());
  }
}

export function attachMusic(el) {
  music = el || null;
  if (music) musicDuck.base = music.volume ?? 1.0;
}

function duckStart() {
  if (!music || !musicDuck.enabled) return;
  try {
    const base = musicDuck.base;
    const tgt  = Math.max(0, base * 0.72); // ~ -2.8 dB
    music.volume = tgt;
    // restore a moment later
    setTimeout(() => {
      if (music) music.volume = base;
    }, 160);
  } catch {}
}

// ---- builders --------------------------------------------------------------
function osc(typ = "sine", f = 440) {
  const o = ctx.createOscillator();
  o.type = typ;
  o.frequency.setValueAtTime(f, now());
  return o;
}

function gain(v = 1.0) {
  const g = ctx.createGain();
  g.gain.setValueAtTime(v, now());
  return g;
}

function noise(duration = 0.1) {
  // White noise buffer
  const sr = ctx.sampleRate;
  const len = Math.max(1, Math.floor(duration * sr));
  const buf = ctx.createBuffer(1, len, sr);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  return src;
}

// ---- tiny variation & rate limit ------------------------------------------
function vary(value, cents = 15) {
  // cents -> frequency ratio
  const r = Math.pow(2, (Math.random() * 2 - 1) * (cents / 1200));
  return value * r;
}

let lastSeedAt = 0;
const SEED_MIN_GAP = 0.028; // seconds; drops some events if too dense

// ---- primitives ------------------------------------------------------------
function blip({ freq = 880, dur = 0.08, gainDb = -10, type = "triangle", attack = 0.004, release = 0.09 } = {}) {
  if (!unlocked || muted) return;
  ensureCtx(); duckStart();

  const t0 = now();
  const g = gain(0.0001);
  const o = osc(type, freq);

  // convert dB-ish control to linear
  const lin = Math.pow(10, gainDb / 20);
  // ADSR
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(Math.max(0.0002, lin), t0 + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur + release);

  o.connect(g).connect(comp);
  o.start(t0);
  o.stop(t0 + dur + release + 0.02);
}

function hit({ toneFreq = 480, dur = 0.12, gainDb = -8, noiseAmt = 0.4 } = {}) {
  if (!unlocked || muted) return;
  ensureCtx(); duckStart();
  const t0 = now();

  // tone
  const o = osc("square", toneFreq);
  const gt = gain(0.0001);
  gt.gain.setValueAtTime(0.0001, t0);
  gt.gain.exponentialRampToValueAtTime(Math.pow(10, gainDb / 20), t0 + 0.005);
  gt.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(gt).connect(comp);
  o.start(t0); o.stop(t0 + dur + 0.02);

  // noise burst
  const n = noise(dur);
  const gn = gain(noiseAmt);
  gn.gain.setValueAtTime(noiseAmt, t0);
  gn.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  n.connect(gn).connect(comp);
  n.start(t0); n.stop(t0 + dur + 0.02);
}

// ---- public SFX ------------------------------------------------------------
export const sfx = {
  seedPit() {
    const t = now();
    if (t - lastSeedAt < SEED_MIN_GAP) return; // rate-limit
    lastSeedAt = t;
    blip({
      freq: vary(980, 22),
      dur: 0.06,
      gainDb: -6,        // louder
      type: "triangle",
      attack: 0.003,
      release: 0.08,
    });
  },
  seedStore() {
    const t = now();
    if (t - lastSeedAt < SEED_MIN_GAP) return; // rate-limit
    lastSeedAt = t;
    blip({
      freq: vary(760, 18),
      dur: 0.09,
      gainDb: -4,        // louder than pit
      type: "sine",
      attack: 0.003,
      release: 0.1,
    });
  },
  capture() {
    // percussive “thud + grit”
    hit({ toneFreq: 520, dur: 0.12, gainDb: -4, noiseAmt: 0.35 });
    setTimeout(() => hit({ toneFreq: 360, dur: 0.12, gainDb: -5, noiseAmt: 0.28 }), 70);
  },
  sweep() {
    // short whoosh-like click
    hit({ toneFreq: 420, dur: 0.10, gainDb: -5, noiseAmt: 0.22 });
  },
  win() {
    // bright triad arpeggio (C major-ish), slightly louder
    const base = 660; // ~E5
    const notes = [base, base * 1.333, base * 1.5]; // E5, G5, A5-ish sweet
    notes.forEach((f, i) => {
      setTimeout(() => blip({ freq: f, dur: 0.18, gainDb: -3, type: "triangle", attack: 0.002, release: 0.16 }), i * 120);
    });
    // sparkle on top
    setTimeout(() => blip({ freq: base * 2, dur: 0.14, gainDb: -6, type: "sine", attack: 0.002, release: 0.12 }), 280);
  },
};
