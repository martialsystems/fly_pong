/* Browser opponent: ONNX when loaded, lag heuristic otherwise. */
import { C } from "./constants.js";
import { applyVecnorm, encodeMirroredRight } from "./obs.js";
import { dyFromAction, lagOpponentDy } from "./physics.js";

const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

export const PRESETS = {
  easy: 0.25,
  normal: 0.55,
  unfair: 0.9,
  neural: 1.0,
};

export function skillDelay(skill) {
  return Math.round((1 - skill) * 8);
}

export function skillNoopProb(skill) {
  return (1 - skill) * 0.4;
}

export class SkillGate {
  constructor() {
    this.buf = [];
  }

  reset() {
    this.buf = [];
  }

  apply(action, skill, rng) {
    const delay = skillDelay(skill);
    const noopP = skillNoopProb(skill);
    let a = action;
    if (rng() < noopP) a = 0;
    if (delay <= 0) return a;
    this.buf.push(a);
    if (this.buf.length > delay) return this.buf.shift();
    return 0;
  }
}

export function heuristicAction(state) {
  const dy = lagOpponentDy(state);
  if (dy < 0) return 1;
  if (dy > 0) return 2;
  return 0;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("failed to load " + src));
    document.head.appendChild(s);
  });
}

export class NeuralOpponent {
  constructor() {
    this.session = null;
    this.norm = null;
    this.ort = null;
    this.status = "heuristic";
    this.error = "";
  }

  async load() {
    try {
      const normResp = await fetch("models/norm.json");
      if (!normResp.ok) throw new Error("norm.json " + normResp.status);
      this.norm = await normResp.json();
      if (!window.ort) {
        await loadScript(ORT_CDN + "ort.min.js");
      }
      this.ort = window.ort;
      if (!this.ort) throw new Error("onnxruntime-web missing");
      if (this.ort.env && this.ort.env.wasm) {
        this.ort.env.wasm.wasmPaths = ORT_CDN;
      }
      this.session = await this.ort.InferenceSession.create("models/pong.onnx", {
        executionProviders: ["wasm"],
      });
      this.status = "onnx";
    } catch (err) {
      this.session = null;
      this.status = "heuristic";
      this.error = String(err && err.message ? err.message : err);
    }
  }

  async action(state) {
    if (!this.session || !this.norm || !this.ort) return heuristicAction(state);
    const raw = encodeMirroredRight(state);
    const obs = applyVecnorm(raw, this.norm);
    const inputName = this.session.inputNames[0];
    const tensor = new this.ort.Tensor("float32", obs, [1, 6]);
    const feeds = {};
    feeds[inputName] = tensor;
    try {
      const out = await this.session.run(feeds);
      const name = this.session.outputNames[0];
      const data = out[name].data;
      const value = Number(data.length === 1 ? data[0] : argmax(data));
      const a = value | 0;
      if (a === 0 || a === 1 || a === 2) return a;
      return heuristicAction(state);
    } catch (err) {
      this.error = String(err && err.message ? err.message : err);
      return heuristicAction(state);
    }
  }
}

function argmax(data) {
  let best = 0;
  let bestV = data[0];
  for (let i = 1; i < data.length; i++) {
    if (data[i] > bestV) {
      bestV = data[i];
      best = i;
    }
  }
  return best;
}

export function actionToOppDy(action) {
  return dyFromAction(action, C.paddleSpeed);
}
