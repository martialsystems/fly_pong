import { C } from "./constants.js";
import { NeuralOpponent, PRESETS, SkillGate, actionToOppDy } from "./ai.js";
import { clipPaddle, dyFromAction, initialState, step } from "./physics.js";

const canvas = document.getElementById("court");
const ctx = canvas.getContext("2d");
const scoreEl = document.getElementById("score");
const statusEl = document.getElementById("ai-status");
const skillEl = document.getElementById("skill");
const skillValEl = document.getElementById("skill-val");
const startBtn = document.getElementById("start");
const resetBtn = document.getElementById("reset");

canvas.width = C.width;
canvas.height = C.height;

const opponent = new NeuralOpponent();
const gate = new SkillGate();

let state = initialState(1.0, 0.0);
let running = false;
let raf = 0;
let keyAction = 0;
let mouseActive = false;
let mouseY = paddleCenterY();
let lastTs = 0;
let acc = 0;
let ticking = false;
const FRAME_MS = 1000 / C.fps;

function paddleCenterY() {
  return (C.height - C.paddleH) / 2;
}

function setSkill(v) {
  const skill = Math.max(0, Math.min(1, Number(v)));
  skillEl.value = String(skill);
  skillValEl.textContent = skill.toFixed(2);
  return skill;
}

function currentSkill() {
  return Number(skillEl.value);
}

function canvasGameY(clientY) {
  const rect = canvas.getBoundingClientRect();
  const y = ((clientY - rect.top) * C.height) / rect.height;
  return y - C.paddleH / 2;
}

function serveAngle() {
  return (Math.random() * 2 - 1) * C.serveAngleMax;
}

function draw() {
  ctx.fillStyle = "#0c0c12";
  ctx.fillRect(0, 0, C.width, C.height);
  ctx.strokeStyle = "#2a2a3a";
  ctx.setLineDash([6, 10]);
  ctx.beginPath();
  ctx.moveTo(C.width / 2, 0);
  ctx.lineTo(C.width / 2, C.height);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#e6e6e6";
  ctx.fillRect(C.agentX, state.agent_y, C.paddleW, C.paddleH);
  ctx.fillStyle = "#b4b4c8";
  ctx.fillRect(C.oppX, state.opp_y, C.paddleW, C.paddleH);
  ctx.fillStyle = "#ffd250";
  ctx.beginPath();
  ctx.arc(state.ball_x, state.ball_y, C.ballR, 0, Math.PI * 2);
  ctx.fill();
  scoreEl.textContent = `${state.agent_score}  ${state.opp_score}`;
}

function updateStatus() {
  const mode = opponent.status === "onnx" ? "neural (ONNX)" : "heuristic";
  const extra = opponent.error && opponent.status !== "onnx" ? ` · ${opponent.error}` : "";
  statusEl.textContent = `AI: ${mode}${extra}`;
}

async function tick() {
  let agentDy;
  if (mouseActive) {
    agentDy = clipPaddle(mouseY) - state.agent_y;
  } else {
    agentDy = dyFromAction(keyAction);
  }
  const raw = await opponent.action(state);
  const gated = gate.apply(raw, currentSkill(), Math.random);
  const oppDy = actionToOppDy(gated);
  const pair = step(state, agentDy, oppDy, serveAngle());
  state = pair[0];
  draw();
  if (state.terminated) {
    running = false;
    startBtn.textContent = "Play again";
  }
}

function loop(ts) {
  if (!running) return;
  if (!lastTs) lastTs = ts;
  acc += ts - lastTs;
  lastTs = ts;
  if (ticking) {
    raf = requestAnimationFrame(loop);
    return;
  }
  ticking = true;
  (async () => {
    while (acc >= FRAME_MS && running) {
      acc -= FRAME_MS;
      await tick();
    }
    ticking = false;
    if (running) raf = requestAnimationFrame(loop);
  })();
}

function start() {
  if (state.terminated) {
    state = initialState(Math.random() < 0.5 ? -1 : 1, serveAngle());
    gate.reset();
  }
  running = true;
  lastTs = 0;
  acc = 0;
  startBtn.textContent = "Playing";
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(loop);
}

function hardReset() {
  running = false;
  cancelAnimationFrame(raf);
  state = initialState(1.0, 0.0);
  gate.reset();
  startBtn.textContent = "Start";
  draw();
}

window.addEventListener("keydown", (e) => {
  if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
    keyAction = 1;
    mouseActive = false;
    e.preventDefault();
  } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
    keyAction = 2;
    mouseActive = false;
    e.preventDefault();
  } else if (e.key === " " || e.key === "Enter") {
    if (!running) start();
    e.preventDefault();
  }
});

window.addEventListener("keyup", (e) => {
  if (
    e.key === "ArrowUp" ||
    e.key === "ArrowDown" ||
    e.key === "w" ||
    e.key === "W" ||
    e.key === "s" ||
    e.key === "S"
  ) {
    keyAction = 0;
  }
});

canvas.addEventListener("mousemove", (e) => {
  mouseActive = true;
  mouseY = canvasGameY(e.clientY);
});

canvas.addEventListener("mouseleave", () => {
  mouseActive = false;
});

canvas.addEventListener(
  "touchmove",
  (e) => {
    if (!e.touches.length) return;
    mouseActive = true;
    mouseY = canvasGameY(e.touches[0].clientY);
    e.preventDefault();
  },
  { passive: false }
);

canvas.addEventListener("touchend", () => {
  mouseActive = false;
});

skillEl.addEventListener("input", () => setSkill(skillEl.value));

document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const name = btn.getAttribute("data-preset");
    if (PRESETS[name] !== undefined) setSkill(PRESETS[name]);
  });
});

startBtn.addEventListener("click", start);
resetBtn.addEventListener("click", hardReset);

draw();
setSkill(skillEl.value || PRESETS.normal);
updateStatus();
opponent.load().then(updateStatus);
