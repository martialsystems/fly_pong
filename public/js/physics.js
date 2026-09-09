/* Clone of fly_pong/physics.py. Keep the formulas in lockstep. */
import { C } from "./constants.js";

export function clip(value, lo, hi) {
  if (value < lo) return lo;
  if (value > hi) return hi;
  return value;
}

export function paddleSpan() {
  return C.height - C.paddleH;
}

export function clipPaddle(y) {
  return clip(y, 0.0, paddleSpan());
}

export function dyFromAction(action, speed) {
  const s = speed === undefined ? C.paddleSpeed : speed;
  const a = action | 0;
  if (a === 1) return -s;
  if (a === 2) return s;
  return 0.0;
}

export function lagOpponentDy(state) {
  const target = state.ball_y - C.paddleH / 2.0;
  const speed = C.paddleSpeed * C.oppLag;
  const center = state.opp_y + C.paddleH / 2.0;
  if (center > target) return -speed;
  return speed;
}

export function serve(state, direction, angle) {
  const speed = C.ballSpeed;
  const out = Object.assign({}, state);
  out.ball_x = C.width / 2.0;
  out.ball_y = C.height / 2.0;
  out.ball_vx = direction * speed * Math.cos(angle);
  out.ball_vy = speed * Math.sin(angle);
  out.serve_dir = direction;
  return out;
}

export function initialState(serveDir, angle) {
  const dir = serveDir === undefined ? 1.0 : serveDir;
  const ang = angle === undefined ? 0.0 : angle;
  const mid = paddleSpan() / 2.0;
  const state = {
    ball_x: C.width / 2.0,
    ball_y: C.height / 2.0,
    ball_vx: 0.0,
    ball_vy: 0.0,
    agent_y: mid,
    opp_y: mid,
    agent_score: 0,
    opp_score: 0,
    terminated: false,
    serve_dir: dir,
  };
  return serve(state, dir, ang);
}

function paddleOverlap(ball_x, ball_y, px, py) {
  const r = C.ballR;
  const pw = C.paddleW;
  const ph = C.paddleH;
  return px - r <= ball_x && ball_x <= px + pw + r && py - r <= ball_y && ball_y <= py + ph + r;
}

export function step(state, agentDy, oppDy, serveAngle) {
  const ang = serveAngle === undefined ? 0.0 : serveAngle;
  if (state.terminated) {
    return [Object.assign({}, state), 0.0];
  }
  const out = Object.assign({}, state);
  out.agent_y = clipPaddle(state.agent_y + agentDy);
  out.opp_y = clipPaddle(state.opp_y + oppDy);

  let ball_x = state.ball_x + state.ball_vx;
  let ball_y = state.ball_y + state.ball_vy;
  let ball_vx = state.ball_vx;
  let ball_vy = state.ball_vy;

  const r = C.ballR;
  const h = C.height;
  const w = C.width;
  const pw = C.paddleW;
  const ph = C.paddleH;
  const gain = C.bounceGain;
  const spin = C.spin;

  if (ball_y <= r) {
    ball_y = r;
    ball_vy = Math.abs(ball_vy);
  } else if (ball_y >= h - r) {
    ball_y = h - r;
    ball_vy = -Math.abs(ball_vy);
  }

  if (ball_vx < 0 && paddleOverlap(ball_x, ball_y, C.agentX, out.agent_y)) {
    ball_vx = Math.abs(ball_vx) * gain;
    const offset = (ball_y - (out.agent_y + ph / 2.0)) / (ph / 2.0);
    ball_vy = ball_vy + offset * spin;
    ball_x = C.agentX + pw + r;
  }

  if (ball_vx > 0 && paddleOverlap(ball_x, ball_y, C.oppX, out.opp_y)) {
    ball_vx = -Math.abs(ball_vx) * gain;
    const offset = (ball_y - (out.opp_y + ph / 2.0)) / (ph / 2.0);
    ball_vy = ball_vy + offset * spin;
    ball_x = C.oppX - r;
  }

  let reward = 0.0;
  if (ball_x < -r) {
    out.opp_score = (state.opp_score | 0) + 1;
    reward = -1.0;
    if (out.opp_score >= C.maxScore) {
      out.terminated = true;
      out.ball_x = ball_x;
      out.ball_y = clip(ball_y, r, h - r);
      out.ball_vx = ball_vx;
      out.ball_vy = ball_vy;
      return [out, reward];
    }
    return [serve(out, -state.serve_dir, ang), reward];
  }

  if (ball_x > w + r) {
    out.agent_score = (state.agent_score | 0) + 1;
    reward = 1.0;
    if (out.agent_score >= C.maxScore) {
      out.terminated = true;
      out.ball_x = ball_x;
      out.ball_y = clip(ball_y, r, h - r);
      out.ball_vx = ball_vx;
      out.ball_vy = ball_vy;
      return [out, reward];
    }
    return [serve(out, -state.serve_dir, ang), reward];
  }

  out.ball_x = ball_x;
  out.ball_y = ball_y;
  out.ball_vx = ball_vx;
  out.ball_vy = ball_vy;
  return [out, reward];
}
