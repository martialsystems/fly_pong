/* Clone of fly_pong/obs.py */
import { C } from "./constants.js";
import { clip, paddleSpan } from "./physics.js";

export function encode(state) {
  const span = paddleSpan();
  const scale = C.velScale;
  return [
    clip(state.ball_x / C.width, 0, 1),
    clip(state.ball_y / C.height, 0, 1),
    clip(state.ball_vx / scale, -1, 1),
    clip(state.ball_vy / scale, -1, 1),
    clip(state.agent_y / span, 0, 1),
    clip(state.opp_y / span, 0, 1),
  ];
}

export function encodeMirroredRight(state) {
  const span = paddleSpan();
  const scale = C.velScale;
  return [
    clip(1.0 - state.ball_x / C.width, 0, 1),
    clip(state.ball_y / C.height, 0, 1),
    clip((-state.ball_vx) / scale, -1, 1),
    clip(state.ball_vy / scale, -1, 1),
    clip(state.opp_y / span, 0, 1),
    clip(state.agent_y / span, 0, 1),
  ];
}

export function applyVecnorm(obs, norm) {
  const eps = norm.epsilon === undefined ? 1e-8 : norm.epsilon;
  const clipObs = norm.clip_obs === undefined ? 10.0 : norm.clip_obs;
  const out = new Float32Array(6);
  for (let i = 0; i < 6; i++) {
    const std = Math.sqrt(norm.var[i] + eps);
    let v = (obs[i] - norm.mean[i]) / std;
    if (v > clipObs) v = clipObs;
    if (v < -clipObs) v = -clipObs;
    out[i] = v;
  }
  return out;
}
