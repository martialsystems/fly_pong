#!/usr/bin/env node
import fs from "fs";
import { dyFromAction, initialState, step } from "./physics.js";
import { encode, encodeMirroredRight } from "./obs.js";

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let state = spec.initial ? spec.initial : initialState(spec.serve_dir, spec.angle);
const frames = [];
for (const frame of spec.actions) {
  const agentDy =
    frame.agent_dy !== undefined ? frame.agent_dy : dyFromAction(frame.agent_action);
  const oppDy = frame.opp_dy !== undefined ? frame.opp_dy : dyFromAction(frame.opp_action);
  const ang = frame.serve_angle === undefined ? 0.0 : frame.serve_angle;
  const pair = step(state, agentDy, oppDy, ang);
  state = pair[0];
  frames.push({
    state,
    reward: pair[1],
    obs: encode(state),
    obs_mirror: encodeMirroredRight(state),
  });
}
process.stdout.write(JSON.stringify({ final: state, frames }));
