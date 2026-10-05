// 赛道保持回归测试: 验证横向 PD 级联外环 + 物理护栏, 三物种都能贴道跑完 25m 且永不冲出跑道。
// 与页面 stepOnce 相同的闭环(MuJoCo WASM + ONNX 策略 + 门控指令 + 平滑起步)。
// 运行: node test-lanekeep.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const imp = (p) => import(pathToFileURL(path.join(ROOT, p)).href);

const ortSrc = fs.readFileSync(path.join(ROOT, 'vendor/ort/ort.min.js'), 'utf8');
const require_ = createRequire(import.meta.url);
const ort = new Function('require', ortSrc + '\n;return ort;')(require_);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(path.join(ROOT, 'vendor/ort')).href + '/';

const loadMujoco = (await imp('vendor/mujoco/mujoco.js')).default;
const mj = await loadMujoco();
const { Sim, makeRng, FENCE_INNER_Y } = await imp('src/sim.js');
const { PolicyRunner, PolicySession, steerCmd } = await imp('src/policy.js');
const { SPECIES } = await imp('src/robots.js');

const TICK = 0.01;
const LANE_W = 1.35;
const laneY = (i) => (i - 2) * LANE_W;

async function loadAssets() {
  const assets = { g1: null, species: new Map() };
  const g1 = SPECIES[0];
  const g1Xml = fs.readFileSync(path.join(ROOT, 'assets/g1_29dof.xml'), 'utf8');
  const g1Meshes = new Map();
  for (const name of [...new Set([...g1Xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))]) {
    g1Meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/meshes', name))));
  }
  assets.g1 = { xml: g1Xml, meshes: g1Meshes };
  for (const sp of SPECIES) {
    if (sp.id === 'g1') continue;
    const extraFiles = new Map();
    for (const f of sp.extraFiles ?? []) {
      extraFiles.set(f, fs.readFileSync(path.join(ROOT, 'assets', sp.id, f), 'utf8'));
    }
    const xml = fs.readFileSync(path.join(ROOT, sp.xmlFile.replace('./assets/', 'assets/')), 'utf8');
    const allXml = [xml, ...extraFiles.values()].join('\n');
    const meshes = new Map();
    const dir = path.join(ROOT, 'assets', sp.id, 'meshes');
    for (const name of [...new Set([...allXml.matchAll(/file="([^"]+\.(?:STL|stl|obj))"/g)].map((m) => m[1].split('/').pop()))]) {
      meshes.set(name, new Uint8Array(fs.readFileSync(path.join(dir, name))));
    }
    assets.species.set(sp.id, { xml, extraFiles, meshes });
  }
  return assets;
}

const assets = await loadAssets();
const sim = await Sim.load(mj, assets, () => {});
const sessions = new Map();
for (const sp of SPECIES) {
  const s = new PolicySession(ort, sp.contract);
  await s.load(new Uint8Array(fs.readFileSync(path.join(ROOT, sp.policyFile.replace('./assets/', 'assets/')))));
  sessions.set(sp.id, s);
}

// 单台 25m: 记录横向偏差与是否触护栏。noSteer=true 时关掉外环, 用来验证护栏兜底。
async function raceSolo(sp, vx, lane, seed, { noSteer = false } = {}) {
  const robot = sim.addRobot(sp);
  try {
    return await raceSoloInner(sp, vx, lane, seed, { noSteer, robot });
  } finally {
    try { robot.data.delete(); } catch (e) { /* 忽略 */ }
  }
}

async function raceSoloInner(sp, vx, lane, seed, { noSteer = false, robot } = {}) {
  const runner = new PolicyRunner(sp.contract);
  runner.period = sp.dt * sp.decim;
  sim.resetRobot(robot, lane, makeRng(seed), sp.noise, 0);
  const cmd = new Float32Array(3);
  let acc = runner.period, t = 0;
  let curVx = 0, maxDev = 0, maxAbsY = 0;
  for (let s = 0; s < Math.ceil(90 / TICK); s++) {
    t += TICK;
    const running = s > 100; // 1s 站立后发枪
    curVx += Math.max(-2.5 * TICK, Math.min(2.5 * TICK, (running ? vx : 0) - curVx));
    const q = robot.data.qpos, qv = robot.data.qvel;
    cmd[0] = curVx;
    cmd[2] = running && !noSteer ? Math.max(-(sp.yawCap ?? 1), Math.min(sp.yawCap ?? 1, steerCmd(q, qv, lane, sp.steer))) : 0;
    acc += TICK;
    if (acc >= runner.period - 1e-9) {
      acc -= runner.period;
      runner.buildAndPushObs(q, qv, cmd, runner.period);
      const outs = await sessions.get(sp.id).inferAll([runner]);
      runner.applyAction(outs[0]);
    }
    for (let k = 0; k < Math.round(TICK / sp.dt); k++) {
      runner.pd(q, qv, robot.data.ctrl);
      mj.mj_step(robot.model, robot.data);
    }
    maxDev = Math.max(maxDev, Math.abs(q[1] - lane));
    maxAbsY = Math.max(maxAbsY, Math.abs(q[1]));
    if (q[0] >= 25) return { done: true, fell: false, t, maxDev, maxAbsY, hitFence: maxAbsY >= FENCE_INNER_Y - 0.15 };
    if (q[2] < sp.fallZ || !Number.isFinite(q[2])) return { done: false, fell: true, t, maxDev, maxAbsY, hitFence: maxAbsY >= FENCE_INNER_Y - 0.15 };
  }
  return { done: false, fell: false, t, maxDev, maxAbsY, hitFence: maxAbsY >= FENCE_INNER_Y - 0.15 };
}

let pass = true;
console.log('1) 三物种贴道跑(全速包线, 典型车道):');
for (const sp of SPECIES) {
  const lanes = [-3.375, 0.675, 3.375];
  for (let k = 0; k < lanes.length; k++) {
    const r = await raceSolo(sp, sp.maxV, lanes[k], 400 + k);
    const ok = r.done && !r.fell && r.maxDev < 0.55;
    pass = pass && ok;
    console.log(`  ${ok ? '✓' : '✗'} ${sp.emoji} ${sp.name} v=${sp.maxV} 道 y=${lanes[k]}: ${r.done ? `完赛 ${r.t.toFixed(1)}s` : '未完赛'} 最大偏差=${r.maxDev.toFixed(2)}m${r.fell ? ' 摔倒' : ''}${r.hitFence ? ' ⚠触护栏' : ''}`);
  }
}

console.log('2) G1 关掉航向外环(固有弧线跑偏) -> 验证物理护栏兜底:');
{
  const r = await raceSolo(SPECIES[0], 1.2, -3.375, 777, { noSteer: true });
  const ok = r.maxAbsY <= FENCE_INNER_Y + 0.1 && Number.isFinite(r.maxAbsY);
  pass = pass && ok;
  console.log(`  ${ok ? '✓' : '✗'} 无外环 G1: |y|最大=${r.maxAbsY.toFixed(2)}m (护栏内侧面 ${FENCE_INNER_Y}m) ${r.hitFence ? '被护栏挡住, 未冲出 ✓' : '未触护栏, 未越界 ✓'}`);
}

console.log('3) 全物种混合比赛(顶格速度, 平滑起步, 与页面同一闭环):');
const robots = SPECIES.map((sp, i) => {
  const runner = new PolicyRunner(sp.contract);
  runner.period = sp.dt * sp.decim;
  return {
    species: sp,
    sim: sim.addRobot(sp),
    runner,
    cmd: new Float32Array(3),
    laneY: laneY(i),
    name: `${i + 1}号·${sp.short}`,
    targetSpeed: Math.min(sp.maxV, sp.maxV * (1 + (makeRng(100 + i)() * 2 - 1) * 0.06)),
    curVx: 0,
    fallZ: sp.fallZ,
    noise: sp.noise ?? 0.02,
    finished: false, finishTime: 0, penalty: 0, falls: 0,
    fallen: false, fallenAt: 0, x: 0, speed: 0, place: 0,
    acc: runner.period,
    maxDev: 0, maxAbsY: 0,
  };
});

const rng = makeRng(20260925);
for (const r of robots) sim.resetRobot(r.sim, r.laneY, rng, r.noise, 0);

let raceClock = 0;
const raceTicks = Math.ceil(95 / TICK);
for (let s = 0; s < raceTicks; s++) {
  raceClock += TICK;
  for (const r of robots) {
    const q = r.sim.data.qpos;
    r.x = q[0];
    r.speed = r.sim.data.qvel[0];
    if (!r.finished && r.x >= 25) { r.finished = true; r.finishTime = raceClock + r.penalty; r.place = robots.filter((x) => x.finished).length + 1; }
    if (r.finished && raceClock > r.finishTime + 1.5) r.targetSpeed = 0;
    if (q[2] < r.fallZ && !r.finished && !r.fallen) { r.fallen = true; r.fallenAt = raceClock; r.falls++; r.penalty += 2; }
    if (r.fallen && raceClock - r.fallenAt >= 1.2) {
      sim.resetRobot(r.sim, r.laneY, rng, 0.02, Math.max(r.x, 0));
      r.runner.reset();
      r.acc = 0;
      r.curVx = 0;
      r.targetSpeed = r.species.maxV; // 与页面一致: 扶起后继续朝目标速度跑
      r.fallen = false;
    }
  }
  const running = raceClock > 1.0; // 与页面一致: 先站立(倒计时)再发枪平滑起步
  for (const r of robots) {
    if (r.fallen) continue;
    if (running) r.curVx += Math.max(-2.5 * TICK, Math.min(2.5 * TICK, r.targetSpeed - r.curVx));
    r.acc += TICK;
    if (r.acc >= r.runner.period - 1e-9) {
      r.acc -= r.runner.period;
      r.cmd[0] = r.curVx;
      r.cmd[2] = Math.max(-(r.species.yawCap ?? 1), Math.min(r.species.yawCap ?? 1, steerCmd(r.sim.data.qpos, r.sim.data.qvel, r.laneY, r.species.steer)));
      r.runner.buildAndPushObs(r.sim.data.qpos, r.sim.data.qvel, r.cmd, r.runner.period);
      const outs = await sessions.get(r.species.id).inferAll([r.runner]);
      r.runner.applyAction(outs[0]);
    }
    for (let k = 0; k < Math.round(TICK / r.species.dt); k++) {
      r.runner.pd(r.sim.data.qpos, r.sim.data.qvel, r.sim.data.ctrl);
      mj.mj_step(r.sim.model, r.sim.data);
    }
    r.maxDev = Math.max(r.maxDev, Math.abs(r.sim.data.qpos[1] - r.laneY));
    r.maxAbsY = Math.max(r.maxAbsY, Math.abs(r.sim.data.qpos[1]));
  }
  if (robots.every((r) => r.finished && raceClock > r.finishTime + 1.5)) break;
}

const standings = [...robots].sort((a, b) => {
  if (a.finished && b.finished) return a.finishTime - b.finishTime;
  if (a.finished) return -1;
  if (b.finished) return 1;
  return b.x - a.x;
});
standings.forEach((r, i) => {
  const ok = r.finished && r.maxDev < 0.55;
  pass = pass && ok;
  console.log(`  ${ok ? '✓' : '✗'} ${i + 1}. ${r.species.emoji} ${r.name.padEnd(12)} ${r.finished ? `${r.finishTime.toFixed(2)}s (均速 ${(25 / r.finishTime).toFixed(2)}m/s)` : `x=${r.x.toFixed(1)}m DNF`} 摔${r.falls}次 最大偏差=${r.maxDev.toFixed(2)}m${r.maxAbsY >= FENCE_INNER_Y - 0.05 ? ' ⚠触护栏' : ''}`);
});

console.log(pass ? '\nALL PASS: 全部贴道完赛且未冲出跑道' : '\nFAILED');
process.exit(pass ? 0 : 1);
