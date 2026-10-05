// 全流程无头回归测试: 全物种(G1 / PM01 / T1 / 天工 / 智元X1 / MicroDuck)混合比赛。
// 每台机器人 = 官方 MJCF 模型 + 官方 ONNX 策略 + 契约化观测 + PD, 全部自由物理。
// 运行: node test-gaits.mjs
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
const { Sim, makeRng } = await imp('src/sim.js');
const { PolicyRunner, PolicySession, steerCmd } = await imp('src/policy.js');
const { SPECIES } = await imp('src/robots.js');

const TICK = 0.01;

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
    let xml = fs.readFileSync(path.join(ROOT, sp.xmlFile.replace('./assets/', 'assets/')), 'utf8');
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

// 单物种 25m 测试(0.85×包线)
async function raceSolo(sp, vx, lane, seed) {
  const robot = sim.addRobot(sp);
  const runner = new PolicyRunner(sp.contract);
  runner.period = sp.dt * sp.decim;
  sim.resetRobot(robot, lane, makeRng(seed), sp.noise, 0);
  const cmd = new Float32Array(3);
  let acc = runner.period, t = 0;
  const ticks = Math.ceil(70 / TICK);
  for (let s = 0; s < ticks; s++) {
    t += TICK;
    const walking = s > 100; // 1s 站立
    if (s === 100) cmd[0] = vx;
    cmd[2] = walking ? Math.max(-(sp.yawCap ?? 1), Math.min(sp.yawCap ?? 1, steerCmd(robot.data.qpos, robot.data.qvel, lane, sp.steer))) : 0;
    acc += TICK;
    let inferred = false;
    if (acc >= runner.period - 1e-9) {
      acc -= runner.period;
      runner.buildAndPushObs(robot.data.qpos, robot.data.qvel, cmd, runner.period);
      const outs = await sessions.get(sp.id).inferAll([runner]);
      runner.applyAction(outs[0]);
      inferred = true;
    }
    for (let k = 0; k < Math.round(TICK / sp.dt); k++) {
      runner.pd(robot.data.qpos, robot.data.qvel, robot.data.ctrl);
      mj.mj_step(robot.model, robot.data);
    }
    void inferred;
    if (robot.data.qpos[0] >= 25) return { x: 25, t, ok: true, fell: false };
    if (robot.data.qpos[2] < sp.fallZ || !Number.isFinite(robot.data.qpos[2])) return { x: robot.data.qpos[0], t, ok: false, fell: true };
  }
  return { x: robot.data.qpos[0], t, ok: false, fell: false };
}

console.log('物种 25m 单测(0.85×包线):');
let allOk = true;
for (const sp of SPECIES) {
  const vx = +(sp.maxV * 0.85).toFixed(2);
  const r = await raceSolo(sp, vx, 0, 42);
  allOk = allOk && r.ok;
  console.log(`  ${r.ok ? '✓' : '✗'} ${sp.emoji} ${sp.name}: x=${r.x.toFixed(1)}m t=${r.t.toFixed(1)}s ${r.fell ? '(摔倒)' : ''}`);
}

// 全物种单台混合比赛(顶格速度)
console.log(`\n${SPECIES.length} 物种混合比赛(顶格速度):`);
const robots = SPECIES.map((sp, i) => {
  const runner = new PolicyRunner(sp.contract);
  runner.period = sp.dt * sp.decim;
  return {
    species: sp,
    sim: sim.addRobot(sp),
    runner,
    cmd: new Float32Array(3),
    laneY: (i - (SPECIES.length - 1) / 2) * 1.35,
    name: `${i + 1}号·${sp.short}`,
    targetSpeed: Math.min(sp.maxV, sp.maxV * (1 + (makeRng(100 + i)() * 2 - 1) * 0.06)),
    curVx: 0,
    fallZ: sp.fallZ,
    noise: sp.noise ?? 0.02,
    finished: false, finishTime: 0, penalty: 0, falls: 0,
    fallen: false, fallenAt: 0, x: 0, speed: 0, place: 0,
    acc: runner.period,
  };
});

const rng = makeRng(20260925);
for (const r of robots) sim.resetRobot(r.sim, r.laneY, rng, r.noise, 0);
// 与页面一致: 指令按 2.5 m/s² 斜坡平滑起步(阶跃指令易摔, 且非比赛真实情况)
let rampClock = 0;

let raceClock = 0;
for (let s = 0; s < Math.ceil(95 / TICK); s++) {
  raceClock += TICK;
  rampClock += TICK;
  // 与页面一致: 发枪前有站立期(倒计时), 之后才按斜率限幅平滑起步
  if (s > 100) for (const r of robots) r.curVx += Math.max(-2.5 * TICK, Math.min(2.5 * TICK, r.targetSpeed - r.curVx));
  void rampClock;
  for (const r of robots) {
    const q = r.sim.data.qpos;
    r.x = q[0];
    r.speed = r.sim.data.qvel[0];
    if (!r.finished && r.x >= 25) { r.finished = true; r.finishTime = raceClock + r.penalty; r.place = robots.filter((x) => x.finished).length; }
    if (r.finished && raceClock > r.finishTime + 1.0) r.curVx = 0;
    if (q[2] < r.fallZ && !r.finished && !r.fallen) {
      r.fallen = true; r.fallenAt = raceClock; r.falls++; r.penalty += 2;
      console.log(`  [FALL] ${r.name} t=${raceClock.toFixed(1)} x=${q[0].toFixed(1)} y=${q[1].toFixed(2)} z=${q[2].toFixed(2)} vx=${r.sim.data.qvel[0].toFixed(2)} vy=${r.sim.data.qvel[1].toFixed(2)}`);
    }
    if (r.fallen && raceClock - r.fallenAt >= 1.2) {
      sim.resetRobot(r.sim, r.laneY, rng, 0.02, Math.max(r.x, 0));
      r.runner.reset();
      r.acc = 0;
      r.curVx = r.targetSpeed;
      r.fallen = false;
    }
  }
  for (const r of robots) {
    if (r.fallen) continue;
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
  }
  if (robots.every((r) => r.finished)) break;
}

const standings = [...robots].sort((a, b) => {
  if (a.finished && b.finished) return a.finishTime - b.finishTime;
  if (a.finished) return -1;
  if (b.finished) return 1;
  return b.x - a.x;
});
console.log('  排名:');
standings.forEach((r, i) => {
  console.log(`   ${i + 1}. ${r.species.emoji} ${r.name.padEnd(12)} ${r.finished ? `${r.finishTime.toFixed(2)}s (均速 ${(25 / r.finishTime).toFixed(2)}m/s)` : `x=${r.x.toFixed(1)}m`}${r.falls ? ` 摔${r.falls}次` : ''}`);
});
const raceOk = standings.every((r) => r.finished);
console.log(allOk && raceOk ? '\nALL PASS: 全部物种完赛' : '\nFAILED');
process.exit(allOk && raceOk ? 0 : 1);
