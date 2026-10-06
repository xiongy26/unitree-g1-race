// 子集加载管线测试(选择屏按需加载 / 赛后换阵容共用的 Sim 管线, Node 无头):
//   1) Sim.load 第 4 参 speciesList 传子集: 只编译所选物种, hasModel 精确成立
//   2) "只编一个"且不含 G1: hasModel('g1') 为 false, modelFor(g1) 显式 throw
//   3) "先编 A 后补编 B"(换阵容增量编译): Sim.empty + compileSpecies 逐个编入, warm VFS 可继续注入
//   4) N=1 计时单跑完赛 与 N=2 紧凑居中混合比赛双完赛(内联完成度判定, 同 test-gaits 闭环)
// 运行: node tools/test-subset.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(TOOLS, '..'); // 仓库根(本测试在 tools/ 下)
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

const byId = (id) => SPECIES.find((s) => s.id === id);
const TICK = 0.01;

// 构造子集 assets(fs 版, 与 test-gaits loadAssets 同构, 只取 ids 内的物种)
function loadSubsetAssets(ids) {
  const assets = { g1: null, species: new Map() };
  const g1 = SPECIES[0];
  if (ids.includes('g1')) {
    const g1Xml = fs.readFileSync(path.join(ROOT, 'assets/g1_29dof.xml'), 'utf8');
    const g1Meshes = new Map();
    for (const name of [...new Set([...g1Xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))]) {
      g1Meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/meshes', name))));
    }
    assets.g1 = { xml: g1Xml, meshes: g1Meshes };
  }
  for (const sp of SPECIES) {
    if (sp.id === 'g1' || !ids.includes(sp.id)) continue;
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

async function loadSessions(ids) {
  const sessions = new Map();
  for (const id of ids) {
    const sp = byId(id);
    const s = new PolicySession(ort, sp.contract);
    await s.load(new Uint8Array(fs.readFileSync(path.join(ROOT, sp.policyFile.replace('./assets/', 'assets/')))));
    sessions.set(id, s);
  }
  return sessions;
}

// 单物种 25m(0.85×包线, 同 test-gaits raceSolo 闭环)
async function raceSolo(sim, sp, vx, lane, seed, sessions) {
  const robot = sim.addRobot(sp);
  const runner = new PolicyRunner(sp.contract);
  runner.period = sp.dt * sp.decim;
  sim.resetRobot(robot, lane, makeRng(seed), sp.noise, 0);
  const cmd = new Float32Array(3);
  let acc = runner.period, t = 0;
  for (let s = 0; s < Math.ceil(70 / TICK); s++) {
    t += TICK;
    const walking = s > 100; // 1s 站立
    if (s === 100) cmd[0] = vx;
    cmd[2] = walking ? Math.max(-(sp.yawCap ?? 1), Math.min(sp.yawCap ?? 1, steerCmd(robot.data.qpos, robot.data.qvel, lane, sp.steer))) : 0;
    acc += TICK;
    if (acc >= runner.period - 1e-9) {
      acc -= runner.period;
      runner.buildAndPushObs(robot.data.qpos, robot.data.qvel, cmd, runner.period);
      const outs = await sessions.get(sp.id).inferAll([runner]);
      runner.applyAction(outs[0]);
    }
    for (let k = 0; k < Math.round(TICK / sp.dt); k++) {
      runner.pd(robot.data.qpos, robot.data.qvel, robot.data.ctrl);
      mj.mj_step(robot.model, robot.data);
    }
    if (robot.data.qpos[0] >= 25) return { x: 25, t, ok: true, fell: false };
    if (robot.data.qpos[2] < sp.fallZ || !Number.isFinite(robot.data.qpos[2])) return { x: robot.data.qpos[0], t, ok: false, fell: true };
  }
  return { x: robot.data.qpos[0], t, ok: false, fell: false };
}

// N 台紧凑居中混合比赛(顶格速度 + 平滑起步 + 摔倒扶起, 同 test-gaits 混赛闭环)
async function raceMixed(sim, specs, sessions) {
  const robots = specs.map((sp, i) => {
    const runner = new PolicyRunner(sp.contract);
    runner.period = sp.dt * sp.decim;
    return {
      species: sp,
      sim: sim.addRobot(sp),
      runner,
      cmd: new Float32Array(3),
      laneY: (i - (specs.length - 1) / 2) * 1.35, // 与页面 laneY(i, N) 同式: 紧凑居中
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
  let raceClock = 0;
  for (let s = 0; s < Math.ceil(95 / TICK); s++) {
    raceClock += TICK;
    // 发枪前站立 1s, 之后按 2.5 m/s² 斜率限幅平滑起步(与页面一致)
    if (s > 100) for (const r of robots) r.curVx += Math.max(-2.5 * TICK, Math.min(2.5 * TICK, r.targetSpeed - r.curVx));
    for (const r of robots) {
      const q = r.sim.data.qpos;
      r.x = q[0];
      r.speed = r.sim.data.qvel[0];
      if (!r.finished && r.x >= 25) { r.finished = true; r.finishTime = raceClock + r.penalty; r.place = robots.filter((x) => x.finished).length; }
      if (r.finished && raceClock > r.finishTime + 1.0) r.curVx = 0;
      if (q[2] < r.fallZ && !r.finished && !r.fallen) {
        r.fallen = true; r.fallenAt = raceClock; r.falls++; r.penalty += 2;
        console.log(`  [FALL] ${r.name} t=${raceClock.toFixed(1)} x=${q[0].toFixed(1)}`);
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
  return robots.sort((a, b) => (a.finished && b.finished) ? a.finishTime - b.finishTime : b.x - a.x);
}

let pass = true;
const check = (name, ok) => { pass = pass && !!ok; console.log(`  ${ok ? '✓' : '✗'} ${name}`); };

console.log('1) Sim.load 子集 [g1, duck]: 只编译所选物种');
const simAB = await Sim.load(mj, loadSubsetAssets(['g1', 'duck']), () => {}, [byId('g1'), byId('duck')]);
check('hasModel(g1) 与 hasModel(duck) 为 true', simAB.hasModel('g1') && simAB.hasModel('duck'));
check('未选物种(pm01/t1/tk/x1) hasModel 全为 false', ['pm01', 't1', 'tk', 'x1'].every((id) => !simAB.hasModel(id)));

console.log('2) 只编一个且不含 G1: [duck]');
const g1sp = byId('g1');
const simDuck = await Sim.load(mj, loadSubsetAssets(['duck']), () => {}, [byId('duck')]);
check('hasModel(duck) 为 true 且 hasModel(g1) 为 false', simDuck.hasModel('duck') && !simDuck.hasModel('g1'));
let threw = false;
try { simDuck.modelFor(g1sp); } catch (e) { threw = true; }
check('modelFor(g1) 显式 throw(不静默返回 undefined)', threw);

console.log('3) 先编 A 后补编 B(换阵容增量编译): empty + compileSpecies(g1) → compileSpecies(duck)');
const simInc = Sim.empty(mj);
const assetsAB = loadSubsetAssets(['g1', 'duck']);
await Sim.compileSpecies(simInc, g1sp, assetsAB.g1, () => {});
check('先编 G1: hasModel(g1) 为 true', simInc.hasModel('g1'));
await Sim.compileSpecies(simInc, byId('duck'), assetsAB.species.get('duck'), () => {});
check('后补编 duck: hasModel(duck) 为 true(warm VFS 增量注入成功)', simInc.hasModel('duck'));

console.log('4) N=1 计时单跑(G1, 模型经 empty+compileSpecies 管线编入)');
const sessionsGD = await loadSessions(['g1', 'duck']);
const solo = await raceSolo(simInc, g1sp, +(g1sp.maxV * 0.85).toFixed(2), 0.675, 42, sessionsGD);
check(`G1 单跑完赛(x=${solo.x.toFixed(1)}m t=${solo.t.toFixed(1)}s${solo.fell ? ' 摔倒' : ''})`, solo.ok);

console.log('5) N=2 紧凑居中混合比赛(G1 + MicroDuck, 车道 ±0.675)');
const robots = await raceMixed(simAB, [g1sp, byId('duck')], sessionsGD);
for (const r of robots) {
  console.log(`   ${r.finished ? `${r.place}.` : 'DNF'} ${r.species.emoji} ${r.name.padEnd(12)} ${r.finished ? `${r.finishTime.toFixed(2)}s` : `x=${r.x.toFixed(1)}m`} 摔${r.falls}次`);
}
check('两台都完赛(N=2, 紧凑居中车道)', robots.every((r) => r.finished));

console.log(pass ? '\nALL PASS: 子集加载管线与小型阵容比赛全部通过' : '\nFAILED');
process.exit(pass ? 0 : 1);
