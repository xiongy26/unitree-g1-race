// X1(shoulder 摆臂策略)航向外环调参: 摆臂策略对 wy 指令极敏感(Python: wy=0.1 即失稳),
// kpYaw=3.0(为旧 leg 策略调的)持续微纠正把速度磨掉。扫增益找 速度/偏差 平衡点。
// 运行: node tools/tune-x1-steer.mjs [配置序号...]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url))); // 脚本在 tools/ 下, ROOT 指向仓库根
const imp = (p) => import(pathToFileURL(path.join(ROOT, p)).href);

const ortSrc = fs.readFileSync(path.join(ROOT, 'vendor/ort/ort.min.js'), 'utf8');
const require_ = createRequire(import.meta.url);
const ort = new Function('require', ortSrc + '\n;return ort;')(require_);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(path.join(ROOT, 'vendor/ort')).href + '/';

const loadMujoco = (await imp('vendor/mujoco/mujoco.js')).default;
const mj = await loadMujoco();
const { Sim, makeRng, FENCE_INNER_Y } = await imp('src/sim.js');
const { PolicyRunner, PolicySession, steerCmd, makeSteer } = await imp('src/policy.js');
const { SPECIES, speciesById } = await imp('src/robots.js');

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
const sp = speciesById('x1');
const session = new PolicySession(ort, sp.contract);
await session.load(new Uint8Array(fs.readFileSync(path.join(ROOT, sp.policyFile.replace('./assets/', 'assets/')))));

async function raceSolo(vx, lane, seed, steer) {
  const robot = sim.addRobot(sp);
  try {
    const runner = new PolicyRunner(sp.contract);
    runner.period = sp.dt * sp.decim;
    sim.resetRobot(robot, lane, makeRng(seed), sp.noise, 0);
    const cmd = new Float32Array(3);
    let acc = runner.period, t = 0, curVx = 0, maxDev = 0, maxAbsY = 0, wyMax = 0;
    for (let s = 0; s < Math.ceil(90 / TICK); s++) {
      t += TICK;
      const running = s > 100;
      curVx += Math.max(-2.5 * TICK, Math.min(2.5 * TICK, (running ? vx : 0) - curVx));
      const q = robot.data.qpos, qv = robot.data.qvel;
      cmd[0] = curVx;
      cmd[2] = running ? Math.max(-(sp.yawCap ?? 1), Math.min(sp.yawCap ?? 1, steerCmd(q, qv, lane, steer))) : 0;
      if (running) wyMax = Math.max(wyMax, Math.abs(cmd[2]));
      acc += TICK;
      if (acc >= runner.period - 1e-9) {
        acc -= runner.period;
        runner.buildAndPushObs(q, qv, cmd, runner.period);
        const outs = await session.inferAll([runner]);
        runner.applyAction(outs[0]);
      }
      for (let k = 0; k < Math.round(TICK / sp.dt); k++) {
        runner.pd(q, qv, robot.data.ctrl);
        mj.mj_step(robot.model, robot.data);
      }
      maxDev = Math.max(maxDev, Math.abs(q[1] - lane));
      maxAbsY = Math.max(maxAbsY, Math.abs(q[1]));
      if (q[0] >= 25) return { done: true, fell: false, t, maxDev, maxAbsY, hitFence: maxAbsY >= FENCE_INNER_Y - 0.15, wyMax };
      if (q[2] < sp.fallZ || !Number.isFinite(q[2])) return { done: false, fell: true, t, maxDev, maxAbsY, hitFence: maxAbsY >= FENCE_INNER_Y - 0.15, wyMax };
    }
    return { done: false, fell: false, t, maxDev, maxAbsY, hitFence: maxAbsY >= FENCE_INNER_Y - 0.15, wyMax };
  } finally {
    try { robot.data.delete(); } catch (e) { /* 忽略 */ }
  }
}

const CONFIGS = [
  { name: 'A 现用 kpYaw3.0/kd0.2', steer: makeSteer({ kpYaw: 3.0, kdYaw: 0.2 }) },
  { name: 'B kpYaw1.0/kd0.1', steer: makeSteer({ kpYaw: 1.0, kdYaw: 0.1 }) },
  { name: 'C kpYaw0.5/kd0.1', steer: makeSteer({ kpYaw: 0.5, kdYaw: 0.1 }) },
  { name: 'D kpYaw0.3/kd0.05+cap0.12', steer: makeSteer({ kpYaw: 0.3, kdYaw: 0.05 }), yawCap: 0.12 },
  { name: 'E kpYaw0.15/kd0.02+cap0.06', steer: makeSteer({ kpYaw: 0.15, kdYaw: 0.02 }), yawCap: 0.06 },
];
const picks = process.argv.slice(2).map(Number);
const list = picks.length ? picks.map((i) => CONFIGS[i]) : CONFIGS;
for (const cfg of list) {
  sp.steer = cfg.steer;
  if (cfg.yawCap) sp.yawCap = cfg.yawCap; else delete sp.yawCap;
  const lanes = [0.675, -3.375];
  let sum = 0, n = 0, dev = 0, falls = 0, det = '';
  for (let k = 0; k < lanes.length; k++) {
    const r = await raceSolo(sp.maxV, lanes[k], 400 + k * 7);
    n++;
    if (r.done) sum += r.t; else falls++;
    dev = Math.max(dev, r.maxDev);
    det += ` [y=${lanes[k]}] ${r.done ? `${r.t.toFixed(1)}s` : '未完赛'} dev=${r.maxDev.toFixed(2)} wyMax=${r.wyMax.toFixed(2)}${r.fell ? ' 摔' : ''}`;
  }
  console.log(`${cfg.name}: 均时=${falls ? 'N/A' : (sum / n).toFixed(1)}s 最坏偏差=${dev.toFixed(2)}m 摔=${falls}${det}`);
}
