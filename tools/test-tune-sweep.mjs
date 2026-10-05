// 物理调参实验: 实测"提高力矩上限 / 提高地面摩擦"能否把策略的稳定速度边界推过 1.6 m/s。
// 历史调参实验, 结论已写入 README「已知实现要点」(G1 策略极限 1.55 m/s)。
// XML 修改全部在内存字符串上做, 不动 assets 原文件。运行: node tools/test-tune-sweep.mjs
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require_ = createRequire(import.meta.url);
const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url))); // 脚本在 tools/ 下, ROOT 指向仓库根
const imp = (p) => import(pathToFileURL(path.join(ROOT, p)).href);

const ortSrc = fs.readFileSync(path.join(ROOT, 'vendor/ort/ort.min.js'), 'utf8');
const ort = new Function('require', ortSrc + '\n;return ort;')(require_);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(path.join(ROOT, 'vendor/ort')).href + '/';

const loadMujoco = (await imp('vendor/mujoco/mujoco.js')).default;
const mj = await loadMujoco();
const { Sim, buildSceneXml, makeRng } = await imp('src/sim.js');
const { PolicyRunner, PolicySession, CFG, steerCmd } = await imp('src/policy.js');

// 把 XML 中所有 actuatorfrcrange/ctrlrange 的数值乘以 f
function boostTorque(xml, f) {
  return xml.replace(/(actuatorfrcrange|ctrlrange)="(-?[\d.]+)\s+(-?[\d.]+)"/g,
    (m, attr, lo, hi) => `${attr}="${-(parseFloat(lo) * f).toFixed(1)} ${(parseFloat(hi) * f).toFixed(1)}"`);
}
function floorFriction(xml, mu) {
  return xml.replace(/friction="[\d.]+ /, `friction="${mu} `);
}

const baseXml = fs.readFileSync(path.join(ROOT, 'assets/g1_29dof.xml'), 'utf8');
const SIM_TIME = 45;

async function loadSim(xmlTransform, sceneTransform) {
  const assets = {
    xml: xmlTransform(baseXml),
    meshes: new Map(),
    sceneXml: sceneTransform ? sceneTransform(buildSceneXml()) : buildSceneXml(),
  };
  for (const name of [...new Set([...baseXml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))]) {
    assets.meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/meshes', name))));
  }
  const sim = await Sim.load(mj, assets, () => {});
  const session = new PolicySession(ort);
  await session.load(new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/policy.onnx'))));
  return { sim, session };
}

async function raceRobot(sim, session, seed, laneY, targetSpeed) {
  const robot = { data: new mj.MjData(sim.model) };
  const runner = new PolicyRunner();
  sim.resetRobot(robot, laneY, makeRng(seed), 0.025, 0);
  const cmd = new Float32Array(3);
  let maxV = 0;
  const N = Math.ceil(SIM_TIME / 0.02);
  for (let s = 0; s < N; s++) {
    if (s === 0 || s === 60) cmd[0] = targetSpeed;
    cmd[2] = steerCmd(robot.data.qpos, laneY);
    runner.pushObs(runner.buildObs(robot.data.qpos, robot.data.qvel, cmd));
    const out = await session.inferAll([runner]);
    runner.applyAction(out[0]);
    for (let k = 0; k < CFG.decimation; k++) {
      runner.pd(robot.data.qpos, robot.data.qvel, robot.data.ctrl);
      mj.mj_step(sim.model, robot.data);
    }
    maxV = Math.max(maxV, robot.data.qvel[0]);
    if (robot.data.qpos[0] >= 25) return { ok: true, t: s * 0.02, maxV };
    if (robot.data.qpos[2] < 0.45) return { ok: false, x: robot.data.qpos[0], maxV };
  }
  return { ok: false, x: robot.data.qpos[0], maxV };
}

const configs = [
  { name: 'baseline(现状)',        xml: (x) => x, scene: null },
  { name: '摩擦x2 (mu=2.0)',      xml: (x) => x, scene: (s) => floorFriction(s, 2.0) },
  { name: '力矩x1.5',              xml: (x) => boostTorque(x, 1.5), scene: null },
  { name: '力矩x1.5 + 摩擦x2',     xml: (x) => boostTorque(x, 1.5), scene: (s) => floorFriction(s, 2.0) },
  { name: '力矩x2 + 摩擦x2',       xml: (x) => boostTorque(x, 2.0), scene: (s) => floorFriction(s, 2.0) },
];
const speeds = [1.6, 1.8, 2.0, 2.25, 2.5];

for (const cfg of configs) {
  const { sim, session } = await loadSim(cfg.xml, cfg.scene);
  console.log(`\n=== ${cfg.name} ===`);
  for (const v of speeds) {
    const r = await raceRobot(sim, session, 20260924, -0.675, v);
    if (r.ok) console.log(`  cmd=${v.toFixed(2)}: ✓ 完赛 t=${r.t.toFixed(1)}s 均速=${(25 / (r.t - 1.2)).toFixed(2)}m/s 峰值=${r.maxV.toFixed(2)}`);
    else console.log(`  cmd=${v.toFixed(2)}: ✗ 摔倒@${r.x.toFixed(1)}m 峰值=${r.maxV.toFixed(2)}`);
  }
}
process.exit(0);
