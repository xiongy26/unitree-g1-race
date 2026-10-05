// 速度扫描实验: 实测 ONNX 速度策略在超出训练范围(>1.0 m/s)的指令下能否稳定跑。
// 对每个指令速度测两种发令方式: 阶跃(GO 后直接给满指令)与斜坡(按 3 m/s² 缓升)。
// 运行: node tools/test-speed-sweep.mjs [起始速度] [结束速度] [步长]
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
const { Sim, makeRng } = await imp('src/sim.js');
const { PolicyRunner, PolicySession, steerCmd } = await imp('src/policy.js');
const { SPECIES } = await imp('src/robots.js');
const G1_CONTRACT = SPECIES[0].contract;

const assets = {
  g1: { xml: fs.readFileSync(path.join(ROOT, 'assets/g1_29dof.xml'), 'utf8'), meshes: new Map() },
  species: new Map(),
};
for (const name of [...new Set([...assets.g1.xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))]) {
  assets.g1.meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/meshes', name))));
}
const sim = await Sim.load(mj, assets, () => {});
const session = new PolicySession(ort, G1_CONTRACT);
await session.load(new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/policy.onnx'))));

const SIM_TIME = 45; // 每档最多仿 45s(25m @ 0.55m/s 也能完赛)

// cmdRamp: >0 按该斜率 m/s² 缓升; <0 两段式软起步(先 1.2 m/s 起步再按 |ramp| m/s² 缓升); 0 阶跃
async function raceRobot(seed, laneY, targetSpeed, cmdRamp) {
  const robot = { data: new mj.MjData(sim.model) };
  const runner = new PolicyRunner(G1_CONTRACT);
  sim.resetRobot(robot, laneY, makeRng(seed), 0.025, 0);
  const cmd = new Float32Array(3);
  let cmdNow = 0, fall = false, maxX = 0, maxV = 0;
  const N = Math.ceil(SIM_TIME / 0.02);
  for (let s = 0; s < N; s++) {
    if (s >= 60) { // 前 1.2s 站立(对应倒计时), 之后发令
      if (cmdRamp > 0) cmdNow = Math.min(targetSpeed, cmdNow + cmdRamp * 0.02);
      else if (cmdRamp < 0) cmdNow = cmdNow === 0 ? 1.2 : Math.min(targetSpeed, cmdNow + (-cmdRamp) * 0.02);
      else cmdNow = targetSpeed;
    }
    cmd[0] = cmdNow;
    cmd[2] = steerCmd(robot.data.qpos, robot.data.qvel, laneY);
    runner.buildAndPushObs(robot.data.qpos, robot.data.qvel, cmd, runner.period);
    const out = await session.inferAll([runner]);
    runner.applyAction(out[0]);
    for (let k = 0; k < 10; k++) { // decimation 10 (dt 0.002 x 10 = 20ms 策略周期)
      runner.pd(robot.data.qpos, robot.data.qvel, robot.data.ctrl);
      mj.mj_step(sim.model, robot.data);
    }
    const x = robot.data.qpos[0];
    if (x > maxX) maxX = x;
    const v = robot.data.qvel[0];
    if (v > maxV) maxV = v;
    if (x >= 25) return { ok: true, t: s * 0.02, avg: 25 / (s * 0.02 - 1.2), maxV, yErr: robot.data.qpos[1] - laneY };
    if (robot.data.qpos[2] < 0.45) { fall = true; break; }
  }
  return { ok: false, fall, x: maxX, maxV, avg: (maxX - 0) / Math.max(SIM_TIME - 1.2, 1), yErr: robot.data.qpos[1] - laneY };
}

const from = parseFloat(process.argv[2] ?? '1.0');
const to = parseFloat(process.argv[3] ?? '3.0');
const step = parseFloat(process.argv[4] ?? '0.25');
const ramps = (process.argv[5] ?? '0,3').split(',').map(Number);
const seeds = (process.argv[6] ?? '20260924').split(',').map(Number);
const speeds = [];
for (let v = from; v <= to + 1e-9; v += step) speeds.push(+v.toFixed(2));

console.log(`速度扫描: 指令 ${speeds.join(' / ')} m/s, 斜坡 ${ramps.join('/')} m/s², 种子 ${seeds.join('/')}`);
console.log('mode     | cmd   | 完赛  | 用时    | 均速    | 峰值速度 | 横向偏差 | 结局');
for (const v of speeds) {
  for (const ramp of ramps) {
    const mode = ramp > 0 ? `ramp${String(ramp).padEnd(4).slice(0, 4)}` : (ramp < 0 ? 'stage ' : 'step  ');
    const results = [];
    for (const sd of seeds) {
      const r = await raceRobot(sd, -0.675, v, ramp);
      results.push({ sd, r });
    }
    const okN = results.filter((x) => x.r.ok).length;
    if (okN === results.length) {
      const r = results[0].r;
      console.log(`${mode} | ${v.toFixed(2)} | ${okN}/${results.length}  | ${r.t.toFixed(1).padStart(5)}s | ${r.avg.toFixed(2).padStart(5)}m/s | ${r.maxV.toFixed(2).padStart(6)}m/s | ${r.yErr.toFixed(2).padStart(5)}m | 全部完赛`);
    } else {
      const det = results.map((x) => `s${x.sd % 1000}:${x.r.ok ? '✓' : `✗@${x.r.x?.toFixed(1) ?? '?'}m`}`).join(' ');
      console.log(`${mode} | ${v.toFixed(2)} | ${okN}/${results.length}  |       -  |       - | ${Math.max(...results.map((x) => x.r.maxV)).toFixed(2).padStart(6)}m/s |        - | ${det}`);
    }
  }
}
process.exit(0);
