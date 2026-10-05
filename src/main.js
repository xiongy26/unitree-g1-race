// 双足短跑大赛 · 主程序
// MuJoCo WASM 物理 + 多物种官方 ONNX 策略 + three.js 渲染, 全部在浏览器本地运行。
//
// 每台机器人 = 自己的官方策略模型(ONNX) + 契约化观测 + 关节空间 PD + 航向外环,
// 全部自由物理(无任何骨盆/轨道辅助)。策略周期按物种而异(G1/T1 50Hz, SA01 100Hz),
// 物理以 100Hz 细分推进, 各物种按自己的 dt×decim 落子。

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { CFG, PolicyRunner, PolicySession, steerCmd } from './policy.js';
import { Sim, buildSceneXml, makeRng } from './sim.js';
import { buildTrack, addLights, buildEnvironment, updateScenery, RobotVisual, TEAM_COLORS, laneY } from './scene.js';
import { Race } from './race.js';
import { RaceRecorder } from './recorder.js';
import { SPECIES } from './robots.js';

const $ = (id) => document.getElementById(id);
const bootlog = $('bootlog');
function log(msg, cls = '') {
  const div = document.createElement('div');
  div.textContent = msg;
  if (cls) div.className = cls;
  bootlog.appendChild(div);
  bootlog.scrollTop = bootlog.scrollHeight;
  $('bootbar').value = Math.min(99, $('bootbar').value + 2);
}

// 主循环节拍: 各物种策略周期(20ms/10ms)的最小公因数
const TICK_DT = 0.01;

const G1_SPEC = SPECIES[0];

const app = {
  mujoco: null,
  sim: null,
  sessionBySpecies: new Map(),
  robots: [],
  race: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  sun: null,
  camMode: 'over',
  simSpeed: 1,
  paused: false,
  baseSpeed: 1.55,
  robotCount: 6,
  lineup: 'mixed',
  seed: 20260923,
  rtf: 0,
};

// ---------- 资产加载 ----------
async function fetchSpeciesAssets(spec) {
  const extraFiles = new Map();
  for (const f of spec.extraFiles ?? []) {
    extraFiles.set(f, await (await fetch('./assets/' + spec.id + '/' + f)).text());
  }
  const xml = await (await fetch(spec.xmlFile)).text();
  const allXml = [xml, ...extraFiles.values()].join('\n');
  const meshFiles = [...new Set([...allXml.matchAll(/file="([^"]+\.(?:STL|stl|obj))"/g)].map((m) => m[1].split('/').pop()))];
  const meshes = new Map();
  await Promise.all(meshFiles.map(async (name) => {
    const buf = await (await fetch('./assets/' + spec.id + '/meshes/' + name)).arrayBuffer();
    meshes.set(name, new Uint8Array(buf));
  }));
  return { xml, extraFiles, meshes };
}

async function loadAllAssets() {
  log('获取 G1 MJCF 与网格 ...');
  const g1Xml = await (await fetch(G1_SPEC.xmlFile)).text();
  const meshFiles = [...new Set([...g1Xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))];
  const g1Meshes = new Map();
  await Promise.all(meshFiles.map(async (name) => {
    const buf = await (await fetch('./assets/meshes/' + name)).arrayBuffer();
    g1Meshes.set(name, new Uint8Array(buf));
  }));
  const assets = { g1: { xml: g1Xml, meshes: g1Meshes }, species: new Map() };
  for (const sp of SPECIES) {
    if (sp.id === 'g1') continue;
    log(`获取 ${sp.name} 官方模型资产 ...`);
    assets.species.set(sp.id, await fetchSpeciesAssets(sp));
  }
  return assets;
}

async function loadSpeciesPolicies() {
  for (const sp of SPECIES) {
    const res = await fetch(sp.policyFile);
    if (!res.ok) { log(`${sp.name}: 策略缺失(${sp.policyFile}), 该物种不可用`, 'err'); continue; }
    const buf = new Uint8Array(await res.arrayBuffer());
    const session = new PolicySession(ort, sp.contract);
    await session.load(buf);
    app.sessionBySpecies.set(sp.id, session);
    log(`${sp.name}: 策略就绪(${(buf.length / 1024) | 0}KB, obs ${sp.contract.numObs}, ${session.batched ? '支持批量' : '逐台'})`);
  }
}

// ---------- 阵容 ----------
function pickLineup(i, rng) {
  const pool = SPECIES.filter((s) => app.sessionBySpecies.has(s.id) && app.sim.hasModel(s.id));
  if (app.lineup === 'g1') return G1_SPEC;
  if (app.lineup === 'random') return pool[Math.floor(rng() * pool.length) % pool.length];
  return pool[i % pool.length];
}

// ---------- 机器人构建 ----------
function buildRobots(n) {
  for (const r of app.robots) {
    app.scene.remove(r.visual.group);
    r.visual.dispose();
    try { r.sim.data.delete(); } catch (e) { /* 忽略 */ }
  }
  app.robots = [];
  const rng = makeRng(app.seed ^ 0x51ed270b);
  for (let i = 0; i < n; i++) {
    const species = pickLineup(i, rng);
    const simRobot = app.sim.addRobot(species);
    const model = app.sim.modelFor(species);
    const color = TEAM_COLORS[i % TEAM_COLORS.length];
    const visual = new RobotVisual(app.mujoco, model, color, i + 1, species.labelH, species.id, species.visGroups);
    app.scene.add(visual.group);

    const runner = new PolicyRunner(species.contract);
    runner.period = species.dt * species.decim;
    app.robots.push({
      sim: simRobot,
      species,
      runner,
      visual,
      cmd: new Float32Array(3),
      laneY: laneY(i),
      name: `${i + 1}号·${species.short}`,
      color,
      targetSpeed: 0,
      curVx: 0,
      acc: runner.period, // 策略周期累积器(首拍立即推理)
      fallZ: species.fallZ,
      noise: species.noise ?? 0.02,
      finished: false, finishTime: 0, penalty: 0, falls: 0,
      fallen: false, fallenAt: 0, x: 0, speed: 0, place: 0,
    });
  }
  app.race.robots = app.robots;
  applySpeeds();
  app.race.resetAll(app.seed);
}

function applySpeeds() {
  const rng = makeRng(app.seed ^ 0x9e3779b9);
  const scale = app.baseSpeed / 1.55;
  for (const r of app.robots) {
    r.targetSpeed = Math.min(r.species.maxV, Math.max(0.1, r.species.maxV * scale * (1 + (rng() * 2 - 1) * 0.06)));
    // 不直接跳变 curVx: 比赛中拖动滑块时由 stepOnce 的斜率限幅平滑过渡, 避免指令阶跃摔机
  }
}

// ---------- 仿真步进(每 TICK_DT 一次) ----------
async function stepOnce() {
  const robots = app.robots;
  // 仅比赛/完赛阶段给前进指令, 就绪/倒计时一律 cmd=0(防抢跑)
  const running = app.race.state === 'racing' || app.race.state === 'finished';

  for (const r of robots) {
    if (r.fallen) continue;
    const d = (running ? r.targetSpeed : 0) - r.curVx;
    r.curVx += Math.max(-2.5 * TICK_DT, Math.min(2.5 * TICK_DT, d));
  }

  // 到期的策略: 构建观测 + 推理 + 写动作
  const due = [];
  for (const r of robots) {
    if (r.fallen) continue;
    r.acc += TICK_DT;
    if (r.acc >= r.runner.period - 1e-9) {
      r.acc -= r.runner.period;
      const q = r.sim.data.qpos;
      r.cmd[0] = r.curVx;
      r.cmd[1] = 0;
      r.cmd[2] = running ? Math.max(-(r.species.yawCap ?? 1.0), Math.min(r.species.yawCap ?? 1.0, steerCmd(q, r.sim.data.qvel, r.laneY, r.species.steer))) : 0;
      r.runner.buildAndPushObs(q, r.sim.data.qvel, r.cmd, r.runner.period);
      due.push(r);
    }
  }
  // 按物种分组批量推理
  const groups = new Map();
  for (const r of due) {
    if (!groups.has(r.species.id)) groups.set(r.species.id, []);
    groups.get(r.species.id).push(r);
  }
  for (const [sid, rs] of groups) {
    const session = app.sessionBySpecies.get(sid);
    if (!session) continue;
    const outs = await session.inferAll(rs.map((r) => r.runner));
    rs.forEach((r, i) => r.runner.applyAction(outs[i]));
  }

  // 物理子步: 每台按自己的 dt 推进 TICK_DT
  for (const r of robots) {
    if (r.fallen) continue;
    const steps = Math.max(1, Math.round(TICK_DT / r.species.dt));
    for (let k = 0; k < steps; k++) {
      r.runner.pd(r.sim.data.qpos, r.sim.data.qvel, r.sim.data.ctrl);
      app.sim.step(r);
    }
  }
  app.race.tick(TICK_DT);
}

// ---------- 相机 ----------
const camPos = new THREE.Vector3(-7, -13, 10);
const camAim = new THREE.Vector3(0, 0, 1);

function updateCamera(dt) {
  const leader = app.robots.reduce((best, r) => (!best || r.x > best.x ? r : best), null);
  const lx = leader ? leader.x : 0;
  const ly = leader ? leader.laneY : 0;
  if (app.camMode === 'free') {
    app.controls.enabled = true;
    app.controls.update();
    return;
  }
  app.controls.enabled = false;
  let want, aim;
  if (app.camMode === 'leader') {
    want = new THREE.Vector3(lx - 3.8, ly * 0.35, 1.7);
    aim = new THREE.Vector3(lx + 2.6, ly * 0.5, 0.85);
  } else {
    want = new THREE.Vector3(lx - 6, -13, 10);
    aim = new THREE.Vector3(lx + 3, 0, 0.4);
  }
  const k = 1 - Math.exp(-dt * 3.5);
  camPos.lerp(want, k);
  camAim.lerp(aim, k);
  app.camera.position.copy(camPos);
  app.camera.lookAt(camAim);
}

// ---------- HUD ----------
function fmtTime(t) {
  if (!t && t !== 0) return '-';
  const m = Math.floor(t / 60), s = t - m * 60;
  return m > 0 ? `${m}:${s.toFixed(2).padStart(5, '0')}` : `${s.toFixed(2)}s`;
}

let hudTick = 0;
function updateHUD() {
  const st = app.race.state;
  const pill = $('status');
  if (st === 'ready') { pill.textContent = '就绪'; pill.className = 'pill warn'; }
  else if (st === 'countdown') { pill.textContent = '即将开始'; pill.className = 'pill warn'; }
  else if (st === 'racing') { pill.textContent = '比赛进行中'; pill.className = 'pill ok'; }
  else { pill.textContent = '已完赛'; pill.className = 'pill bad'; }
  $('clock').textContent = app.race.state === 'countdown' ? '00.0s' : `${app.race.raceClock.toFixed(1)}s`;
  $('rt').textContent = `RTF ${app.rtf.toFixed(2)} · 策略×${app.sessionBySpecies.size} · 推理 ${inferMsAvg().toFixed(2)}ms`;

  const cd = $('countdown');
  if (st === 'countdown' && app.race.countdown > 1.0) {
    cd.style.display = 'block';
    cd.textContent = String(Math.ceil(app.race.countdown - 1));
  } else if (st === 'countdown') {
    cd.style.display = 'block';
    cd.textContent = 'GO!';
  } else {
    cd.style.display = 'none';
  }

  if (++hudTick % 3 !== 0) return;
  const rows = app.race.standings();
  const lb = $('lb');
  lb.innerHTML = '';
  for (const r of rows) {
    const div = document.createElement('div');
    div.className = 'row' + (r.finished ? ' done' : '');
    const prog = Math.max(0, Math.min(100, (r.x / 25) * 100));
    const stateTxt = r.finished
      ? `🏁 第${r.place}名 ${fmtTime(r.finishTime)}`
      : r.fallen ? '😵 摔倒-扶起中' : `${r.speed.toFixed(2)} m/s`;
    div.innerHTML = `
      <span class="dot" style="background:${r.color}"></span>
      <span class="nm">${r.species.emoji}${r.name}</span>
      <span class="bar"><i style="width:${prog}%"></i></span>
      <span class="st">${stateTxt}</span>`;
    lb.appendChild(div);
  }
}

function inferMsAvg() {
  let s = 0, n = 0;
  for (const sess of app.sessionBySpecies.values()) { s += sess.inferMs; n++; }
  return n ? s / n : 0;
}

function showResults() {
  const rows = app.race.standings();
  const tb = $('res-rows');
  tb.innerHTML = '';
  const medal = ['🥇', '🥈', '🥉'];
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    const place = r.finished ? (medal[r.place - 1] || `${r.place}`) : 'DNF';
    const time = r.finished ? fmtTime(r.finishTime) : `跑了 ${r.x.toFixed(1)}m`;
    const avg = r.finished ? (25 / r.finishTime).toFixed(2) : '-';
    tr.innerHTML = `<td>${place}</td><td><span class="dot" style="background:${r.color}"></span>${r.species.emoji} ${r.name}</td>
      <td>${time}</td><td>${avg}</td><td>${r.falls}</td>`;
    tb.appendChild(tr);
  });
  $('results').classList.add('show');
}

// ---------- 主循环 ----------
let acc = 0;
let busy = false;
let simAdvanced = 0;
let realAccum = 0;
let simLastT = performance.now();

function simTick() {
  if (busy || app.paused || app.sessionBySpecies.size === 0) { simLastT = performance.now(); return; }
  const now = performance.now();
  const dt = Math.min((now - simLastT) / 1000, 0.1);
  simLastT = now;
  realAccum += dt;
  if (realAccum > 0.5) { app.rtf = simAdvanced / realAccum; simAdvanced = 0; realAccum = 0; }
  busy = true;
  (async () => {
    try {
      acc += dt * app.simSpeed;
      let n = 0;
      while (acc >= TICK_DT && n < 16) {
        await stepOnce();
        acc -= TICK_DT; simAdvanced += TICK_DT; n++;
      }
      if (acc > 0.5) acc = 0;
    } catch (e) {
      console.error('step error', e);
      log('仿真出错: ' + e.message, 'err');
      app.paused = true;
    } finally {
      busy = false;
    }
  })();
}

let lastRenderT = 0;
function renderTick() {
  if (!app.renderer || !app.race || app.sessionBySpecies.size === 0) return;
  const now = performance.now();
  const dt = Math.min((now - lastRenderT) / 1000 || 0.016, 0.1);
  lastRenderT = now;
  for (const r of app.robots) r.visual.update(r.sim.data);
  updateCamera(dt);
  updateScenery(dt, now / 1000); // 场馆环境更新
  updateHUD();
  const lx = app.robots.reduce((b, r) => (!b || r.x > b.x ? r : b), null)?.x ?? 0;
  app.sun.position.set(lx - 10, -8, 20);
  app.sun.target.position.set(lx, 0, 0);
  app.renderer.render(app.scene, app.camera);
  app.recorder?.onFrame(); // 必须紧跟 render: 同一任务内 WebGL 绘图缓冲仍有效
  if (app.race.state === 'finished' && !$('results').classList.contains('show')) showResults();
}

function frame() {
  requestAnimationFrame(frame);
  lastRenderT = performance.now();
  renderTick();
}
let renderFallbackStarted = false;
function startLoops() {
  if (renderFallbackStarted) return;
  renderFallbackStarted = true;
  setInterval(simTick, 4);
  setInterval(() => {
    if (performance.now() - lastRenderT > 120) {
      lastRenderT = performance.now();
      renderTick();
    }
  }, 100);
  requestAnimationFrame(frame);
}

// ---------- UI 事件 ----------
function wireUI() {
  const startRace = () => {
    $('results').classList.remove('show');
    app.seed = (Math.random() * 0xffffffff) >>> 0;
    app.race.newSeed(app.seed);
    if (app.lineup === 'random') buildRobots(app.robotCount);
    applySpeeds();
    app.race.start();
  };
  $('btn-start').onclick = startRace;
  $('btn-again').onclick = startRace;
  $('res-close').onclick = () => $('results').classList.remove('show');

  for (const b of document.querySelectorAll('#count-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#count-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.robotCount = parseInt(b.dataset.n, 10);
      buildRobots(app.robotCount);
    };
  }
  for (const b of document.querySelectorAll('#lineup-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#lineup-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.lineup = b.dataset.v;
      buildRobots(app.robotCount);
    };
  }
  $('spd').oninput = (e) => {
    app.baseSpeed = parseFloat(e.target.value);
    $('spd-val').textContent = app.baseSpeed.toFixed(2) + ' m/s';
    applySpeeds();
  };
  for (const b of document.querySelectorAll('#speed-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#speed-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.simSpeed = parseFloat(b.dataset.v);
    };
  }
  for (const b of document.querySelectorAll('#cam-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#cam-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.camMode = b.dataset.m;
      if (app.camMode === 'free') {
        const leader = app.robots.reduce((a, r) => (!a || r.x > a.x ? r : a), null);
        app.controls.target.set(leader ? leader.x : 0, 0, 0.8);
        app.controls.update();
      }
    };
  }
  $('btn-pause').onclick = () => {
    app.paused = !app.paused;
    $('btn-pause').textContent = app.paused ? '继续' : '暂停';
  };
  $('btn-rec').onclick = () => app.recorder.toggle();
  $('rec-panels').onchange = (e) => {
    if (!app.recorder) return;
    app.recorder.includePanels = e.target.checked;
    app.recorder.dirty = true; // 录制中重新勾上时立即重光栅面板层, 不等下次 DOM 变更
  };
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); $('btn-start').click(); }
    if (e.code === 'KeyR') { e.preventDefault(); app.recorder?.toggle(); }
  });
}

// ---------- 启动 ----------
async function main() {
  try {
    log('加载 MuJoCo WASM(约 10MB, 首次稍慢)...');
    app.mujoco = await loadMujoco();

    log('获取全部物种资产(约 160MB)并逐模型编译, 本地也需约 1 分钟, 进度见下方逐条日志 ...');
    const assets = await loadAllAssets();
    app.sim = await Sim.load(app.mujoco, assets, log);

    log('加载各物种官方 ONNX 策略 ...');
    const ort = window.ort;
    ort.env.wasm.wasmPaths = './vendor/ort/';
    ort.env.wasm.numThreads = 1;
    await loadSpeciesPolicies();

    log('初始化 three.js 场景 ...');
    const canvasWrap = $('view');
    app.renderer = new THREE.WebGLRenderer({ antialias: true });
    app.renderer.setSize(window.innerWidth, window.innerHeight);
    app.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    app.renderer.shadowMap.enabled = true;
    app.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    app.renderer.outputColorSpace = THREE.SRGBColorSpace;
    app.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    app.renderer.toneMappingExposure = 1.05;
    canvasWrap.appendChild(app.renderer.domElement);

    app.scene = new THREE.Scene();
    app.scene.background = new THREE.Color(0xdce9ed);
    app.scene.fog = new THREE.Fog(0xdce9ed, 85, 210);
    app.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 400);
    app.camera.up.set(0, 0, 1);
    app.camera.position.copy(camPos);
    app.controls = new OrbitControls(app.camera, app.renderer.domElement);
    app.controls.enableDamping = true;
    app.controls.target.set(0, 0, 0.8);
    app.sun = addLights(app.scene);
    buildTrack(app.scene);
    buildEnvironment(app.scene);

    app.race = new Race([], app.sim, null);
    app.recorder = new RaceRecorder({ getSceneCanvas: () => app.renderer.domElement });
    window.__recorder = app.recorder;
    buildRobots(app.robotCount);
    wireUI();

    window.addEventListener('resize', () => {
      app.camera.aspect = window.innerWidth / window.innerHeight;
      app.camera.updateProjectionMatrix();
      app.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    $('bootbar').value = 100;
    log('一切就绪! 点击「开始比赛」发枪 🏁');
    setTimeout(() => $('boot').classList.add('hidden'), 600);
    startLoops();
    window.__app = app;
    window.__kick = async (steps) => {
      for (let i = 0; i < steps; i++) {
        await stepOnce();
        simAdvanced += TICK_DT;
      }
      renderTick();
      return window.__debug();
    };
    window.__debug = () => ({
      state: app.race.state, clock: app.race.raceClock, rtf: app.rtf,
      robots: app.robots.map((r) => {
        const q = r.sim.data.qpos;
        return {
          name: r.name, species: r.species.id,
          x: +(q[0] ?? NaN).toFixed?.(2) ?? NaN,
          y: +(q[1] ?? NaN).toFixed?.(2) ?? NaN,
          z: +(q[2] ?? NaN).toFixed?.(2) ?? NaN,
          v: +(r.sim.data.qvel[0] ?? NaN).toFixed?.(2) ?? NaN,
          falls: r.falls, done: r.finished,
        };
      }),
    });
  } catch (e) {
    console.error(e);
    log('启动失败: ' + (e.message || e), 'err');
    log('请确认通过本地 HTTP 服务访问本页(如 node server.js)。', 'err');
  }
}

main();
