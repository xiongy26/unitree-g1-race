// 双足短跑大赛 · 主程序
// MuJoCo WASM 物理 + 多物种官方 ONNX 策略 + three.js 渲染, 全部在浏览器本地运行。
//
// 每台机器人 = 自己的官方策略模型(ONNX) + 契约化观测 + 关节空间 PD + 航向外环,
// 全部自由物理(无任何骨盆/轨道辅助)。策略周期按物种而异(G1/T1 50Hz, SA01 100Hz),
// 物理以 100Hz 细分推进, 各物种按自己的 dt×decim 落子。
//
// 启动分两阶段: 阶段 0 只装基础件(WASM + 空体育场 + 渲染循环), 先出物种选择屏;
// 阶段 1 按所选阵容增量加载(资产/模型/策略三件套按物种幂等缓存), 赛后可回
// 选择屏换阵容——已加载物种零等待复用, 新增物种走增量加载。

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { CFG, PolicyRunner, PolicySession, steerCmd } from './policy.js';
import { Sim, makeRng } from './sim.js';
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

// 阶段 1 加载进度: 里程碑制(每物种三个里程碑: 资产到位/模型编译/策略就绪, 收尾 +1
// 机器人构建), 总数 = N×3 + 1, 随所选物种数 N 线性推进。逐 10MB 注入等过程文字
// 走 log() 只刷 bootlog, bar 只在里程碑经 bump() 推进——无需预估字节数。
let phaseDone = 0, phaseTotal = 1;
function bump(msg) {
  log(msg);
  phaseDone++;
  $('bootbar').value = Math.min(99, Math.round((phaseDone / phaseTotal) * 100));
}

// 主循环节拍: 各物种策略周期(20ms/10ms)的最小公因数
const TICK_DT = 0.01;

const app = {
  mujoco: null,
  sim: null,                    // Sim 实例(阶段 0 为空壳, 模型随所选物种增量编入)
  sessionBySpecies: new Map(),  // 物种 id -> ONNX 会话(跨阵容缓存复用)
  assetsCache: new Map(),       // 物种 id -> 资产 { xml, extraFiles, meshes }(fetch 过不重拉)
  robots: [],
  race: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  sun: null,
  camMode: 'free',
  simSpeed: 1,
  paused: false,
  baseSpeed: 1.55,
  selectedIds: ['g1', 'pm01', 'duck'], // 选择屏当前勾选(默认阵容: G1+PM01+MicroDuck)
  seed: 20260923,
  rtf: 0,
  loading: false,               // 阶段 1 加载互斥(防连点重复进入)
  resultsDismissed: false,      // 用户已手动关闭结算面板: 防完赛自动弹窗每帧顶回, 发枪时复位
};

// ---------- 资产/模型/策略按需加载(ensure 三件套, 缓存键均为物种 id) ----------
async function fetchSpeciesAssets(spec) {
  const extraFiles = new Map();
  for (const f of spec.extraFiles ?? []) {
    extraFiles.set(f, await (await fetch('./assets/' + spec.id + '/' + f)).text());
  }
  const xml = await (await fetch(spec.xmlFile)).text();
  const allXml = [xml, ...extraFiles.values()].join('\n');
  const meshFiles = [...new Set([...allXml.matchAll(/file="([^"]+\.(?:STL|stl|obj))"/g)].map((m) => m[1].split('/').pop()))];
  const meshes = new Map();
  // G1 网格在 assets/meshes/(robots.js 的 meshesDir), 其余物种在各自 assets/<id>/meshes/
  const meshDir = spec.meshesDir ?? './assets/' + spec.id + '/meshes/';
  await Promise.all(meshFiles.map(async (name) => {
    const buf = await (await fetch(meshDir + name)).arrayBuffer();
    meshes.set(name, new Uint8Array(buf));
  }));
  return { xml, extraFiles, meshes };
}

async function ensureSpeciesAssets(sp) {
  let entry = app.assetsCache.get(sp.id);
  if (!entry) {
    entry = await fetchSpeciesAssets(sp);
    app.assetsCache.set(sp.id, entry);
  }
  return entry;
}

async function ensureSpeciesModel(sp) {
  if (app.sim.hasModel(sp.id)) return; // 已编入(本阵容或上次阵容)零等待复用
  const entry = await ensureSpeciesAssets(sp);
  await Sim.compileSpecies(app.sim, sp, entry, log);
  if (!app.sim.hasModel(sp.id)) throw new Error(`${sp.name} 模型编译失败`);
}

async function ensureSession(sp) {
  if (app.sessionBySpecies.has(sp.id)) return;
  const res = await fetch(sp.policyFile);
  if (!res.ok) throw new Error(`${sp.name}: 策略缺失(${sp.policyFile})`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const session = new PolicySession(window.ort, sp.contract);
  await session.load(buf);
  app.sessionBySpecies.set(sp.id, session);
  log(`${sp.name}: 策略就绪(${(buf.length / 1024) | 0}KB, obs ${sp.contract.numObs}, ${session.batched ? '支持批量' : '逐台'})`);
}

// ---------- 机器人构建 ----------
// specs: 本场阵容物种列表, 第 i 台 = specs[i](一一对应, 由选择屏勾选决定)。
// 开头先完整清理旧机器人(释放 MjvScene+材质 / delete MjData / 移出场景)——
// 换阵容增量重建走同一路径, 无脏残留; 模型/会话/几何缓存跨阵容复用。
function buildRobots(specs) {
  for (const r of app.robots) {
    app.scene.remove(r.visual.group);
    r.visual.dispose();
    try { r.sim.data.delete(); } catch (e) { /* 忽略 */ }
  }
  app.robots = [];
  for (let i = 0; i < specs.length; i++) {
    const species = specs[i];
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
      laneY: laneY(i), // 机号=道号: 第 i 台站画好的第 i+1 号道, 与地面道号/编号牌一致
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

// ---------- 选择屏 ----------
function renderSelectGrid() {
  const grid = $('sel-grid');
  grid.innerHTML = '';
  for (const sp of SPECIES) {
    const card = document.createElement('label');
    card.className = 'sel-card';
    card.dataset.id = sp.id;
    card.innerHTML = `
      <span class="sel-head"><span class="emoji">${sp.emoji}</span><span class="nm">${sp.name}</span>
        <input type="checkbox"></span>
      <span class="blurb">${sp.blurb}</span>
      <span class="size">${sp.sizeHint}</span>`;
    // label 包裹 checkbox: 点卡片任意处原生切换勾选, 语义可访问
    card.querySelector('input').addEventListener('change', (e) => {
      if (e.target.checked) {
        if (!app.selectedIds.includes(sp.id)) app.selectedIds.push(sp.id);
      } else {
        app.selectedIds = app.selectedIds.filter((x) => x !== sp.id);
      }
      updateSelectState();
    });
    grid.appendChild(card);
  }
}

// 同步卡片勾选态/琥珀高亮与开始按钮守卫(0 只禁用 + 副标题提示)
function updateSelectState() {
  for (const card of document.querySelectorAll('#sel-grid .sel-card')) {
    const on = app.selectedIds.includes(card.dataset.id);
    card.classList.toggle('active', on);
    const box = card.querySelector('input');
    if (box.checked !== on) box.checked = on;
  }
  const n = app.selectedIds.length;
  $('sel-count').textContent = n ? `已选 ${n}/6 · 点「开始比赛」加载所选物种` : '至少选择 1 只才能开始';
  $('sel-start').disabled = n === 0;
}

function showSelect() {
  app.paused = true; // 比赛冻结(保持当前姿态静止), launchRace 完成加载后恢复
  $('btn-pause').textContent = '暂停';
  $('countdown').style.display = 'none'; // 倒计时中换将: 内联样式优先级高于 CSS, 需显式归位
  $('results').classList.remove('show');
  document.body.classList.add('mode-select'); // CSS 隐藏比赛 UI(#hdr 保留品牌位)
  updateSelectState();
  $('select').classList.add('show');
}

function hideSelect() {
  document.body.classList.remove('mode-select');
  $('select').classList.remove('show');
}

// ---------- 阶段 1: 按所选阵容增量加载并进入比赛 ----------
async function launchRace() {
  if (app.loading) return;
  const specs = SPECIES.filter((sp) => app.selectedIds.includes(sp.id));
  if (!specs.length) return; // 0 只: #sel-start 已禁用, 此处双保险
  app.loading = true;
  hideSelect();
  // 复用 #boot 遮罩做二次加载进度: 改标题、清空日志、进度归零
  $('boot').classList.remove('hidden');
  $('boot-tag').textContent = `正在加载所选阵容(${specs.length} 只) ...`;
  bootlog.innerHTML = '';
  $('bootbar').value = 0;
  phaseDone = 0;
  phaseTotal = specs.length * 3 + 1;
  try {
    const valid = [];
    for (const sp of specs) {
      // 单物种失败只剔除该物种(显式 err, 绝不静默回退成别的物种模型), 其余照常
      try {
        await ensureSpeciesAssets(sp);
        bump(`${sp.name}: 资产就绪`);
        await ensureSpeciesModel(sp);
        bump(`${sp.name}: 模型编译完成`);
        await ensureSession(sp);
        bump(`${sp.name}: 策略就绪`);
        valid.push(sp);
      } catch (e) {
        log(`${sp.name} 加载失败, 已从本场阵容剔除: ${e.message || e}`, 'err');
      }
    }
    if (!valid.length) {
      log('有效阵容为空, 返回选择屏重新选择。', 'err');
      setTimeout(() => { $('boot').classList.add('hidden'); showSelect(); }, 1500);
      return;
    }
    buildRobots(valid); // 内含旧机器人完整清理 + resetAll(换阵容走同一路径)
    $('results').classList.remove('show'); // 兜底: 完赛结算未关时换将, 防旧结算面板残留
    bump(`机器人构建完成(${valid.length} 台)`);
    $('bootbar').value = 100;
    $('hdr-sub').textContent = `已选 ${valid.length} 物种官方模型+官方策略 · MUJOCO 3.14 WASM`;
    log('一切就绪! 点击「开始比赛」发枪 🏁');
    setTimeout(() => $('boot').classList.add('hidden'), 400);
    app.paused = false;
  } catch (e) {
    // 整链兜底: 未预期异常(如 WASM 堆耗尽)→ 整页重置, reload 后回选择屏 + 默认阵容
    console.error(e);
    log('加载出现未预期错误: ' + (e.message || e), 'err');
    log('3 秒后整页重置 ...', 'err');
    setTimeout(() => location.reload(), 3000);
  } finally {
    app.loading = false;
  }
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
  // 选择屏阶段 robots 为空也要渲染: 空体育场即选择屏背景(可自由视角)
  if (!app.renderer || !app.race) return;
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
  // 完赛自动弹结算; 换阵容回选择屏(mode-select)或加载中(app.loading)时必须挡住,
  // 否则旧 race.state='finished' 会让已关闭的结算面板每帧复活, 且新比赛开始后无人再移除。
  // resultsDismissed: 用户手动关过就不再自动弹, 直到下一次发枪复位。
  if (app.race.state === 'finished' && !$('results').classList.contains('show')
      && !app.loading && !document.body.classList.contains('mode-select')
      && !app.resultsDismissed) showResults();
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
    // 加载中(mode-select 已摘、boot 遮罩显示)或尚未构建阵容(选择屏阶段)一律忽略发枪
    if (app.loading || !app.robots.length) return;
    app.resultsDismissed = false; // 新一局: 结算面板恢复"完赛自动弹"
    $('results').classList.remove('show');
    app.seed = (Math.random() * 0xffffffff) >>> 0;
    app.race.newSeed(app.seed);
    applySpeeds();
    app.race.start();
  };
  $('btn-start').onclick = startRace;
  $('btn-again').onclick = startRace;
  // 手动关闭: 置 resultsDismissed, 否则完赛态下 renderTick 下一帧会把面板顶回来
  $('res-close').onclick = () => { $('results').classList.remove('show'); app.resultsDismissed = true; };
  // 换阵容入口: 结算面板「换阵容」与控制条「阵容」都回选择屏(应用内复位, 不整页 reload)
  $('btn-relineup').onclick = showSelect;
  $('btn-lineup').onclick = showSelect;

  // 选择屏: 卡片勾选在 renderSelectGrid 内接线, 这里接辅助按钮与开始守卫
  $('sel-start').onclick = () => { if (app.selectedIds.length) launchRace(); };
  $('sel-all').onclick = () => { app.selectedIds = SPECIES.map((s) => s.id); updateSelectState(); };
  $('sel-none').onclick = () => { app.selectedIds = []; updateSelectState(); };

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
    const selecting = document.body.classList.contains('mode-select');
    if (e.code === 'Space') {
      e.preventDefault();
      // 选择屏可见时空格=开始加载所选阵容(0 只时按钮禁用即无效), 否则=重新发枪
      if (selecting) $('sel-start').click();
      else $('btn-start').click();
    }
    if (e.code === 'KeyR') {
      if (selecting) return; // 选择屏模式下忽略录制快捷键, 防误录
      e.preventDefault();
      app.recorder?.toggle();
    }
  });
}

// ---------- 阶段 0: 基础件(WASM + 空体育场), 不碰任何物种资产 ----------
async function bootBase() {
  log('加载 MuJoCo WASM(约 10MB, 首次稍慢)...');
  app.mujoco = await loadMujoco();
  app.sim = Sim.empty(app.mujoco); // 空实例: 只装 mj + VFS, 模型随所选物种增量编入
  $('bootbar').value = 40;

  const ort = window.ort;
  // 必须给绝对 URL: ort 在经典脚本内动态 import(wasmPaths + 文件名), 相对说明符按 ort.min.js
  // 所在目录(/vendor/ort/)解析, 会拼出 /vendor/ort/vendor/ort/... 双重前缀导致 404
  ort.env.wasm.wasmPaths = new URL('./vendor/ort/', document.baseURI).href;
  ort.env.wasm.numThreads = 1;

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

  // Race 空壳: 阵容为空, state=ready; buildRobots 时填 robots(全程序化几何, 不依赖物种)
  app.race = new Race([], app.sim, null);
  app.recorder = new RaceRecorder({ getSceneCanvas: () => app.renderer.domElement });
  window.__recorder = app.recorder;
  wireUI();

  window.addEventListener('resize', () => {
    app.camera.aspect = window.innerWidth / window.innerHeight;
    app.camera.updateProjectionMatrix();
    app.renderer.setSize(window.innerWidth, window.innerHeight);
  });

  renderSelectGrid();
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

  $('bootbar').value = 100;
  log('基础场景就绪, 请选择出战物种 🏁');
  showSelect();
  setTimeout(() => $('boot').classList.add('hidden'), 600);
}

// ---------- 启动 ----------
async function main() {
  try {
    await bootBase();
  } catch (e) {
    console.error(e);
    log('启动失败: ' + (e.message || e), 'err');
    log('请确认通过本地 HTTP 服务访问本页(如 node server.js)。', 'err');
  }
}

main();
