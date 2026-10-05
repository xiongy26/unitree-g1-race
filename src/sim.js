// MuJoCo WASM 加载、MEMFS 资产注入、多物种模型编译与机器人实例管理。
//
// 每个物种一份 MjModel(官方 MJCF 原版 + 最小运行时补丁), 每台机器人独立 MjData。
// 同物种机器人共享模型。不要缓存 data.qpos/ctrl 的包装对象, 每次经 data.<field> 现取。

import { SPECIES } from './robots.js';

export const LANE_WIDTH = 1.35;

// 赛道两侧物理护栏(挡墙): 兜底防止机器人冲出跑道(摔倒/打滑/极端漂移时).
// 内侧面 = 红色跑道边(3 条道)外移 0.15m, 正常贴道跑(偏差 <0.5m)永不接触.
// 视觉上对应 scene.js 的红白路缘, geom 为 group 0(MjvOption 默认隐藏).
export const FENCE_INNER_Y = (6 * LANE_WIDTH) / 2 + 0.15; // 4.20
export const FENCE_CENTER_X = 14.5;   // 覆盖 x ∈ [-2, 31](起点前 2m ~ 终点后 6m)
export const FENCE_HALF_LEN = 16.5;
export const FENCE_HALF_T = 0.05;     // 半厚(全厚 0.1)
export const FENCE_HALF_H = 0.3;      // 半高(全高 0.6)

export function fenceGeomsXml() {
  const cy = FENCE_INNER_Y + FENCE_HALF_T;
  // contype=1 conaffinity=7: X1 等官方 MJCF 用 contype/conaffinity 位分离左右腿接触,
  // 护栏位必须显式接受其碰撞位(1/2/4), 否则护栏对它们失效。
  return `
    <geom name="race_fence_left" type="box" pos="${FENCE_CENTER_X} ${-cy} ${FENCE_HALF_H}" size="${FENCE_HALF_LEN} ${FENCE_HALF_T} ${FENCE_HALF_H}" condim="3" friction="0.5 0.005 0.0001" contype="1" conaffinity="7"/>
    <geom name="race_fence_right" type="box" pos="${FENCE_CENTER_X} ${cy} ${FENCE_HALF_H}" size="${FENCE_HALF_LEN} ${FENCE_HALF_T} ${FENCE_HALF_H}" condim="3" friction="0.5 0.005 0.0001" contype="1" conaffinity="7"/>`;
}

// 各物种官方 MJCF 的第一个 <worldbody> 后注入护栏(MuJoCo 对重复 section 合并, 位置无所谓)
function insertFences(xml) {
  if (xml.includes('race_fence')) return xml;
  return xml.replace('<worldbody>', `<worldbody>${fenceGeomsXml()}`);
}

// G1 的"机器人 MJCF"外层比赛场景包装
export function buildSceneXml() {
  return `<mujoco model="g1_race">
  <include file="g1_29dof.xml"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" rgba="0.98 0.98 0.98 1" condim="3" friction="1 0.005 0.0001"/>${fenceGeomsXml()}
  </worldbody>
</mujoco>`;
}

// 递归内联 MJCF 的 <include file="X"/>(各文件为完整 <mujoco> 文档, 取其根内容)。
// 嵌套 include 按所在文件目录解析(如 xml/serial_pm_v2.xml 里的 "assets.xml");
// file 与 = 之间允许空白(智元 X1 的官方 MJCF 为 `file = "..."` 风格)。
function inlineMjcfIncludes(xml, files, baseDir = '', depth = 0) {
  if (depth > 6) return xml;
  const re = /<include\s+file\s*=\s*"([^"]+)"\s*\/>/g;
  return xml.replace(re, (_, f) => {
    const key = baseDir ? baseDir + '/' + f.replace(/^\.\//, '') : f;
    const raw = files.get(key);
    if (raw === undefined) return '';
    const body = raw.replace(/^[\s\S]*?<mujoco[^>]*>/, '').replace(/<\/mujoco>\s*$/i, '');
    const subDir = key.includes('/') ? key.slice(0, key.lastIndexOf('/')) : '';
    return inlineMjcfIncludes(body, files, subDir, depth + 1);
  });
}

// 编译是同步重活(单物种注入+编译最长约 20s): 分块之间让出主线程一拍,
// 让 bootlog 真正刷出进度, 避免页面冻结在上一条旧日志上造成"卡死"错觉。
const breathe = () => new Promise((r) => setTimeout(r, 0));

export class Sim {
  // assets: { g1: {xml, meshes:Map}, species: [{id, xml, meshes:Map}] }
  static async load(mujocoMod, assets, log) {
    const sim = new Sim();
    sim.mj = mujocoMod;

    const vfs = new mujocoMod.MjVFS();
    sim.vfs = vfs;

    // 逐文件注入: 每个文件之间让出一拍(单个 8.9MB 大文件的拷贝也要 2-3s,
    // 不让出就会连续冻结十几秒), 每累计 ~10MB 刷一条进度日志。
    async function inject(prefix, meshes, label) {
      let done = 0, mb = 0, loggedMb = 0;
      const total = meshes.size;
      for (const [name, bytes] of meshes) {
        vfs.addBuffer(prefix + name, bytes);
        done++; mb += bytes.length;
        if (done === total || mb - loggedMb >= 10 * 1048576) {
          log(`${label} ${done}/${total}(${(mb / 1048576) | 0}MB) ...`);
          await breathe();
          loggedMb = mb;
        } else {
          await breathe();
        }
      }
    }

    log('注入 G1 MJCF 与网格 ...');
    vfs.addBuffer('g1_29dof.xml', new TextEncoder().encode(assets.g1.xml));
    await inject('meshes/', assets.g1.meshes, '注入 G1 网格');
    log('编译 G1 模型(29 DoF + 36 网格), 请稍候 ...');
    await breathe();
    sim.model = mujocoMod.MjModel.from_xml_string(buildSceneXml(), vfs);
    if (!sim.model) throw new Error('G1 MjModel 编译失败');
    sim.nq = sim.model.nq; sim.nu = sim.model.nu; sim.nv = sim.model.nv;
    log(`G1 就绪: nq=${sim.nq} nv=${sim.nv} nu=${sim.nu}`);
    await breathe();

    // 各物种官方 MJCF(自带场景)直接编译
    sim.speciesModels = new Map();
    for (const sp of SPECIES) {
      if (sp.id === 'g1') continue;
      const a = assets.species.get(sp.id);
      if (!a) continue;
      let xml = a.xml;
      if (a.extraFiles && a.extraFiles.size > 0) xml = inlineMjcfIncludes(xml, a.extraFiles);
      for (const p of sp.xmlPatches ?? []) xml = xml.split(p.from).join(p.to);
      xml = insertFences(xml);
      const key = sp.id + '_model.xml';
      vfs.addBuffer(key, new TextEncoder().encode(xml));
      // 网格 VFS 键按物种加前缀: 不同厂商的网格文件可能重名(如 pelvis.STL)
      await inject(sp.id + '_meshes/', a.meshes, `注入 ${sp.name} 网格`);
      log(`编译 ${sp.name} 模型, 请稍候 ...`);
      await breathe();
      try {
        const m = mujocoMod.MjModel.from_xml_string(xml, vfs);
        m.opt.timestep = sp.dt; // 与官方部署一致(如 sim2sim 覆盖 timestep)
        sim.speciesModels.set(sp.id, m);
        log(`${sp.name} 模型就绪: nq=${m.nq} nu=${m.nu} dt=${sp.dt}`);
      } catch (e) {
        log(`${sp.name} 模型编译失败: ${String(e.message).slice(0, 120)}`, 'err');
      }
      await breathe();
    }
    return sim;
  }

  // 物种模型编译失败(如 WASM 内存耗尽)时绝不能静默回退成 G1 模型——
  // 那会让该物种拿着自己的契约驱动 G1 的执行器, 行为完全失控(踩过)。
  hasModel(id) {
    if (id === 'g1') return true;
    return this.speciesModels.has(id);
  }

  modelFor(species) {
    if (!species || species.id === 'g1') return this.model;
    const m = this.speciesModels.get(species.id);
    if (!m) throw new Error(`物种 ${species.id} 的 MjModel 未编译成功(见 Sim.load 日志)`);
    return m;
  }

  addRobot(species = null) {
    const model = this.modelFor(species);
    return { data: new this.mj.MjData(model), model, species: species ?? null };
  }

  resetRobot(robot, laneY, rng, noiseScale = 0.02, startX = 0) {
    const { data } = robot;
    const model = this.modelFor(robot.species);
    const zHome = robot.species ? robot.species.zHome : 0.793;
    const qpos = data.qpos, qvel = data.qvel;
    this.mj.mj_resetData(model, data);
    qpos[0] = startX;
    qpos[1] = laneY;
    qpos[2] = zHome;
    qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;
    for (let i = 7; i < model.nq; i++) {
      qpos[i] = (rng() * 2 - 1) * noiseScale;
    }
    // 关节初始: G1 从 mujoco 零位出生(其策略/模型顺序不同, 旧验证路径);
    // 其他物种按契约摆到策略默认站姿再加微扰。契约给 qposIdx 时按索引摆放
    // (X1 腿关节在 qpos 24..35, 非连续前缀), 上肢保持关节同时摆到保持目标。
    const sp = robot.species;
    const def = sp && sp.contract.defaultDof;
    if (def && sp.contract.qposIdx) {
      const idx = sp.contract.qposIdx;
      for (let i = 0; i < def.length; i++) qpos[idx[i]] = def[i] + (rng() * 2 - 1) * noiseScale;
      for (const h of sp.contract.holdJoints ?? []) qpos[h.qpos] = h.target + (rng() * 2 - 1) * noiseScale;
    } else if (def) {
      for (let i = 0; i < def.length; i++) qpos[7 + i] = def[i] + (rng() * 2 - 1) * noiseScale;
    } else {
      for (let i = 7; i < model.nq; i++) qpos[i] = (rng() * 2 - 1) * noiseScale;
    }
    for (let i = 0; i < model.nv; i++) qvel[i] = 0;
    this.mj.mj_forward(model, data);
  }

  step(robot) {
    // 兼容两种机器人包装: 页面运行时 { sim: { data } }, 无头测试 { data }
    const data = robot.sim ? robot.sim.data : robot.data;
    this.mj.mj_step(this.modelFor(robot.species), data);
  }
}

// 可复现的伪随机数(mulberry32)
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
