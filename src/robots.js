// 机器人物种注册表 —— 只收录"官方模型文件 + 官方策略模型"齐备的真实机器人。

import { makeSteer } from './policy.js';
//
// 每个物种:
//   xmlFile/meshesDir  磁盘上的 MJCF 模型资产(官方仓库原版, 仅运行时打最小补丁)
//   policyFile         官方发布的策略权重(ONNX; TorchScript 权重已事先转换)
//   contract           该策略的观测/动作/PD 契约(逐条对齐官方部署脚本/配置)
//   dt×decim           仿真步长与策略频率(与官方部署一致)
//   steer              航向保持外环参数(不填用 policy.js STEER 默认; 见 makeSteer)
// 模型与策略来源(调研定案 2026-09):
//   G1   unitree 官方 MJCF(unitree_ros) + 策略 RoboCubPilot/g1_deploy_mujoco
//   PM01 众擎 engineai_legged_gym(官方 MJCF+STL+ONNX+sim2sim 脚本)
//   T1   Booster booster_gym(官方 MJCF+T1.pt 权重+T1.yaml 部署配置)
//   TK   天工 Tienkung2-Lite: Open-X-Humanoid/TienKung-Lab(官方 MJCF+STL+walk.pt+
//        sim2sim.py, BSD-3-Clause; walk.pt 已转 ONNX, 数值误差 <1.1e-6)
//   X1   智元灵犀 X1: AgibotTech/agibot_x1_infer(官方 serial MJCF+STL+
//        rl_walk_leg_shoulder.onnx+rl_x1_sim.yaml 部署契约, 同仓库 agibot_x1_train 为训练代码)
//   DUCK MicroDuck: pollen-robotics/microduck_rl(官方 scene_allcollisions+
//        robot_allcollisions MJCF+43 STL, Apache-2.0) + HuggingFace
//        pollen-robotics/microduck-policies 官方 velstand.onnx(官方默认行走策略,
//        行走+零指令站立一体, 61 维观测/14 动作/50Hz, manifest.json 契约)
// 落选记录: 优必选 Walker S2(无公开模型+策略对)、傅里叶 GR-1/N1(有模型有权重但
//   obs 契约锁在闭源 SDK/多层配置里无法对齐, 硬接必摔)。
//   (MicroDuck 曾因 HuggingFace 直连不可达落选, 后经 hf-mirror 镜像取回权重转正。)

// ---------- G1 契约(unitree_rl_gym 布局, 见 policy.js) ----------
export const G1_CONTRACT = {
  layout: 'g1',
  numActions: 29,
  numObs: 96,
  inputLen: 480,
  stack: 5,
  stackMode: 'group',        // 组优先堆叠
  angVelScale: 0.2,
  dofVelScale: 0.05,
  actionScale: 0.25,
  kps: [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
  kds: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
  tauLimit: 1e9, // 无额外限幅(力矩上限由 MJCF 的 actuatorfrcrange 把关)
};

// ---------- 众擎 PM01 契约(engineai_rl_lab amp_velocity_flat_pm01 逐条对齐) ----------
// 官方资产: GMR 仓库 MJCF(pm_v2) + rl_lab AMP 速度策略导出 ONNX + params/env.yaml。
// obs(1128) = [joint_pos(23), joint_vel(23), 上次动作(23), 角速度(3), 重力投影(3)]
//              各 15 帧历史(原始值, 无缩放) + 当前速度指令(3)。
// 动作 23 关节(J00..J22), 目标角 = 每关节缩放·a + 0(默认站姿为直立零位)。
// PD@500Hz, 策略@100Hz(decim 5 × dt 0.002); 指令范围 vx 0.5~0.8, yaw ±0.6。
export const PM01_CONTRACT = {
  layout: 'pm01',
  numActions: 23,
  numObs: 1128,
  inputLen: 1128,
  stack: 15,
  stackMode: 'pm01',
  actionScales: [0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
  kps: [110, 70, 70, 110, 30, 30, 110, 70, 70, 110, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
  kds: [5, 3, 3, 5, 0.3, 0.3, 5, 3, 3, 5, 0.3, 0.3, 3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
  tauLimit: [164, 164, 61, 164, 54.9, 54.9, 164, 164, 61, 164, 54.9, 54.9, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61],
  defaultDof: new Array(23).fill(0),
};

// ---------- Booster T1 契约(booster_gym deploy/configs/T1.yaml + utils/policy.py 逐条对齐) ----------
// obs(47) = [重力投影(3), 机体角速度(3), vx, vy, wy, cos(2πφ), sin(2πφ),
//            (q-def)(12), dq·0.1(12), 上次动作(12)], 单帧无堆叠。
// 指令平滑(每策略周期 ±period 限幅) + 步态门控(|cmd|≈0 时步态项清零)。
// PD@500Hz, 策略@50Hz(decim 10 × dt 0.002), 动作限幅 ±1, 目标角 = a·1.0 + 默认角。
export const T1_CONTRACT = {
  layout: 't1',
  numActions: 12,
  numObs: 47,
  inputLen: 47,
  stack: 1,
  stackMode: 'none',
  gaitFrequency: 1.0,
  cmdScales: { vx: 1.0, vy: 1.0, wy: 1.0 },
  gravityScale: 1.0,
  angVelScale: 1.0,
  dofVelScale: 0.1,
  actionScale: 1.0,
  clipObs: 100.0,
  clipAction: 1.0,
  kps: [200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50],
  kds: [5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3],
  tauLimit: [60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15],
  defaultDof: [-0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0],
};

// ---------- 天工 Tienkung2-Lite 契约(TienKung-Lab legged_lab sim2sim.py + walk_cfg.py 逐条对齐) ----------
// 官方资产: TienKung-Lab 官方 MJCF(tienkung2_lite/mjcf/tienkung.xml)+21 STL+
// Exported_policy/walk.pt(TorchScript, 已转 ONNX)。
// obs(75) = [机体角速度(3), 重力投影(3), 指令(3), 关节位置-默认(20), 关节速度(20),
//            上次动作(20), sin(2πφ)左右, cos(2πφ)左右, 摆空相比例(2)] 各值原始无缩放,
//            帧优先堆叠 10 帧 = 750。
// 步态时钟: cycle 0.85s, 左右脚相位偏移 0.38/0.88, 摆空相 0.38/0.38(官方 walk 预设)。
// 动作 20 关节(12 腿+8 臂), 目标角 = a·0.25 + 默认角, 官方 MJCF 的 position 舵机
// 自闭环(kp 与训练刚度一致), 力矩限幅=训练 effort(运行时补 forcerange)。
// 指令范围 vx -0.6~1.0 / vy ±0.5 / wy ±1.57(官方 clip ±1.0)。
export const TIENKUNG_CONTRACT = {
  layout: 'tk',
  numActions: 20,
  numSingleObs: 75,
  numObs: 75,
  inputLen: 750,
  stack: 10,
  stackMode: 'frame',
  gaitCycle: 0.85,
  phaseOffsets: [0.38, 0.88],
  airRatios: [0.38, 0.38],
  clipObs: 100.0,
  clipAction: 100.0,
  actionScale: 0.25,
  // 官方 MJCF 执行器(mujoco)顺序: 髋滚/俯/偏航+膝+踝俯/仰 ×L,R + 肩俯/滚/偏航+肘 ×L,R
  // policyToXml = sim2sim 的 isaac_to_mujoco(动作侧); 观测侧用其逆(运行时自动求)
  policyToXml: [0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 17, 19, 2, 6, 10, 14, 3, 7, 11, 15],
  defaultDof: [0, -0.5, 0, 1.0, -0.5, 0, 0, -0.5, 0, 1.0, -0.5, 0, 0, 0.1, 0, -0.3, 0, -0.1, 0, -0.3],
  actuatorMode: 'position',
  kps: [700, 700, 500, 700, 30, 16.8, 700, 700, 500, 700, 30, 16.8, 60, 20, 10, 10, 60, 20, 10, 10],
  kds: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  tauLimit: [180, 300, 180, 300, 60, 30, 180, 300, 180, 300, 60, 30, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 52.5],
};

// 天工 position 舵机补 forcerange(=训练 effort_limit_sim); 关节名 → 力矩限幅(N·m)
const TIENKUNG_EFFORT = [
  ['hip_roll', 180], ['hip_pitch', 300], ['hip_yaw', 180], ['knee_pitch', 300],
  ['ankle_pitch', 60], ['ankle_roll', 30],
  ['shoulder_pitch', 52.5], ['shoulder_roll', 52.5], ['shoulder_yaw', 52.5], ['elbow_pitch', 52.5],
];
const tienkungPatches = [];
for (const side of ['l', 'r']) {
  for (const [joint, eff] of TIENKUNG_EFFORT) {
    const name = `${joint}_${side}_joint`;
    tienkungPatches.push({
      from: `joint="${name}" kp=`,
      to: `joint="${name}" forcerange="-${eff} ${eff}" kp=`,
    });
  }
}

// ---------- 智元灵犀 X1 契约(agibot_x1_infer rl_x1_sim.yaml + rl_controller.cc 逐条对齐) ----------
// 官方资产: 推理仓库自带 serial MJCF(xyber_x1_flat + xyber_x1_serial)与
// control_module/policy 下的两个官方策略:
//   rl_walk_leg.onnx          12 腿(上肢保持), 步态周期 0.7s, sim2sim 包线 ~0.85 指令
//   rl_walk_leg_shoulder.onnx 12 腿+2 肩俯仰摆臂(walk_leg_arm 模式), 步态周期 1.0s
// 本仓库选用 shoulder 摆臂版(2026-09 提速调研: 同条件多种子对照, leg 版 0.85 指令
// 实跑 ~0.47, shoulder 版 1.4 指令实跑 ~0.72 且 8/8 种子全净——官方训练课程上限
// 1.5, 硬件标称 >2m/s, shoulder 版把包线拉满; 详见 README 调研记录)。
// obs(53) = [sin(2πφ), cos(2πφ), vx·2, vy·2, wy, 关节位置-默认(14), 关节速度×0.05(14),
//            上次动作(14), 机体角速度(3), 欧拉角 rpy(3)], 帧优先堆叠 66 帧 = 3498,
//            首帧整段填充当前观测且动作段清零(与 rl_controller 首帧行为一致)。
// 策略控 12 腿 + 双肩俯仰(顺序 = 配置 joint_list: 髋俯/滚/偏航+膝+踝俯/仰 + 肩俯仰 ×L,R,
// 注意肩俯仰插在两腿之间), 其余 15 执行器按官方 pd_zero+pd_stand 组合增益保持;
// 目标角一阶 LPF(wc=100, 官方 1kHz -> 本仓库 500Hz, alpha=wc·dt=0.2);
// |指令|≤0.05 时步态相位清零(sw_mode)。
// PD@500Hz(官方 1kHz), 策略@100Hz(decim 5 × dt 0.002)。
// 腿部 PD 增益与 leg 版不同(官方 shoulder 控制器: 髋 60/60/40 膝 80 踝 40/30)。
export const X1_CONTRACT = {
  layout: 'x1',
  numActions: 14,
  numSingleObs: 53,
  numObs: 53,
  inputLen: 3498,
  stack: 66,
  stackMode: 'frame',
  historyFillFirst: true,
  cycleTime: 1.0,
  cmdThreshold: 0.05,
  cmdScales: { vx: 2.0, vy: 2.0, wy: 1.0 },
  dofVelScale: 0.05,
  angVelScale: 1.0,
  clipObs: 100.0,
  clipAction: 100.0,
  actionScale: 0.5,
  lpfAlpha: 0.2,
  // 策略顺序: L腿6(ctrl 17..22) + L肩俯仰(ctrl 3) + R腿6(ctrl 23..28) + R肩俯仰(ctrl 10)
  ctrlIdx: [17, 18, 19, 20, 21, 22, 3, 23, 24, 25, 26, 27, 28, 10],
  qposIdx: [24, 25, 26, 27, 28, 29, 10, 30, 31, 32, 33, 34, 35, 17],
  kps: [60, 60, 40, 80, 40, 30, 20, 60, 60, 40, 80, 40, 30, 20],
  kds: [6, 3, 3, 4, 2, 2, 3, 6, 3, 3, 4, 2, 2, 3],
  tauLimit: [150, 50, 50, 150, 18, 18, 150, 150, 50, 50, 150, 18, 18, 150],
  defaultDof: [0.4, 0.05, -0.31, 0.49, -0.21, 0.0, 0.15, -0.4, -0.05, 0.31, 0.49, -0.21, 0.0, 0.15],
  // 上肢保持(pd_zero 覆盖全身 + pd_stand 覆盖肩滚/肘俯的非零目标, 官方组合;
  // 肩俯仰已入策略, 从保持表剔除):
  // 腰 3 × (kp700,kd0.6) 目标 0; 每侧臂 肩滚(300,0.6) 目标 -0.1、
  // 肘俯(300,0.6) 目标 0.3、肩偏航/肘偏航/腕俯/腕仰(30,0.1) 目标 0
  holdJoints: [
    ...[0, 1, 2].map((i) => ({ ctrl: i, qpos: 7 + i, qvel: 6 + i, target: 0, kp: 700, kd: 0.6, lim: 150 })),
    ...[0, 1].flatMap((side) => {
      const b = side * 7 + 3; // l_arm 执行器 3..9, r_arm 10..16
      return [
        { ctrl: b + 1, qpos: 11 + side * 7, qvel: 10 + side * 7, target: -0.1, kp: 300, kd: 0.6, lim: 150 },
        { ctrl: b + 2, qpos: 12 + side * 7, qvel: 11 + side * 7, target: 0, kp: 30, kd: 0.1, lim: 150 },
        { ctrl: b + 3, qpos: 13 + side * 7, qvel: 12 + side * 7, target: 0.3, kp: 300, kd: 0.6, lim: 150 },
        { ctrl: b + 4, qpos: 14 + side * 7, qvel: 13 + side * 7, target: 0, kp: 30, kd: 0.1, lim: 150 },
        { ctrl: b + 5, qpos: 15 + side * 7, qvel: 14 + side * 7, target: 0, kp: 30, kd: 0.1, lim: 150 },
        { ctrl: b + 6, qpos: 16 + side * 7, qvel: 15 + side * 7, target: 0, kp: 30, kd: 0.1, lim: 150 },
      ];
    }),
  ],
};

// X1 手腕 4 关节加 armature: 官方 MJCF 无 armature, 1kHz 隐式阻尼下稳定; 本仓库
// 500Hz 显式 PD 对 5.7e-7 量级腕部惯量会发散, 加 0.002 kg·m²(电机转子经减速器的
// 等效惯量, 量级合理)保证数值稳定, 不改变步行物理。
const x1Patches = [
  { from: 'meshdir="../meshes"', to: 'meshdir="x1_meshes"' },
];
for (const j of ['left_wrist_pitch', 'left_wrist_roll', 'right_wrist_pitch', 'right_wrist_roll']) {
  x1Patches.push({
    from: `name="${j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1"`,
    to: `name="${j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1" armature="0.002"`,
  });
}


// ---------- Pollen MicroDuck 契约(microduck_rl scripts/infer_policy.py --no-bam + HF manifest.json 逐条对齐) ----------
// 官方资产: microduck_rl 官方 scene_allcollisions.xml + robot_allcollisions.xml
// (VelStand 任务的训练模型, 70 碰撞 geom 全量导出) + 43 STL, Apache-2.0;
// 策略为 HuggingFace pollen-robotics/microduck-policies 官方 velstand.onnx
// (manifest: "the default walk policy since v5", 行走+零指令站立一体, 免站立切换)。
// obs(61) 单帧 = [机体角速度(3), 重力投影(3), 关节位置-默认(14), 关节速度(14),
//            上次动作(14), 指令(13): twist(3) + head_pose(4) + body_pose(6)],
//            全部原始值无缩放无裁剪; 关节顺序 = 执行器顺序(恒等映射, qpos 7..20)。
// 动作 14 -> 目标角 = a·1.0 + 默认角(STAND2 站姿, 与 ONNX 元数据 default_joint_pos 一致),
// ctrl 直接写目标角(官方 position 舵机 kp=0.55/forcerange ±0.96 = BAM XL330 执行器在
// vin 7.4V/kp_fw 200 下的官方 to_mujoco 等效参数, 由 bam 包数值验证)。
// PD@200Hz(位置舵机随物理步), 策略@50Hz(decim 4 × dt 0.005, 官方 manifest control_hz=50)。
// 训练指令范围 vx ±0.4 / vy ±0.3 / wy ±1.0(velocity 配置; 开环实测指令跟踪 ~40-50%,
// 站立高 ~0.12m, 零指令自站稳, 固有左偏航漂移 ~0.1 rad/s 由外环修正)。
export const DUCK_CONTRACT = {
  layout: 'duck',
  numActions: 14,
  numObs: 61,
  inputLen: 61,
  stack: 1,
  stackMode: 'none',
  actionScale: 1.0,
  defaultDof: [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530],
  actuatorMode: 'position',
};

// ---------- 物种注册表 ----------
export const SPECIES = [
  {
    id: 'g1',
    name: 'Unitree G1',
    short: 'G1',
    emoji: '🤖',
    xmlFile: './assets/g1_29dof.xml',
    // G1 网格在 assets/meshes/(其余物种在各自 assets/<id>/meshes/), 资产 fetch 按此解析
    meshesDir: './assets/meshes/',
    policyFile: './assets/policy.onnx',
    contract: G1_CONTRACT,
    dt: 0.002,
    decim: 10,
    zHome: 0.793,
    fallZ: 0.45,
    labelH: 1.25,
    visGroups: [1],
    maxV: 1.55,
    noise: 0.025,
    // 选择屏文案(纯展示, 不参与物理/策略)
    blurb: 'Unitree 全尺寸人形, 29 自由度全能型选手',
    sizeHint: '身高约 1.3m · 29 关节',
    // G1 资产为"机器人 MJCF", 需包一层比赛场景
    wrapScene: true,
  },
  {
    id: 'pm01',
    name: '众擎 PM01',
    short: 'PM01',
    emoji: '🦾',
    xmlFile: './assets/pm01/pm_v2.xml',
    extraFiles: ['xml/serial_pm_v2.xml', 'xml/serial_links.xml', 'xml/serial_actuators.xml', 'xml/serial_sensors.xml', 'xml/assets.xml', 'ground.xml'],
    policyFile: './assets/pm01/policy.onnx',
    contract: PM01_CONTRACT,
    dt: 0.002,
    decim: 5,
    zHome: 0.75,
    fallZ: 0.35,
    labelH: 1.35,
    visGroups: [2],
    // 训练指令范围 vx 0.5~0.8(官方 env.yaml ranges, 观测 scale=null 原始值),
    // sim2sim 下策略只跟踪六成(0.8 指令实跑 ~0.52)。实测更高指令:
    // 1.0/1.1 稳定完赛(实跑 0.68~0.77), 1.2 起步必摔, ≥1.11 随机中段摔。
    // maxV=1.0: 比赛 ±6% 速度抖动后最坏 1.06, 留足安全边际(实跑 ~0.68)。
    maxV: 1.0,
    noise: 0.01,
    yawCap: 0.6,
    blurb: '众擎紧凑型人形, 100Hz 高频细腻步态',
    sizeHint: '身高约 1.4m · 23 关节',
    wrapScene: false,
    // 官方地面 friction 0.6, 训练时物理材质为 1.0 -> 对齐
    xmlPatches: [{ from: '../meshes/', to: 'pm01_meshes/' }, { from: 'friction="0.6"', to: 'friction="1"' }],
  },
  {
    id: 't1',
    name: 'Booster T1',
    short: 'T1',
    emoji: '🦿',
    xmlFile: './assets/t1/T1_locomotion.xml',
    policyFile: './assets/t1/policy.onnx',
    contract: T1_CONTRACT,
    dt: 0.002,
    decim: 10,
    zHome: 0.70,
    fallZ: 0.35,
    labelH: 1.30,
    visGroups: [1],
    maxV: 1.2,
    noise: 0.01,
    blurb: 'Booster 高瘦型人形, 大步幅稳定冲刺',
    sizeHint: '身高约 1.45m · 12 关节',
    // T1 策略的 yaw 跟踪迟缓且超调大, 默认外环增益会画 ~2s 周期的 S 形(±0.4m),
    // 实测降低航向增益后 6 组车道/种子最大偏差 0.25~0.35m(默认增益 0.33~0.58m)
    steer: makeSteer({ kpYaw: 1.5, kdYaw: 0.2 }),
    wrapScene: false,
    // 官方 meshdir 共用 meshes/ 键会跨物种重名 -> 改专属前缀; 地面 condim=1(无摩擦) -> 带摩擦
    xmlPatches: [
      { from: 'meshdir="meshes/"', to: 'meshdir="t1_meshes/"' },
      { from: 'condim="1"', to: 'condim="3" friction="1 0.005 0.0001"' },
    ],
  },
  {
    id: 'tk',
    name: '天工 Tienkung2-Lite',
    short: '天工',
    emoji: '🧑‍🚀',
    xmlFile: './assets/tk/tienkung.xml',
    policyFile: './assets/tk/policy.onnx',
    contract: TIENKUNG_CONTRACT,
    dt: 0.005,
    decim: 4,
    zHome: 0.99,
    fallZ: 0.5,
    labelH: 1.75,
    visGroups: [1],
    // 官方指令 clip ±1.0, 训练范围 vx -0.6~1.0; sim2sim 实测 0.8 指令实跑 ~0.81
    // (跟踪 ~1:1, 五物种中最快)。maxV=1.0 留 ±6% 抖动边际由回归测试把关。
    maxV: 1.0,
    noise: 0.01,
    blurb: '天工全尺寸人形, 五物种中的速度担当',
    sizeHint: '身高约 1.7m · 20 关节',
    wrapScene: false,
    // 官方 compiler meshdir="../meshes/" -> VFS 的 meshes/;
    // position 舵机补 forcerange(训练 effort_limit_sim)
    xmlPatches: [{ from: 'meshdir="../meshes/"', to: 'meshdir="tk_meshes/"' }, ...tienkungPatches],
  },
  {
    id: 'x1',
    name: '智元灵犀 X1',
    short: 'X1',
    emoji: '🧍',
    xmlFile: './assets/x1/xyber_x1_flat.xml',
    extraFiles: ['robot/xyber_x1/xyber_x1_serial.xml', 'environment/flat.xml'],
    // 官方摆臂行走策略(带双肩俯仰, walk_leg_arm 模式); leg 版保留在 assets/x1/policy.onnx
    policyFile: './assets/x1/policy_shoulder.onnx',
    contract: X1_CONTRACT,
    dt: 0.002,
    decim: 5,
    zHome: 0.62,
    fallZ: 0.32,
    labelH: 1.45,
    visGroups: [0],
    // 指令 1.3~1.5 实测包线(实跑饱和 ~0.71-0.73 m/s, 跟踪 ~50%): Python 侧 8 种子
    // ±6% 抖动 40s 全净; JS 回归见 test-gaits/test-lanekeep。maxV=1.4 取中值。
    maxV: 1.4,
    // 生成噪声 0.005(其余物种 0.01): X1 小脚+高重心对初始扰动更敏感
    noise: 0.005,
    // 摆臂策略对 wy 指令极敏感(Python 实测: 恒定 wy=0.1 即失稳降速, wy≤0.06 安全),
    // 外环必须"轻手": kpYaw=3.0(旧 leg 策略时代)的持续微纠正会磨掉 ~20% 前进速度
    // (test-x1-tune: 硬增益 40.1s -> 软增益+cap0.06 35.9s, 偏差仅 0.19m, 护栏兜底)
    steer: makeSteer({ kpYaw: 0.15, kdYaw: 0.02 }),
    yawCap: 0.06,
    blurb: '智元灵犀轻量人形, 摆臂小跑姿态',
    sizeHint: '身高约 1.35m · 14 关节',
    wrapScene: false,
    // 官方 compiler meshdir + 手腕 4 关节 armature 数值稳定补丁(见 X1_CONTRACT 注)
    xmlPatches: x1Patches,
  },
  {
    id: 'duck',
    name: 'Pollen MicroDuck',
    short: 'MicroDuck',
    emoji: '🦆',
    xmlFile: './assets/duck/scene_allcollisions.xml',
    extraFiles: ['robot_allcollisions.xml'],
    policyFile: './assets/duck/policy.onnx',
    contract: DUCK_CONTRACT,
    dt: 0.005,
    decim: 4,
    zHome: 0.125,
    fallZ: 0.08,
    labelH: 0.42,
    visGroups: [2],
    // 训练指令范围 vx ±0.4, 策略只跟踪 ~40-50%(0.8 指令实跑 ~0.41, 0.95 指令
    // 8 种子全不摔)。maxV=0.9: 比赛 ±6% 抖动后最坏 0.954, 实跑 ~0.40,
    // 25m 约 62s(90s 超时内)。
    maxV: 0.9,
    noise: 0.005,
    // 默认外环增益即可贴道(3 车道最大偏差 0.20~0.25m); 策略固有左偏漂移由外环修正
    blurb: 'Pollen 迷你鸭式机器人, 短腿扑腾向前',
    sizeHint: '身高约 0.12m · 14 关节',
    wrapScene: false,
    // 官方 compiler meshdir="assets"(网格在 assets/ 子目录) -> VFS 专属前缀
    xmlPatches: [{ from: 'meshdir="assets"', to: 'meshdir="duck_meshes"' }],
  },
];

export function speciesById(id) {
  return SPECIES.find((s) => s.id === id) || SPECIES[0];
}
