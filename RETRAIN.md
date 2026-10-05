# 路线图:重训速度策略(本项目后续提速方向)

> **现状**:当前策略实测硬极限 **1.55 m/s**(README「已知实现要点」,阶跃/斜坡/软起步、
> 力矩与摩擦调参均已排除,6 个随机种子验证)。2026-09 对 HuggingFace(镜像)/GitHub/
> 官方仓库/论文的全量调研确认:**公开世界没有任何可换的更快策略**——所有速度指令型
> 人形策略(无论 G1、T1 还是 Berkeley Humanoid)训练范围都只有 ±1.0~1.5 m/s。
> **因此重训是本项目后续提升速度的唯一路径**,目标:指令范围 2.5~3.0 m/s 的 G1 速度
> 策略,让 25 米冲线进入 10 秒量级。

## 方案 A(推荐):holosoma + FastSAC

[holosoma](https://github.com/amazon-far/holosoma)(Amazon FAR,1.6k★,FastTD3/FastSAC
作者出品):支持 G1 29dof、训练用 IsaacGym、MuJoCo 推理验证、**自动导出 ONNX 到 wandb**,
FastSAC 训 G1 运动单卡仅需数小时。

```bash
git clone https://github.com/amazon-far/holosoma && cd holosoma
bash scripts/setup_isaacgym.sh   # 需要 Linux + RTX GPU

# 唯一必改: 放开速度指令范围(src/holosoma/holosoma/config_values/loco/g1/command.py)
#   "lin_vel_x": [-1.0, 1.0]  ->  [-1.0, 2.5]
# 建议同步检查 reward 的 tracking_lin_vel 权重与 gait/air-time 奖励,让 2.0+ 学出跑步姿态

python src/holosoma/holosoma/train_agent.py exp:g1-29dof-fast-sac simulator:isaacgym --training.seed 1
```

注意:holosoma 的观测契约与本项目不同(相位观测),接入时按其
`config_values/loco/g1/observation.py`(布局:base_ang_vel×0.25, projected_gravity,
cmd_lin, cmd_ang, dof_pos, dof_vel×0.05, actions, sin_phase, cos_phase)与
`action.py` 重写 `src/policy.js` 的 `buildObs`,并同步 PD 增益与默认关节角。

## 方案 B:unitree_rl_lab(同源管线,契约即插即换)

本项目现策略的[训练管线](https://github.com/unitreerobotics/unitree_rl_lab)
(IsaacLab + RSL-RL),优点是训练配置与本项目 `src/policy.js` 的 CFG 逐项对应,
训完无需改观测代码。需要 Linux + RTX GPU,训练约 2~6 小时。

关键改动(在 G1 29dof 速度任务配置中,对照 `src/policy.js` 保持契约一致):

| 配置项 | 现值(推测) | 改为 | 说明 |
|---|---|---|---|
| `commands.ranges.lin_vel_x` | `[-0.8, 1.0~1.5]` | `[-1.0, 2.5~3.0]` | 上限决定能跑多快;一次别跳太猛,可分两轮课程 |
| 步态奖励 | 步行奖励 | 增大 air-time / 提高步频奖励、放宽足端落点 | 1.5 m/s 以上需要明显的"跑"姿,纯步行奖励学不出速度 |
| `sim.dt`/`decimation` | 0.005 / 4 (50Hz) | 保持不变 | 必须与部署频率一致 |
| obs/动作布局 | 96 维×5 帧堆叠、actionScale 0.25 | **不要动** | 动了就无法即插即换 |

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=<g1-29dof-velocity> --headless
```

## 导出 ONNX(两方案通用)

```bash
# RSL-RL checkpoint(.pt) -> JIT/ONNX,可用本项目现策略同源的转换脚本:
git clone https://github.com/RoboCubPilot/g1_deploy_mujoco
python scripts/convert_jit_to_onnx.py <policy.pt> --out policy_fast.onnx
```

校验:方案 B 的输入名/形状应为 `obs [N,480]`(96×5,组优先堆叠)、输出 `action [N,29]`;
方案 A 按其 wandb 自动导出的 ONNX 为准(形状见其配置)。

## 接入本项目

```bash
cp policy_fast.onnx assets/policy.onnx
```

然后对照新训练配置同步 `src/policy.js` 的 `CFG`(方案 B 且按上表没动结构时通常无需改):

- `kps`/`kds`:训练用的 stiffness/damping(逐关节,策略关节顺序)
- `defaultAnglesPolicyOrder`:训练的默认关节角
- 观测缩放/堆叠帧数有变动才需要改 `buildObs`/`pushObs`

验证三步:

```bash
node tools/test-speed-sweep.mjs 1.5 3.0 0.25 0 101,202,303,404,505,606  # 实测新包线
# 把回归速度上限改到新边界后:
node test-straightline.mjs
```

最后把 `index.html` 滑条上限与 `main.js` 的钳制值改到实测稳定边界(留 0.05 余量),
README 的速度说明同步更新。

## 展望:动作跟踪路线(真跑步姿态)

若目标是 3 m/s+ 的人类短跑姿态(蹬摆/腾空),速度指令范式之外还有动作跟踪路线
([BeyondMimic/ProtoMotions](https://github.com/HybridRobotics/beyond_mimic)、ResMimic):
机器人回放跑步动捕片。公开研究已验证 G1 可达,但控制范式是"跟踪参考动作"而非
"跟踪速度指令",比赛逻辑(发令/配速差异/摔倒判定)需整体重写,量级为新项目,
暂作为远期方向。

## 为其他物种接入真 RL 权重(每物种独立策略槽)

页面运行时对所有物种走同一条管线:`assets/<id>/policy.onnx` + `robots.js` 里该物种的
`contract`(观测/动作/PD 契约)。替换权重时对齐训练配置与契约即可:

1. **契约**:在 `robots.js` 该物种 `contract` 里对齐 `angVelScale/dofVelScale/
   actionScale/policyToXml/kps/kds/stack/stackMode/numObs` 等字段(现有 G1/PM01/
   T1/天工/X1 五种布局都是现成参考; 天工展示如何用 `actuatorMode='position'`
   适配官方位置舵机, X1 展示 `ctrlIdx/qposIdx/holdJoints/lpfAlpha` 的组合控制器)。
2. **导出 ONNX**:输入 `[N, inputLen]`(float32), 输出 `[N, numActions]`;
   TorchScript 参照 `tools/convert_and_calib.py` 的 `torch.jit.load → torch.onnx.export`。
3. **覆盖** `assets/<id>/policy.onnx` → 刷新页面即生效, 运行时代码零改动。

## 接入新物种(模型+策略齐备的机器人)

本仓库的物种接入是纯数据工作, 运行时代码只改 `src/policy.js` 的观测布局分支:

0. **Python 侧先验证**: 仿照 `sim2sim_check.py` 用官方 MJCF+权重按部署脚本逐条
   复现观测/PD/相位, Python 里能走 10m+ 再动 JS(本次天工/X1 都靠它排错)。
1. **下载官方资产**: MJCF(或 URDF+网格)放到 assets/<id>/(目录名=物种 id),
   策略权重(ONNX 直接用; TorchScript .pt 用 CPU torch 转换: torch.jit.load →
   torch.onnx.export(动态 batch), 导出后务必与 torch 前向数值比对(应 <1e-5))。
   一键下载脚本示例见 `download_assets.sh`。
2. **对齐观测契约**: 找官方部署脚本/配置里的 obs 布局(各分量顺序/缩放/堆叠方式/
   相位时钟/指令缩放)、PD 增益、力矩限幅、默认站姿、动作缩放、仿真步长与策略频率。
3. **注册物种**: 在 src/robots.js 的 SPECIES 加一条(id/名称/资产路径/contract/dt/
   decim/zHome/fallZ/maxV/visGroups), MJCF 需要的运行时补丁写进 xmlPatches
   (meshdir 要用物种专属前缀 `<id>_meshes/`, 避免跨物种网格重名)。
4. **验证**: test-gaits.mjs 的单测+混合赛自动覆盖新物种; test-lanekeep.mjs 验证
   贴道与护栏; 浏览器里确认渲染与摔倒判定。
