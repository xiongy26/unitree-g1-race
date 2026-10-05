# 🏃 双足短跑大赛(Bipedal Sprint Race)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![MuJoCo 3.14 WASM](https://img.shields.io/badge/physics-MuJoCo_3.14_WASM-orange)
![ONNX Runtime Web 1.19.2](https://img.shields.io/badge/inference-ONNX_Runtime_Web_1.19.2-yellowgreen)
![three.js r160](https://img.shields.io/badge/render-three.js_r160-lightgrey)

在浏览器里让 **6 种真实双足机器人**同场进行 25 米短跑比赛:

| 物种 | 速度包线 | 模型来源 | 策略来源 |
|---|---|---|---|
| 🤖 Unitree G1(29 DoF) | 1.55 m/s | unitree_ros 官方 MJCF+网格(BSD-3-Clause) | [g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco) ONNX(**上游未声明许可,仅限学习研究**) |
| 🦾 众擎 PM01(23 DoF) | 1.00 m/s | [engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab) 官方 MJCF+网格(BSD-3-Clause) | 同仓库官方 AMP 速度策略 ONNX(`model_19999.onnx`) |
| 🦿 Booster T1(12 DoF) | 1.20 m/s | [booster_gym](https://github.com/BoosterRobotics/booster_gym) 官方 MJCF+网格(Apache-2.0) | 同仓库官方 `T1.pt`(已转 ONNX) |
| 🧑‍🚀 天工 Tienkung2-Lite(20 DoF) | 1.00 m/s | [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) 官方 MJCF+网格(自定义许可:BSD-3-Clause 系归属) | 同仓库官方 `Exported_policy/walk.pt`(TorchScript, 已转 ONNX) |
| 🧍 智元灵犀 X1(12+17 DoF) | 1.40 m/s | [agibot_x1_infer](https://github.com/AgibotTech/agibot_x1_infer) 官方 serial MJCF+网格(Mulan PSL v2) | 同仓库官方 `rl_walk_leg_shoulder.onnx`(摆臂版) + `rl_x1_sim.yaml` 部署契约 |
| 🦆 Pollen MicroDuck(14 DoF) | 0.90 m/s | [microduck_rl](https://github.com/pollen-robotics/microduck_rl) 官方 MJCF+网格(Apache-2.0) | HuggingFace [microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies) 官方 `velstand.onnx`(Apache-2.0) |

**收录标准:官方机器人模型文件 + 官方策略模型,两者齐备才收录;缺一不加。**
每一台都由自己的神经网络策略(ONNX @ onnxruntime-web)在 50/100Hz 闭环控制,
全部自由物理(无任何骨盆/轨道辅助)。物理由 **MuJoCo 3.14 WebAssembly** 求解,
three.js 渲染,纯前端本地运行。

## 在线体验(GitHub Pages)

本项目已部署到 GitHub Pages: [https://xiongy26.github.io/unitree-g1-race/](https://xiongy26.github.io/unitree-g1-race/), 纯浏览器运行、无需安装 Node。
首次打开需下载全部模型资产(约 185MB, 有 HTTP 缓存, 二次加载显著变快)。
需要支持 WebAssembly SIMD 的现代浏览器(Chrome / Edge / Firefox / Safari 近两年的版本)。

## 运行方式

```bash
cd g1-race
node server.js 8137        # 或任意静态服务器: python -m http.server 8137
# 浏览器打开 http://127.0.0.1:8137/
```

**运行要求**:

- Node.js ≥ 18(仅用于起本地静态服务器);
- 浏览器需为支持 WebAssembly SIMD 的现代版本(Chrome / Edge / Firefox / Safari 近两年的版本);
- 仓库含约 200MB 模型资产与 vendor 产物, 克隆较大属预期。

> 必须通过 HTTP 访问(ES Module 与 WASM 跨域限制)。启动分两段开销: ~160MB 资产的
> 本地传输(二次刷新走 ETag 协商缓存, 304 零传输) + 六套模型的网格注入与编译
> (约 45s 的同步计算, 与传输无关; 进度会逐条刷出, 单段最长冻结 ~5s, 并非卡死)。

## 玩法

1. 点击 **「▶ 开始比赛」**(或按空格)发枪, 3-2-1 倒计时后六台机器人起跑;
2. 右上角实时排名, 25 米终点线冲线后弹出结算面板;
3. 每次摔倒罚时 2 秒并在原地扶起; 所有选手完赛(或 90 秒超时)后比赛结束;
4. 想留下影像: 点 **「● 录制 MP4」**(或按 R), 停止后自动下载 —— 3D 画面与
   所有面板(排名/时钟/倒计时/结算)都会录进视频。

### 控制项

| 控件 | 说明 |
|---|---|
| 机器人数 | 2~6 台同场竞技(默认 6) |
| 阵容 | 六强混战(默认) / 全 G1 / 随机 |
| 目标速度 | 0.30~1.55 m/s, **按各物种包线等比缩放** |
| 仿真倍速 | 0.5× / 1× / 2× / 3× / 4× |
| 视角 | 跟随领跑(低机位) / 全景 / 自由(拖拽旋转、滚轮缩放) |
| 录制 | 开始/停止一键录制, 停止自动下载; Chrome/Edge/Safari 直出 **H.264 MP4**, 其余浏览器回退 WebM |

## 技术实现

```
浏览器
├─ MuJoCo 3.14 WASM(@mujoco/mujoco 官方绑定)
│    · 每物种一份 MjModel(官方 MJCF 原版 + 最小运行时补丁), 每台机器人独立 MjData
│    · 关节空间 PD/位置舵机控制(各物种自己的 kp/kd/力矩限幅/步长: G1/T1/X1 500Hz,
│      SA01 1kHz, 天工/MicroDuck 200Hz)
├─ ONNX Runtime Web(WASM 后端, 单线程)
│    · 每物种独立策略 @ 各自频率(G1/T1/天工/MicroDuck 50Hz, PM01/X1 100Hz), 观测契约逐条对齐
│      官方部署代码:
│      G1  : unitree_rl_gym 布局, 96×5 组优先堆叠 = 480
│      SA01: [相位, 指令×2, 关节, 关节速度, 上次动作, 角速度, 欧拉角] 47×15 帧优先 = 705
│      T1  : [重力投影, 角速度, 指令, 步态时钟, 关节, 关节速度, 上次动作] 47 单帧
│      天工: [角速度, 重力投影, 指令, 关节, 关节速度, 上次动作, sin/cos(2πφ)×2,
│              摆空相比例] 75×10 帧优先 = 750(全原始值无缩放)
│      X1  : [sin/cos(2πφ), 指令×2 缩放, wy, 关节, 关节速度, 上次动作, 角速度, 欧拉角]
│              53×66 帧优先 = 3498(首帧整段填充, |指令|≤0.05 步态相位清零;
│              动作 14 = 12 腿+双肩俯仰摆臂)
│      MicroDuck: [角速度, 重力投影, 关节, 关节速度, 上次动作, 指令 13 维
│              (twist 3 + head 4 + body 6)] 61 单帧(全原始值无缩放)
│    · 动作 -> 目标角 = a·actionScale + 默认角(各物种自己的缩放/默认站姿)
├─ three.js: mjv_updateScene 管线取每个 geom 世界位姿(官方 STL 网格), 室外蓝色赛道/体育场/阴影
├─ 录制(src/recorder.js): 离屏合成画布每帧叠两层 —— WebGL 画布 + SVG foreignObject
│   光栅化的面板层(内嵌页面样式表, MutationObserver 标脏 + 50ms 节流, 面板静止零开销;
│   画布按物理像素创建[×devicePixelRatio 上限 2, 宽上限 2560], 面板按 CSS 像素布局
│   整体 scale 放大光栅化, 码率按像素数给足[10~24Mbps, 1080p60≈16Mbps]);
│   captureStream(60) -> MediaRecorder 编码, Chrome/Edge/Safari 直出 H.264 MP4,
│   其余回退 WebM; 控制条「面板」勾选框决定是否合成面板层(录制中可实时切换, 默认含),
│   录制控件带 data-norec 不会出现在视频里
├─ 航向保持外环(横向 PD 级联): 车道偏差 P + 横向速度阻尼 -> 期望航向 -> 航向误差 P
│   + 偏航阻尼 -> 各策略的 yaw 角速度指令(按物种调增益; X1 摆臂策略对 wy 极敏感,
│   需"轻手" kpYaw=0.15+cap 0.06, 见已知要点);
│   赛道两侧另有物理挡墙兜底(红白路缘)
└─ 比赛逻辑: 倒计时发枪(倒计时期间指令清零防抢跑)、实时排名、摔倒罚时扶起、结算面板
```

### 已知实现要点(踩坑记录)

- **G1 策略极限 1.55 m/s**;速度指令更高时起步必摔(详见 `tools/test-speed-sweep.mjs`)。
- **PM01 速度包线 1.0 m/s**:官方 env.yaml 训练指令范围 vx 0.5~0.8(观测原始值
  无缩放), sim2sim 下策略只跟踪六成(0.8 指令实跑 ~0.52)。给更高指令可再提速:
  1.0/1.1 稳定完赛(实跑 0.68~0.77), 1.2 起步必摔, ≥1.11 随机中段摔。
  `maxV` 取 1.0, 比赛 ±6% 抖动后最坏 1.06, 留足安全边际(实际均速 ~0.68, 比原
  0.8 包线快 ~30%)。
- **G1 的策略顺序 ≠ mujoco 顺序**:默认角必须经 `policyToXml` 重排成 mujoco 顺序
  再喂给 PD/obs(否则观测错乱、起步即摔——本次多物种化时踩过)。
- **SA01 obs 里的欧拉角**:官方 sim2sim 用 rpy(非四元数/重力投影), 且相位时钟按
  仿真时间 / cycle_time(0.8s) 计算, 帧优先堆叠 15 帧(与 G1 的组优先不同)。
- **T1 的步态门控**:指令平滑(每周期 ±period 限幅), |cmd|≈0 时 cos/sin 与指令项
  全部清零; 关节速度归一 0.1(非 0.05)。
- **天工(TK)的位置舵机**:官方 MJCF 用 `<position kp=...>` 执行器(kp=训练刚度,
  阻尼由关节被动 damping 提供), sim2sim 直接 `ctrl=目标角`。本仓库显式 PD
  (motor+kd)在 dt=0.005 下数值发散, 因此契约支持 `actuatorMode='position'`
  原样保留位置舵机、仅补 `forcerange`(=训练 effort 限幅)。验证: Python 侧
  0.8 m/s 指令实测 0.81 m/s(跟踪 ~1:1, 全场最快); obs 侧映射是 sim2sim 的
  `mujoco_to_isaac`(与动作侧 `isaac_to_mujoco` 互逆, 接反立即发散)。
- **智元 X1 的"组合控制器"**:官方 rl_walk_leg 策略只控 12 个腿关节, 上肢 17 个
  执行器由 pd_zero(全身)+pd_stand(肩/肘非零目标)按官方增益表保持——契约用
  `ctrlIdx/qposIdx` 定位腿执行器、`holdJoints` 保持上肢。66 帧历史首帧须整段填充
  当前观测(动作段清零, 与 rl_controller.cc 首帧行为一致), 否则起步踉跄。
- **X1 的腕部 armature 补丁**:官方 MJCF 无 armature(1kHz 隐式积分稳定), 本仓库
  500Hz 显式 PD 对 5.7e-7 量级腕部惯量数值发散, 给 4 个腕关节加
  `armature="0.002"`(电机转子经减速器的等效惯量, 量级合理, 不改变步行物理)。
- **MicroDuck 的"BAM 等效"位置舵机**:官方策略按 Rhoban BAM(Better Actuator Models)
  XL330 电压舵机模型训练(vin 7.4V / firmware kp 200, 电压钳位+反电动势+负载相关摩擦),
  本仓库沿用官方 `infer_policy.py --no-bam` 部署回退——直接用官方 MJCF 的 position
  舵机(kp=0.55 / forcerange ±0.96 / 关节阻尼 0.053 / armature 0.0018), 该参数恰为
  BAM `to_mujoco` 在标称电压下的官方等效(用 bam 包 m6 参数 kt=0.366, R=2.81 数值验证:
  kt·vin·e·kp/R=0.5547, vin·kt/R=0.963)。Python 侧状态注入对比确认 obs 与官方逐位一致
  (误差 ~1e-9), 开环轨迹与官方脚本一致。
- **MicroDuck 行为特征**:velstand 策略零指令自站稳(倒计时免站立网络)、指令跟踪
  ~40-50%(0.8 指令实跑 ~0.41 m/s), 训练范围 vx ±0.4 / wy ±1.0; 固有左偏航漂移
  (~-0.1 rad/s)由外环修正(默认增益即可, 3 车道最大偏差 0.20~0.25m); 起步慢热
  (~2s 内加速), maxV=0.9 时比赛 ±6% 抖动最坏 0.954 指令 8 种子全不摔。
- **MicroDuck 网格瘦身(WASM 2GB 上限)**:官方 43 STL 共 471k 三角面(24MB), 六物种
  同场编译时 MuJoCo WASM 堆顶到 2GB 上限, 鸭子模型 `Could not allocate memory`
  编译失败被静默回退成 G1 模型(无头测试里表现为"鸭子摔得诡异")。把 12 个 1MB 上限
  大网格抽稀到 4000 三角面(凸包几乎不变, 比赛相机距离无视觉差), 总量降到 272k 面
  /15MB 后六模型编译稳定。同时 `Sim.modelFor` 对编译失败的物种改为显式抛错、
  阵容池按"策略+模型双就绪"过滤(此前会静默回退成 G1 模型, 行为完全失控)。
- **跨物种网格重名**: 不同厂商 MJCF 网格可能重名(如 G1 与天工都有 `pelvis.STL`),
  VFS 键按物种加前缀(`<id>_meshes/...`), MJCF 的 meshdir 由 xmlPatches 重写。
- **护栏 contype/conaffinity**: X1 用碰撞位分离左右腿(脚 contype 2/4), 护栏必须
  显式 `contype="1" conaffinity="7"` 才能挡住它; 对默认 1/1 的物种无影响。
- **T1 官方 MJCF 地面 condim=1(无摩擦)**, 直接用会打滑, 运行时打补丁换成摩擦地面。
- **`mjvGeom.dataid` 损坏**: 本 WASM 构建对 mesh geom 返回 2 倍 dataid, 需用
  `model.geom_dataid[objid]` 还原。
- **TorchScript → ONNX**: `T1.pt`/天工 `walk.pt` 用 `torch.jit.load +
  torch.onnx.export`(动态 batch)转换, 转换后与 torch 前向误差 <1e-6;
  转换脚本思路见 RETRAIN.md / tools/convert_and_calib.py。
- **rAF 不可靠**: 仿真由 4ms 定时器驱动、渲染由 rAF + 100ms 定时器兜底。
- **跑出跑道问题的两层修复(2026-09-25)**:
  1. 航向外环从"纯跟踪 P 控制"升级为**横向 PD 级联**(车道偏差 P + 横向速度阻尼 D →
     期望航向 → 航向误差 P + 偏航阻尼 D)。纯 P 纯跟踪在高速下欠阻尼会画 S 形;
     T1/X1 策略的 yaw 跟踪迟缓/漂移, 需按物种调低/调高航向增益(`species.steer`,
     见 `makeSteer`; X1 默认增益 10 种子中 1 摔, kpYaw=3.0 后 20 种子全净)。
     修复后各物种 25m 最大横向偏差: G1 ≤0.14m / PM01 ≤0.15m / T1 ≤0.42m /
     天工 ≤0.09m / X1 ≤0.16m / MicroDuck ≤0.25m(默认增益)。
  2. **物理护栏兜底**: 每个物种的 MJCF 在编译前注入赛道两侧挡墙(`sim.js`
     `fenceGeomsXml`, 内侧面 y=±4.20), 视觉上对应红白路缘。正常贴道跑永不接触;
     摔倒/打滑/极端漂移时被挡在跑道内。无头测试用"关闭外环的 G1"(固有弧线偏置)
     验证护栏兜底有效。
- **X1 换装官方摆臂策略提速(2026-09-26)**:原 `rl_walk_leg.onnx`(12 腿)换成同仓库
  `rl_walk_leg_shoulder.onnx`(12 腿+双肩俯仰, `walk_leg_arm` 模式)。要点:
  1. **契约逐项都变**:obs 47→53(多出的 6 维=两肩俯仰的位置/速度/上次动作), 动作 12→14
     (肩俯仰插在两腿之间: ctrlIdx/qposIdx 为 `L腿6, L肩, R腿6, R肩`), 步态周期 0.7→1.0s,
     腿部 PD 增益也不同(髋 60/60/40、膝 80、踝 40/30, 官方 shoulder 控制器配置)。
  2. **速度包线 0.85→1.4**:sim2sim 实跑从 ~0.47 提到 ~0.72 m/s(跟踪饱和在 ~50%,
     指令 1.3~1.5 无差别; 8 种子 ±6% 抖动 40s 全净)。比赛 43.99s→35.81s(+23%)。
  3. **摆臂策略对 wy 指令极敏感**:Python 实测恒定 wy=0.1 即失稳降速(wy≤0.06 安全)。
     航向外环必须"轻手":kpYaw 3.0(旧策略时代)→0.15 且 yawCap=0.06。外环硬增益下
     偶发打满 ±1.0 的 wy 会磨掉 ~20% 前进速度(`tools/tune-x1-steer.mjs`: 硬增益 40.1s →
     软增益 35.9s, 偏差仅 0.19m);贴道偏差 0.18~0.20m, 护栏兜底仍在。
- **起步与调速全平滑**: 发枪后 `curVx` 由斜率限幅(2.5 m/s²)加速到目标速度、
  倒计时期间指令清零(防抢跑)、比赛中拖动速度滑块不再产生指令阶跃——阶跃易造成
  踉跄, 踉跄正是斜向冲出车道的常见诱因。修复后混合比赛 0~1 次摔倒(此前 1-3 次)。
- **无头回归与页面同闭环**: 混合赛测试与页面一样先站立(倒计时)再斜坡起步;
  测试里所有 `steerCmd` 调用必须传 `species.steer`(漏传 = 用默认增益, X1 会摔——
  本次五物种化踩过)。

## 回归测试

```bash
node test-gaits.mjs                # 每物种 0.85×包线 25m 单测 + 六物种混合比赛(全自由物理)
node test-straightline.mjs         # G1 六道六速直线跑回归(含 1.55 极限速度)
node test-lanekeep.mjs             # 赛道保持回归: 六物种贴道跑 + 物理护栏兜底 + 混合比赛
node tools/test-boot-profile.mjs   # 启动剖析(tools/): 逐物种网格注入/模型编译耗时(诊断"打开慢")
node tools/test-speed-sweep.mjs    # G1 速度包线扫描(tools/)
```

## 文件结构

```
g1-race/
├── index.html            # UI / importmap / 启动遮罩 / 阵容选择
├── server.js             # 极简静态服务器(node server.js [port])
├── test-gaits.mjs        # 无头回归: 六物种 25m 单测 + 混合比赛
├── test-straightline.mjs # G1 直线跑回归
├── test-lanekeep.mjs     # 赛道保持回归: 贴道跑 + 护栏兜底 + 混合比赛
├── sim2sim_check.py      # 新物种契约的 Python 级 sim2sim 验证(接入前先跑通它)
├── download_assets.sh    # 天工/X1/duck 官方资产一键下载(复现 assets/; G1/PM01/T1 获取方式见脚本头注释)
├── RETRAIN.md            # 提速重训指南 + 新物种接入流程
├── LICENSE               # 本项目代码的 MIT 许可证
├── THIRD_PARTY_NOTICES.md # 第三方组件/模型/权重的上游许可映射与全文
├── tools/                # 诊断/调参/一次性工具(归档, 不影响比赛与回归测试)
│   ├── test-boot-profile.mjs   # 启动剖析: 逐物种网格注入/模型编译耗时(诊断"打开慢")
│   ├── test-speed-sweep.mjs    # G1 速度包线扫描(阶跃/斜坡/软起步)
│   ├── test-tune-sweep.mjs     # 历史调参实验(力矩/摩擦扫描, 结论已写入「已知实现要点」)
│   ├── tune-x1-steer.mjs       # X1 航向外环调参工具(扫 kpYaw/yawCap, 输出均时/偏差/摔倒)
│   ├── x1_shoulder_check.py    # X1 两官方策略(leg/shoulder)的 Python 侧对照: 速度包线 + wy 敏感度
│   └── convert_and_calib.py    # walk.pt→ONNX 转换 + 站立高度标定(资产已入库, 复现用)
├── src/
│   ├── main.js           # 启动、多物种调度(各策略周期)、阵容、相机、HUD
│   ├── robots.js         # 🧬 物种注册表: 模型/策略来源 + 观测/PD 契约 + 调研记录
│   ├── policy.js         # 契约化观测构建(6 种布局)、ONNX 会话、PD/位置舵机/保持关节
│   ├── sim.js            # MuJoCo 加载、多物种模型编译(VFS 注入+补丁)、实例管理
│   ├── scene.js          # three.js 赛道与机器人可视化(mjv 管线) + 环境景观
│   │                     #   (蓝色橡胶跑道/白色标线/开放天空/草坪内场/体育场看台/高杆灯/
│   │                     #    双侧观众区/广告围挡/技术席/摄影机位/计时台)
│   ├── recorder.js       # 比赛录制: WebGL+面板合成画布 -> MediaRecorder MP4/WebM 自动下载
│   └── race.js           # 比赛状态机(倒计时/排名/摔倒罚时/结算)
├── assets/
│   ├── g1_29dof.xml + meshes/ + policy.onnx + meshlist.txt
│   │                            # Unitree G1(官方模型 + 第三方策略; meshlist.txt 为
│   │                            #   G1 网格清单, 供排查网格缺失)
│   ├── pm01/pm_v2.xml + meshes/ + policy.onnx    # 众擎 PM01(官方模型 + 官方策略)
│   ├── t1/T1_locomotion.xml + meshes/ + policy.onnx + T1.pt
│   │                            # Booster T1(官方模型, 权重已转 ONNX;
│   │                            #   T1.pt 为官方 TorchScript 原始权重, 仅溯源、运行时不加载)
│   ├── tk/tienkung.xml + meshes/ + policy.onnx + walk.pt
│   │                            # 天工 Tienkung2-Lite(官方模型, walk.pt 已转 ONNX;
│   │                            #   walk.pt 为官方 TorchScript 原始权重, 仅溯源、运行时不加载)
│   ├── x1/xyber_x1_flat.xml + meshes/ + policy_shoulder.onnx + policy.onnx
│   │                            # 智元灵犀 X1(官方模型 + 官方摆臂策略 rl_walk_leg_shoulder;
│   │                            #   policy.onnx 保留旧 leg 版可随时切回)
│   └── duck/scene_allcollisions.xml + robot_allcollisions.xml + meshes/ + policy.onnx
│                                  # Pollen MicroDuck(官方模型 + 官方 velstand.onnx; 12 个大网格
│                                  #   已从 1MB 上限抽稀到 4000 三角面以适配 WASM 2GB 堆)
└── vendor/               # 本地化的 mujoco-wasm / three / onnxruntime-web(仅 WASM CPU 后端所需文件)
```

## 接入新物种

见 [RETRAIN.md](RETRAIN.md):只要 GitHub 上存在某机器人的官方 MJCF/URDF + 可部署
策略权重(ONNX 或 TorchScript), 按"下载资产 → 对齐观测契约 → 转权重 → 注册物种"
四步即可加入比赛, 运行时代码零改动(建议先用 `sim2sim_check.py` 在 Python 侧
验证契约, 再移植到 `src/policy.js`)。

## 来源与致谢

各资产目录 ↔ 上游仓库 ↔ 许可证的精确映射与许可证全文见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), 简表:

- Unitree G1:模型 [unitree_ros](https://github.com/unitreerobotics/unitree_ros)(BSD-3-Clause) /
  [loco-lab](https://github.com/JacobEGarcia/loco-lab), 策略
  [RoboCubPilot/g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)
  (**上游未声明许可证, 仅限学习研究**)
- 众擎 PM01:模型+策略 [engineai-robotics/engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab)
  (BSD-3-Clause)
- Booster T1:模型+权重 [BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym)
  (Apache-2.0)
- 天工 Tienkung2-Lite:[Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
  (自定义许可: BSD-3-Clause 系多段归属)
- 智元灵犀 X1:[AgibotTech/agibot_x1_infer](https://github.com/AgibotTech/agibot_x1_infer)(Mulan PSL v2) /
  [AgibotTech/agibot_x1_train](https://github.com/AgibotTech/agibot_x1_train)
- Pollen MicroDuck:模型 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) /
  策略 [pollen-robotics/microduck-policies](https://huggingface.co/pollen-robotics/microduck-policies)
  (均 Apache-2.0; 执行器等效性验证用 [Rhoban/bam](https://github.com/Rhoban/bam) 的 XL330 m6 模型参数)
- 物理引擎:[MuJoCo](https://mujoco.readthedocs.io)(Apache-2.0)
- 推理引擎:[onnxruntime-web](https://github.com/microsoft/onnxruntime)(MIT)
- 渲染:[three.js](https://threejs.org)(MIT)

## 许可证

- 本项目代码以 [MIT](LICENSE) 许可证发布。
- `vendor/` 内的前端依赖与 `assets/` 内的机器人模型、策略权重按其各自上游许可证授权,
  逐项映射与许可证全文见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- ⚠️ **G1 策略权重(`assets/policy.onnx`)的上游仓库未声明任何许可证**, 随本仓库再分发
  仅限学习研究使用;真机部署请遵循各厂商与原策略作者的相关许可。如您是权利人并对
  该权重的分发有疑虑, 请通过 issue 联系处理。
