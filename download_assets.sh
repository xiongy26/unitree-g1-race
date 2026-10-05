#!/bin/bash
# 一次性资产下载: 天工 Tienkung2-Lite(TienKung-Lab) + 智元灵犀X1(agibot_x1_infer)
#                 + Pollen MicroDuck(microduck_rl + HuggingFace microduck-policies)
#
# 未覆盖的物种(上游未提供可直链的成品资产包, 需手动获取, 见各上游):
#   G1  : 模型 https://github.com/unitreerobotics/unitree_ros (MJCF+网格, BSD-3-Clause)
#         策略 https://github.com/RoboCubPilot/g1_deploy_mujoco (ONNX, 上游未声明许可)
#   PM01: 模型+策略 https://github.com/engineai-robotics/engineai_rl_lab (BSD-3-Clause)
#   T1  : 模型+权重 https://github.com/BoosterRobotics/booster_gym (Apache-2.0)
# 许可证详情见 THIRD_PARTY_NOTICES.md。
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
RAW="https://raw.githubusercontent.com"

# ---------- 天工 ----------
# 注意: 本仓库的天工资产目录是 assets/tk(非 assets/tienkung)
TK_BASE="$RAW/Open-X-Humanoid/TienKung-Lab/main/legged_lab/assets/tienkung2_lite"
mkdir -p "$ROOT/assets/tk/meshes"
curl -sf --retry 3 "$TK_BASE/mjcf/tienkung.xml" -o "$ROOT/assets/tk/tienkung.xml"
TK_MESHES="pelvis hip_roll_l_link hip_pitch_l_link hip_yaw_l_link knee_pitch_l_link ankle_pitch_l_link ankle_roll_l_link \
hip_roll_r_link hip_pitch_r_link hip_yaw_r_link knee_pitch_r_link ankle_pitch_r_link ankle_roll_r_link \
shoulder_pitch_l_link shoulder_roll_l_link shoulder_yaw_l_link elbow_pitch_l_link \
shoulder_pitch_r_link shoulder_roll_r_link shoulder_yaw_r_link elbow_pitch_r_link"
for m in $TK_MESHES; do
  test -s "$ROOT/assets/tk/meshes/$m.STL" || curl -sf --retry 3 "$TK_BASE/meshes/$m.STL" -o "$ROOT/assets/tk/meshes/$m.STL"
done
curl -sf --retry 3 "$RAW/Open-X-Humanoid/TienKung-Lab/main/Exported_policy/walk.pt" -o "$ROOT/assets/tk/walk.pt"
echo "tk done: $(du -sh "$ROOT/assets/tk" | cut -f1)"

# ---------- 智元 X1 ----------
X1_BASE="$RAW/AgibotTech/agibot_x1_infer/main/src/module/sim_module/model"
mkdir -p "$ROOT/assets/x1/robot/xyber_x1" "$ROOT/assets/x1/environment" "$ROOT/assets/x1/meshes"
curl -sf --retry 3 "$X1_BASE/mjcf/xyber_x1_flat.xml" -o "$ROOT/assets/x1/xyber_x1_flat.xml"
curl -sf --retry 3 "$X1_BASE/mjcf/robot/xyber_x1/xyber_x1_serial.xml" -o "$ROOT/assets/x1/robot/xyber_x1/xyber_x1_serial.xml"
curl -sf --retry 3 "$X1_BASE/mjcf/environment/flat.xml" -o "$ROOT/assets/x1/environment/flat.xml"
X1_MESHES="base_link_simple lumbar_yaw lumbar_roll lumbar_pitch \
left_shoulder_pitch left_shoulder_roll left_shoulder_yaw left_elbow_pitch left_elbow_yaw left_wrist_pitch left_wrist_roll \
right_shoulder_pitch right_shoulder_roll right_shoulder_yaw right_elbow_pitch right_elbow_yaw right_wrist_pitch right_wrist_roll \
left_hip_pitch left_hip_roll left_hip_yaw left_knee_pitch left_ankle_pitch left_ankle_roll \
right_hip_pitch right_hip_roll right_hip_yaw right_knee_pitch right_ankle_pitch right_ankle_roll"
for m in $X1_MESHES; do
  test -s "$ROOT/assets/x1/meshes/$m.STL" || curl -sf --retry 3 "$X1_BASE/meshes/$m.STL" -o "$ROOT/assets/x1/meshes/$m.STL"
done
# 官方行走策略 x2: rl_walk_leg_shoulder.onnx(摆臂版, 本仓库现用) +
#                  rl_walk_leg.onnx(leg 版, 保留可切回)
curl -sf --retry 3 "$RAW/AgibotTech/agibot_x1_infer/main/src/module/control_module/policy/rl_walk_leg_shoulder.onnx" -o "$ROOT/assets/x1/policy_shoulder.onnx"
curl -sf --retry 3 "$RAW/AgibotTech/agibot_x1_infer/main/src/module/control_module/policy/rl_walk_leg.onnx" -o "$ROOT/assets/x1/policy.onnx"
echo "x1 done: $(du -sh "$ROOT/assets/x1" | cut -f1)"

# ---------- Pollen MicroDuck ----------
# 官方 MJCF+网格来自 microduck_rl(VelStand 任务的训练模型 scene_allcollisions);
# 官方策略 velstand.onnx 发布在 HuggingFace(直连不可达时用 hf-mirror 镜像)。
# 注意: 本仓库对 12 个 1MB 上限的大网格做了 4000 三角面抽稀以适配 MuJoCo WASM
# 2GB 堆(见 README「已知实现要点」), 此处下载的是官方原版网格。
DUCK_BASE="$RAW/pollen-robotics/microduck_rl/main/src/mjlab_microduck/robot/microduck"
HF_BASE="${HF_BASE:-https://hf-mirror.com/pollen-robotics/microduck-policies/resolve/main}"
mkdir -p "$ROOT/assets/duck/meshes"
curl -sf --retry 3 "$DUCK_BASE/scene_allcollisions.xml" -o "$ROOT/assets/duck/scene_allcollisions.xml"
curl -sf --retry 3 "$DUCK_BASE/robot_allcollisions.xml" -o "$ROOT/assets/duck/robot_allcollisions.xml"
# 网格清单直接从 robot_allcollisions.xml 提取(与运行时同一正则口径)
for m in $(grep -o 'mesh file="[^"]*\.stl"' "$ROOT/assets/duck/robot_allcollisions.xml" | sed 's/mesh file="//;s/"//' | sort -u); do
  test -s "$ROOT/assets/duck/meshes/$m" || curl -sf --retry 3 "$DUCK_BASE/assets/$m" -o "$ROOT/assets/duck/meshes/$m"
done
curl -sfL --retry 3 "$HF_BASE/velstand.onnx" -o "$ROOT/assets/duck/policy.onnx"
echo "duck done: $(du -sh "$ROOT/assets/duck" | cut -f1)"
