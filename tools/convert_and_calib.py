# 一次性脚本: walk.pt -> ONNX 转换 + 数值校验 + 站立高度标定
import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 脚本在 tools/ 下, ROOT 指向仓库根

# ---------- 1. TienKung walk.pt -> ONNX ----------
pt_path = os.path.join(ROOT, "assets/tk/walk.pt")
onnx_path = os.path.join(ROOT, "assets/tk/policy.onnx")
jit = torch.jit.load(pt_path, map_location="cpu")
jit.eval()

IN_LEN, OUT_LEN = 750, 20  # 75 obs x 10 history
example = torch.zeros(1, IN_LEN)
with torch.no_grad():
    ref = jit(example).numpy()
torch.onnx.export(
    jit, example, onnx_path,
    input_names=["obs"], output_names=["action"],
    dynamic_axes={"obs": {0: "N"}, "action": {0: "N"}},
    opset_version=17,
)
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
errs = []
for _ in range(8):
    x = rng.standard_normal((3, IN_LEN)).astype(np.float32) * 0.7
    got = sess.run(None, {"obs": x})[0]
    with torch.no_grad():
        want = jit(torch.from_numpy(x)).numpy()
    errs.append(np.abs(got - want).max())
print(f"[TK] onnx export max_err={max(errs):.2e}  out_shape={got.shape}")

# ---------- 2. X1 rl_walk_leg.onnx 形状校验 ----------
x1 = os.path.join(ROOT, "assets/x1/policy.onnx")
s2 = ort.InferenceSession(x1, providers=["CPUExecutionProvider"])
i0 = s2.get_inputs()[0]
o0 = s2.get_outputs()[0]
print(f"[X1] input {i0.name} shape={i0.shape}  output {o0.name} shape={o0.shape}")
probe = np.zeros((1, 3102), np.float32)
out = s2.run(None, {i0.name: probe})[0]
print(f"[X1] probe run ok, out dim={out.shape}")

# ---------- 3. 站立高度标定 ----------
import mujoco

def settle(xml, patch=None, q=None):
    s = open(xml, encoding="utf-8").read()
    for a, b in (patch or []):
        s = s.replace(a, b)
    m = mujoco.MjModel.from_xml_string(s)
    d = mujoco.MjData(m)
    if q is not None:
        d.qpos[7:7 + len(q)] = q
        d.ctrl[:] = q  # 位置执行器目标 = 默认站姿(与官方 sim2sim 静止一致)
    mujoco.mj_forward(m, d)
    z0 = d.qpos[2]
    for _ in range(4000):  # 8s 落地稳定
        mujoco.mj_step(m, d)
    return z0, d.qpos[2], d.qpos[0:3].copy()

# TienKung: 运动学标定站立高度(默认站姿脚底相对基座), 不做动力学平衡
tk_default = [0, -0.5, 0, 1.0, -0.5, 0, 0, -0.5, 0, 1.0, -0.5, 0,
              0, 0.1, 0, -0.3, 0, -0.1, 0, -0.3]
tk_xml = os.path.join(ROOT, "assets/tk/tienkung.xml")
s = open(tk_xml, encoding="utf-8").read().replace("../meshes/", os.path.join(ROOT, "assets/tk/meshes") + "/")
m = mujoco.MjModel.from_xml_string(s)
d = mujoco.MjData(m)
d.qpos[7:7 + 20] = tk_default
mujoco.mj_forward(m, d)
min_toe = min(d.geom_xpos[g][2] - m.geom_size[g][1] for g in range(m.ngeom)
              if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER.value and d.geom_xpos[g][2] < 0.5)
print(f"[TK] default pose: base z={d.qpos[2]:.3f}, toe bottom rel base z={min_toe - d.qpos[2]:.3f}"
      f" -> zHome ≈ {-min_toe:.3f}")

# X1: serial MJCF 需要 include, 直接读 flat 组合文件
x1_flat = os.path.join(ROOT, "assets/x1/xyber_x1_flat.xml")
base = os.path.dirname(x1_flat)
serial = open(os.path.join(base, "robot/xyber_x1/xyber_x1_serial.xml"), encoding="utf-8").read()
env = open(os.path.join(base, "environment/flat.xml"), encoding="utf-8").read()
# 手动内联(与 sim.js 的 inlineMjcfIncludes 同逻辑)
import re
flat = open(x1_flat, encoding="utf-8").read()
def body_of(txt):
    txt = re.sub(r"^[\s\S]*?<mujoco[^>]*>", "", txt)
    return re.sub(r"</mujoco>\s*$", "", txt)
flat = flat.replace('<include file = "robot/xyber_x1/xyber_x1_serial.xml" />', body_of(serial))
flat = flat.replace('<include file = "environment/flat.xml" />', body_of(env))
flat = flat.replace('meshdir="../meshes"', 'meshdir="assets/x1/meshes"')
# 官方 rl_walk_leg 初始站姿(qpos 顺序: lumbar3 + l_arm7 + r_arm7 + l_leg6 + r_leg6)
x1_q = [0, 0, 0,
        0.15, -0.1, 0, 0.3, 0, 0, 0,
        0.15, -0.1, 0, 0.3, 0, 0, 0,
        0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
        -0.4, -0.05, 0.31, 0.49, -0.21, 0.0]
# 手腕加 armature 数值稳定补丁(与 robots.js 的 xmlPatches 一致)
for j in ["left_wrist_pitch", "left_wrist_roll", "right_wrist_pitch", "right_wrist_roll"]:
    flat = flat.replace(f'name="{j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1"',
                        f'name="{j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1" armature="0.002"')
z0x, z1x, px = settle_from_str = (None, None, None)
m = mujoco.MjModel.from_xml_string(flat)
d = mujoco.MjData(m)
d.qpos[7:7 + 29] = x1_q
mujoco.mj_forward(m, d)
z0x = d.qpos[2]
for _ in range(4000):
    # 位置保持 PD(官方 pd_zero/pd_stand/pd_walk 组合), 500Hz
    q = d.qpos[7:36].copy(); dq = d.qvel[6:35].copy()
    tgt = np.array(x1_q)
    kp = np.array([700]*3 + [300,300,30,300,30,30,30] + [300,300,30,300,30,30,30] +
                  [30,40,35,100,35,35] + [30,40,35,100,35,35], dtype=float)
    kd = np.array([0.6]*3 + [0.6,0.6,0.1,0.6,0.1,0.1,0.1] + [0.6,0.6,0.1,0.6,0.1,0.1,0.1] +
                  [3,3,4,10,0.5,0.5] + [3,3,4,10,0.5,0.5], dtype=float)
    tau = kp * (tgt - q) - kd * dq
    lim = np.array([150]*3 + [150,150,150,150,150,150,150] + [150,150,150,150,150,150,150] +
                   [150,50,50,150,18,18] + [150,50,50,150,18,18], dtype=float)
    d.ctrl[:] = np.clip(tau, -lim, lim)
    mujoco.mj_step(m, d)
print(f"[X1] spawn z={z0x:.3f} settled z={d.qpos[2]:.3f} xy=({d.qpos[0]:.2f},{d.qpos[1]:.2f})")
