# X1 两个官方策略的 sim2sim 对照: rl_walk_leg.onnx(12腿, 现用) vs
# rl_walk_leg_shoulder.onnx(12腿+2肩俯仰摆臂, walk_leg_arm 模式).
# 契约均逐条对齐 agibot_x1_infer rl_x1_sim.yaml 两个 controller 配置:
#   leg     : kps [30,40,35,100,35,35]x2, cycle 0.7, obs 47x66
#   shoulder: kps [60,60,40,80,40,30,20]x2, cycle 1.0, obs 53x66, 肩俯仰入策略
# 速度指令按比赛同款 2.5 m/s^2 斜坡起步, 量测 t>=2s 后的平均速度.
import os
import re
import sys

import numpy as np
import mujoco
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 脚本在 tools/ 下, ROOT 指向仓库根
np.set_printoptions(precision=3, suppress=True)


def gravity_orientation(q):
    w, x, y, z = q
    return np.array([
        2 * (-z * x + w * y),
        -2 * (z * y + w * x),
        1 - 2 * (w * w + z * z),
    ])


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def load_model():
    base = os.path.join(ROOT, "assets/x1")
    flat = open(os.path.join(base, "xyber_x1_flat.xml"), encoding="utf-8").read()

    def body_of(t):
        t = re.sub(r"^[\s\S]*?<mujoco[^>]*>", "", t)
        return re.sub(r"</mujoco>\s*$", "", t)
    serial = open(os.path.join(base, "robot/xyber_x1/xyber_x1_serial.xml"), encoding="utf-8").read()
    env = open(os.path.join(base, "environment/flat.xml"), encoding="utf-8").read()
    flat = flat.replace('<include file = "robot/xyber_x1/xyber_x1_serial.xml" />', body_of(serial))
    flat = flat.replace('<include file = "environment/flat.xml" />', body_of(env))
    flat = flat.replace('meshdir="../meshes"', f'meshdir="{os.path.join(ROOT, "assets/x1/meshes")}"')
    for j in ["left_wrist_pitch", "left_wrist_roll", "right_wrist_pitch", "right_wrist_roll"]:
        flat = flat.replace(f'name="{j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1"',
                            f'name="{j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1" armature="0.002"')
    m = mujoco.MjModel.from_xml_string(flat)
    m.opt.timestep = 0.002
    return m


def hold_specs(with_shoulder):
    # 29 执行器顺序: lumbar3, l_arm7, r_arm7, l_leg6, r_leg6
    # 策略不控、按 pd_zero/pd_stand 保持的关节; 肩俯仰入策略时从保持表中剔除
    holds = []
    for i in range(3):
        holds.append((i, 7 + i, 6 + i, 0.0, 700, 0.6, 150))
    for side in range(2):
        b = side * 7 + 3
        qp = 10 + side * 7
        entry = [
            (b + 0, qp + 0, None, 0.15, 300, 0.6, 150),   # shoulder_pitch(入策略时跳过)
            (b + 1, qp + 1, None, -0.1, 300, 0.6, 150),   # shoulder_roll
            (b + 2, qp + 2, None, 0.0, 30, 0.1, 150),     # shoulder_yaw
            (b + 3, qp + 3, None, 0.3, 300, 0.6, 150),    # elbow_pitch
            (b + 4, qp + 4, None, 0.0, 30, 0.1, 150),     # elbow_yaw
            (b + 5, qp + 5, None, 0.0, 30, 0.1, 150),     # wrist_pitch
            (b + 6, qp + 6, None, 0.0, 30, 0.1, 150),     # wrist_roll
        ]
        for h in entry:
            if with_shoulder and h[0] % 7 == 3:  # 每侧臂的第1个 = shoulder_pitch(ctrl 3/10)
                continue
            holds.append(h)
    return holds


def run(m, policy, vx, seconds=10.0, verbose=False, seed=0, jitter=0.0, wy=0.0):
    """policy: 'leg' | 'shoulder'. 返回 (ok, 平均速度 m/s, 终点x).
    seed>0: 初始关节角/基座位姿加噪声 + 指令 ±jitter 抖动(模拟比赛条件).
    wy: 恒定偏航角速度指令(量化转向指令的速度代价)."""
    d = mujoco.MjData(m)

    if policy == "leg":
        na, obs, cycle = 12, 47, 0.7
        kps = np.array([30, 40, 35, 100, 35, 35] * 2, np.float32)
        kds = np.array([3, 3, 4, 10, 0.5, 0.5] * 2, np.float32)
        lim = np.array([150, 50, 50, 150, 18, 18] * 2, np.float32)
        default = np.array([0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
                            -0.4, -0.05, 0.31, 0.49, -0.21, 0.0], np.float32)
        # 策略序 -> (ctrl, qpos): 腿 17..28 / 24..35 恒等
        p_ctrl = list(range(17, 29))
        p_qpos = list(range(24, 36))
        onnx_file = os.path.join(ROOT, "assets/x1/policy.onnx")
    else:
        na, obs, cycle = 14, 53, 1.0
        kps = np.array([60, 60, 40, 80, 40, 30, 20] * 2, np.float32)
        kds = np.array([6, 3, 3, 4, 2, 2, 3] * 2, np.float32)
        lim = np.array([150, 50, 50, 150, 18, 18, 150] * 2, np.float32)
        default = np.array([0.4, 0.05, -0.31, 0.49, -0.21, 0.0, 0.15,
                            -0.4, -0.05, 0.31, 0.49, -0.21, 0.0, 0.15], np.float32)
        # 策略序: L腿6 + L肩俯仰 + R腿6 + R肩俯仰
        p_ctrl = [17, 18, 19, 20, 21, 22, 3, 23, 24, 25, 26, 27, 28, 10]
        p_qpos = [24, 25, 26, 27, 28, 29, 10, 30, 31, 32, 33, 34, 35, 17]
        onnx_file = os.path.join(ROOT, "assets/x1/policy_shoulder.onnx")

    p_qvel = [q - 1 for q in p_qpos]
    holds = hold_specs(policy == "shoulder")

    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    stack, dec, cmd_th, alpha = 66, 5, 0.05, 100.0 * 0.002
    hist = np.zeros((stack, obs), np.float32)
    action = np.zeros(na, np.float32)
    cmd = np.zeros(3, np.float32)
    lpf_t = default.copy()
    leg_def = np.array([0.4, 0.05, -0.31, 0.49, -0.21, 0.0,
                        -0.4, -0.05, 0.31, 0.49, -0.21, 0.0], np.float32)
    d.qpos[2] = 0.62
    d.qpos[10:17] = [0.15, -0.1, 0, 0.3, 0, 0, 0]
    d.qpos[17:24] = [0.15, -0.1, 0, 0.3, 0, 0, 0]
    d.qpos[24:36] = leg_def
    rng = np.random.default_rng(seed)
    if seed > 0:
        d.qpos[24:36] += rng.uniform(-0.03, 0.03, 12)   # 关节初始扰动
        d.qpos[2] += rng.uniform(-0.005, 0.005)
        d.qpos[3:7] = d.qpos[3:7] / np.linalg.norm(d.qpos[3:7])
        vx = vx * (1.0 + rng.uniform(-jitter, jitter))   # 比赛同款 ±6% 速度抖动
    mujoco.mj_forward(m, d)
    filled = False
    sim_t = 0.0
    x_ref, t_ref = None, None
    nsteps = int(seconds / (0.002 * dec))
    for st in range(nsteps):
        if st >= 20:
            # 比赛同款斜坡: 2.5 m/s^2
            cmd[0] = min(vx, cmd[0] + 2.5 * 0.002 * dec)
            cmd[2] = wy
        phase = 0.0 if np.linalg.norm(cmd) <= cmd_th else (sim_t / cycle) % 1.0
        o = np.concatenate([
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            [cmd[0] * 2.0, cmd[1] * 2.0, cmd[2] * 1.0],
            d.qpos[p_qpos] - default,
            d.qvel[p_qvel] * 0.05,
            action,
            d.qvel[3:6] * 1.0,
            quat_to_rpy(d.qpos[3:7]),
        ]).astype(np.float32)
        if not filled:
            hist[:] = o
            hist[:, 5 + 2 * na:5 + 3 * na] = 0.0
            filled = True
        else:
            hist = np.roll(hist, -1, axis=0)
            hist[-1] = o
        action = np.clip(sess.run(None, {"input": hist.reshape(-1)[None]})[0][0], -100, 100).astype(np.float32)
        pos_des = action * 0.5 + default
        for _ in range(dec):
            lpf_t += alpha * (pos_des - lpf_t)
            tau = kps * (lpf_t - d.qpos[p_qpos]) - kds * d.qvel[p_qvel]
            d.ctrl[p_ctrl] = np.clip(tau, -lim, lim)
            for (c, qp, qv, tgt, kp, kd, lm) in holds:
                th = kp * (tgt - d.qpos[qp]) - kd * d.qvel[qp - 1]
                d.ctrl[c] = np.clip(th, -lm, lm)
            mujoco.mj_step(m, d)
            sim_t += 0.002
        if x_ref is None and sim_t >= 2.0:
            x_ref, t_ref = d.qpos[0], sim_t
        if verbose and st % 100 == 0:
            print(f"    t={sim_t:5.1f}s x={d.qpos[0]:6.2f} z={d.qpos[2]:.3f} yaw={quat_to_rpy(d.qpos[3:7])[2]:5.2f}")
        if d.qpos[2] < 0.3:
            v = (d.qpos[0] - x_ref) / max(1e-6, sim_t - t_ref) if x_ref is not None else 0.0
            print(f"    FELL at t={sim_t:.1f}s x={d.qpos[0]:.2f} (v_avg={v:.2f})")
            return False, v, d.qpos[0]
    v = (d.qpos[0] - x_ref) / (sim_t - t_ref)
    print(f"    OK t={sim_t:.0f}s x={d.qpos[0]:.2f} (v_avg={v:.2f} m/s, cmd={vx})")
    return True, v, d.qpos[0]


if __name__ == "__main__":
    m = load_model()
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    speeds = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    plans = {
        "leg": speeds or [0.5, 0.85, 0.9, 1.0, 1.1],
        "shoulder": speeds or [0.5, 0.85, 1.0, 1.2, 1.4, 1.6],
    }
    for name in ([which] if which in ("leg", "shoulder") else ["leg", "shoulder"]):
        print(f"== {name} ==")
        for vx in plans[name]:
            print(f"  vx={vx}")
            run(m, name, vx)
