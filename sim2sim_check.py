# Sim2sim 契约验证: 用与本仓库 JS 运行时完全一致的管线跑官方策略
#   TK (天工 Tienkung2-Lite): motor+显式PD(kd=0, 被动阻尼同官方), 75obsx10帧@50Hz
#   X1 (智元灵犀X1): 12腿PD+上肢保持, 47obsx66帧@100Hz, 动作LPF
# 跑 12m 直线, 验证契约方向正确性(不追求完美调参)
import os
import re

import numpy as np
import mujoco
import onnxruntime as ort

ROOT = os.path.dirname(os.path.abspath(__file__))
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


def run_tk(seconds=14.0, vx=0.6):
    xml = open(os.path.join(ROOT, "assets/tk/tienkung.xml"), encoding="utf-8").read()
    xml = xml.replace("../meshes/", "assets/tk/meshes/")
    # 与 JS 运行时一致的执行器补丁: position -> motor(力矩), ctrlrange=训练 effort 限幅
    effort = {"hip_roll": 180, "hip_pitch": 300, "hip_yaw": 180, "knee_pitch": 300,
              "ankle_pitch": 60, "ankle_roll": 30, "shoulder_pitch": 52.5,
              "shoulder_roll": 52.5, "shoulder_yaw": 52.5, "elbow_pitch": 52.5}
    import re
    def repl(mo):
        joint = mo.group(2)
        e = next(v for k, v in effort.items() if joint.startswith(k))
        return f'<motor name="{mo.group(1)}" joint="{joint}" ctrlrange="-{e} {e}"/>'
    xml = re.sub(r'<position name="([^"]+)" joint="([^"]+)" kp="[^"]+"/>', repl, xml)
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = 0.005
    d = mujoco.MjData(m)

    default_dof = np.array([0, -0.5, 0, 1.0, -0.5, 0, 0, -0.5, 0, 1.0, -0.5, 0,
                            0, 0.1, 0, -0.3, 0, -0.1, 0, -0.3], np.float32)
    # sim2sim mujoco_to_isaac: 每个策略位 i 对应的 mujoco 关节号
    mj_to_isaac = np.array([0, 6, 12, 16, 1, 7, 13, 17, 2, 8, 14, 18,
                            3, 9, 15, 19, 4, 10, 5, 11])
    # sim2sim isaac_to_mujoco(=逆映射): target_mj[j] = a[isa_to_mj[j]] + default[j]
    isa_to_mj = np.array([0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 17, 19,
                          2, 6, 10, 14, 3, 7, 11, 15])
    kps = np.array([700, 700, 500, 700, 30, 16.8] * 2 + [60, 20, 10, 10] * 2, np.float32)
    lim = np.array([180, 300, 180, 300, 60, 30] * 2 + [52.5] * 8, np.float32)

    sess = ort.InferenceSession(os.path.join(ROOT, "assets/tk/policy.onnx"),
                                providers=["CPUExecutionProvider"])
    OBS, STACK, DEC, CYCLE = 75, 10, 4, 0.85
    hist = np.zeros((STACK, OBS), np.float32)
    action = np.zeros(20, np.float32)
    cmd = np.array([0.0, 0.0, 0.0], np.float32)
    d.qpos[2] = 0.99
    d.qpos[7:27] = default_dof
    mujoco.mj_forward(m, d)
    step_count = 0
    nsteps = int(seconds / (0.005 * DEC))
    for st in range(nsteps):
        if st == 20:
            cmd[0] = vx
        # ---- 观测(75) ----
        phase = (step_count * 0.005 * DEC / CYCLE + np.array([0.38, 0.88])) % 1.0
        obs = np.concatenate([
            d.qvel[3:6] * 1.0,                       # 机体系角速度(原始)
            gravity_orientation(d.qpos[3:7]),        # 重力投影
            cmd,
            (d.qpos[7:27] - default_dof)[mj_to_isaac],
            d.qvel[6:26][mj_to_isaac],
            np.clip(action, -100, 100),
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
            [0.38, 0.38],                            # phase_ratio
        ]).astype(np.float32)
        hist = np.roll(hist, -1, axis=0)
        hist[-1] = obs
        x = hist.reshape(-1)[None]
        action = np.clip(sess.run(None, {"obs": x})[0][0], -100, 100).astype(np.float32)
        target = action[isa_to_mj] * 0.25 + default_dof
        # ---- PD @ 200Hz, dt 0.005, kd=0(被动阻尼同官方) ----
        for _ in range(DEC):
            tau = kps * (target - d.qpos[7:27])
            d.ctrl[:] = np.clip(tau, -lim, lim)
            mujoco.mj_step(m, d)
        step_count += 1
        if st % 100 == 0:
            print(f"[TK] t={st*0.02:5.1f}s x={d.qpos[0]:6.2f} z={d.qpos[2]:.3f}"
                  f" yaw={quat_to_rpy(d.qpos[3:7])[2]:5.2f}")
        if d.qpos[2] < 0.5:
            print(f"[TK] FELL at t={st*0.02:.1f}s x={d.qpos[0]:.2f}")
            return False
    print(f"[TK] OK: 12m 目标, 终点 x={d.qpos[0]:.2f} (vx={vx})")
    return d.qpos[0] > 6.0


def run_x1(seconds=14.0, vx=0.5):
    base = os.path.join(ROOT, "assets/x1")
    flat = open(os.path.join(base, "xyber_x1_flat.xml"), encoding="utf-8").read()
    import re
    def body_of(t):
        t = re.sub(r"^[\s\S]*?<mujoco[^>]*>", "", t)
        return re.sub(r"</mujoco>\s*$", "", t)
    serial = open(os.path.join(base, "robot/xyber_x1/xyber_x1_serial.xml"), encoding="utf-8").read()
    env = open(os.path.join(base, "environment/flat.xml"), encoding="utf-8").read()
    flat = flat.replace('<include file = "robot/xyber_x1/xyber_x1_serial.xml" />', body_of(serial))
    flat = flat.replace('<include file = "environment/flat.xml" />', body_of(env))
    flat = flat.replace('meshdir="../meshes"', 'meshdir="assets/x1/meshes"')
    for j in ["left_wrist_pitch", "left_wrist_roll", "right_wrist_pitch", "right_wrist_roll"]:
        flat = flat.replace(f'name="{j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1"',
                            f'name="{j}_joint" type="hinge" pos="0 0 0" axis="0 0 -1" damping="1" armature="0.002"')
    m = mujoco.MjModel.from_xml_string(flat)
    m.opt.timestep = 0.002  # 运行时 500Hz(官方 1kHz), 策略仍 100Hz
    d = mujoco.MjData(m)

    # 29 执行器顺序: lumbar3, l_arm7, r_arm7, l_leg6, r_leg6
    # 腿(策略顺序): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll × L,R
    leg_kp = np.array([30, 40, 35, 100, 35, 35] * 2, np.float32)
    leg_kd = np.array([3, 3, 4, 10, 0.5, 0.5] * 2, np.float32)
    leg_lim = np.array([150, 50, 50, 150, 18, 18] * 2, np.float32)
    leg_def = np.array([0.4, 0.05, -0.31, 0.49, -0.21, 0.0, -0.4, -0.05, 0.31, 0.49, -0.21, 0.0], np.float32)
    leg_ctrl_idx = np.arange(17, 29)      # 12 个腿执行器
    leg_qpos_idx = np.arange(24, 36)      # 对应绝对 qpos 下标(基座7+关节序号17..28)
    leg_qvel_idx = leg_qpos_idx - 1       # qvel: 基座6+关节序号
    hold_ctrl = np.arange(0, 17)
    hold_qpos = np.arange(7, 24)
    hold_qvel = hold_qpos - 1
    hold_tgt = np.array([0, 0, 0,
                         0.15, -0.1, 0, 0.3, 0, 0, 0,
                         0.15, -0.1, 0, 0.3, 0, 0, 0], np.float32)
    hold_kp = np.array([700, 700, 700,
                        300, 300, 30, 300, 30, 30, 30,
                        300, 300, 30, 300, 30, 30, 30], np.float32)
    hold_kd = np.array([0.6, 0.6, 0.6,
                        0.6, 0.6, 0.1, 0.6, 0.1, 0.1, 0.1,
                        0.6, 0.6, 0.1, 0.6, 0.1, 0.1, 0.1], np.float32)

    sess = ort.InferenceSession(os.path.join(ROOT, "assets/x1/policy.onnx"),
                                providers=["CPUExecutionProvider"])
    OBS, STACK, DEC, CYCLE, CMD_TH = 47, 66, 5, 0.7, 0.05
    hist = np.zeros((STACK, OBS), np.float32)
    action = np.zeros(12, np.float32)
    cmd = np.array([0.0, 0.0, 0.0], np.float32)
    lpf_t = leg_def.copy()
    d.qpos[2] = 0.62
    d.qpos[10:17] = [0.15, -0.1, 0, 0.3, 0, 0, 0]   # l_arm(qpos 10..16)
    d.qpos[17:24] = [0.15, -0.1, 0, 0.3, 0, 0, 0]   # r_arm(qpos 17..23)
    d.qpos[24:36] = leg_def                          # 腿(qpos 24..35)
    mujoco.mj_forward(m, d)
    filled = False
    sim_t = 0.0
    nsteps = int(seconds / (0.002 * DEC))
    for st in range(nsteps):
        if st == 20:
            cmd[0] = vx
        # ---- 观测(47) ----
        phase = 0.0 if np.linalg.norm(cmd) <= CMD_TH else (sim_t / CYCLE) % 1.0
        obs = np.concatenate([
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            [cmd[0] * 2.0, cmd[1] * 2.0, cmd[2] * 1.0],
            (d.qpos[leg_qpos_idx] - leg_def) * 1.0,
            d.qvel[leg_qvel_idx] * 0.05,
            action,
            d.qvel[3:6] * 1.0,
            quat_to_rpy(d.qpos[3:7]),
        ]).astype(np.float32)
        if not filled:
            hist[:] = obs
            hist[:, 29:41] = 0.0  # 首帧动作段清零(与官方一致)
            filled = True
        else:
            hist = np.roll(hist, -1, axis=0)
            hist[-1] = obs
        x = hist.reshape(-1)[None]
        action = np.clip(sess.run(None, {"input": x})[0][0], -100, 100).astype(np.float32)
        pos_des = action * 0.5 + leg_def
        # ---- PD @ 500Hz + 动作一阶 LPF(wc=100) ----
        for _ in range(DEC):
            lpf_t += 100.0 * 0.002 * (pos_des - lpf_t)
            tau = leg_kp * (lpf_t - d.qpos[leg_qpos_idx]) - leg_kd * d.qvel[leg_qvel_idx]
            d.ctrl[leg_ctrl_idx] = np.clip(tau, -leg_lim, leg_lim)
            th = hold_kp * (hold_tgt - d.qpos[hold_qpos]) - hold_kd * d.qvel[hold_qvel]
            d.ctrl[hold_ctrl] = np.clip(th, -150, 150)
            mujoco.mj_step(m, d)
            sim_t += 0.002
        if st % 50 == 0:
            print(f"[X1] t={sim_t:5.1f}s x={d.qpos[0]:6.2f} z={d.qpos[2]:.3f}"
                  f" yaw={quat_to_rpy(d.qpos[3:7])[2]:5.2f}")
        if d.qpos[2] < 0.3:
            print(f"[X1] FELL at t={sim_t:.1f}s x={d.qpos[0]:.2f}")
            return False
    print(f"[X1] OK: 终点 x={d.qpos[0]:.2f} (vx={vx})")
    return d.qpos[0] > 6.0


def run_duck(seconds=8.0, vx=0.8):
    # MicroDuck(Pollen Robotics): velstand.onnx 官方默认行走策略(行走+零指令站立一体),
    # 官方 infer_policy.py --no-bam 契约逐条对齐:
    #   模型  scene_allcollisions.xml + robot_allcollisions.xml(官方 VelStand 训练模型)
    #   执行器 官方 MJCF position 舵机原样(kp=0.55 kv=0 forcerange ±0.96, 关节阻尼 0.053)
    #   obs(61) 单帧 = [机体角速度(3), 重力投影(3), 关节位置-默认(14), 关节速度(14),
    #            上次动作(14), 指令(13): [vx,vy,wy + head(4) + body(6) 置零]]
    #   动作 14 -> 目标角 = a·1.0 + 默认角, ctrl 直接写目标角
    #   dt 0.005 × decim 4 = 50Hz; 初始 trunk z=0.125 + 默认站姿(官方 main)
    #   训练指令范围 vx ±0.4 / vy ±0.3 / wy ±1.0(velocity 配置)
    base = os.path.join(ROOT, "assets/duck")
    scene = open(os.path.join(base, "scene_allcollisions.xml"), encoding="utf-8").read()
    robot = open(os.path.join(base, "robot_allcollisions.xml"), encoding="utf-8").read()

    def body_of(t):
        t = re.sub(r"^[\s\S]*?<mujoco[^>]*>", "", t)
        return re.sub(r"</mujoco>\s*$", "", t)
    scene = scene.replace('<include file="robot_allcollisions.xml" />', body_of(robot))
    scene = scene.replace('meshdir="assets"', 'meshdir="assets/duck/meshes"')
    m = mujoco.MjModel.from_xml_string(scene)
    m.opt.timestep = 0.005
    d = mujoco.MjData(m)

    default_dof = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                            0.3491, 0.3491, 0.0, 0.0,
                            0.0, 0.0873, 0.4579, 0.0049, -0.4530], np.float32)
    sess = ort.InferenceSession(os.path.join(base, "policy.onnx"),
                                providers=["CPUExecutionProvider"])
    DEC = 4
    action = np.zeros(14, np.float32)
    d.qpos[2] = 0.125
    d.qpos[7:21] = default_dof
    d.ctrl[:] = default_dof  # 官方: set_position_targets(default_pose) 后 mj_forward
    mujoco.mj_forward(m, d)
    sim_t = 0.0
    stand_steps = int(1.5 / (0.005 * DEC))
    nsteps = stand_steps + int(seconds / (0.005 * DEC))
    x_start = None
    for st in range(nsteps):
        if st == stand_steps:
            cmd = np.array([vx, 0, 0] + [0] * 10, np.float32)
            x_start = d.qpos[0]
        elif st == 0:
            cmd = np.zeros(13, np.float32)
        obs = np.concatenate([
            d.qvel[3:6] * 1.0,                        # 机体系角速度(原始)
            gravity_orientation(d.qpos[3:7]),         # 重力投影
            d.qpos[7:21] - default_dof,
            d.qvel[6:20] * 1.0,
            action,
            cmd,
        ]).astype(np.float32)
        action = sess.run(None, {"obs": obs[None]})[0][0].astype(np.float32)
        target = action * 1.0 + default_dof
        for _ in range(DEC):
            d.ctrl[:] = target
            mujoco.mj_step(m, d)
            sim_t += 0.005
        if st % 50 == 0:
            v = 0.0 if x_start is None or sim_t <= 1.5 else (d.qpos[0] - x_start) / (sim_t - 1.5)
            print(f"[DUCK] t={sim_t:5.1f}s x={d.qpos[0]:6.2f} z={d.qpos[2]:.3f}"
                  f" yaw={quat_to_rpy(d.qpos[3:7])[2]:5.2f} v_avg={v:.2f}")
        if d.qpos[2] < 0.06:
            print(f"[DUCK] FELL at t={sim_t:.1f}s x={d.qpos[0]:.2f}")
            return False
    v = (d.qpos[0] - x_start) / (sim_t - 1.5)
    print(f"[DUCK] OK: 指令 vx={vx}, 实测均速 {v:.2f} m/s, 终点 x={d.qpos[0]:.2f}")
    return v > 0.15


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "tk"
    vx = float(sys.argv[2]) if len(sys.argv) > 2 else None
    ok = {"tk": run_tk, "x1": run_x1, "duck": run_duck}[which](vx=vx)
    print("PASS" if ok else "FAIL")
