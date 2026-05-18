"""
里程碑三：逆运动学 + 轨迹规划。
目标：对比手动示教轨迹与逆运动学自动规划轨迹。

操作说明：
  鼠标拖拽：调整视角
  空格：随机目标，自动 IK 规划
  R：完整流程（设置固定目标 → 手动录制 → 演示手动轨迹 → 演示 IK 轨迹）
  ESC：退出
"""

import mujoco
import numpy as np
import glfw
import json
import os
import time

# ==================== 模型定义 ====================
MODEL_XML = r"""
<mujoco model="3dof_arm_5cm">
  <compiler angle="degree"/>
  <option timestep="0.01"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.94 0.96 1.0" rgb2="0.90 0.93 0.97" width="512" height="512"/>
    <material name="mat_floor" texture="grid" texrepeat="5 5"/>
    <material name="mat_link1" rgba="0.2 0.5 0.8 1"/>
    <material name="mat_link2" rgba="0.8 0.4 0.2 1"/>
    <material name="mat_link3" rgba="0.3 0.7 0.3 1"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" material="mat_floor"/>

    <body name="base" pos="0 0 0.02">
      <geom name="base_geom" type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>

      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180" damping="2"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.05" size="0.012" material="mat_link1"/>

        <body name="link2" pos="0 0 0.05">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-150 150" damping="1.5"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.05 0 0" size="0.01" material="mat_link2"/>

          <body name="link3" pos="0.05 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-150 150" damping="1.0"/>
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0.05 0 0" size="0.008" material="mat_link3"/>

            <site name="ee_site" pos="0.05 0 0" type="sphere" size="0.012" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>

    <!-- 目标初始位置（与02一致） -->
    <body name="target" pos="-0.08 0.03 0.04">
      <geom name="target_geom" type="box" size="0.008 0.008 0.008" rgba="0 1 0 0.9"/>
    </body>
  </worldbody>

  <actuator>
    <position joint="joint1" kp="80"/>
    <position joint="joint2" kp="60"/>
    <position joint="joint3" kp="40"/>
  </actuator>
</mujoco>
"""


# ==================== 逆运动学 ====================
def solve_ik(model, data, target_pos, site_id, max_iter=500, tol=1e-3):
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3"),
    ]
    q = np.array([data.qpos[jid] for jid in joint_ids])

    for _ in range(max_iter):
        for i, jid in enumerate(joint_ids):
            data.qpos[jid] = q[i]
        mujoco.mj_forward(model, data)

        ee_pos = data.site_xpos[site_id].copy()
        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            return q

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, joint_ids]

        damping = 0.01
        delta_q = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), error)
        q += delta_q * 0.5
        for i, jid in enumerate(joint_ids):
            q[i] = np.clip(q[i], model.jnt_range[jid][0], model.jnt_range[jid][1])
    return None


# ==================== 多段 IK 轨迹规划 ====================
def plan_trajectory(model, data, start_q, target_pos, site_id, steps=100):
    """
    在笛卡尔空间取多个中间路径点，每段分别求解 IK，
    然后在关节空间平滑插值，避免穿模。
    """
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3"),
    ]

    # 获取起始末端位置
    for i, jid in enumerate(joint_ids):
        data.qpos[jid] = start_q[i]
    mujoco.mj_forward(model, data)
    start_pos = data.site_xpos[site_id].copy()

    # 在笛卡尔空间中生成中间路径点
    num_waypoints = 10
    waypoints = []
    for k in range(1, num_waypoints + 1):
        t = k / num_waypoints
        alpha = t * t * (3 - 2 * t)
        wp = start_pos + alpha * (target_pos - start_pos)
        waypoints.append(wp)

    # 对每个路径点求解 IK
    all_q = [start_q.tolist()]
    current_q = start_q.copy()

    for wp in waypoints:
        q = solve_ik(model, data, wp, site_id)
        if q is None:
            continue
        all_q.append(q.tolist())
        for i, jid in enumerate(joint_ids):
            data.qpos[jid] = q[i]
        current_q = q

    # 确保终点在内
    end_q = solve_ik(model, data, target_pos, site_id)
    if end_q is not None and not np.allclose(end_q, all_q[-1], atol=1e-3):
        all_q.append(end_q.tolist())

    if len(all_q) < 2:
        return [start_q.tolist()]

    # 均匀分配步数到各段
    segment_steps = steps // (len(all_q) - 1)
    trajectory = []

    for seg in range(len(all_q) - 1):
        q_start = np.array(all_q[seg])
        q_end = np.array(all_q[seg + 1])
        for i in range(segment_steps + 1):
            t = i / segment_steps
            alpha = t * t * (3 - 2 * t)
            q = q_start + alpha * (q_end - q_start)
            trajectory.append(q.tolist())

    while len(trajectory) < steps + 1:
        trajectory.append(trajectory[-1])

    return trajectory[:steps + 1]


# ==================== 保存/加载轨迹 ====================
def save_trajectory(trajectory):
    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.json"
    filepath = os.path.join("models", filename)
    with open(filepath, "w") as f:
        json.dump(trajectory, f)
    return filename


# ==================== 主函数 ====================
def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    joint_names = ["joint1", "joint2", "joint3"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    speed_deg_per_sec = 30.0

    print("=" * 60)
    print("三连杆机械臂 - IK 轨迹规划 vs 手动示教")
    print("  R：完整对比流程")
    print("  空格：随机目标 + IK 自动规划")
    print("  ESC：退出")
    print("=" * 60)

    if not glfw.init():
        return
    window = glfw.create_window(1200, 900, "MuJoCo IK Planning", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.08, 0, 0.08]
    cam.distance = 0.5
    cam.azimuth = 130
    cam.elevation = -25
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

    FIXED_TARGET = [-0.08, 0.03, 0.04]

    # ---- 状态机 ----
    state = "idle"
    trajectory = []
    manual_trajectory = None
    ik_trajectory = None
    replay_frame = 0
    replay_counter = 0
    replay_speed_manual = 2
    replay_speed_ik = 9
    timer_start = 0

    key_r_pressed = False
    key_space_pressed = False

    timestep = model.opt.timestep
    angle_increment = np.deg2rad(speed_deg_per_sec) * timestep

    last_x, last_y = 0, 0
    button_left, button_right, button_middle = False, False, False

    print("\n按 R 开始完整对比流程。\n")

    while not glfw.window_should_close(window):
        # ---- 鼠标视角 ----
        current_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        current_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        current_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        current_x, current_y = glfw.get_cursor_pos(window)

        if current_left and not button_left:
            last_x, last_y = current_x, current_y
        if current_right and not button_right:
            last_x, last_y = current_x, current_y
        if current_middle and not button_middle:
            last_x, last_y = current_x, current_y

        dx = current_x - last_x
        dy = current_y - last_y
        last_x, last_y = current_x, current_y

        if current_left:
            cam.azimuth += dx * 0.3
            cam.elevation -= dy * 0.3
            cam.elevation = max(-90, min(90, cam.elevation))
        if current_right:
            cam.lookat[0] -= dx * 0.0005 * cam.distance
            cam.lookat[1] += dy * 0.0005 * cam.distance
        if current_middle:
            cam.distance += dy * 0.01 * cam.distance
            cam.distance = max(0.05, min(2.0, cam.distance))

        button_left, button_right, button_middle = current_left, current_right, current_middle

        ctrl_pressed = (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                        glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)

        # ---- R 键 ----
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            if not key_r_pressed and state == "idle":
                model.body_pos[target_body_id] = FIXED_TARGET
                mujoco.mj_forward(model, data)
                data.ctrl[0] = 0
                data.ctrl[1] = 0
                data.ctrl[2] = 0
                mujoco.mj_step(model, data)
                state = "manual_record"
                trajectory = []
                init_qpos = [data.qpos[joint_ids[i]] for i in range(3)]
                trajectory.append(init_qpos)
                print("\n" + "=" * 40)
                print("🔴 阶段1：手动录制。请用箭头键操控机械臂触碰目标！")
                print("   触碰目标后自动停止。")
            key_r_pressed = True
        else:
            key_r_pressed = False

        # ---- 空格键 ----
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            if not key_space_pressed and state == "idle":
                max_radius = 0.12
                min_radius = 0.04
                base_center = np.array([0.0, 0.0, 0.10])
                while True:
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(-np.pi/6, np.pi/3)
                    r = np.random.uniform(min_radius, max_radius)
                    new_x = base_center[0] + r * np.cos(phi) * np.cos(theta)
                    new_y = base_center[1] + r * np.cos(phi) * np.sin(theta)
                    new_z = base_center[2] + r * np.sin(phi)
                    if new_z > 0.04 and np.linalg.norm([new_x, new_y]) > 0.04:
                        break

                model.body_pos[target_body_id] = [new_x, new_y, new_z]
                mujoco.mj_forward(model, data)
                target_pos = data.geom_xpos[target_geom_id].copy()

                for jid in joint_ids:
                    data.qpos[jid] = 0
                mujoco.mj_forward(model, data)

                start_q = np.zeros(3)
                ik_trajectory = plan_trajectory(model, data, start_q, target_pos, ee_site_id, steps=100)
                if ik_trajectory is not None and len(ik_trajectory) > 1:
                    state = "replay_ik"
                    replay_frame = 0
                    replay_counter = 0
                    print(f"\n✅ IK 规划成功 → 演示自动轨迹")
                else:
                    print("❌ IK 规划失败")
            key_space_pressed = True
        else:
            key_space_pressed = False

        # ---- ESC ----
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # ============ 状态处理 ============

        # ---- 手动录制 ----
        if state == "manual_record":
            if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
                data.ctrl[0] = data.qpos[joint_ids[0]] + angle_increment
            if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
                data.ctrl[0] = data.qpos[joint_ids[0]] - angle_increment
            if ctrl_pressed:
                if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                    data.ctrl[2] = data.qpos[joint_ids[2]] + angle_increment
                if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                    data.ctrl[2] = data.qpos[joint_ids[2]] - angle_increment
            else:
                if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                    data.ctrl[1] = data.qpos[joint_ids[1]] + angle_increment
                if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                    data.ctrl[1] = data.qpos[joint_ids[1]] - angle_increment

        # ---- 演示手动轨迹 ----
        elif state == "replay_manual":
            if replay_frame < len(manual_trajectory):
                target_qpos = manual_trajectory[replay_frame]
                for i in range(3):
                    data.ctrl[i] = target_qpos[i]
                replay_counter += 1
                if replay_counter >= replay_speed_manual:
                    replay_counter = 0
                    replay_frame += 1
            else:
                if timer_start == 0:
                    timer_start = glfw.get_time()
                    print("   手动轨迹演示完毕，即将演示 IK 轨迹...")
                if glfw.get_time() - timer_start > 1.5:
                    for jid in joint_ids:
                        data.qpos[jid] = 0
                    mujoco.mj_forward(model, data)
                    target_pos = data.geom_xpos[target_geom_id].copy()
                    start_q = np.zeros(3)
                    ik_trajectory = plan_trajectory(model, data, start_q, target_pos, ee_site_id, steps=100)
                    if ik_trajectory is not None and len(ik_trajectory) > 1:
                        state = "replay_ik"
                        replay_frame = 0
                        replay_counter = 0
                        timer_start = 0
                        print(f"   阶段3：演示 IK 轨迹（{len(ik_trajectory)} 帧）")
                    else:
                        print("❌ IK 规划失败")
                        state = "idle"

        # ---- 演示 IK 轨迹 ----
        elif state == "replay_ik":
            if replay_frame < len(ik_trajectory):
                target_qpos = ik_trajectory[replay_frame]
                for i in range(3):
                    data.ctrl[i] = target_qpos[i]
                replay_counter += 1
                if replay_counter >= replay_speed_ik:
                    replay_counter = 0
                    replay_frame += 1
            else:
                ee_pos = data.site_xpos[ee_site_id]
                target_pos = data.geom_xpos[target_geom_id]
                dist = np.linalg.norm(ee_pos - target_pos)
                print(f"   IK 轨迹完成，末端距目标：{dist:.4f}m")
                print("=" * 40 + "\n")
                state = "idle"

        # ---- 物理步进 ----
        mujoco.mj_step(model, data)

        # ---- 录制：去重记录 ----
        if state == "manual_record":
            current_qpos = [data.qpos[joint_ids[i]] for i in range(3)]
            if len(trajectory) == 0 or not np.allclose(current_qpos, trajectory[-1], atol=1e-3):
                trajectory.append(current_qpos)

        # ---- 触碰检测 ----
        if state == "manual_record":
            ee_pos = data.site_xpos[ee_site_id]
            target_pos = data.geom_xpos[target_geom_id]
            distance = np.linalg.norm(ee_pos - target_pos)
            if distance < 0.02:
                manual_trajectory = trajectory.copy()
                # save_trajectory(manual_trajectory)
                trajectory = []
                print(f"✅ 触碰目标！手动轨迹已保存（{len(manual_trajectory)} 帧）")
                print("   阶段2：演示手动录制的轨迹...")
                data.ctrl[0] = 0
                data.ctrl[1] = 0
                data.ctrl[2] = 0
                mujoco.mj_step(model, data)
                state = "replay_manual"
                replay_frame = 0
                replay_counter = 0
                timer_start = 0

        # ---- 渲染 ----
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()
    print("程序已退出。")


if __name__ == "__main__":
    main()