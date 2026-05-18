"""
里程碑二：示教轨迹记录与复现。
目标：录制手动控制的关节状态序列，自动复现。

操作说明：
  手动控制：←/→ 关节1  ↑/↓ 关节2  Ctrl+↑/↓ 关节3
  R：开始录制（末端触碰目标时自动停止）
  P：复现最近一次录制的轨迹
  空格：复位
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

    <!-- 固定目标位置 -->
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


def save_trajectory(trajectory):
    """保存轨迹，按时间戳自动命名"""
    os.makedirs("models", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{timestamp}.json"
    filepath = os.path.join("models", filename)
    with open(filepath, "w") as f:
        json.dump(trajectory, f)
    print(f"轨迹已保存：{filepath}（共 {len(trajectory)} 个状态点）")
    return filename


def load_latest_trajectory():
    """列出所有轨迹文件，按状态点数量升序排列（最短最优）"""
    if not os.path.exists("models"):
        print("轨迹目录不存在")
        return None, None

    files = [f for f in os.listdir("models") if f.startswith("trajectory_") and f.endswith(".json")]
    if not files:
        print("没有找到轨迹文件")
        return None, None

    # 读取所有轨迹并获取长度
    file_info = []
    for fname in files:
        filepath = os.path.join("models", fname)
        with open(filepath, "r") as f:
            traj = json.load(f)
        file_info.append((fname, len(traj), traj))

    # 按状态点数量升序排列（越短越优）
    file_info.sort(key=lambda x: x[1])

    print("\n轨迹排行榜（按状态点数升序，越短越优）：")
    for rank, (fname, length, traj) in enumerate(file_info, 1):
        # 计算得分：状态点越少得分越高
        max_points = max(info[1] for info in file_info)
        min_points = min(info[1] for info in file_info)
        if max_points == min_points:
            score = 100
        else:
            # 归一化到 0-100 分，越短分越高
            score = 100 * (1 - (length - min_points) / (max_points - min_points))
        print(f"  {rank}. [{score:.0f}分] {fname}（{length} 个状态点）")

    # 默认选排名第一（最短的）
    best = file_info[0]
    print(f"\n自动选择最优轨迹：{best[0]}（{best[1]} 个状态点）")
    return best[2], best[0]


def main():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    joint_names = ["joint1", "joint2", "joint3"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    speed_deg_per_sec = 30.0

    print("=" * 50)
    print("三连杆机械臂 - 示教轨迹记录与复现")
    print("  R：开始录制（触碰目标自动停止）")
    print("  P：复现最近一次轨迹（流畅回放，忽略停顿）")
    print("  ←/→：关节1  ↑/↓：关节2  Ctrl+↑/↓：关节3")
    print("  空格：复位  ESC：退出")
    print("=" * 50)

    if not glfw.init():
        return
    window = glfw.create_window(1200, 900, "MuJoCo 示教与复现", None, None)
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

    # 状态变量
    trajectory = []
    recorded_trajectory = None
    is_recording = False
    is_replaying = False
    replay_frame = 0
    replay_step_counter = 0

    key_r_pressed = False
    key_p_pressed = False

    timestep = model.opt.timestep
    angle_increment = np.deg2rad(speed_deg_per_sec) * timestep

    # 鼠标变量
    last_x, last_y = 0, 0
    button_left, button_right, button_middle = False, False, False

    # 回放速度：每 N 个物理步进切换一帧
    replay_speed = 3

    print("\n目标已固定。按 R 开始录制，操作机械臂触碰目标。\n")

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

        # ---- 键盘控制 ----
        ctrl_pressed = (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                        glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)

                # R 键：开始录制
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            if not key_r_pressed and not is_recording and not is_replaying:
                # 复位并执行一步物理，确保状态同步
                data.ctrl[0] = 0
                data.ctrl[1] = 0
                data.ctrl[2] = 0
                mujoco.mj_step(model, data)  # 先步进一步让状态生效
                is_recording = True
                trajectory = []
                # 立即记录初始状态
                init_qpos = [data.qpos[joint_ids[i]] for i in range(3)]
                trajectory.append(init_qpos)
                print("🔴 开始录制！操作机械臂触碰目标...")
            key_r_pressed = True
        else:
            key_r_pressed = False

        # P 键：复现轨迹
        if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
            if not key_p_pressed and not is_recording:
                traj, fname = load_latest_trajectory()
                if traj is not None:
                    recorded_trajectory = traj
                    is_replaying = True
                    replay_frame = 0
                    replay_step_counter = 0
                    data.ctrl[0] = 0
                    data.ctrl[1] = 0
                    data.ctrl[2] = 0
                    print(f"▶ 开始复现：{fname}")
            key_p_pressed = True
        else:
            key_p_pressed = False

        # 手动控制（仅在非复现模式下有效）
        if not is_replaying:
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

            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                data.ctrl[0] = 0
                data.ctrl[1] = 0
                data.ctrl[2] = 0
        else:
            # 复现模式：固定速度逐帧回放
            if replay_frame < len(recorded_trajectory):
                target_qpos = recorded_trajectory[replay_frame]
                for i in range(3):
                    data.ctrl[i] = target_qpos[i]
                replay_step_counter += 1
                if replay_step_counter >= replay_speed:
                    replay_step_counter = 0
                    replay_frame += 1
            else:
                is_replaying = False
                print("复现完成。")

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # ---- 物理步进 ----
        mujoco.mj_step(model, data)

        # ---- 录制：去重记录 ----
        if is_recording:
            current_qpos = [data.qpos[joint_ids[i]] for i in range(3)]
            if len(trajectory) == 0 or not np.allclose(current_qpos, trajectory[-1], atol=1e-3):
                trajectory.append(current_qpos)

        # ---- 触碰检测（仅在录制模式下） ----
        if is_recording:
            ee_pos = data.site_xpos[ee_site_id]
            target_pos = data.geom_xpos[target_geom_id]
            distance = np.linalg.norm(ee_pos - target_pos)
            if distance < 0.02:
                is_recording = False
                recorded_trajectory = trajectory.copy()
                save_trajectory(recorded_trajectory)
                trajectory = []
                print("✅ 触碰目标，录制自动停止！按 P 复现轨迹。")

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