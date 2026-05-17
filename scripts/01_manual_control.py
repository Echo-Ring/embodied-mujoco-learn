"""
里程碑一：手动键盘控制。
目标：用键盘控制 MuJoCo 中的三自由度机械臂，建立关节-连杆-末端的心智模型。

操作说明：
  ←/→ 箭头：控制关节1（底座旋转）正转/反转
  ↑/↓ 箭头：控制关节2（大臂俯仰）正转/反转
  Ctrl+↑/↓：控制关节3（小臂俯仰）正转/反转
  空格：复位所有关节
  ESC：退出

鼠标操作：
  左键拖拽：旋转视角
  右键拖拽：平移视角
  中键/滚轮：缩放视角

游戏规则：
  控制机械臂末端（红色小球）触碰绿色目标方块得分。
"""

import mujoco
import numpy as np
import glfw

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

    <!-- 底座 -->
    <body name="base" pos="0 0 0.02">
      <geom name="base_geom" type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>

      <!-- 连杆1：竖直向上，关节1绕Z轴旋转 -->
      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180" damping="2"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.05" size="0.012" material="mat_link1"/>

        <!-- 连杆2：从连杆1顶端开始 -->
        <body name="link2" pos="0 0 0.05">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-150 150" damping="1.5"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.05 0 0" size="0.01" material="mat_link2"/>

          <!-- 连杆3 -->
          <body name="link3" pos="0.05 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-150 150" damping="1.0"/>
            <geom name="link3_geom" type="capsule" fromto="0 0 0 0.05 0 0" size="0.008" material="mat_link3"/>

            <!-- 末端标记点 -->
            <site name="ee_site" pos="0.05 0 0" type="sphere" size="0.009" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>

    <!-- 目标方块（静态 body，通过代码移动） -->
    <body name="target" pos="0.08 0.04 0.1">
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


# ==================== 主函数 ====================
def main():
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    # 2. 关节索引和转速
    joint_names = ["joint1", "joint2", "joint3"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    speed_deg_per_sec = 30.0

    print("=" * 50)
    print("三连杆机械臂 - 键盘手动控制（按住连续运动）")
    print("  ←/→ 箭头：关节1（底座旋转）")
    print("  ↑/↓ 箭头：关节2（大臂俯仰）")
    print("  Ctrl+↑/↓：关节3（小臂俯仰）")
    print("  空格：复位所有关节")
    print("  ESC：退出")
    print("=" * 50)

    # 3. 初始化 glfw 窗口
    if not glfw.init():
        print("无法初始化 GLFW")
        return

    window = glfw.create_window(1200, 900, "MuJoCo 机械臂控制", None, None)
    if not window:
        glfw.terminate()
        print("无法创建窗口")
        return

    glfw.make_context_current(window)

    # 4. 初始化 MuJoCo 可视化
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

    # 5. 获取 ID
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

    score = 0
    reached = False
    contact_frames = 0

    print("\n查看器已打开，请用箭头键控制机械臂。")
    print("用末端红色小球触碰绿色方块得分！\n")

    # 6. 主循环
    timestep = model.opt.timestep
    angle_increment = np.deg2rad(speed_deg_per_sec) * timestep

    # 鼠标交互变量
    last_x, last_y = 0, 0
    button_left = False
    button_right = False
    button_middle = False

    while not glfw.window_should_close(window):
        # ---- 鼠标视角控制 ----
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

        button_left = current_left
        button_right = current_right
        button_middle = current_middle

        # ---- 键盘控制 ----
        # ←/→ 箭头：关节1（底座旋转）
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            data.ctrl[0] = data.qpos[joint_ids[0]] + angle_increment
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            data.ctrl[0] = data.qpos[joint_ids[0]] - angle_increment

        

        # Ctrl + ↑/↓：关节3（小臂俯仰）
        ctrl_pressed = (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                        glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
        if ctrl_pressed:
            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                data.ctrl[2] = data.qpos[joint_ids[2]] + angle_increment
            if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                data.ctrl[2] = data.qpos[joint_ids[2]] - angle_increment
        # ↑/↓ 箭头：关节2（大臂俯仰）
        else:
            if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
                data.ctrl[1] = data.qpos[joint_ids[1]] + angle_increment
            if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
                data.ctrl[1] = data.qpos[joint_ids[1]] - angle_increment

        # 空格：复位
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            data.ctrl[0] = 0
            data.ctrl[1] = 0
            data.ctrl[2] = 0
            print("关节已复位到零位")

        # ESC：退出
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # ---- 物理步进 ----
        mujoco.mj_step(model, data)

        # ---- 碰撞检测 ----
        ee_pos = data.site_xpos[ee_site_id]
        target_pos = data.geom_xpos[target_geom_id]
        distance = np.linalg.norm(ee_pos - target_pos)
        touch_threshold = 0.015

        if distance < touch_threshold:
            contact_frames += 1
            if contact_frames == 3 and not reached:
                reached = True
                score += 1
                print(f"✅ 触碰目标！得分：{score}")

                # 复位关节
                data.ctrl[0] = 0
                data.ctrl[1] = 0
                data.ctrl[2] = 0

                # 在机械臂工作空间内随机生成新目标
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

        else:
            contact_frames = 0
            reached = False

        # ---- 渲染 ----
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    print(f"\n游戏结束，最终得分：{score}")
    glfw.terminate()
    print("程序已退出。")


if __name__ == "__main__":
    main()