from unittest import TestCase
import time
import numpy as np
import mujoco
import mujoco.viewer

from src.model import Skydio
from src.controller import FormationController
from src.formation import LineFormation
from src.parameter import Parameter
from src.motion_planning import JointParameter, TrajectoryParameter, TrajectoryPlanner, QuinticVelocityParameter


class TestProcess(TestCase):
    def test_independent_to_formation(self):
        # -------------------------- 1. 加载仿真模型 --------------------------
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene4.xml")
        data = mujoco.MjData(model)
        dt = model.opt.timestep  # 仿真步长（0.01s）

        # 初始化状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # -------------------------- 2. 初始化参数 --------------------------
        skydio = Skydio()
        count = 4  # 4架无人机
        kps = [1, 0.5, 0.5, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]

        # 编队参数（直线编队，间距0.8m）
        formation = LineFormation(2, count)
        # 控制器（初始为独立模式）
        formation_controller = FormationController(
            kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0, use_formation=False
        )

        # -------------------------- 3. 轨迹规划（三阶段参数）--------------------------
        # 阶段1：独立运动时间（各无人机执行原有轨迹）
        independent_time = 15  
        # 阶段2：聚拢时间（从独立轨迹过渡到编队初始位置）
        gather_time = 10  
        # 阶段3：编队运动时间
        formation_duration = 20  
        total_simulation_time = independent_time + gather_time + formation_duration

        # -------------------------- 4. 独立模式轨迹（复用原有逻辑）--------------------------
        trajectory_planners_list = []  # 存储每架无人机的独立轨迹规划器

        # 无人机0: 螺旋上升
        init_rotate_time1 = 5
        q0_1 = np.array([0.0, 0.0, 0.0, 0.0])
        q1_1 = q0_1 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init1 = JointParameter(q0_1, q1_1)
        vel_param_init1 = QuinticVelocityParameter(init_rotate_time1)
        traj_param_init1 = TrajectoryParameter(joint_param_init1, vel_param_init1)
        trajectory_planner_init1 = TrajectoryPlanner(traj_param_init1)

        radius_init1 = 1.0
        z_final1 = 0.6
        total_angle1 = 2 * 3 * np.pi
        phase_count1 = 3000 * 3
        phase_time1 = 0.005
        trajectory_planners1 = [trajectory_planner_init1]
        waypoints1 = [q1_1]
        for i in range(phase_count1):
            angle = i * (total_angle1 / phase_count1)
            radius = radius_init1 - (radius_init1 / phase_count1) * i
            x = q0_1[0] + radius * np.cos(angle)
            y = q0_1[1] + radius * np.sin(angle)
            z = q0_1[2] + (z_final1 / phase_count1) * i
            yaw = angle + np.pi
            waypoints1.append(np.array([x, y, z, yaw]))
            joint_param = JointParameter(waypoints1[i], waypoints1[i + 1])
            vel_param = QuinticVelocityParameter(phase_time1)
            trajectory_planners1.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))
        trajectory_planners_list.append(trajectory_planners1)

        # 无人机1: 圆形飞行（作为基准机，编队阶段继续此轨迹）
        z_init2 = 0.3
        init_rotate_time2 = 5
        q0_2 = np.array([0.0, 0.0, z_init2, 0.0])
        q1_2 = q0_2 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init2 = JointParameter(q0_2, q1_2)
        vel_param_init2 = QuinticVelocityParameter(init_rotate_time2)
        trajectory_planner_init2 = TrajectoryPlanner(TrajectoryParameter(joint_param_init2, vel_param_init2))

        radius_xy2 = 1.4
        radius_z2 = 0.8
        total_angle2 = 2 * 3 * np.pi
        phase_count2 = 3000 * 3
        phase_time2 = 0.005
        trajectory_planners2 = [trajectory_planner_init2]
        waypoints2 = [q1_2]
        for i in range(phase_count2):
            angle = i * (total_angle2 / phase_count2)
            x = q0_2[0] + radius_xy2 * np.cos(angle)
            y = q0_2[1] + radius_xy2 * np.sin(angle)
            z = radius_z2 * np.sin(4 * angle)
            yaw = angle + np.pi
            waypoints2.append(np.array([x, y, z, yaw]))
            joint_param = JointParameter(waypoints2[i], waypoints2[i + 1])
            vel_param = QuinticVelocityParameter(phase_time2)
            trajectory_planners2.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))
        trajectory_planners_list.append(trajectory_planners2)

        # 无人机2: 八字飞行
        z_init3 = 0.1
        init_rotate_time3 = 5
        q0_3 = np.array([0.0, 0.0, z_init3, 0.0])
        q1_3 = q0_3 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init3 = JointParameter(q0_3, q1_3)
        vel_param_init3 = QuinticVelocityParameter(init_rotate_time3)
        trajectory_planner_init3 = TrajectoryPlanner(TrajectoryParameter(joint_param_init3, vel_param_init3))

        radius_xy3 = 0.6
        radius_z3 = 0.3
        circle_steps_per3 = 4500
        phase_count3 = circle_steps_per3 * 2
        phase_time3 = 0.005
        trajectory_planners3 = [trajectory_planner_init3]
        waypoints3 = [q1_3]
        total_z_cycle3 = 2 * np.pi
        for i in range(phase_count3):
            if i <= circle_steps_per3:
                angle = i * (2 * np.pi / circle_steps_per3)
            else:
                angle = 2 * np.pi - (i - circle_steps_per3) * (2 * np.pi / circle_steps_per3)
            x = q0_3[0] + radius_xy3 * np.cos(angle)
            y = q0_3[1] + radius_xy3 * np.sin(angle)
            z_angle = (i / phase_count3) * total_z_cycle3
            z = q0_3[2] + radius_z3 * np.cos(z_angle)
            yaw = angle + np.pi
            waypoints3.append(np.array([x, y, z, yaw]))
            joint_param = JointParameter(waypoints3[i], waypoints3[i + 1])
            vel_param = QuinticVelocityParameter(phase_time3)
            trajectory_planners3.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))
        trajectory_planners_list.append(trajectory_planners3)

        # 无人机3: 圆形飞行
        z_init4 = 0.1
        init_rotate_time4 = 5
        q0_4 = np.array([0.0, -0.1, z_init4, 0.0])
        q1_4 = q0_4 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init4 = JointParameter(q0_4, q1_4)
        vel_param_init4 = QuinticVelocityParameter(init_rotate_time4)
        trajectory_planner_init4 = TrajectoryPlanner(TrajectoryParameter(joint_param_init4, vel_param_init4))

        radius4 = 1
        y_final4 = 0.6
        total_angle4 = 2 * 5 * np.pi
        phase_count4 = 3000 * 3
        phase_time4 = 0.005
        trajectory_planners4 = [trajectory_planner_init4]
        waypoints4 = [q1_4]
        for i in range(phase_count4):
            angle = i * (total_angle4 / phase_count4)
            x = q0_4[0] + radius4 * np.cos(angle)
            y = q0_4[1] + (y_final4 / phase_count4) * i
            z = radius4 * np.sin(angle)
            yaw = angle + np.pi
            waypoints4.append(np.array([x, y, z, yaw]))
            joint_param = JointParameter(waypoints4[i], waypoints4[i + 1])
            vel_param = QuinticVelocityParameter(phase_time4)
            trajectory_planners4.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))
        trajectory_planners_list.append(trajectory_planners4)

        # -------------------------- 5. 聚拢阶段轨迹规划 --------------------------
        # 基准机选择：无人机1（索引1）保持独立轨迹终点，作为聚拢目标点
        base_uav = 1
        # 计算基准机在独立阶段结束时的位置（作为聚拢目标）
        base_end_idx = min(int(independent_time / phase_time2) + 1, len(waypoints2) - 1)
        base_target_pos = waypoints2[base_end_idx]  # [x,y,z,yaw]

        # 其他无人机的聚拢目标位置（围绕基准机形成初始小编队）
        # gather_targets = [
        #     base_target_pos + np.array([0, 0, 0, 0]),  # 无人机0：
        #     base_target_pos,  # 无人机1：基准机自身
        #     base_target_pos + np.array([0.0, 0, 0, 0]),  # 无人机2：
        #     base_target_pos + np.array([0.0, 0, 0, 0])   # 无人机3：
        # ]
        
        gather_targets = [
            np.array([0, 0, 0, 0]),  # 无人机0：
            np.array([0, 0, 0, 0]), 
            np.array([0, 0, 0, 0]), 
            np.array([0, 0, 0, 0]),
        ]
        # 计算各无人机在独立阶段结束时的位置（聚拢起点）
        gather_starts = []
        for uav_id in range(count):
            phase_time = [phase_time1, phase_time2, phase_time3, phase_time4][uav_id]
            waypoints = [waypoints1, waypoints2, waypoints3, waypoints4][uav_id]
            start_idx = min(int(independent_time / phase_time) + 1, len(waypoints) - 1)
            gather_starts.append(waypoints[start_idx])

        # 为每个无人机创建聚拢轨迹规划器
        gather_planners = []
        for uav_id in range(count):
            # 从独立终点到聚拢目标的平滑轨迹（五次多项式）
            joint_param = JointParameter(gather_starts[uav_id], gather_targets[uav_id])
            vel_param = QuinticVelocityParameter(gather_time)
            gather_planners.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))

        # -------------------------- 6. 编队阶段轨迹（复用基准机轨迹）--------------------------
        # 从基准机轨迹中提取编队阶段的轨迹段
        formation_start_idx = base_end_idx + int(gather_time / phase_time2)  # 聚拢结束时基准机的位置索引
        formation_trajectory = waypoints2[formation_start_idx: formation_start_idx + int(formation_duration / phase_time2)]
        formation_phase_count = len(formation_trajectory) - 1
        formation_phase_time = formation_duration / formation_phase_count if formation_phase_count > 0 else 0.005

        # -------------------------- 7. 仿真循环 --------------------------
        time_step_num = round(total_simulation_time / dt)
        times = np.linspace(0, total_simulation_time, time_step_num)
        qds_list = [np.zeros((time_step_num, 6)) for _ in range(count)]  # 目标位置存储

        time_num = 0
        # 阶段标志
        in_independent = True    # 独立模式
        in_gather = False        # 聚拢模式
        in_formation = False     # 编队模式
        gather_start_time = 0
        formation_start_time = 0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_simulation_time:
                step_start = time.time()
                current_time = data.time

                # -------------------------- 阶段切换逻辑 --------------------------
                if in_independent and current_time >= independent_time:
                    # 从独立模式切换到聚拢模式
                    in_independent = False
                    in_gather = True
                    gather_start_time = current_time
                    formation_controller.use_formation = False  # 聚拢阶段仍禁用编队
                    print(f"切换到聚拢模式，持续{gather_time}秒")

                if in_gather and current_time >= gather_start_time + gather_time:
                    # 从聚拢模式切换到编队模式
                    in_gather = False
                    in_formation = True
                    formation_start_time = current_time
                    formation_controller.use_formation = True  # 启用编队
                    print(f"切换到编队模式，持续{formation_duration}秒")

                # -------------------------- 轨迹更新 --------------------------
                if in_independent:
                    # 阶段1：执行独立轨迹
                    for uav_id in range(count):
                        planners = trajectory_planners_list[uav_id]
                        phase_time = [phase_time1, phase_time2, phase_time3, phase_time4][uav_id]
                        init_rotate_time = [init_rotate_time1, init_rotate_time2, init_rotate_time3, init_rotate_time4][uav_id]

                        if current_time < init_rotate_time:
                            joint_pos = planners[0].interpolate(current_time)
                        else:
                            phase_idx = int((current_time - init_rotate_time) / phase_time) + 1
                            phase_idx = min(phase_idx, len(planners) - 1)
                            local_time = current_time - (init_rotate_time + (phase_idx - 1) * phase_time)
                            joint_pos = planners[phase_idx].interpolate(local_time)
                        qds_list[uav_id][time_num, [0, 1, 2, 5]] = joint_pos

                elif in_gather:
                    # 阶段2：执行聚拢轨迹
                    local_time = current_time - gather_start_time
                    for uav_id in range(count):
                        joint_pos = gather_planners[uav_id].interpolate(local_time)
                        qds_list[uav_id][time_num, [0, 1, 2, 5]] = joint_pos

                else:
                    # 阶段3：执行编队轨迹（基准机为领航机）
                    local_time = current_time - formation_start_time
                    if local_time <= formation_duration and formation_phase_count > 0:
                        phase_idx = int(local_time / formation_phase_time)
                        phase_idx = min(phase_idx, formation_phase_count - 1)
                        t = local_time - phase_idx * formation_phase_time
                        q_prev = formation_trajectory[phase_idx]
                        q_next = formation_trajectory[phase_idx + 1]
                        joint_pos = q_prev + (t / formation_phase_time) * (q_next - q_prev)
                        qds_list[base_uav][time_num, [0, 1, 2, 5]] = joint_pos  # 仅设置领航机目标
                    else:
                        qds_list[base_uav][time_num, [0, 1, 2, 5]] = formation_trajectory[-1]

                # -------------------------- 控制逻辑 --------------------------
                # 设置目标位置
                for i in range(count):
                    if in_independent or in_gather:
                        # 独立和聚拢阶段：所有无人机使用自身目标
                        parameters[i].dposd = qds_list[i][time_num, [0, 1, 2]]
                        parameters[i].psid = qds_list[i][time_num, 5]
                    else:
                        # 编队阶段：仅设置领航机目标，跟随机由控制器计算
                        if i == base_uav:
                            parameters[i].dposd = qds_list[base_uav][time_num, [0, 1, 2]]
                            parameters[i].psid = qds_list[base_uav][time_num, 5]

                # 读取当前状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 计算控制力矩
                torques = formation_controller.control(parameters, formation)

                # 应用控制并仿真
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                # 可视化
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(current_time % 2)
                viewer.sync()

                # 控制帧率
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                time_num += 1
                if time_num >= time_step_num:
                    break
