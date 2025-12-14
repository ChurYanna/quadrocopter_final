from unittest import TestCase
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.model import Skydio
from src.controller import FormationController
from src.formation import LineFormation
from src.parameter import Parameter
from src.motion_planning import JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner


class TestCooperation(TestCase):
    """无人机螺旋上升（半径减小）运动测试"""
    def test_single_drone_spiral_motion1(self):
         # 加载1架无人机的仿真模型
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器
        skydio = Skydio()
        count = 1
        kps = [1.0, 1.0, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        formation = LineFormation(0, count)  
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        # 关键修改：添加初始旋转阶段（与test_formation_control_triangle_count3保持一致）
        # 阶段1：先旋转π角度（180度），耗时5秒，调整初始朝向
        init_rotate_time = 5  # 初始旋转时间（与同类测试保持一致）
        q0 = np.array([0.0, 0.0, 0.0, 0.0])  # 初始位置和姿态
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])  # 仅旋转yaw角180度
        joint_param_init = JointParameter(q0, q1) #关节位置参数
        vel_param_init = QuinticVelocityParameter(init_rotate_time)#五次多项式速度规划
        traj_param_init = TrajectoryParameter(joint_param_init, vel_param_init)#轨迹参数
        trajectory_planner_init = TrajectoryPlanner(traj_param_init)#轨迹插值

        # 圆形轨迹参数：在初始旋转后执行）+高度增加
        radius_init = 1
        radius = 1
        z_fixed = 0.0
        z_final = 1
        total_angle = 2*3* np.pi
        phase_count = 3000*3
        phase_time = 0.01
        circle_total_time = phase_count * phase_time  
        total_time = init_rotate_time + circle_total_time  # 总时间=初始旋转(5s)+圆形轨迹(32s)=37s

        # 生成圆形轨迹关键点
        waypoints = []
        for i in range(phase_count + 1):
            angle = i * (total_angle / phase_count)
            radius -= (radius_init/phase_count)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z_fixed += (z_final/phase_count)
            yaw = angle + np.pi  # 偏航角在初始旋转基础上叠加（保持朝前飞行）
            waypoints.append(np.array([x, y, z_fixed, yaw]))

        # 创建轨迹规划器列表（先添加初始旋转阶段，再添加圆形轨迹阶段）
        trajectory_planners = [trajectory_planner_init]  # 初始旋转作为第一个阶段
        for i in range(phase_count):
            joint_param = JointParameter(waypoints[i], waypoints[i+1])
            vel_param = QuinticVelocityParameter(phase_time)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners.append(TrajectoryPlanner(traj_param))

        # 初始化轨迹数据存储
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        # 计算每个时刻的目标位置（分阶段处理）
        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            # 阶段0：初始旋转（0~5秒）
            if timei < init_rotate_time:
                joint_position = trajectory_planners[0].interpolate(timei)
            # 阶段1~16：圆形轨迹（5秒后开始）
            else:
                circle_time = timei - init_rotate_time
                phase_idx = int(circle_time // phase_time)
                if phase_idx >= phase_count:
                    phase_idx = phase_count - 1
                local_time = circle_time % phase_time
                joint_position = trajectory_planners[phase_idx + 1].interpolate(local_time)  # +1跳过初始阶段
            qds[i, [0, 1, 2, 5]] = joint_position
        
        # 启动仿真可视化
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 设置领航机目标位置
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 更新无人机状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 计算控制力矩并执行仿真
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)


                # 可视化设置
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # 控制帧率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # 记录状态数据
                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q
                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break
    
    """无人机圆形飞行"""
    def test_single_drone_circle_motion2(self):
         # 加载1架无人机的仿真模型
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器
        skydio = Skydio()
        count = 1
        kps = [1.0, 1.0, 0.5, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        formation = LineFormation(0, count)  
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        # 阶段1：先旋转π角度（180度），耗时5秒，调整初始朝向
        z_init = 0.2
        init_rotate_time = 5  # 初始旋转时间（与同类测试保持一致）
        q0 = np.array([0.0, 0.0, z_init,0.0])  # 初始位置和姿态,z高度为0.3
        q1 = q0 + np.array([0.0, 0.0, 0, np.pi])  # 仅旋转yaw角180度
        joint_param_init = JointParameter(q0, q1) #关节位置参数
        vel_param_init = QuinticVelocityParameter(init_rotate_time)#五次多项式速度规划
        traj_param_init = TrajectoryParameter(joint_param_init, vel_param_init)#轨迹参数
        trajectory_planner_init = TrajectoryPlanner(traj_param_init)#轨迹插值

        #轨迹关键参数
        radius_xy = 2
        radius_z = 1
        total_angle = 2*3*np.pi
        phase_count = 3000*3
        phase_time = 0.005
        circle_total_time = phase_count * phase_time  
        total_time = init_rotate_time + circle_total_time

        #生成空间圆形轨迹关键点
        waypoints = []
        for i in range(phase_count + 1):
            angle = i * (total_angle / phase_count)
            x = radius_xy * np.cos(angle)
            y = radius_xy * np.sin(angle)
            z_fixed =radius_z*np.cos(4*angle) + z_init
            yaw = angle + np.pi  # 偏航角在初始旋转基础上叠加（保持朝前飞行）
            waypoints.append(np.array([x, y, z_fixed, yaw]))

        # 创建轨迹规划器列表（先添加初始旋转阶段，再添加圆形轨迹阶段）
        trajectory_planners = [trajectory_planner_init]  # 初始旋转作为第一个阶段
        for i in range(phase_count):
            joint_param = JointParameter(waypoints[i], waypoints[i+1])
            vel_param = QuinticVelocityParameter(phase_time)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners.append(TrajectoryPlanner(traj_param))

        # 初始化轨迹数据存储
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        # 计算每个时刻的目标位置（分阶段处理）
        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            # 阶段0：初始旋转（0~5秒）
            if timei < init_rotate_time:
                joint_position = trajectory_planners[0].interpolate(timei)
            # 阶段1~N：圆形轨迹（5秒后开始）
            else:
                circle_time = timei - init_rotate_time
                phase_idx = int(circle_time // phase_time)
                if phase_idx >= phase_count:
                    phase_idx = phase_count - 1
                local_time = circle_time % phase_time
                joint_position = trajectory_planners[phase_idx + 1].interpolate(local_time)  # +1跳过初始阶段
            qds[i, [0, 1, 2, 5]] = joint_position

        # 启动仿真可视化
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 设置领航机目标位置
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 更新无人机状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 计算控制力矩并执行仿真
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)


                # 可视化设置
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # 控制帧率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # 记录状态数据
                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q
                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    """无人机八字飞行"""
    def test_single_drone_bazi_motion3(self):
        # 加载1架无人机的仿真模型
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器
        skydio = Skydio()
        count = 1
        kps = [1.0, 1.0, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        formation = LineFormation(0, count)  
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        # 阶段1：先旋转π角度（180度），耗时5秒，调整初始朝向
        init_rotate_time = 5  # 初始旋转时间（与同类测试保持一致）
        q0 = np.array([0.0, 0.0, 0.0 ,0.0])  # 初始位置和姿态,z高度为0.3
        q1 = q0 + np.array([0.0, 0.0, 0.3, np.pi])  # 仅旋转yaw角180度
        joint_param_init = JointParameter(q0, q1) #关节位置参数
        vel_param_init = QuinticVelocityParameter(init_rotate_time)#五次多项式速度规划
        traj_param_init = TrajectoryParameter(joint_param_init, vel_param_init)#轨迹参数
        trajectory_planner_init = TrajectoryPlanner(traj_param_init)#轨迹插值\

        # -------------------------- 轨迹关键参数（按需求配置）--------------------------
        radius_xy = 1.0  # XY平面圆半径（与原圆形飞行一致，保证8字大小）
        radius_z = 0.6   # Z轴sin波动振幅（与原一致）
        circle_steps_per = 3000  # 每圈步数（按需求设置）
        phase_count = circle_steps_per * 2  # 总步数：正向1圈+反向1圈=6000步
        phase_time = 0.005  # 每步时间（与原一致，保证平滑度）
        circle_total_time = phase_count * phase_time  # 8字轨迹总时间：6000×0.005=30秒
        total_time = init_rotate_time + circle_total_time  # 仿真总时间：5+30=35秒

        # -------------------------- 生成8字轨迹waypoints（正向逆时针+反向顺时针）--------------------------
        waypoints = []
        total_z_cycle = 2 * np.pi  # z轴一个完整周期（对应两圈轨迹）

        for i in range(phase_count + 1):
            # 分阶段计算角度：0~3000步（正向逆时针），3001~6000步（反向顺时针）
            if i <= circle_steps_per:
                # 第一阶段：正向逆时针（角度0→2π，递增）
                angle = i * (2 * np.pi / circle_steps_per)
            else:
                # 第二阶段：反向顺时针（角度2π→0，递减），i从3001开始对应角度2π - 步进度
                angle = 2 * np.pi - (i - circle_steps_per) * (2 * np.pi / circle_steps_per)

            # 计算XY坐标（正向/反向角度直接驱动，形成两个交叉圆，构成8字）
            x = radius_xy * np.cos(angle)
            y = radius_xy * np.sin(angle)

            # 计算Z轴坐标：两圈对应sin一个完整周期（0→2π）
            z_angle = (i / phase_count) * total_z_cycle  # 按总步数均匀分配z轴周期
            z_fixed = radius_z * np.cos(z_angle) + 0.3  # 叠加初始高度0.3，与旋转后状态对齐

            # 计算yaw角：始终朝向当前飞行方向（与角度同步，正向递增/反向递减）
            yaw = angle + np.pi  # 保持与原圆形飞行一致的朝向逻辑，确保朝前飞行

            # 添加waypoint（x,y,z,yaw）
            waypoints.append(np.array([x, y, z_fixed, yaw]))


        # 创建轨迹规划器列表（先添加初始旋转阶段，再添加圆形轨迹阶段）
        trajectory_planners = [trajectory_planner_init]  # 初始旋转作为第一个阶段
        for i in range(phase_count):
            joint_param = JointParameter(waypoints[i], waypoints[i+1])
            vel_param = QuinticVelocityParameter(phase_time)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners.append(TrajectoryPlanner(traj_param))

        # 初始化轨迹数据存储
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        # 计算每个时刻的目标位置（分阶段处理）
        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            # 阶段0：初始旋转（0~5秒）
            if timei < init_rotate_time:
                joint_position = trajectory_planners[0].interpolate(timei)
            # 阶段1~N：圆形轨迹（5秒后开始）
            else:
                circle_time = timei - init_rotate_time
                phase_idx = int(circle_time // phase_time)
                if phase_idx >= phase_count:
                    phase_idx = phase_count - 1
                local_time = circle_time % phase_time
                joint_position = trajectory_planners[phase_idx + 1].interpolate(local_time)  # +1跳过初始阶段
            qds[i, [0, 1, 2, 5]] = joint_position

        # 启动仿真可视化
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 设置领航机目标位置
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 更新无人机状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 计算控制力矩并执行仿真
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)


                # 可视化设置
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # 控制帧率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # 记录状态数据
                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q
                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    def test_single_drone_circle_motion4(self):
        # 加载1架无人机的仿真模型
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器
        skydio = Skydio()
        count = 1
        kps = [1.0, 1.0, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        formation = LineFormation(0, count)  
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        # 阶段1：先旋转π角度（180度），耗时5秒，调整初始朝向
        init_rotate_time = 5  # 初始旋转时间（与同类测试保持一致）
        q0 = np.array([0.0, 0.0, 0.0, 0.0])  # 初始位置和姿态
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])  # 仅旋转yaw角180度
        joint_param_init = JointParameter(q0, q1) #关节位置参数
        vel_param_init = QuinticVelocityParameter(init_rotate_time)#五次多项式速度规划
        traj_param_init = TrajectoryParameter(joint_param_init, vel_param_init)#轨迹参数
        trajectory_planner_init = TrajectoryPlanner(traj_param_init)#轨迹插值

        #圆形轨迹参数
        radius = 1
        y = 0
        y_final = 0.6
        total_angle = 2*3* np.pi
        phase_count = 3000*3
        phase_time = 0.005
        circle_total_time = phase_count * phase_time  
        total_time = init_rotate_time + circle_total_time  # 总时间=初始旋转(5s)+圆形轨迹(32s)=37s

        #生成圆形轨迹关键点
        waypoints = []
        for i in range(phase_count + 1):
            angle = i * (total_angle / phase_count)
            x = radius * np.cos(angle)
            y += (y_final/phase_count)
            z = radius * np.sin(angle)
            yaw = angle + np.pi  # 偏航角在初始旋转基础上叠加（保持朝前飞行）
            waypoints.append(np.array([x, y, z, yaw]))

        # 创建轨迹规划器列表（先添加初始旋转阶段，再添加圆形轨迹阶段）
        trajectory_planners = [trajectory_planner_init]  # 初始旋转作为第一个阶段
        for i in range(phase_count):
            joint_param = JointParameter(waypoints[i], waypoints[i+1])
            vel_param = QuinticVelocityParameter(phase_time)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners.append(TrajectoryPlanner(traj_param))

        # 初始化轨迹数据存储
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        # 计算每个时刻的目标位置（分阶段处理）
        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            # 阶段0：初始旋转（0~5秒）
            if timei < init_rotate_time:
                joint_position = trajectory_planners[0].interpolate(timei)
            # 阶段1~16：圆形轨迹（5秒后开始）
            else:
                circle_time = timei - init_rotate_time
                phase_idx = int(circle_time // phase_time)
                if phase_idx >= phase_count:
                    phase_idx = phase_count - 1
                local_time = circle_time % phase_time
                joint_position = trajectory_planners[phase_idx + 1].interpolate(local_time)  # +1跳过初始阶段
            qds[i, [0, 1, 2, 5]] = joint_position
        
        # 启动仿真可视化
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 设置领航机目标位置
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 更新无人机状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 计算控制力矩并执行仿真
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)


                # 可视化设置
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # 控制帧率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # 记录状态数据
                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q
                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    def test_cooperation(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene4.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器（4架无人机）
        skydio = Skydio()
        count = 4
        kps = [1, 0.5, 0.5, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.1, 0.1, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        # 使用LineFormation但间距为0，实现独立控制
        formation = LineFormation(1.0, count)
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0,use_formation=False)

        # 为每架无人机创建轨迹规划器（复用四个单体动作的轨迹逻辑）
        trajectory_planners_list = []
        total_times = []
        phase_counts = []
        phase_times = []
        init_rotate_times = []

        #----------------------------------------------------------------------------------

        # 无人机1: 螺旋上升（radius减小）
        # 阶段1：初始旋转
        init_rotate_time1 = 5
        q0_1 = np.array([0.0, 0.0, 0.0, 0.0])  # 初始位置错开，避免碰撞
        q1_1 = q0_1 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init1 = JointParameter(q0_1, q1_1)
        vel_param_init1 = QuinticVelocityParameter(init_rotate_time1)
        traj_param_init1 = TrajectoryParameter(joint_param_init1, vel_param_init1)
        trajectory_planner_init1 = TrajectoryPlanner(traj_param_init1)

            # 螺旋轨迹参数
        radius_init1 = 1.0
        z_final1 = 0.6
        total_angle1 = 2*3*np.pi
        phase_count1 = 3000*3
        phase_time1 = 0.005
        circle_total_time1 = phase_count1 * phase_time1
        total_time1 = init_rotate_time1 + circle_total_time1

        # 生成螺旋轨迹关键点
        waypoints1 = []
        for i in range(phase_count1 + 1):
            angle = i * (total_angle1 / phase_count1)
            radius = radius_init1 - (radius_init1/phase_count1)*i
            x = q0_1[0] + radius * np.cos(angle)
            y = q0_1[1] + radius * np.sin(angle)
            z = q0_1[2] + (z_final1/phase_count1)*i
            yaw = angle + np.pi
            waypoints1.append(np.array([x, y, z, yaw]))

        # 创建轨迹规划器列表
        trajectory_planners1 = [trajectory_planner_init1]
        for i in range(phase_count1):
            joint_param = JointParameter(waypoints1[i], waypoints1[i+1])
            vel_param = QuinticVelocityParameter(phase_time1)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners1.append(TrajectoryPlanner(traj_param))

        trajectory_planners_list.append(trajectory_planners1)
        total_times.append(total_time1)
        phase_counts.append(phase_count1)
        phase_times.append(phase_time1)
        init_rotate_times.append(init_rotate_time1)


        #--------------------------------------------------------------------------------------------

        # 无人机2: 圆形飞行（motion2）
        # 阶段1：初始旋转
        z_init2 = 0.3
        init_rotate_time2 = 5
        q0_2 = np.array([0.0, 0.0, z_init2, 0.0])  # 初始位置错开
        q1_2 = q0_2 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init2 = JointParameter(q0_2, q1_2)
        vel_param_init2 = QuinticVelocityParameter(init_rotate_time2)
        traj_param_init2 = TrajectoryParameter(joint_param_init2, vel_param_init2)
        trajectory_planner_init2 = TrajectoryPlanner(traj_param_init2)

        # 圆形轨迹参数
        radius_xy2 = 1.4
        radius_z2 = 0.8
        total_angle2 = 2*3*np.pi
        phase_count2 = 3000*3
        phase_time2 = 0.005
        circle_total_time2 = phase_count2 * phase_time2
        total_time2 = init_rotate_time2 + circle_total_time2

        # 生成圆形轨迹关键点
        waypoints2 = []
        for i in range(phase_count2 + 1):
            angle = i * (total_angle2 / phase_count2)
            x = q0_2[0] + radius_xy2 * np.cos(angle)
            y = q0_2[1] + radius_xy2 * np.sin(angle)
            z = radius_z2* np.sin(4*angle) 
            yaw = angle + np.pi
            waypoints2.append(np.array([x, y, z, yaw]))

        # 创建轨迹规划器列表
        trajectory_planners2 = [trajectory_planner_init2]
        for i in range(phase_count2):
            joint_param = JointParameter(waypoints2[i], waypoints2[i+1])
            vel_param = QuinticVelocityParameter(phase_time2)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners2.append(TrajectoryPlanner(traj_param))

        trajectory_planners_list.append(trajectory_planners2)
        total_times.append(total_time2)
        phase_counts.append(phase_count2)
        phase_times.append(phase_time2)
        init_rotate_times.append(init_rotate_time2)

        #------------------------------------------------------------------------------------------

        # 无人机3: 八字飞行（motion3）
        # 阶段1：初始旋转
        z_init3 = 0.1
        init_rotate_time3 = 5
        q0_3 = np.array([0.0, 0.0, z_init3, 0.0])  # 初始位置错开
        q1_3 = q0_3 +np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init3 = JointParameter(q0_3, q1_3)
        vel_param_init3 = QuinticVelocityParameter(init_rotate_time3)
        traj_param_init3 = TrajectoryParameter(joint_param_init3, vel_param_init3)
        trajectory_planner_init3 = TrajectoryPlanner(traj_param_init3)

        # 八字轨迹参数
        radius_xy3 = 0.6
        radius_z3 = 0.3
        circle_steps_per3 = 4500
        phase_count3 = circle_steps_per3 * 2
        phase_time3 = 0.005
        circle_total_time3 = phase_count3 * phase_time3
        total_time3 = init_rotate_time3 + circle_total_time3

        # 生成八字轨迹关键点
        waypoints3 = []
        total_z_cycle3 =  2*np.pi
        for i in range(phase_count3 + 1):
            if i <= circle_steps_per3:
                angle = i * (2 * np.pi / circle_steps_per3)
            else:
                angle = 2 * np.pi - (i - circle_steps_per3) * (2 * np.pi / circle_steps_per3)
                
            x = q0_3[0] + radius_xy3 * np.cos(angle)
            y = q0_3[1] + radius_xy3 * np.sin(angle)
            z_angle = (i / phase_count3) * total_z_cycle3
            z = q0_3[2] + radius_z3 * np.cos(z_angle)  #Z值计算
            yaw = angle + np.pi
            waypoints3.append(np.array([x, y, z, yaw]))

        # 创建轨迹规划器列表
        trajectory_planners3 = [trajectory_planner_init3]
        for i in range(phase_count3):
            joint_param = JointParameter(waypoints3[i], waypoints3[i+1])
            vel_param = QuinticVelocityParameter(phase_time3)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners3.append(TrajectoryPlanner(traj_param))

        trajectory_planners_list.append(trajectory_planners3)
        total_times.append(total_time3)
        phase_counts.append(phase_count3)
        phase_times.append(phase_time3)
        init_rotate_times.append(init_rotate_time3)

        #----------------------------------------------------------------------------------

        # 无人机4: 圆形飞行（motion4）
        # 阶段1：初始旋转
        z_init4 = 0.1
        init_rotate_time4 = 5
        q0_4 = np.array([0.0, -0.1, z_init4, 0.0])  # 初始位置错开
        q1_4 = q0_4 + np.array([0.0, 0.0,0.0, np.pi])
        joint_param_init4 = JointParameter(q0_4, q1_4)
        vel_param_init4 = QuinticVelocityParameter(init_rotate_time4)
        traj_param_init4 = TrajectoryParameter(joint_param_init4, vel_param_init4)
        trajectory_planner_init4 = TrajectoryPlanner(traj_param_init4)

        # 圆形轨迹参数
        radius4 = 1
        y_final4 = 0.6
        total_angle4 = 2*5*np.pi
        phase_count4 = 3000*3
        phase_time4 = 0.005
        circle_total_time4 = phase_count4 * phase_time4
        total_time4 = init_rotate_time4 + circle_total_time4

        # 生成圆形轨迹关键点
        waypoints4 = []
        for i in range(phase_count4 + 1):
            angle = i * (total_angle4 / phase_count4)
            x = q0_4[0] + radius4 * np.cos(angle)
            y = q0_4[1] + (y_final4/phase_count4)*i
            z = radius4 * np.sin(angle)
            yaw = angle + np.pi
            waypoints4.append(np.array([x, y, z, yaw]))

        # 创建轨迹规划器列表
        trajectory_planners4 = [trajectory_planner_init4]
        for i in range(phase_count4):
            joint_param = JointParameter(waypoints4[i], waypoints4[i+1])
            vel_param = QuinticVelocityParameter(phase_time4)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            trajectory_planners4.append(TrajectoryPlanner(traj_param))

        trajectory_planners_list.append(trajectory_planners4)
        total_times.append(total_time4)
        phase_counts.append(phase_count4)
        phase_times.append(phase_time4)
        init_rotate_times.append(init_rotate_time4)

        #--------------------------------------------------------------------------------

        # 计算仿真总时间（取四架无人机中最长的时间）
        total_simulation_time = max(total_times)

        dt = model.opt.timestep  # 仿真时间步长

        time_step_num = round(total_simulation_time / model.opt.timestep)
        
        # 为每架无人机初始化目标位置存储
        qds_list = [np.zeros((time_step_num, 6)) for _ in range(count)]
        times = np.linspace(0, total_simulation_time, time_step_num)

        # 预计算每架无人机每个时刻的目标位置
        for drone_id in range(count):
            joint_position = np.zeros(4)
            for i, timei in enumerate(times):
                # 检查当前时间是否超过该无人机的总时间
                if timei > total_times[drone_id]:
                    # 超过后保持最后一个位置
                    joint_position = qds_list[drone_id][i-1, [0, 1, 2, 5]]
                else:
                    # 阶段0：初始旋转
                    if timei < init_rotate_times[drone_id]:
                        joint_position = trajectory_planners_list[drone_id][0].interpolate(timei)
                    # 阶段1：执行对应轨迹
                    else:
                        circle_time = timei - init_rotate_times[drone_id]
                        phase_idx = int(circle_time // phase_times[drone_id])
                        if phase_idx >= phase_counts[drone_id]:
                            phase_idx = phase_counts[drone_id] - 1
                        local_time = circle_time % phase_times[drone_id]
                        joint_position = trajectory_planners_list[drone_id][phase_idx + 1].interpolate(local_time)
                
                qds_list[drone_id][i, [0, 1, 2, 5]] = joint_position

        # -------------------------- 新增：初始化轨迹记录数组 --------------------------
        # 记录每架无人机的实际位置 (x, y, z)
        trajectories = [np.zeros((time_step_num, 3)) for _ in range(count)]
        # 记录每架无人机的目标位置（可选）
        target_trajectories = [np.zeros((time_step_num, 3)) for _ in range(count)]
        # 启动仿真可视化
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_simulation_time:
                step_start = time.time()

                # 为每架无人机设置目标位置
                for drone_id in range(count):
                    if time_num < time_step_num:
                        parameters[drone_id].dposd = qds_list[drone_id][time_num, [0, 1, 2]]
                        parameters[drone_id].psid = qds_list[drone_id][time_num, 5]
                        # 记录目标位置
                        target_trajectories[drone_id][time_num] = qds_list[drone_id][time_num, [0, 1, 2]]
                # 更新所有无人机状态
                for drone_id in range(count):
                    parameters[drone_id].q = data.qpos[7 * drone_id: 7 * (drone_id + 1)]
                    parameters[drone_id].dq = data.qvel[6 * drone_id: 6 * (drone_id + 1)]
                    
                    # -------------------------- 新增：记录实际位置 --------------------------
                    # 从qpos中提取x, y, z坐标（前3个值）
                    trajectories[drone_id][time_num] = parameters[drone_id].q[:3]
                
                # 计算控制力矩并执行仿真
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                # 可视化设置
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # 控制帧率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                time_num += 1
                if time_num >= time_step_num:
                    break

        # -------------------------- 新增：绘制轨迹图 --------------------------
        
        # 1. 计算5秒对应的时间步索引（向下取整）
        skip_seconds = 5.0
        skip_steps = int(skip_seconds / dt)  # 例如：dt=0.01时，5秒对应500步

        # 2. 过滤轨迹数据（只保留5秒后的点）
        # 注意：如果总仿真时间 < 5秒，会保留空数组，这里加个判断避免错误
        filtered_trajectories = []
        filtered_targets = []
        for i in range(count):
            # 实际轨迹过滤：从skip_steps开始截取，同时去除全零的无效点
            valid_actual = trajectories[i][~np.all(trajectories[i] == 0, axis=1)]  # 先过滤全零点
            filtered_actual = valid_actual[skip_steps:] if len(valid_actual) > skip_steps else valid_actual
            filtered_trajectories.append(filtered_actual)

            # 目标轨迹过滤（同上）
            valid_target = target_trajectories[i][~np.all(target_trajectories[i] == 0, axis=1)]
            filtered_target = valid_target[skip_steps:] if len(valid_target) > skip_steps else valid_target
            filtered_targets.append(filtered_target)

        # 绘制过滤后的轨迹
        self.plot_trajectories(filtered_trajectories, filtered_targets)

    # -------------------------- 新增：轨迹绘制方法 --------------------------
    # 轨迹绘制方法（无需修改核心逻辑，仅使用过滤后的数据）
    def plot_trajectories(self, actual_trajectories, target_trajectories=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['r', 'g', 'b', 'm']
        labels = [f'无人机 {i+1}' for i in range(len(actual_trajectories))]
        
        # 绘制实际轨迹（已过滤前5秒）
        for i, traj in enumerate(actual_trajectories):
            if len(traj) == 0:
                continue  # 跳过空数据
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=colors[i], label=f'{labels[i]} 实际轨迹（5秒后）')
            # 标记起点（5秒时的位置）和终点（剩余轨迹的终点）
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                    color=colors[i], marker='o', s=100, label=f'{labels[i]} 5秒起点')
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                    color=colors[i], marker='x', s=100, label=f'{labels[i]} 终点')
        
        # 绘制目标轨迹（已过滤前5秒）
        if target_trajectories is not None:
            for i, traj in enumerate(target_trajectories):
                if len(traj) == 0:
                    continue
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        color=colors[i], linestyle='--', alpha=0.5, label=f'{labels[i]} 目标轨迹（5秒后）')
        
        ax.set_xlabel('X 坐标 (m)')
        ax.set_ylabel('Y 坐标 (m)')
        ax.set_zlabel('Z 坐标 (m)')
        ax.set_title('多无人机协同运动轨迹（已去除前5秒数据）')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        plt.show()