
from unittest import TestCase

import time
import numpy as np

import mujoco
import mujoco.viewer

from src.model import Skydio
from src.controller import FormationController
from src.formation import (
    TriangleFormation, 
    LineFormation,
    FourFormation,
    )
from src.parameter import Parameter
from src.motion_planning import JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner




class TestFormation(TestCase):

    def test_formation_control_line_count3(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene3.xml")# 加载3架无人机的仿真模型
        data = mujoco.MjData(model)# 仿真数据（存储状态、控制量等）
        # 设置初始状态（物理状态、控制量）
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)#  forward计算，更新初始状态

        skydio = Skydio()#无人机模型
        count = 3#数量
        kps = [1, 1, 10, 50.0, 50.0, 50.0]  #pid控制器比例系数
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# 积分系数
        kds = [0.0, 0.0, 0.0, 10, 10, 10]# 微分系数

        parameters = [Parameter() for _ in range(count)]# 存储每架无人机的状态参数（位置、速度等）
        formation = LineFormation(1.0, count)# 直线编队（间距1.0，数量3）
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)# 编队控制器
        # 第1段轨迹（0-5秒）：从q0到q1（偏航角变化）
        time1 = 5
        q0 = np.array([0.0, 0.0, 0.0, 0.0])# 起始位置（x,y,z,偏航角）
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])# 目标位置
        joint_parameter1 = JointParameter(q0, q1)# 关节位置参数
        velocity_parameter1 = QuinticVelocityParameter(time1)# 五次多项式速度规划（平滑加速/减速）
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)# 轨迹参数
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)# 轨迹插值器
        #后续六段同理
        time2 = 5
        q2 = np.array([1, 0.0, 0.0, np.pi])
        joint_parameter2 = JointParameter(q1, q2)
        velocity_parameter2 = QuinticVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(joint_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = 5
        q3 = q2 + np.array([-1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter3 = JointParameter(q2, q3)
        velocity_parameter3 = QuinticVelocityParameter(time3)
        trajectory_parameter3 = TrajectoryParameter(joint_parameter3, velocity_parameter3)
        trajectory_planner3 = TrajectoryPlanner(trajectory_parameter3)

        time4 = 5
        q4 = q3 + np.array([-1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter4 = JointParameter(q3, q4)
        velocity_parameter4 = QuinticVelocityParameter(time4)
        trajectory_parameter4 = TrajectoryParameter(joint_parameter4, velocity_parameter4)
        trajectory_planner4 = TrajectoryPlanner(trajectory_parameter4)

        time5 = 5
        q5 = q4 + np.array([1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter5 = JointParameter(q4, q5)
        velocity_parameter5 = QuinticVelocityParameter(time5)
        trajectory_parameter5 = TrajectoryParameter(joint_parameter5, velocity_parameter5)
        trajectory_planner5 = TrajectoryPlanner(trajectory_parameter5)

        time6 = 5
        q6 = q5 + np.array([1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter6 = JointParameter(q5, q6)
        velocity_parameter6 = QuinticVelocityParameter(time6)
        trajectory_parameter6 = TrajectoryParameter(joint_parameter6, velocity_parameter6)
        trajectory_planner6 = TrajectoryPlanner(trajectory_parameter6)

        time7 = 5
        q7 = q6 + np.array([0.0, 0.0, 0.0, 0.0])
        joint_parameter7 = JointParameter(q6, q7)
        velocity_parameter7 = QuinticVelocityParameter(time7)
        trajectory_parameter7 = TrajectoryParameter(joint_parameter7, velocity_parameter7)
        trajectory_planner7 = TrajectoryPlanner(trajectory_parameter7)
        #仿真循环（执行控制逻辑）
        total_time = time1 + time2 + time3 + time4 + time5 + time6 + time7# 总仿真时间（35秒）
        time_step_num = round(total_time / model.opt.timestep)# 总步数（根据仿真步长计算）
        qds = np.zeros((time_step_num, 6))# 存储每个时间步的期望关节位置
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)# 时间序列

        joint_position = np.zeros(4)
        # 预计算每个时间步的期望位置
        for i, timei in enumerate(times):
            if timei < time1:
                joint_position = trajectory_planner1.interpolate(timei)# 第1段轨迹插值
            elif timei < time1 + time2:
                joint_position = trajectory_planner2.interpolate(timei - time1) # 第2段轨迹插值
            elif timei < time1 + time2 + time3:
                joint_position = trajectory_planner3.interpolate(timei - time1 - time2)
            elif timei < time1 + time2 + time3 + time4:
                joint_position = trajectory_planner4.interpolate(timei - time1 - time2 - time3)
            elif timei < time1 + time2 + time3 + time4 + time5:
                joint_position = trajectory_planner5.interpolate(timei - time1 - time2 - time3 - time4)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6:
                joint_position = trajectory_planner6.interpolate(timei - time1 - time2 - time3 - time4 - time5)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6 + time7:
                joint_position = trajectory_planner7.interpolate(timei - time1 - time2 - time3 - time4 - time5 - time6)
            qds[i, [0, 1, 2, 5]] = joint_position

        time_num = 0
        # 启动Mujoco可视化 viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()#记录开始时间
                # 设置第0架无人机的期望位置和偏航角（作为编队参考）
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]
                # 读取每架无人机的当前状态（位置q、速度dq）
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]
                # 计算控制力矩
                torques = formation_controller.control(parameters, formation)
                # 应用控制量并执行仿真步
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)# 设置控制力矩
                mujoco.mj_step(model, data)# 执行物理仿真步
                
                # 可视化设置（每隔1秒显示/隐藏接触点）
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()# 同步可视化
                 # 控制仿真速度（确保实时性）
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                # 记录实际状态
                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q

                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    def test_formation_control_triangle_count3(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene3.xml")
        data = mujoco.MjData(model)

        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        skydio = Skydio()
        count = 3
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]

        parameters = [Parameter() for _ in range(count)]
        formation = TriangleFormation(1.0, count)
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        time1 = 5
        q0 = np.array([0.0, 0.0, 0.0, 0.0])
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_parameter1 = JointParameter(q0, q1)
        velocity_parameter1 = QuinticVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        time2 = 5
        q2 = np.array([1, 0.0, 0.0, np.pi])
        joint_parameter2 = JointParameter(q1, q2)
        velocity_parameter2 = QuinticVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(joint_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = 5
        q3 = q2 + np.array([-1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter3 = JointParameter(q2, q3)
        velocity_parameter3 = QuinticVelocityParameter(time3)
        trajectory_parameter3 = TrajectoryParameter(joint_parameter3, velocity_parameter3)
        trajectory_planner3 = TrajectoryPlanner(trajectory_parameter3)

        time4 = 5
        q4 = q3 + np.array([-1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter4 = JointParameter(q3, q4)
        velocity_parameter4 = QuinticVelocityParameter(time4)
        trajectory_parameter4 = TrajectoryParameter(joint_parameter4, velocity_parameter4)
        trajectory_planner4 = TrajectoryPlanner(trajectory_parameter4)

        time5 = 5
        q5 = q4 + np.array([1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter5 = JointParameter(q4, q5)
        velocity_parameter5 = QuinticVelocityParameter(time5)
        trajectory_parameter5 = TrajectoryParameter(joint_parameter5, velocity_parameter5)
        trajectory_planner5 = TrajectoryPlanner(trajectory_parameter5)

        time6 = 5
        q6 = q5 + np.array([1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter6 = JointParameter(q5, q6)
        velocity_parameter6 = QuinticVelocityParameter(time6)
        trajectory_parameter6 = TrajectoryParameter(joint_parameter6, velocity_parameter6)
        trajectory_planner6 = TrajectoryPlanner(trajectory_parameter6)

        time7 = 5
        q7 = q6 + np.array([0.0, 0.0, 0.0, 0.0])
        joint_parameter7 = JointParameter(q6, q7)
        velocity_parameter7 = QuinticVelocityParameter(time7)
        trajectory_parameter7 = TrajectoryParameter(joint_parameter7, velocity_parameter7)
        trajectory_planner7 = TrajectoryPlanner(trajectory_parameter7)

        total_time = time1 + time2 + time3 + time4 + time5 + time6 + time7
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            if timei < time1:
                joint_position = trajectory_planner1.interpolate(timei)
            elif timei < time1 + time2:
                joint_position = trajectory_planner2.interpolate(timei - time1)
            elif timei < time1 + time2 + time3:
                joint_position = trajectory_planner3.interpolate(timei - time1 - time2)
            elif timei < time1 + time2 + time3 + time4:
                joint_position = trajectory_planner4.interpolate(timei - time1 - time2 - time3)
            elif timei < time1 + time2 + time3 + time4 + time5:
                joint_position = trajectory_planner5.interpolate(timei - time1 - time2 - time3 - time4)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6:
                joint_position = trajectory_planner6.interpolate(timei - time1 - time2 - time3 - time4 - time5)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6 + time7:
                joint_position = trajectory_planner7.interpolate(timei - time1 - time2 - time3 - time4 - time5 - time6)
            qds[i, [0, 1, 2, 5]] = joint_position

        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                torques = formation_controller.control(parameters, formation)

                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q

                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    def test_formation_control_line_count5(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene5.xml")
        data = mujoco.MjData(model)

        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        skydio = Skydio()
        count = 5
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]

        parameters = [Parameter() for _ in range(count)]
        formation = LineFormation(1.0, count)
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=0.5)

        time1 = 5
        q0 = np.array([0.0, 0.0, 0.0, 0.0])
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_parameter1 = JointParameter(q0, q1)
        velocity_parameter1 = QuinticVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        time2 = 5
        q2 = np.array([1, 0.0, 0.0, np.pi])
        joint_parameter2 = JointParameter(q1, q2)
        velocity_parameter2 = QuinticVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(joint_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = 5
        q3 = q2 + np.array([-1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter3 = JointParameter(q2, q3)
        velocity_parameter3 = QuinticVelocityParameter(time3)
        trajectory_parameter3 = TrajectoryParameter(joint_parameter3, velocity_parameter3)
        trajectory_planner3 = TrajectoryPlanner(trajectory_parameter3)

        time4 = 5
        q4 = q3 + np.array([-1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter4 = JointParameter(q3, q4)
        velocity_parameter4 = QuinticVelocityParameter(time4)
        trajectory_parameter4 = TrajectoryParameter(joint_parameter4, velocity_parameter4)
        trajectory_planner4 = TrajectoryPlanner(trajectory_parameter4)

        time5 = 5
        q5 = q4 + np.array([1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter5 = JointParameter(q4, q5)
        velocity_parameter5 = QuinticVelocityParameter(time5)
        trajectory_parameter5 = TrajectoryParameter(joint_parameter5, velocity_parameter5)
        trajectory_planner5 = TrajectoryPlanner(trajectory_parameter5)

        time6 = 5
        q6 = q5 + np.array([1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter6 = JointParameter(q5, q6)
        velocity_parameter6 = QuinticVelocityParameter(time6)
        trajectory_parameter6 = TrajectoryParameter(joint_parameter6, velocity_parameter6)
        trajectory_planner6 = TrajectoryPlanner(trajectory_parameter6)

        time7 = 5
        q7 = q6 + np.array([0.0, 0.0, 0.0, 0.0])
        joint_parameter7 = JointParameter(q6, q7)
        velocity_parameter7 = QuinticVelocityParameter(time7)
        trajectory_parameter7 = TrajectoryParameter(joint_parameter7, velocity_parameter7)
        trajectory_planner7 = TrajectoryPlanner(trajectory_parameter7)

        total_time = time1 + time2 + time3 + time4 + time5 + time6 + time7
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            if timei < time1:
                joint_position = trajectory_planner1.interpolate(timei)
            elif timei < time1 + time2:
                joint_position = trajectory_planner2.interpolate(timei - time1)
            elif timei < time1 + time2 + time3:
                joint_position = trajectory_planner3.interpolate(timei - time1 - time2)
            elif timei < time1 + time2 + time3 + time4:
                joint_position = trajectory_planner4.interpolate(timei - time1 - time2 - time3)
            elif timei < time1 + time2 + time3 + time4 + time5:
                joint_position = trajectory_planner5.interpolate(timei - time1 - time2 - time3 - time4)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6:
                joint_position = trajectory_planner6.interpolate(timei - time1 - time2 - time3 - time4 - time5)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6 + time7:
                joint_position = trajectory_planner7.interpolate(timei - time1 - time2 - time3 - time4 - time5 - time6)
            qds[i, [0, 1, 2, 5]] = joint_position

        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                torques = formation_controller.control(parameters, formation)

                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q

                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    def test_formation_control_triangle_count5(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene5.xml")
        data = mujoco.MjData(model)

        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        skydio = Skydio()
        count = 5
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]

        parameters = [Parameter() for _ in range(count)]
        formation = TriangleFormation(1.0, count)
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=0.5)

        time1 = 5
        q0 = np.array([0.0, 0.0, 0.0, 0.0])
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_parameter1 = JointParameter(q0, q1)
        velocity_parameter1 = QuinticVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        time2 = 5
        q2 = np.array([1, 0.0, 0.0, np.pi])
        joint_parameter2 = JointParameter(q1, q2)
        velocity_parameter2 = QuinticVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(joint_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = 5
        q3 = q2 + np.array([-1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter3 = JointParameter(q2, q3)
        velocity_parameter3 = QuinticVelocityParameter(time3)
        trajectory_parameter3 = TrajectoryParameter(joint_parameter3, velocity_parameter3)
        trajectory_planner3 = TrajectoryPlanner(trajectory_parameter3)

        time4 = 5
        q4 = q3 + np.array([-1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter4 = JointParameter(q3, q4)
        velocity_parameter4 = QuinticVelocityParameter(time4)
        trajectory_parameter4 = TrajectoryParameter(joint_parameter4, velocity_parameter4)
        trajectory_planner4 = TrajectoryPlanner(trajectory_parameter4)

        time5 = 5
        q5 = q4 + np.array([1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter5 = JointParameter(q4, q5)
        velocity_parameter5 = QuinticVelocityParameter(time5)
        trajectory_parameter5 = TrajectoryParameter(joint_parameter5, velocity_parameter5)
        trajectory_planner5 = TrajectoryPlanner(trajectory_parameter5)

        time6 = 5
        q6 = q5 + np.array([1.0, 1.0, 0.0, np.pi / 2])
        joint_parameter6 = JointParameter(q5, q6)
        velocity_parameter6 = QuinticVelocityParameter(time6)
        trajectory_parameter6 = TrajectoryParameter(joint_parameter6, velocity_parameter6)
        trajectory_planner6 = TrajectoryPlanner(trajectory_parameter6)

        time7 = 5
        q7 = q6 + np.array([0.0, 0.0, 0.0, 0.0])
        joint_parameter7 = JointParameter(q6, q7)
        velocity_parameter7 = QuinticVelocityParameter(time7)
        trajectory_parameter7 = TrajectoryParameter(joint_parameter7, velocity_parameter7)
        trajectory_planner7 = TrajectoryPlanner(trajectory_parameter7)

        total_time = time1 + time2 + time3 + time4 + time5 + time6 + time7
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            if timei < time1:
                joint_position = trajectory_planner1.interpolate(timei)
            elif timei < time1 + time2:
                joint_position = trajectory_planner2.interpolate(timei - time1)
            elif timei < time1 + time2 + time3:
                joint_position = trajectory_planner3.interpolate(timei - time1 - time2)
            elif timei < time1 + time2 + time3 + time4:
                joint_position = trajectory_planner4.interpolate(timei - time1 - time2 - time3)
            elif timei < time1 + time2 + time3 + time4 + time5:
                joint_position = trajectory_planner5.interpolate(timei - time1 - time2 - time3 - time4)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6:
                joint_position = trajectory_planner6.interpolate(timei - time1 - time2 - time3 - time4 - time5)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6 + time7:
                joint_position = trajectory_planner7.interpolate(timei - time1 - time2 - time3 - time4 - time5 - time6)
            qds[i, [0, 1, 2, 5]] = joint_position

        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                torques = formation_controller.control(parameters, formation)

                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q

                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break

    def test_formation_control(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene5.xml")
        data = mujoco.MjData(model)

        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        skydio = Skydio()
        count = 5
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]

        parameters = [Parameter() for _ in range(count)]
        triangle_formation = TriangleFormation(1.0, count)
        line_formation = LineFormation(1.0, count)
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=0.5)

        time1 = 5
        q0 = np.array([0.0, 0.0, 0.0, 0.0])
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_parameter1 = JointParameter(q0, q1)
        velocity_parameter1 = QuinticVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        time2 = 5
        q2 = np.array([1, 0.0, 0.0, np.pi])
        joint_parameter2 = JointParameter(q1, q2)
        velocity_parameter2 = QuinticVelocityParameter(time2)
        trajectory_parameter2 = TrajectoryParameter(joint_parameter2, velocity_parameter2)
        trajectory_planner2 = TrajectoryPlanner(trajectory_parameter2)

        time3 = 5
        q3 = q2 + np.array([-1.0, 1.0, 1.0, np.pi / 2])
        joint_parameter3 = JointParameter(q2, q3)
        velocity_parameter3 = QuinticVelocityParameter(time3)
        trajectory_parameter3 = TrajectoryParameter(joint_parameter3, velocity_parameter3)
        trajectory_planner3 = TrajectoryPlanner(trajectory_parameter3)

        time4 = 5
        q4 = q3 + np.array([-1.0, -1.0, -1.0, np.pi / 2])
        joint_parameter4 = JointParameter(q3, q4)
        velocity_parameter4 = QuinticVelocityParameter(time4)
        trajectory_parameter4 = TrajectoryParameter(joint_parameter4, velocity_parameter4)
        trajectory_planner4 = TrajectoryPlanner(trajectory_parameter4)

        time5 = 5
        q5 = q4 + np.array([1.0, -1.0, 0.0, np.pi / 2])
        joint_parameter5 = JointParameter(q4, q5)
        velocity_parameter5 = QuinticVelocityParameter(time5)
        trajectory_parameter5 = TrajectoryParameter(joint_parameter5, velocity_parameter5)
        trajectory_planner5 = TrajectoryPlanner(trajectory_parameter5)

        time6 = 5
        q6 = q5 + np.array([1.0, 1.0, -1.0, np.pi / 2])
        joint_parameter6 = JointParameter(q5, q6)
        velocity_parameter6 = QuinticVelocityParameter(time6)
        trajectory_parameter6 = TrajectoryParameter(joint_parameter6, velocity_parameter6)
        trajectory_planner6 = TrajectoryPlanner(trajectory_parameter6)

        time7 = 5
        q7 = q6 + np.array([0.0, 0.0, 1.0, 0.0])
        joint_parameter7 = JointParameter(q6, q7)
        velocity_parameter7 = QuinticVelocityParameter(time7)
        trajectory_parameter7 = TrajectoryParameter(joint_parameter7, velocity_parameter7)
        trajectory_planner7 = TrajectoryPlanner(trajectory_parameter7)

        total_time = time1 + time2 + time3 + time4 + time5 + time6 + time7
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        qs = np.zeros_like(qds)
        times = np.linspace(0, total_time, time_step_num)

        joint_position = np.zeros(4)
        for i, timei in enumerate(times):
            if timei < time1:
                joint_position = trajectory_planner1.interpolate(timei)
            elif timei < time1 + time2:
                joint_position = trajectory_planner2.interpolate(timei - time1)
            elif timei < time1 + time2 + time3:
                joint_position = trajectory_planner3.interpolate(timei - time1 - time2)
            elif timei < time1 + time2 + time3 + time4:
                joint_position = trajectory_planner4.interpolate(timei - time1 - time2 - time3)
            elif timei < time1 + time2 + time3 + time4 + time5:
                joint_position = trajectory_planner5.interpolate(timei - time1 - time2 - time3 - time4)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6:
                joint_position = trajectory_planner6.interpolate(timei - time1 - time2 - time3 - time4 - time5)
            elif timei < time1 + time2 + time3 + time4 + time5 + time6 + time7:
                joint_position = trajectory_planner7.interpolate(timei - time1 - time2 - time3 - time4 - time5 - time6)
            qds[i, [0, 1, 2, 5]] = joint_position

        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                if data.time < total_time / 2:
                    formation = triangle_formation
                else:
                    formation = line_formation
                torques = formation_controller.control(parameters, formation)

                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                qds[time_num, :] = parameters[0].qd
                qs[time_num, :] = parameters[0].q

                qds[time_num, :3] = parameters[0].dqd[:3]
                qs[time_num, :3] = parameters[0].dq[:3]

                time_num += 1
                if time_num >= time_step_num:
                    break
    
    def test_formation_control_triangle_circle_count3(self):
        # 加载3架无人机的仿真模型
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene3.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器（沿用三角形编队测试的参数）
        skydio = Skydio()
        count = 3
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        formation = TriangleFormation(1.0, count)  # 保持三角形编队模式
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        # 关键修改：添加初始旋转阶段（与test_formation_control_triangle_count3保持一致）
        # 阶段1：先旋转π角度（180度），耗时5秒，调整初始朝向
        init_rotate_time = 5  # 初始旋转时间（与同类测试保持一致）
        q0 = np.array([0.0, 0.0, 0.0, 0.0])  # 初始位置和姿态
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])  # 仅旋转yaw角180度
        joint_param_init = JointParameter(q0, q1)
        vel_param_init = QuinticVelocityParameter(init_rotate_time)
        traj_param_init = TrajectoryParameter(joint_param_init, vel_param_init)
        trajectory_planner_init = TrajectoryPlanner(traj_param_init)

        # 圆形轨迹参数：16个阶段，每个阶段2秒，总时间32秒（在初始旋转后执行）
        radius = 0.6
        z_fixed = 0.0
        total_angle = 2 * np.pi
        phase_count = 16
        phase_time = 2
        circle_total_time = phase_count * phase_time  # 圆形轨迹总时间32秒
        total_time = init_rotate_time + circle_total_time  # 总时间=初始旋转(5s)+圆形轨迹(32s)=37s

        # 生成圆形轨迹关键点
        waypoints = []
        for i in range(phase_count + 1):
            angle = i * (total_angle / phase_count)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
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
    def test_formation_control_spiral_transform_count3(self):
        # 加载3架无人机的仿真模型
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene3.xml")
        data = mujoco.MjData(model)

        # 初始化模型状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化无人机参数和控制器
        skydio = Skydio()
        count = 3
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        
        # 定义两种队形
        triangle_formation = TriangleFormation(1.0, count)
        line_formation = LineFormation(1.0, count)  # 若需前后直线，替换为ParallelLineFormation
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0)

        # -------------------------- 阶段1：初始旋转（5秒，z=0.0） --------------------------
        init_rotate_time = 5
        q0 = np.array([0.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 0.0, 0.0, np.pi])
        joint_param_init = JointParameter(q0, q1)
        vel_param_init = QuinticVelocityParameter(init_rotate_time)
        trajectory_planner_init = TrajectoryPlanner(TrajectoryParameter(joint_param_init, vel_param_init))

        # -------------------------- 阶段2：螺旋上升（20秒，z从0→0.5） --------------------------
        spiral_up_time = 20
        spiral_radius = 0.6
        start_z_up = 0.0
        end_z_up = 0.5  # 高度调整为0.5m
        total_angle = 2 * np.pi
        phase_count_spiral_up = 16

        spiral_up_waypoints = []
        for i in range(phase_count_spiral_up + 1):
            angle = i * (total_angle / phase_count_spiral_up)
            x = spiral_radius * np.cos(angle)
            y = spiral_radius * np.sin(angle)
            z = start_z_up + (end_z_up - start_z_up) * (i / phase_count_spiral_up)  # 线性上升
            yaw = angle + np.pi
            spiral_up_waypoints.append(np.array([x, y, z, yaw]))

        spiral_up_planners = []
        for i in range(phase_count_spiral_up):
            joint_param = JointParameter(spiral_up_waypoints[i], spiral_up_waypoints[i+1])
            vel_param = QuinticVelocityParameter(spiral_up_time / phase_count_spiral_up)
            spiral_up_planners.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))

        # -------------------------- 阶段3：直线飞行（10秒，z=0.5，仅前进0.2m） --------------------------
        forward_time = 10
        last_spiral_pos = spiral_up_waypoints[-1].copy()
        forward_target = np.array([
            last_spiral_pos[0] + 0.2,  # 核心修改：直线飞行距离从2m改为0.2m
            last_spiral_pos[1],        # y不变
            last_spiral_pos[2],        # z固定0.5（高度不变）
            last_spiral_pos[3]         # yaw不变
        ])
        joint_param_forward = JointParameter(last_spiral_pos, forward_target)
        vel_param_forward = QuinticVelocityParameter(forward_time)
        trajectory_planner_forward = TrajectoryPlanner(TrajectoryParameter(joint_param_forward, vel_param_forward))

        # -------------------------- 阶段4：调整方向（5秒，z=0.5） --------------------------
        line_rotate_time = 5
        rotate_target = np.array([
            forward_target[0],
            forward_target[1],
            forward_target[2],  # z固定0.5
            forward_target[3] + np.pi/2  # 转向90度
        ])
        joint_param_rotate = JointParameter(forward_target, rotate_target)
        vel_param_rotate = QuinticVelocityParameter(line_rotate_time)
        trajectory_planner_rotate = TrajectoryPlanner(TrajectoryParameter(joint_param_rotate, vel_param_rotate))

        # -------------------------- 阶段5：螺旋下降（20秒，z从0.5→0.0） --------------------------
        spiral_down_time = 20
        start_z_down = end_z_up  # 0.5m
        end_z_down = 0.0        # 回到初始高度
        phase_count_spiral_down = 16

        spiral_down_waypoints = []
        for i in range(phase_count_spiral_down + 1):
            angle = i * (total_angle / phase_count_spiral_down)
            # 基于直线飞行终点画圆（仅前进0.2m，位置更紧凑）
            x = forward_target[0] + spiral_radius * np.cos(angle)
            y = forward_target[1] + spiral_radius * np.sin(angle)
            z = start_z_down - (start_z_down - end_z_down) * (i / phase_count_spiral_down)  # 线性下降
            yaw = rotate_target[3] + angle
            spiral_down_waypoints.append(np.array([x, y, z, yaw]))

        spiral_down_planners = []
        for i in range(phase_count_spiral_down):
            joint_param = JointParameter(spiral_down_waypoints[i], spiral_down_waypoints[i+1])
            vel_param = QuinticVelocityParameter(spiral_down_time / phase_count_spiral_down)
            spiral_down_planners.append(TrajectoryPlanner(TrajectoryParameter(joint_param, vel_param)))

        # 总时间（不变）
        total_time = init_rotate_time + spiral_up_time + forward_time + line_rotate_time + spiral_down_time

        # 初始化轨迹数据
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        times = np.linspace(0, total_time, time_step_num)

        # 计算每个时刻的目标位置
        for i, timei in enumerate(times):
            if timei < init_rotate_time:
                joint_position = trajectory_planner_init.interpolate(timei)
            
            elif timei < init_rotate_time + spiral_up_time:
                t = timei - init_rotate_time
                phase_idx = min(int(t // (spiral_up_time / phase_count_spiral_up)), phase_count_spiral_up - 1)
                joint_position = spiral_up_planners[phase_idx].interpolate(t % (spiral_up_time / phase_count_spiral_up))
            
            elif timei < init_rotate_time + spiral_up_time + forward_time:
                t = timei - (init_rotate_time + spiral_up_time)
                joint_position = trajectory_planner_forward.interpolate(t)
            
            elif timei < init_rotate_time + spiral_up_time + forward_time + line_rotate_time:
                t = timei - (init_rotate_time + spiral_up_time + forward_time)
                joint_position = trajectory_planner_rotate.interpolate(t)
            
            else:
                t = timei - (init_rotate_time + spiral_up_time + forward_time + line_rotate_time)
                phase_idx = min(int(t // (spiral_down_time / phase_count_spiral_down)), phase_count_spiral_down - 1)
                joint_position = spiral_down_planners[phase_idx].interpolate(t % (spiral_down_time / phase_count_spiral_down))
            
            qds[i, [0, 1, 2, 5]] = joint_position

        # 启动仿真
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 设置领航机目标
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 更新无人机状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 队形变换逻辑
                spiral_up_end = init_rotate_time + spiral_up_time
                forward_end = spiral_up_end + forward_time
                if data.time < spiral_up_end:
                    formation = triangle_formation
                elif data.time < forward_end:
                    formation = line_formation  # 直线飞行阶段完成队形变换
                else:
                    formation = line_formation

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
    
    def test_formation_four_dynamic_converge(self):
        """测试四无人机动态靠拢编队（2号机在原点）"""
        # 加载4架无人机模型（假设scene4.xml对应4架）
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene4.xml")
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 初始化参数
        skydio = Skydio()
        count = 4  # 固定4架无人机
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        
        # 初始化四机编队（初始间距2米，最小0.5米）
        initial_spacing = 0.2
        min_spacing = 0.1
        formation = FourFormation(initial_spacing, count, min_spacing)
        formation_controller = FormationController(
            kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0, use_formation=True
        )

        # 复用原有7段轨迹（总时间35秒）
        # （轨迹规划逻辑与test_formation_control_line_count3一致，此处省略）
        time1 = 5
        q0 = np.array([initial_spacing, 0.0, 0.0, 0.0])  # 领航机初始位置（确保2号机在原点(0,0,0)）
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])
        joint_parameter1 = JointParameter(q0, q1)
        velocity_parameter1 = QuinticVelocityParameter(time1)
        trajectory_parameter1 = TrajectoryParameter(joint_parameter1, velocity_parameter1)
        trajectory_planner1 = TrajectoryPlanner(trajectory_parameter1)

        # 后续轨迹段（q2-q7）与原测试用例一致（省略）
        # ...

        # 预计算领航机轨迹（与原逻辑一致）
        total_time = 35  # 总时间35秒
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        times = np.linspace(0, total_time, time_step_num)
        # （轨迹插值逻辑省略）

        # 仿真循环
        time_num = 0
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 1. 动态更新编队间距（前5秒内完成靠拢）
                formation.set_spacing(data.time, adjust_time=5.0)

                # 2. 设置领航机（0号）目标
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 3. 更新无人机状态
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 4. 计算控制力矩
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                # 可视化同步
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


    def test_formation_control_switch(self):
        """
        Test direct switch from independent movement to formation flight.
        Phases:
        1. Independent phase: ONLY the leader UAV moves. Followers remain stationary.
        2. Formation phase: All UAVs switch to formation flight.
        """
        # Load the simulation model with 5 UAVs
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene5.xml")
        data = mujoco.MjData(model)
        dt = model.opt.timestep

        # Initialize the simulation state
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # Initialize controller parameters
        skydio = Skydio()
        uav_count = 5
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]

        parameters = [Parameter() for _ in range(uav_count)]
        line_formation = LineFormation(1.0, uav_count)

        # Controller starts in independent mode
        formation_controller = FormationController(
            kps, kis, kds, skydio, uav_count, ts=0.01, 
            position_gain=0.5, use_formation=False
        )

        # -------------------------- Phase 1: Independent Movement --------------------------
        independent_duration = 15.0
        formation_duration = 15.0
        total_simulation_time = independent_duration + formation_duration
        time_step_num = round(total_simulation_time / dt)

        # Plan independent trajectory for the leader UAV only
        leader_trajectory = []
        waypoints = [
            (0.0, np.array([0.0, 0.0, 0.0, 0.0])),
            (5.0, np.array([0.0, 0.0, 0.0, np.pi])),
            (10.0, np.array([1.0, 0.0, 0.0, np.pi])),
            (15.0, np.array([1.0, 1.0, 0.0, np.pi*(3/2)])),
            (30.0, np.array([-1.0, -1.0, 0.0, np.pi])) # Extend for formation phase
        ]
        
        for i in range(len(waypoints) - 1):
            t_start, q_start = waypoints[i]
            t_end, q_end = waypoints[i + 1]
            duration = t_end - t_start
            joint_param = JointParameter(q_start, q_end)
            vel_param = QuinticVelocityParameter(duration)
            traj_param = TrajectoryParameter(joint_param, vel_param)
            leader_trajectory.append((t_start, t_end, TrajectoryPlanner(traj_param)))

        # -------------------------- Simulation Loop --------------------------
        time_num = 0
        formation_started = False

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_simulation_time:
                step_start = time.time()
                current_time = data.time

                # -------------------------- Mode Switch Logic --------------------------
                if current_time >= independent_duration and not formation_started:
                    print("!!! Switching to formation mode NOW !!!")
                    formation_started = True
                    formation_controller.use_formation = True # Enable formation

                # -------------------------- Update Target Positions --------------------------
                if not formation_started:
                    # Phase 1: Only update the leader's target
                    for (t_start, t_end, planner) in leader_trajectory:
                        if t_start <= current_time < t_end:
                            local_time = current_time - t_start
                            joint_pos = planner.interpolate(local_time)
                            parameters[0].dposd = joint_pos[:3]
                            parameters[0].psid = joint_pos[3]
                            break
                    # Followers: Do not set any target, they will remain stationary
                else:
                    # Phase 2: Formation mode - Only update the leader's target
                    for (t_start, t_end, planner) in leader_trajectory:
                        if t_start <= current_time < t_end:
                            local_time = current_time - t_start
                            joint_pos = planner.interpolate(local_time)
                            parameters[0].dposd = joint_pos[:3]
                            parameters[0].psid = joint_pos[3]
                            break

                # -------------------------- Update UAV States and Control --------------------------
                for i in range(uav_count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                torques = formation_controller.control(parameters, line_formation)

                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                # -------------------------- Visualization and Sync --------------------------
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                time_num += 1
                if time_num >= time_step_num:
                    break