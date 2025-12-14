from unittest import TestCase
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.model import Skydio
from src.controller import FormationController
from src.formation import FourFormation,LineFormation,TriangleFormation # 导入修改后的四机编队类
from src.parameter import Parameter
from src.motion_planning import JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner


class TestFourFormation(TestCase):
    """四无人机编队测试：固定间距编队+轨迹跟踪"""

    def test_four_formation(self):
        # -------------------------- 1. 加载仿真模型（4架无人机）--------------------------
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene4.xml")
        data = mujoco.MjData(model)
        dt = model.opt.timestep  # 仿真时间步长（默认0.01s）

        # 初始化物理状态和控制量
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # -------------------------- 2. 初始化参数 --------------------------
        # 无人机模型参数
        skydio = Skydio()
        count = 4  # 固定4架无人机
        # PID控制器参数
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]

        # 四机编队参数（仅保留固定间距1.0m）
        fixed_spacing = 0.8
        formation = TriangleFormation(fixed_spacing,4)  # 使用固定间距
        # 编队控制器（启用编队模式）
        formation_controller = FormationController(
            kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0, use_formation=True
        )

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
      