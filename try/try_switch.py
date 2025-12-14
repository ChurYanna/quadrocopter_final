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
    )
from src.parameter import Parameter
from src.motion_planning import JointParameter, QuinticVelocityParameter, TrajectoryParameter, TrajectoryPlanner

class TryFormation(TestCase):

    def line_switch(self):
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene3.xml")# 加载3架无人机的仿真模型
        data = mujoco.MjData(model)# 仿真数据（存储状态、控制量等）
        #设置初始状态
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        skydio = Skydio()#无人机模型
        count = 3#数量
        kps = [1, 1, 10, 50.0, 50.0, 50.0]  #pid控制器比例系数
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# 积分系数
        kds = [0.0, 0.0, 0.0, 10, 10, 10]# 微分系数

        parameters = [Parameter() for _ in range(count)]# 存储每架无人机的状态参数（位置、速度等）
        formation = LineFormation(1.0, count)# 直线编队（间距1.0，数量3）
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2,use_formation=False)# 编队控制器

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

        # 根据阶段设置是否使用编队控制：如果你想让前 3 段为 False、后面为 True，设置如下
        phase_switch_time = time1 + time2 + time3
        formation_flags = np.zeros(time_step_num, dtype=bool)
        # 前三段（times <= phase_switch_time）保持 False，后面阶段设为 True
        formation_flags[times > phase_switch_time] = True

        # 计算切换发生的索引（用于诊断打印）
        try:
            switch_index = int(round(phase_switch_time / model.opt.timestep))
        except Exception:
            switch_index = None
        # 临时降采样窗口：在切换发生后暂时把 position_gain 设为 0 的步数（例如 0.5s）
        # 临时降采样窗口：在切换发生后让 controller 处理 position_gain 临时置零并平滑恢复的时长
        # 我们不在试验脚本直接修改 controllers，改由 FormationController.apply_temporary_position_gain_zero 处理

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
        # 记录上一次 switch_state 以便检测状态变化并打印诊断
        prev_switch_state = formation_controller.switch_state
        # 当检测到状态变化时，在下一个控制输出后打印该时刻的 torques 以便诊断
        print_torques_next = False
        # 启动Mujoco可视化 viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()#记录开始时间
                # 设置第0架无人机的期望位置和偏航角（作为编队参考）
                # 每一步根据预设阶段开关切换编队模式（通过 request/update 接口）
                if time_num == 0:
                    prev_flag = bool(formation_flags[0])
                # detect rising/falling edge
                if formation_flags[time_num] and not prev_flag:
                    formation_controller.request_enable_formation()
                if (not formation_flags[time_num]) and prev_flag:
                    formation_controller.request_disable_formation()
                prev_flag = bool(formation_flags[time_num])

                # If a phase switch occurs, request the FormationController to handle
                # the temporary zeroing and smooth restore of position gains and
                # align internal _qd_prev to leader. This keeps transition logic
                # fully inside the controller implementation (module boundary).
                if switch_index is not None and time_num == switch_index:
                    try:
                        formation_controller.apply_temporary_position_gain_zero(0.5)
                        formation_controller.sync_qd_prev_to_leader(parameters)
                        print(f'*** REQUESTED: controller apply_temporary_position_gain_zero and sync_qd_prev at time_num={time_num} ***')
                    except Exception:
                        print('*** WARNING: failed to request controller temp zero/sync ***')

                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # advance formation_controller switch state machine (may modify parameters[*].dposd during blending)
                formation_controller.update_switch(model.opt.timestep, parameters, formation)
                # 检测 switch_state 是否发生变化，若发生记录并打印原因/相关标志
                if formation_controller.switch_state != prev_switch_state:
                    try:
                        gate_ok = formation_controller.is_gate_ok(parameters)
                    except Exception:
                        gate_ok = None
                    print(f'*** SWITCH STATE CHANGE at time_num={time_num}: {prev_switch_state} -> {formation_controller.switch_state} , gate_ok={gate_ok}, rendezvous_timer={formation_controller.rendezvous_timer} ***')
                    # 触发在本次控制输出后打印 torques
                    print_torques_next = True
                    prev_switch_state = formation_controller.switch_state
                # --- 诊断打印：在切换附近打印关键量，帮助定位突变原因 ---
                if switch_index is not None and abs(time_num - switch_index) <= 3:
                    print('--- DEBUG SWITCH WINDOW ---')
                    print(f'time_num={time_num}, sim_time={data.time:.4f}, switch_index={switch_index}, use_formation={formation_controller.use_formation}')
                    # 计算编队误差（如果能访问 formation）用于参考
                    try:
                        deltas = formation.cal_deltas(parameters[0].psi)
                    except Exception:
                        deltas = None
                    for ii in range(count):
                        vc = formation_controller._velocity_controllers[ii]
                        qd_prev = getattr(vc, '_qd_prev', None)
                        ts = getattr(vc, '_ts', None)
                        dposd = parameters[ii].dposd
                        pos = parameters[ii].pos
                        psi = parameters[ii].psi
                        # 估计 dqd
                        est_dqd = None
                        if (qd_prev is not None) and (ts is not None) and ts != 0:
                            est_dqd = (dposd - qd_prev) / ts
                        # 计算相对位置与期望偏移（如果可用）
                        rel_pos = None
                        err_norm = None
                        if deltas is not None:
                            try:
                                rel_pos = (parameters[ii].pos - parameters[0].pos)
                                e_vec = deltas[ii] - np.array(rel_pos).T
                                err_norm = np.linalg.norm(e_vec)
                            except Exception:
                                err_norm = None
                        # 距离到领航机
                        try:
                            dist_to_leader = np.linalg.norm(parameters[ii].pos - parameters[0].pos)
                        except Exception:
                            dist_to_leader = None
                        # 估算 velocity controller 会输出的 u1（忽略 PID 内部输出，近似）
                        try:
                            pg = vc.position_gain
                        except Exception:
                            pg = 0.0
                        u1x = (0.0 if est_dqd is None else est_dqd[0]) + pg * (0.0 if e_vec is None else (e_vec[0] if isinstance(e_vec, (list, tuple, np.ndarray)) else 0.0))
                        u1y = (0.0 if est_dqd is None else est_dqd[1]) + pg * (0.0 if e_vec is None else (e_vec[1] if isinstance(e_vec, (list, tuple, np.ndarray)) else 0.0))
                        u1z = (0.0 if est_dqd is None else est_dqd[2]) + formation_controller._model.g + (pg * (0.0 if e_vec is None else (e_vec[2] if isinstance(e_vec, (list, tuple, np.ndarray)) else 0.0)))
                        # 计算合成 u1 和期望角（近似，不含 PID 输出）
                        try:
                            thetad = np.arctan((u1x * np.cos(psi) + u1y * np.sin(psi)) / u1z)
                            phid = np.arctan((u1x * np.sin(psi) - u1y * np.cos(psi)) * np.cos(thetad) / u1z)
                            u1 = formation_controller._model.m * u1z / (np.cos(phid) * np.cos(thetad))
                        except Exception:
                            thetad = None
                            phid = None
                            u1 = None
                        print(f'  drone={ii}: dposd={dposd}, qd_prev={qd_prev}, est_dqd={est_dqd}, err_norm={err_norm}, dist_to_leader={dist_to_leader}, psi={psi}, approx_u1={u1}, phid={phid}, thetad={thetad}')
                    print('--- END DEBUG ---')
                # 读取每架无人机的当前状态（位置q、速度dq）
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]
                # 计算控制力矩
                torques = formation_controller.control(parameters, formation)
                # 应用控制量并执行仿真步
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)# 设置控制力矩
                mujoco.mj_step(model, data)# 执行物理仿真步
                # 如果上一步检测到 switch_state 变化，则在产生了实际 torques 后打印扭矩用于诊断
                if print_torques_next:
                    try:
                        print(f'*** TORQUES AT SWITCH (time_num={time_num}) torques.shape={None if torques is None else np.array(torques).shape} ***')
                        # 每架无人机的 4 元控制量（推力 + 3 个转矩）
                        for di in range(count):
                            t = torques[4 * di: 4 * (di + 1)]
                            print(f'  drone={di} torques={t}, norm={np.linalg.norm(t)}')
                    except Exception as e:
                        print('*** ERROR printing torques:', e)
                    print_torques_next = False
                
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

    def circle_switch(self, formation_start_time: float = 25.0, radius: float = 0.6, circle_total_time: float = 32.0,
                      phase_count: int = 1000):
        """
        让领航机沿圆轨迹飞行（更平滑：默认100段），并在 formation_start_time 触发编队过渡流程。

        参数:
        - formation_start_time: 从该仿真时间开始触发编队切换（独立->过渡->编队）
        - radius: 圆轨迹半径（米）
        - circle_total_time: 圆轨迹总时长（秒），默认与测试用例一致为32s
        - phase_count: 将圆轨迹划分的段数，默认100段以获得更平滑的轨迹
        """
        # 载入3机场景
        model = mujoco.MjModel.from_xml_path("assets/skydio_x2/scene3.xml")
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, np.append(model.key_qpos, model.key_qvel), mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_setState(model, data, model.key_ctrl[0, :], mujoco.mjtState.mjSTATE_CTRL)
        mujoco.mj_forward(model, data)

        # 控制器与编队
        skydio = Skydio()
        count = 3
        kps = [1, 1, 10, 50.0, 50.0, 50.0]
        kis = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        kds = [0.0, 0.0, 0.0, 10, 10, 10]
        parameters = [Parameter() for _ in range(count)]
        # 采用三角编队以对齐参考测试
        formation = TriangleFormation(1.0, count)
        formation_controller = FormationController(kps, kis, kds, skydio, count, ts=0.01, position_gain=2.0, use_formation=False)

        # 初始旋转阶段（5s），仅改变偏航，便于统一参考姿态
        init_rotate_time = 5.0
        q0 = np.array([0.0, 0.0, 0.2, 0.0])
        q1 = q0 + np.array([0.0, 0.0, 0.0, np.pi])
        planner_init = TrajectoryPlanner(TrajectoryParameter(JointParameter(q0, q1), QuinticVelocityParameter(init_rotate_time)))

        # 圆轨迹（phase_count 段，总时长 circle_total_time）
        phase_time = circle_total_time / max(1, int(phase_count))
        total_angle = 2 * np.pi
        z_fixed = 0.0
        waypoints = []
        for i in range(phase_count + 1):
            angle = i * (total_angle / phase_count)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            yaw = angle + np.pi
            waypoints.append(np.array([x, y, z_fixed, yaw]))

        circle_planners = []
        for i in range(phase_count):
            jp = JointParameter(waypoints[i], waypoints[i + 1])
            vp = QuinticVelocityParameter(phase_time)
            circle_planners.append(TrajectoryPlanner(TrajectoryParameter(jp, vp)))

        # 预计算轨迹
        total_time = init_rotate_time + circle_total_time
        time_step_num = round(total_time / model.opt.timestep)
        qds = np.zeros((time_step_num, 6))
        times = np.linspace(0, total_time, time_step_num)

        joint_position = np.zeros(4)
        for i, t in enumerate(times):
            if t < init_rotate_time:
                joint_position = planner_init.interpolate(t)
            else:
                local = t - init_rotate_time
                idx = int(local // phase_time)
                if idx >= phase_count:
                    idx = phase_count - 1
                joint_position = circle_planners[idx].interpolate(local % phase_time)
            qds[i, [0, 1, 2, 5]] = joint_position

        # 根据 formation_start_time 生成编队使能标志
        formation_flags = np.zeros(time_step_num, dtype=bool)
        formation_flags[times >= float(formation_start_time)] = True
        try:
            switch_index = int(round(float(formation_start_time) / model.opt.timestep))
        except Exception:
            switch_index = None

        # 仿真循环
        time_num = 0
        prev_switch_state = formation_controller.switch_state
        print_torques_next = False
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= total_time:
                step_start = time.time()

                # 领航机期望
                parameters[0].dposd = qds[time_num, [0, 1, 2]]
                parameters[0].psid = qds[time_num, 5]

                # 编队开关沿时间触发
                if time_num == 0:
                    prev_flag = bool(formation_flags[0])
                if formation_flags[time_num] and not prev_flag:
                    formation_controller.request_enable_formation()
                if (not formation_flags[time_num]) and prev_flag:
                    formation_controller.request_disable_formation()
                prev_flag = bool(formation_flags[time_num])

                # 在切换点请求控制器内部进行 position_gain 临时置零与 _qd_prev 对齐
                if switch_index is not None and time_num == switch_index:
                    try:
                        formation_controller.apply_temporary_position_gain_zero(0.5)
                        formation_controller.sync_qd_prev_to_leader(parameters)
                        print(f'*** REQUESTED: controller apply_temporary_position_gain_zero and sync_qd_prev at time_num={time_num} ***')
                    except Exception:
                        print('*** WARNING: failed to request controller temp zero/sync ***')

                # 更新所有无人机状态（从仿真器读）
                for i in range(count):
                    parameters[i].q = data.qpos[7 * i: 7 * (i + 1)]
                    parameters[i].dq = data.qvel[6 * i: 6 * (i + 1)]

                # 推进过渡状态机
                formation_controller.update_switch(model.opt.timestep, parameters, formation)

                # 状态变迁诊断
                if formation_controller.switch_state != prev_switch_state:
                    try:
                        gate_ok = formation_controller.is_gate_ok(parameters)
                    except Exception:
                        gate_ok = None
                    print(f'*** SWITCH STATE CHANGE at time_num={time_num}: {prev_switch_state} -> {formation_controller.switch_state} , gate_ok={gate_ok}, rendezvous_timer={formation_controller.rendezvous_timer} ***')
                    print_torques_next = True
                    prev_switch_state = formation_controller.switch_state

                # 控制与仿真步进
                torques = formation_controller.control(parameters, formation)
                mujoco.mj_setState(model, data, torques, mujoco.mjtState.mjSTATE_CTRL)
                mujoco.mj_step(model, data)

                # 切换点扭矩打印
                if print_torques_next:
                    try:
                        print(f'*** TORQUES AT SWITCH (time_num={time_num}) torques.shape={None if torques is None else np.array(torques).shape} ***')
                        for di in range(count):
                            tqs = torques[4 * di: 4 * (di + 1)]
                            print(f'  drone={di} torques={tqs}, norm={np.linalg.norm(tqs)}')
                    except Exception as e:
                        print('*** ERROR printing torques:', e)
                    print_torques_next = False

                # 可视化
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
                viewer.sync()

                # 保持实时
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                time_num += 1
                if time_num >= time_step_num:
                    break
