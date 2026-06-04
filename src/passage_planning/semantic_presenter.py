from __future__ import annotations

from typing import Any

from .structures import FormationState, MissionPreference, ObstacleField, PassagePlan, ValidationResult


def obstacle_function_cn(function: str) -> str:
    if function == 'aperture':
        return '穿越型障碍'
    if function == 'solid':
        return '实体绕行障碍'
    return '未知障碍'


def passage_mode_cn(mode: str) -> str:
    mapping = {
        'formation': '保持编队整体通过',
        'snake_sequence': '蛇形时序穿越',
        'bypass': '局部绕行/越顶',
    }
    return mapping.get(str(mode), str(mode))


def target_policy_cn(policy: str) -> str:
    mapping = {
        'predictive_center_crossing': '预测洞口中心穿越',
        'edge_bypass': '实体边缘绕行',
        'side_bypass_left': '左侧绕行',
        'side_bypass_right': '右侧绕行',
        'split_by_lane_bypass': '按队形分流绕行',
        'overpass': '顶部越障',
        'underpass': '底部下穿',
        'hybrid_over_or_side': '侧绕/越顶自适应',
        'hold_and_reform': '等待并恢复编队',
    }
    return mapping.get(str(policy), str(policy))


def summarize_scene(field: ObstacleField, formation: FormationState, mission: MissionPreference) -> dict[str, Any]:
    aperture_count = sum(1 for obstacle in field.obstacles if obstacle.function == 'aperture')
    solid_count = sum(1 for obstacle in field.obstacles if obstacle.function == 'solid')
    obstacle_lines = []
    for index, obstacle in enumerate(field.obstacles, start=1):
        if obstacle.function == 'aperture':
            detail = (
                f'第 {index} 个障碍：{obstacle_function_cn(obstacle.function)}，'
                f'洞口宽度约 {float(obstacle.aperture_width or 0.0):.2f} m，'
                f'通过高度约 {float(obstacle.pass_z):.2f} m，'
                f'横向运动幅值约 {float(obstacle.motion_amplitude):.2f} m，'
                f'风险等级 {obstacle.risk_level}'
            )
        else:
            size_text = _solid_size_text(obstacle.size_x, obstacle.size_y, obstacle.size_z)
            detail = (
                f'第 {index} 个障碍：{obstacle_function_cn(obstacle.function)}，'
                f'{size_text}，风险等级 {obstacle.risk_level}'
            )
        obstacle_lines.append(detail)

    return {
        'summary': (
            f'SFSC 传感器融合结构化上下文已形成：包含 {formation.num_uavs} 架无人机状态，'
            f'队形横向宽度约 {formation.width_y:.2f} m；局部障碍上下文共 {field.count} 个障碍，'
            f'其中 {aperture_count} 个穿越型、{solid_count} 个实体型。'
        ),
        'details': {
            '上下文名称': 'SFSC - Sensor-Fused Structured Context',
            '含义': '由非视觉传感器/状态估计/无人机通信/快速感知模块组织出的结构化高层上下文',
            '无人机数量': int(formation.num_uavs),
            '当前队形宽度': f'{formation.width_y:.2f} m',
            '单机安全宽度': f'{formation.single_file_width:.2f} m',
            '任务偏好': mission.priority,
            '允许临时解散': bool(mission.allow_disband),
            '最后恢复编队': bool(mission.recover_after_last_obstacle),
            '障碍上下文列表': obstacle_lines,
        },
    }


def summarize_llm_input(scene_context: dict[str, Any]) -> dict[str, Any]:
    team = scene_context.get('uav_team', {})
    obstacles = scene_context.get('obstacle_field', [])
    mission = scene_context.get('mission_preference', {})
    lines = []
    for index, obstacle in enumerate(obstacles, start=1):
        function = str(obstacle.get('function', 'unknown'))
        if function == 'aperture':
            lines.append(
                f'第 {index} 个障碍输入为穿越型：洞口宽度 {float(obstacle.get("aperture_width") or 0.0):.2f} m，'
                f'通过高度 {float(obstacle.get("pass_z") or 0.0):.2f} m，'
                f'运动模型 {obstacle.get("motion_model", "unknown")}'
            )
        else:
            size_x = obstacle.get('size_x')
            size_y = obstacle.get('size_y')
            size_z = obstacle.get('size_z')
            if size_x is None or size_y is None or size_z is None:
                size_text = '尺寸未知'
            else:
                size_text = f'{float(size_x):.2f} x {float(size_y):.2f} x {float(size_z):.2f} m'
            lines.append(
                f'第 {index} 个障碍输入为实体型：尺寸 {size_text}，'
                f'建议从侧方或顶部绕行，运动模型 {obstacle.get("motion_model", "unknown")}'
            )
    return {
        'summary': (
            'LLM 接收的是 SFSC 传感器融合结构化上下文，而不是 MuJoCo 原始状态或底层速度指令。'
        ),
        'details': {
            '输入上下文': 'SFSC - Sensor-Fused Structured Context',
            '上下文来源说明': '由非视觉传感器信息、无人机状态、队形通信状态、距离/障碍估计和规划状态快速编码形成',
            '任务类型': scene_context.get('task', 'unknown'),
            '无人机数量': int(team.get('num_uavs', 0)),
            '初始队形': team.get('formation', 'unknown'),
            '最大前向参考速度': _fmt_float(team.get('nominal_speed_x'), 'm/s'),
            '任务优先级': mission.get('priority', 'unknown'),
            '允许策略模式': scene_context.get('allowed_modes', []),
            '障碍输入摘要': lines,
        },
    }


def summarize_plan(plan: PassagePlan, label: str = '策略计划') -> dict[str, Any]:
    obstacle_lines = [
        (
            f'第 {strategy.obstacle_index + 1} 个障碍采用 {passage_mode_cn(strategy.mode)}，'
            f'目标策略为 {target_policy_cn(strategy.target_policy)}，'
            f'观察点约 x={strategy.observe_x:.2f}，释放点约 x={strategy.clear_x:.2f}'
        )
        for strategy in plan.obstacle_strategies
    ]
    slot_lines = [
        (
            f'UAV {slot.drone_id}: 第 {slot.order_index + 1} 个通过，'
            f'名义时间窗 {slot.nominal_entry_time:.2f}-{slot.nominal_exit_time:.2f} s'
        )
        for slot in plan.slots
    ]
    return {
        'summary': (
            f'{label}：{passage_mode_cn(plan.mode)}，通行顺序 {list(plan.passing_order)}，'
            f'时间槽间隔约 {plan.time_slot_interval:.2f} s，策略来源 {plan.source}。'
        ),
        'details': {
            '计划编号': plan.plan_id,
            '策略来源': plan.source,
            '整体模式': passage_mode_cn(plan.mode),
            '置信度': f'{float(plan.confidence):.2f}',
            '通行顺序': list(plan.passing_order),
            '时间槽间隔': f'{float(plan.time_slot_interval):.2f} s',
            '穿越后恢复编队': bool(plan.recover_after_last_obstacle),
            '单机时间窗': slot_lines,
            '障碍策略': obstacle_lines,
        },
    }


def summarize_validation(validation: ValidationResult, used_fallback: bool = False) -> dict[str, Any]:
    if validation.valid and not used_fallback:
        status = '安全验证通过，策略可执行。'
    elif validation.valid and used_fallback:
        status = 'LLM 策略未被直接采用，系统已回退到确定性安全策略。'
    else:
        status = '安全验证未通过，策略不可直接执行。'
    issue_lines = [
        f'{issue.severity}: {issue.code} - {issue.message}'
        for issue in validation.issues
    ]
    return {
        'summary': status,
        'details': {
            '验证是否通过': bool(validation.valid),
            '是否发生修复': bool(validation.repaired),
            '是否使用回退策略': bool(used_fallback),
            '问题数量': len(validation.issues),
            '问题列表': issue_lines,
            '验证消息': list(validation.messages),
        },
    }


def summarize_stage_transition(from_stage: str, to_stage: str) -> dict[str, Any]:
    stage_cn = {
        'rotate': '起始转向',
        'formation': '编队接近障碍区',
        'snake': '障碍区内时序通行',
        'reform': '通过后恢复编队',
        'post_formation': '恢复后巡航',
        'complete': '任务完成',
    }
    return {
        'summary': f'执行阶段从“{stage_cn.get(from_stage, from_stage)}”切换到“{stage_cn.get(to_stage, to_stage)}”。',
        'details': {
            '上一阶段': stage_cn.get(from_stage, from_stage),
            '当前阶段': stage_cn.get(to_stage, to_stage),
        },
    }


def summarize_adaptive_order(obstacle_label: str, order: list[int] | tuple[int, ...]) -> dict[str, Any]:
    return {
        'summary': f'{obstacle_label} 触发自适应通行顺序更新：{list(order)}。',
        'details': {
            '障碍': obstacle_label,
            '更新后通行顺序': [int(item) for item in order],
            '含义': '根据无人机到下一个窄洞的距离和横向对齐状态，重新确定谁先穿越。',
        },
    }


def _solid_size_text(size_x: float | None, size_y: float | None, size_z: float | None) -> str:
    if size_x is None or size_y is None or size_z is None:
        return '尺寸未知'
    return f'尺寸约 {float(size_x):.2f} x {float(size_y):.2f} x {float(size_z):.2f} m'


def _fmt_float(value: Any, unit: str = '') -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'unknown'
    suffix = f' {unit}' if unit else ''
    return f'{number:.2f}{suffix}'
