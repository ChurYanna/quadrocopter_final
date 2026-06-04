# 多模态复杂场景 Demo 底层调参记录

记录对象：`tests.test_multimodal_complex_strategy_zone.TestMultimodalComplexStrategyZone.test_multimodal_complex_strategy_zone_viewer`

## 当前运行模式

- 起飞前全局 LLM：关闭，使用确定性基准策略作为安全兜底。
- 在线高层策略：开启，`ONLINE_REPLANNING_PROVIDER = 'qwen_vlm'`。
- 面板复核型 VLM：关闭，避免重复提交视觉复核请求。
- 控制入口：VLM 输出必须通过 `StrategyValidator`，通过后合并进在线控制缓存。
- 安全投影：`PRESERVE_ONLINE_PASSING_ORDER = True`，VLM 不能改写底层全局通行顺序。
- 局部窗口投影：VLM 可以提前 `observe_x`、延长 `clear_x`，但不能推迟观察点或缩短释放距离。

## 底层稳定化关键改动

1. 复杂场景间距调整
   - 为在线 VLM 预留约 8s 推理窗口，主障碍 x 位置调整为：
     `gate1=7`，`over_beam=19`，`gate2=32`，`cone_cluster=45`，
     `tall_tower=58`，`gate3=72`，`side_box_d=86`，`gate4=102`。
   - 相邻主障碍间距基本保持在 `12m` 以上，避免 VLM 结果返回时已经错过 `observe_x`。
   - `ONLINE_LOOKAHEAD_DISTANCE = 26.0`，`ONLINE_DETAIL_REVEAL_DISTANCE = 18.5`，使 SFSC 细节在更早位置触发 VLM。
   - `FINAL_TARGET_X = 112.0`，保留最后一门后的恢复空间。

2. 窄门几何修正
   - 删除 `gate2_ring_bottom`，避免第二个窄门中心通行区域被小横杠干扰。
   - `gate3` 横向洞口扩展约 `1.2x`，洞口宽度约 `1.48m`。

3. 最后一门稳定化
   - `gate4` 几何本身并非异常窄，核心问题是动态门心预测和门后回撤过早。
   - 为 `gate4` 单独设置：
     - `center_lookahead_time = 0.16`
     - `max_center_lead_y = 0.14`
   - 高速穿门使用严格 commit 容差：
     - `GATE_COMMIT_ALIGNMENT_Y_TOL = 0.34`
     - `GATE_COMMIT_ALIGNMENT_Z_TOL = 0.46`
   - 最后一门单独延长清除距离：
     - `FINAL_GATE_PASS_CLEAR_X = 2.80`
     - `POST_GATE_HOLD_CLEAR_X = 2.80`
   - 门后等待位后移并放缓：
     - `POST_GATE_HOLD_X = 3.40`
     - `POST_GATE_HOLD_STAGGER_X = 0.55`
     - `POST_GATE_HOLD_KP_XY = 0.62`
     - `POST_GATE_HOLD_MAX_SPEED_XY = 0.90`

4. 穿洞控制节奏
   - 横向追踪变柔：
     - `REACTIVE_Y_KP = 1.30`
     - `CMD_SLEW_RATE_XY = 1.0`
   - Z 轴对齐放松：
     - `GATE_ALIGNMENT_Z_TOL = 0.50`
     - `BYPASS_ALIGNMENT_Z_TOL = 0.50`
   - 对齐后进入承诺通过段：
     - `GATE_ALIGNED_PASS_SPEED_X = 1.34`
     - `SOLID_BYPASS_ALIGNED_PASS_SPEED_X = 1.18`
     - `GATE_SKIP_TEMPORAL_LIMIT_AFTER_ALIGN = True`

5. 实体绕障
   - 实体障碍提前激活并保持足够释放距离：
     - `SOLID_BYPASS_ACTIVATE_X = 7.20`
     - `SOLID_BYPASS_CLEAR_X = 4.00`
   - 每个实体障碍保留局部参数覆盖：
     - `over_beam`：顶部越障，低速对齐后越顶。
     - `cone_cluster`：按原始横向位置分侧绕行。
     - `tall_tower`：右侧绕行。
     - `side_box_d`：分侧绕行。
   - 绕障阶段局部避碰开启，仅作用于 `active_solid`：
     - `SOLID_BYPASS_COLLISION_GUARD_ENABLE = True`
     - `SOLID_BYPASS_GUARD_MIN_GAP_X = 1.50`
     - `SOLID_BYPASS_GUARD_HARD_GAP_X = 1.20`

## 高层 VLM 启用原则

- VLM 输入：最前方 UAV 的前视 RGB + SFSC 非视觉结构化上下文。
- VLM 角色：局部策略增强，不替代底层稳定控制。
- 执行侧约束：即使 VLM 返回新的 `passing_order`，复杂 demo 合并执行计划时也会保留底层顺序和时间槽。
- 执行侧还会过滤 0.25m 以下的小幅窗口噪声，并将 `clear_x` 延长幅度限制在 0.80m 内。
- 推荐改动范围：
  - 调整单个障碍物的 `observe_x` / `clear_x`。
  - 在验证器允许范围内细化 `mode` / `target_policy`。
  - 当确定性兜底已经安全时确认兜底策略。
- 禁止改动：
  - 障碍区执行中的全局 `passing_order`。
  - 低层速度、推力、电机控制。
  - 不存在的障碍物编号。

## 面板链路优化记录

- 在线 VLM 事件现在区分两个层次：
  - `online_replanning`：展示 VLM 原始候选是否相对兜底提出改写。
  - `execution_feedback`：展示候选经过安全验证和安全投影后，实际是否进入底层执行。
- 面板固定展示闭环：
  - `VLM 原始建议`
  - `安全验证`
  - `时效性验证`
  - `安全投影`
  - `实际进入底层执行`
- 即使安全投影后没有实际改变底层计划，也会记录“VLM 策略完成安全闭环复核”，用于说明 VLM 参与了判断，但执行层选择保持当前安全计划。
- 若 VLM 返回时最前方 UAV 已经过于接近候选障碍的 `observe_x`，或剩余 TTC 不足，会记录“VLM 策略因时效性过期未进入执行”。这类结果只作为认知证据保存，不允许改写底层执行缓存。
- 字段命名拆分为：
  - `VLM候选计划编号` / `VLM候选来源`
  - `实际执行计划编号` / `实际执行来源`
  这样可以避免把 VLM 原始建议误读为直接控制命令。
- 默认时效性阈值：
  - `ONLINE_VLM_MIN_REMAINING_X = 2.0`
  - `ONLINE_VLM_MIN_REMAINING_TTC = 2.5`
  - `ONLINE_VLM_TTC_MIN_SPEED_X = 0.25`

## 论文包装建议

这版 demo 可以作为“稳定底层控制 + 异步多模态高层策略 + 返回时效性门控”的复杂场景实验。底层保证连续安全通行，VLM 负责基于视觉形态和 SFSC 进行可验证、不过期的局部策略修正，体现多模态高层认知和底层控制稳定性的分层融合。

更完整的算法包装建议是：“时效性门控的安全投影多模态 VLM 协同通行框架”。它不是 VLM 直接飞控，而是让 VLM 在认知层提出 PassagePlan 候选，再由验证器、时效性门控和投影器将候选压入底层可执行边界。该结构能同时强调多模态智能性、实时安全性和工程可落地性。
