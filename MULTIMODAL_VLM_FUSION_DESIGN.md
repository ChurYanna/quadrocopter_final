# SFSC + 前视 RGB 多模态 VLM 融合设计

## 1. 核心定位

多模态扩展不是把视觉图像和结构化信息重复喂给模型，而是把两类互补证据交给 VLM/LLM：

- **前视 RGB**：来自当前最前方无人机的机载相机，保留真实视觉外观证据，例如障碍物形状、孔洞轮廓、遮挡关系、局部通道形态。
- **SFSC**：传感器融合结构化上下文，由非视觉传感器和系统状态快速编码得到，例如位姿、机间距离、障碍距离、队形宽度、速度能力、已规划状态、风险等级。

因此，多模态输入不是：

```text
结构化语义 -> 画成图 -> 再给 VLM
```

而是：

```text
前视 RGB 原始视觉证据 + SFSC 非视觉结构化约束 -> VLM 高层策略判断
```

这能避免“绕弯子”的问题，也能解释为什么结构化上下文和视觉图像不是重复信息。

## 2. 为什么只取最前方无人机视角

当前视觉采样策略是 `frontmost_uav`：

```text
每次关键帧触发时，系统读取 5 架无人机的 x 方向位置；
选择 x 最大的无人机作为当前探路机；
使用该无人机的机载前视 RGB 相机保存关键帧。
```

原因：

- 最前方无人机更接近当前待处理障碍，视觉信息最直接。
- 后方无人机视角容易被前方无人机遮挡，导致 VLM 误判障碍形态。
- 多无人机同时给图会增加 VLM 输入冗余，降低实时性和解释清晰度。
- 单一探路视角更接近真实系统中的 scout/leader perception 逻辑。

这并不代表其它无人机的信息丢失。其它无人机的位置、速度、队形状态、机间距离仍然通过 SFSC 输入 VLM。

## 3. VLM 输入包应该包含什么

建议定义一个统一的多模态输入包：

```json
{
  "task": "multi_uav_dynamic_obstacle_passage",
  "event": "online_replanning_triggered",
  "timestamp": 57.23,
  "front_rgb": {
    "image_path": "logs/visual_frames/00057.23_online_submitted.png",
    "camera_name": "uav3_front_rgb",
    "source_uav_id": 3,
    "selection_policy": "frontmost_uav",
    "view_type": "onboard_front_rgb"
  },
  "sfsc": {
    "uav_team": "...",
    "visible_obstacles": "...",
    "mission_preference": "...",
    "deterministic_reference_plan": "...",
    "current_strategy_buffer": "...",
    "safety_constraints": "..."
  },
  "request": {
    "expected_role": "high_level_strategy_advisor",
    "output_format": "json_only",
    "must_not_output": [
      "motor thrust",
      "raw velocity command",
      "unvalidated collision decision"
    ]
  }
}
```

其中 SFSC 重点包含：

- 无人机数量、队形宽度、每架无人机当前位置和速度能力。
- 当前最前方无人机 ID，以及它为什么被选为视觉来源。
- 有限视野内的障碍列表、相对距离、粗略/详细可见状态。
- 每个障碍的功能类型：穿越型、实体绕行型、可越顶型等。
- 洞口宽度、通过高度、实体尺寸、运动趋势和风险等级。
- 当前已接受策略、待规划障碍、已清障障碍。
- 确定性 fallback 或参考策略。
- 安全验证硬约束。

前视 RGB 重点提供：

- 障碍外观是否和 SFSC 描述一致。
- 孔洞边界是否清晰。
- 障碍是否存在遮挡、倾斜、复杂形状或视觉异常。
- 局部通道是否存在额外视觉风险。

## 4. VLM 应该输出什么

VLM 不应该直接输出低层控制指令，而应该输出“可验证的高层策略建议”。推荐输出结构：

```json
{
  "version": "1.0",
  "source": "vlm",
  "confidence": 0.82,
  "visual_assessment": {
    "image_understanding": "front obstacle appears to be a moving aperture",
    "sfsc_consistency": "consistent",
    "occlusion_risk": "low",
    "morphology_risk": "medium"
  },
  "strategy_suggestion": {
    "mode": "snake_sequence",
    "target_policy": "predictive_center_crossing",
    "passing_order": [0, 1, 2, 3, 4],
    "reason": "formation width is larger than visible aperture; sequential crossing is safer"
  },
  "obstacle_strategies": [
    {
      "obstacle_id": "gate4",
      "mode": "snake_sequence",
      "target_policy": "predictive_center_crossing",
      "risk_note": "dynamic aperture, high pass-z alignment demand"
    }
  ],
  "safety_notes": [
    "keep validator mandatory",
    "do not advance if vertical clearance is unsafe"
  ]
}
```

输出字段分成三类：

- **视觉解释字段**：说明 VLM 从图像中看到了什么，以及图像是否支持 SFSC。
- **策略建议字段**：给出穿越、侧绕、越顶、等待、重排序等高层策略。
- **安全约束字段**：指出风险，但最终由安全验证器裁决。

## 5. VLM 输出不能做什么

VLM 禁止直接输出：

- 电机推力。
- 单帧速度控制。
- 绕过验证器的最终控制决策。
- 与 SFSC 硬约束冲突的通行顺序。
- 未覆盖所有无人机的时间槽。

如果 VLM 输出不完整、不合法或超时，系统必须继续使用确定性 fallback。

## 6. 和当前异步在线重规划如何结合

当前系统已经有异步在线 LLM 重规划：

```text
新障碍进入有限视野
        ↓
确定性 fallback 立即生成可执行策略
        ↓
异步提交 LLM/VLM 请求
        ↓
模型返回后进入安全验证器
        ↓
验证通过才合并到在线策略缓冲区
```

多模态 VLM 加入后，这条链路不变，只是请求内容从：

```text
SFSC -> LLM
```

升级为：

```text
frontmost UAV RGB + SFSC -> VLM
```

这样不会因为 VLM 慢而卡住 viewer，也不会因为视觉误判直接影响底层控制。

VLM 触发应限制在规划相关关键帧，例如初始观察、进入障碍区、在线感知发现新障碍、自适应顺序更新等。恢复编队、任务完成和最终状态帧只保留视觉证据，不再提交 VLM，避免模型对已经结束的场景继续输出无意义绕行策略。

## 7. 面板应该如何呈现

SFSC + VLM 证据链面板建议分成四类信息：

```text
1. SFSC / 多模态输入
   - 当前最前方 UAV 是谁
   - 前视 RGB 图片路径
   - 可见障碍和 SFSC 摘要
   - 本次为什么触发 VLM

2. VLM 宏观判断
   - 图像中障碍形态判断
   - 图像与 SFSC 是否一致
   - 遮挡风险、形态风险、通行风险

3. 候选策略 / 安全验证
   - VLM 原始建议穿越、侧绕、越顶还是等待
   - 原始建议相对确定性兜底改变了哪些窗口或策略
   - `StrategyValidator` 是否接受该候选

4. 底层执行反馈
   - VLM 原始建议
   - 安全验证
   - 时效性验证
   - 安全投影
   - 实际进入底层执行
   - 最终执行指标
```

这能让导师直观看到：

```text
前视图像看到了什么
SFSC 提供了什么非视觉约束
VLM 如何综合两者给出原始候选策略
验证器是否接受候选
返回时该候选是否仍然来得及执行
安全投影如何把候选压缩到可执行安全边界内
底层实际执行了什么
```

面板只展示压缩后的关键字段，例如链路阶段、一句话说明、策略是否进入控制、建议模式、风险等级、安全投影结果和执行计划变化。完整 VLM 输入包、VLM 输出和 SFSC 事件仍保存在 `logs/vlm_packets/` 与 `logs/semantic_trace_history.jsonl` 中，方便后续论文复盘。

关键展示链路固定为：

```text
VLM 原始建议 -> 安全验证 -> 时效性验证 -> 安全投影 -> 实际执行
```

这条链路是论文中避免“VLM 直接控制无人机”质疑的核心。VLM 只产生认知层 PassagePlan 候选；`StrategyValidator` 负责形式化合法性检查；Timeliness Gate 负责判断模型返回时障碍是否仍处于可提前介入窗口；Safety Projection 负责把候选策略限制在底层控制器可稳定执行的安全边界内；OnlineStrategyBuffer 和底层控制器只消费通过时效性门控和投影后的执行计划。

## 8. 推荐实现顺序

1. 固定当前 `frontmost_uav` 前视 RGB 关键帧采样。当前已完成。
2. 在面板中展示“front RGB + SFSC”的多模态输入候选，并保存 `logs/vlm_packets/*.json` 输入包。当前已完成。
3. 定义 VLM 输出 JSON schema，但暂时不接入控制。当前已完成。
4. 调用 mock VLM 做离线解释，面板显示 VLM 判断结果，并保存 `*_mock_result.json`。当前已完成。
5. 接入真实 Qwen VLM，使用异步后台线程提交请求，返回后保存结果并更新面板；失败时记录错误。当前已完成。
6. 将 VLM 输出改为标准 `PassagePlan` JSON，并接入安全验证器；只有验证通过的 VLM 策略才能进入在线控制缓存。当前已完成。
7. 做对比实验：SFSC-only、RGB-only、SFSC+RGB、同步 VLM、异步 VLM。

## 9. 论文包装要点

可以将该部分包装为：

```text
Hierarchical Multimodal Evidence Fusion for LLM-Assisted Multi-UAV Passage
```

当前更贴近最终系统的表述是：

```text
Safety-Projected Multimodal VLM Planning for Cooperative Multi-UAV Passage
```

中文可写为：

```text
面向多无人机协同通行的安全投影多模态 VLM 策略规划框架
```

核心创新可以拆成三点：

1. **前视 RGB 与 SFSC 的互补融合**：前视 RGB 提供机载真实视觉外观证据，SFSC 提供由非视觉传感器、队形状态和机间通信编码得到的结构化约束，避免把结构化语义重新画图再喂给模型的重复路径。
2. **VLM 候选策略与确定性兜底并行**：确定性策略先进入控制缓存保证实时连续性，VLM 异步返回后只作为高层候选策略参与局部优化，解决云端模型延迟和飞控实时性冲突。
3. **延迟鲁棒的安全闭环**：VLM 原始 PassagePlan 必须经过 `StrategyValidator`、返回时效性门控和 per-obstacle safety projection，才能进入底层执行缓存；因此系统允许 VLM 提升高层决策质量，但不允许过期或超出执行边界的 VLM 输出破坏底层稳定性。

最终算法链路可以概括为：

```text
Sensor/Communication Fusion -> SFSC
Frontmost UAV Camera -> Onboard Front RGB
SFSC + Front RGB -> VLM Candidate PassagePlan
Candidate PassagePlan -> StrategyValidator
Validated Candidate -> Timeliness Gate
Fresh Candidate -> Safety Projection
Projected Plan -> Online Per-Obstacle Strategy Buffer
Buffer -> Stable Snake/Bypass Execution Controller
```

中文可表述为：

```text
面向多无人机协同通行的分层多模态证据融合机制
```

核心创新点：

- 使用 SFSC 表示非视觉传感器融合约束，避免让 VLM 处理所有底层数值。
- 使用当前最前方无人机前视 RGB 保留真实视觉证据，避免上帝视角和语义图伪多模态。
- 使用异步机制解决 VLM 延迟问题。
- 使用安全验证器隔离 VLM 的不确定性，保证高层建议不会直接污染底层控制。
- 使用证据链面板展示“视觉证据 + SFSC 约束 + VLM 建议 + 安全验证 + 执行反馈”的完整闭环。
