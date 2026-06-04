# SV-STCP Quadrocopter Avoidance

面向复杂动态混合障碍区的多无人机 **SFSC + VLM 语义验证时空协同通行算法**。

本仓库是一个基于 MuJoCo 的多无人机协同通行研究工程。当前核心工作不是普通单机避障，而是研究多架无人机在动态窄洞、实体障碍、悬空障碍、顶部/底部可通行区域混合出现时，如何通过：

```text
SFSC 传感器融合结构化上下文
        +
前视 RGB 视觉证据
        +
VLM 高层 PassagePlan 候选
        +
确定性安全验证 / 时效性门控 / 安全投影
        +
底层稳定执行
```

完成可解释、可验证、不中断控制循环的协同通行。

算法名称：

```text
SV-STCP: Sensor-Fused/Semantics-Verified Spatio-Temporal Cooperative Passage
```

中文可表述为：

```text
传感器融合结构化上下文验证的多无人机动态混合障碍时空协同通行算法
```

## 1. 项目核心思想

本项目的核心不是让 VLM 直接控制无人机，而是把 VLM 放在高层策略候选生成位置。底层控制权始终由确定性执行器、安全验证器和在线策略缓存共同约束。

```text
多模态语义层：
    SFSC 非视觉结构化上下文 + 当前最前方 UAV 前视 RGB
    生成 VLM 高层 PassagePlan 候选

安全准入层：
    Strategy validation
    Timeliness Gate
    Safety Projection
    Online Strategy Buffer

底层执行层：
    蛇形时序穿越
    实体侧绕 / 越顶 / 下穿
    机间防碰撞约束
    per-obstacle 执行缓存
    清障后恢复编队
```

也就是说：

- VLM 不输出电机推力；
- VLM 不直接输出每一帧速度；
- VLM 输出的是高层通行策略候选；
- VLM 策略必须通过安全验证、返回时效性门控和安全投影；
- 如果 VLM 超时、失败、过期或输出不安全，系统继续执行确定性 fallback；
- 在线 VLM 重规划采用异步机制，不阻塞 MuJoCo viewer 和控制循环；
- VLM 的作用重点体现在策略级选择，例如侧绕、越顶、下穿、窗口延长和局部策略确认，而不是简单数值微调。

## 2. 当前已实现能力

当前稳定 demo 已实现以下闭环：

- 五架无人机初始横向编队飞行；
- 动态窄洞/窄门的蛇形时序穿越；
- 动态洞口中心预测与 y/z 对齐；
- 对齐完成后的快速穿越，避免在门内反复慢速修正；
- 实体障碍路线族：左绕、右绕、分流绕行、顶部越障、底部下穿；
- 悬空实体障碍的底部净空描述与 VLM 路线选择提示；
- 实体障碍越顶/下穿几何可行性验证；
- per-obstacle 在线策略缓存，每架无人机按自身 x 位置选择当前障碍策略；
- 绕障阶段机间防碰撞建议约束；
- VLM 异步在线策略请求；
- 返回时效性门控，根据 remaining_x 和 TTC 判断候选是否仍可进入执行；
- 安全投影，限制过晚 observe_x、过短 clear_x 和过小窗口噪声；
- 当前执行策略、确定性兜底、VLM 候选和实际采纳结果的实时面板展示；
- 详细语义证据链日志，用于复盘 VLM 是否真正改变高层策略。

## 3. 仓库结构

```text
assets/skydio_x2/
    MuJoCo 飞行器模型、贴图和主场景 XML

src/
    无人机模型、控制器、几何工具、运动规划和算法模块

src/passage_planning/
    SV-STCP 核心算法模块
    包括 SFSC 编码、可通行性分析、策略规划、VLM 多模态输入、
    安全验证、在线感知、异步重规划、证据链面板和指标统计

tests/
    当前稳定 demo、策略框架测试、在线感知与重规划测试、
    VLM 复杂混合障碍 demo 和兼容性策略接口测试

MULTIMODAL_VLM_FUSION_DESIGN.md
    SFSC + 前视 RGB 多模态 VLM 设计说明

MULTIMODAL_COMPLEX_DEMO_TUNING_LOG.md
    复杂混合障碍 demo 底层调参和策略演进记录

SV_STCP_STAGE_ALGORITHM_REPORT.docx
    给导师查看的阶段性算法报告

SV_STCP_DETAILED_ALGORITHM_REPORT.md
    更详细的方法记录
```

## 4. 环境准备

推荐使用 Python 3.10 或 3.11。

```bash
conda create -n quadrocopter python=3.10 -y
conda activate quadrocopter
pip install -r requirements.txt
```

如果使用 Ubuntu，并且需要打开实时证据链面板，建议确认系统具备 Tk：

```bash
sudo apt-get install python3-tk
```

MuJoCo viewer 需要图形显示环境。如果在服务器或远程终端运行，需要确保 X11/桌面显示可用。

## 5. Qwen VLM API 配置

复杂 demo 默认使用 DashScope/Qwen VLM 作为在线高层策略候选生成器。请准备本地环境变量：

```bash
export DASHSCOPE_API_KEY='你的 DashScope API Key'
export DASHSCOPE_MODEL='qwen-plus'
export DASHSCOPE_VLM_MODEL='qwen3-vl-flash'
export DASHSCOPE_VLM_MAX_TOKENS='900'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
```

`DASHSCOPE_VLM_MODEL` 默认建议使用 flash 轻量版本，便于降低前视 RGB + SFSC 在线重规划的等待时间。

`DASHSCOPE_VLM_MAX_TOKENS` 用于限制 VLM JSON 输出长度，避免模型生成过长解释拖慢在线返回。

也可以新建本地配置文件：

```bash
cp .env.local.example .env.local
```

然后填入自己的 key。`.env.local` 已被 `.gitignore` 排除，不会提交到仓库。

如果没有配置 Qwen API，系统仍会触发 deterministic fallback，但无法展示真实在线 VLM 返回、候选验证和安全投影的完整效果。

## 6. 快速验证：不打开 viewer

以下测试用于验证核心算法模块，不需要图形界面：

```bash
python -m unittest \
  tests.test_online_perception \
  tests.test_online_replanner \
  tests.test_passage_strategy_framework \
  tests.test_llm_strategy_interface \
  tests.test_passage_metrics \
  tests.test_complex_strategy_framework
```

这些测试覆盖：

- SFSC 障碍上下文编码；
- 可通行性分析；
- 时空协同策略生成；
- 高层 PassagePlan JSON 导入导出；
- 策略候选解析与 fallback；
- 安全验证；
- 在线感知窗口；
- 在线局部重规划；
- 指标统计。

说明：仓库中仍保留 `test_llm_strategy_interface` 这类历史命名测试，用于验证通用高层策略 JSON 兼容性。当前主链路已经转向 VLM 多模态输入。

## 7. 运行主 demo：复杂混合障碍 + VLM + 证据链面板

当前推荐主 demo：

```bash
python -m unittest tests.test_multimodal_complex_strategy_zone.TestMultimodalComplexStrategyZone.test_multimodal_complex_strategy_zone_viewer
```

运行后会看到：

1. MuJoCo viewer 中五架无人机通过复杂混合障碍区；
2. 实时面板显示当前确定性兜底策略、VLM 是否起作用、核心策略改变和实际进入底层的执行结果；
3. 在线重规划事件显示 fallback accepted、VLM submitted、VLM candidate returned、timeliness rejected、safety projected、execution refreshed 等状态；
4. demo 完成后语义面板不会自动关闭，需要手动关闭，便于复盘 VLM 和底层执行状态。

主 demo 场景包含：

- 多个动态穿越型障碍；
- 多个实体绕行障碍；
- 悬空实体障碍；
- 顶部越障与底部下穿候选；
- 不同通过高度的动态孔洞；
- 在线有限视野重规划；
- Qwen VLM 高层策略生成；
- 安全验证、时效性门控、安全投影与 fallback；
- 最终恢复编队。

旧版动态混合障碍 demo 仍可运行，但当前论文第一版建议以复杂多模态 demo 作为主展示版本。

## 8. SFSC + 前视 RGB 多模态输入

SFSC 是 Sensor-Fused Structured Context，中文为“传感器融合结构化上下文”。它不是直接把 XML 或仿真底层变量交给 VLM，也不是把视觉图像重新画成语义图后再喂给模型。

在真实系统中，SFSC 对应的是 IMU/里程计、雷达/测距、机间通信、控制器状态和快速非 VLM 感知模块融合后的结构化上下文。在当前 MuJoCo demo 中，系统用仿真运行时状态构造同样格式的 SFSC，保证实验可复现。

最终采用的多模态输入形式是：

```text
当前最前方 UAV 前视 RGB
        +
SFSC 非视觉结构化上下文
        +
确定性兜底 PassagePlan
        +
路线决策摘要
        ↓
Qwen VLM 高层策略候选
        ↓
安全验证
        ↓
返回时效性门控
        ↓
安全投影
        ↓
Online Strategy Buffer
        ↓
底层稳定执行
```

这样设计的核心原因是：

- 只给前视 RGB，VLM 很难稳定估计距离、尺寸、洞口宽度、底部净空和队形约束；
- 只给 SFSC，会丢失真实视觉形态、遮挡关系和局部通道外观；
- SFSC 提供可验证的数值约束；
- RGB 提供真实外观证据；
- VLM 在二者结合下做高层路线族判断，而不是凭图猜控制指令。

当前视觉关键帧来自“当前 x 方向最前方 UAV”的机载前视 RGB 相机。系统会在 `uav0_front_rgb` 到 `uav4_front_rgb` 之间动态选择最前方无人机视角，避免固定后方无人机被前机遮挡。

## 9. 在线 VLM 策略准入链路

在线策略链路如下：

```text
有限视野发现待规划障碍
        ↓
确定性 fallback 立即进入控制缓存
        ↓
保存当前前视 RGB
        ↓
构造 SFSC + RGB 多模态输入包
        ↓
异步提交 Qwen VLM 请求
        ↓
VLM 返回 PassagePlan 候选
        ↓
安全验证器检查格式、几何、路线族和时序
        ↓
Timeliness Gate 检查 remaining_x 和 TTC
        ↓
Safety Projection 限制高风险改写
        ↓
合并进 per-obstacle 在线执行缓存
```

其中 Timeliness Gate 解决的是 VLM 推理延迟问题。即使 VLM 返回格式正确，如果无人机已经太接近对应障碍，候选也不会进入执行，只保留为证据链记录。

```text
remaining_x = deadline_x - x_front
TTC = remaining_x / max(v_front, v_min)
```

只有当剩余距离和 TTC 同时超过阈值，VLM 候选才允许进入安全投影。

## 10. 证据链面板说明

当前面板围绕四个问题组织：

```text
1. 当前兜底 / 执行状态
   当前障碍原本确定性策略是什么，每架无人机正在执行什么避障状态。

2. VLM 是否起作用
   VLM 是否改变兜底，是否只是确认兜底，还是提出了路线族改变。

3. 实际进入底层的改动
   候选是否通过安全验证、TTC 是否通过、最终是否写入执行缓存。

4. 详细证据链
   保留视觉帧、SFSC 输入包、候选返回、验证结果和执行反馈。
```

录屏或汇报时建议同时展示 MuJoCo viewer 和该面板，这样可以直接说明：

```text
SFSC + 前视 RGB 输入
        ↓
VLM 原始候选
        ↓
安全验证
        ↓
时效性门控
        ↓
安全投影
        ↓
实际底层执行
```

## 11. 核心创新点

### 11.1 SFSC + RGB 的互补式多模态输入

创新点不在于简单把图像和文本拼接给 VLM，而在于明确划分两类证据的职责：SFSC 提供可验证数值约束，RGB 提供真实形态证据。

这解决了 VLM 单看图时几何估计不稳定，以及确定性结构化上下文缺少视觉外观理解的问题。

### 11.2 VLM 作为高层策略候选生成器

VLM 只生成 PassagePlan 候选，不直接控制无人机。这使大模型能力可以进入机器人决策，同时安全责任仍由确定性验证和执行层承担。

### 11.3 确定性 fallback + 异步 VLM

VLM 延迟不可避免。系统先执行确定性安全策略，后台请求 VLM，返回后再验证和合并。这样既保留 VLM 的语义策略能力，又避免控制循环卡顿。

### 11.4 Timeliness Gate

返回时效性门控判断 VLM 候选是否仍处于可提前介入窗口。过期候选不会进入执行，避免近场突然改写策略。

### 11.5 实体路线族拆分

实体绕障不再只有通用 edge bypass，而是拆成左绕、右绕、分流、越顶、下穿等路线族。这样 VLM 的作用可以体现在真正的策略级改变上。

### 11.6 per-obstacle 策略缓存

复杂混合障碍中，前机可能已经处理下一障碍，后机仍在处理上一障碍。按障碍缓存、按无人机位置取用策略，可以避免全队错误同步切换。

### 11.7 可解释证据链

系统不只展示“用了 VLM”，而是展示兜底是什么、VLM 改了什么、候选是否过期、最终是否进入执行。这对于论文汇报和算法可信性非常关键。

## 12. 常见问题

### 12.1 viewer 打不开

请确认当前机器具备图形显示环境。远程服务器需要配置 X11 转发或桌面环境。

### 12.2 Qwen VLM 请求失败

检查：

```bash
echo $DASHSCOPE_API_KEY
echo $DASHSCOPE_VLM_MODEL
echo $DASHSCOPE_BASE_URL
```

如果没有配置 API key，系统会 fallback，但无法复现完整真实 VLM 在线效果。

### 12.3 demo 结束后 unittest 不退出

这是预期行为。为了方便复盘，证据链面板在 demo 完成后会保持打开。手动关闭面板窗口后，unittest 会继续结束。

### 12.4 如何只验证算法、不打开界面

运行第 6 节中的非 viewer 单元测试即可。

## 13. 当前阶段结论

当前仓库已经形成一套完整的多无人机动态混合障碍通行算法闭环：

```text
SFSC 结构化上下文
前视 RGB 视觉证据
VLM 高层策略候选
确定性 fallback
安全验证
返回时效性门控
安全投影
per-obstacle 在线策略缓存
底层稳定执行
证据链展示
```

后续建议围绕该稳定版本开展系统实验和论文整理，包括：

- 无 VLM / 有 VLM 对比；
- 无 Timeliness Gate / 有 Timeliness Gate 对比；
- 无路线族拆分 / 有路线族拆分对比；
- 无安全投影 / 有安全投影对比；
- 成功率、总耗时、最小机间距、障碍净空、VLM 延迟和 fallback 次数统计。
