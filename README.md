# SV-STCP Quadrocopter Avoidance

面向复杂动态混合障碍区的多无人机 SFSC 验证时空协同通行算法。

本仓库是一个基于 MuJoCo 的多无人机协同避障研究工程。当前核心工作不是普通单机避障，而是研究多架无人机在动态窄洞、实体绕行障碍、顶部越障障碍混合出现时，如何通过“SFSC 传感器融合结构化上下文 + LLM 高层规划 + 安全验证 + 底层稳定执行”完成协同通行。

算法名称：

```text
SV-STCP: Sensor-Fused/Semantics-Verified Spatio-Temporal Cooperative Passage
```

中文可表述为：

```text
传感器融合结构化上下文验证的多无人机动态混合障碍时空协同通行算法
```

## 1. 项目核心思想

本项目的核心不是让 LLM 直接控制无人机，而是将系统分成三层：

```text
SFSC/LLM 层：融合非视觉传感器上下文，生成高层通行策略
验证/规划层：检查策略合法性，提供确定性 fallback
执行/控制层：完成连续飞行控制、时序穿越、侧绕、越顶和恢复编队
```

也就是说：

- LLM 不输出电机推力；
- LLM 不直接输出每一帧速度；
- LLM 输出的是高层通行策略；
- 所有 LLM 策略必须通过安全验证器；
- 如果 LLM 超时、失败或输出不安全，系统会自动回退确定性策略；
- 在线 LLM 重规划采用异步机制，不阻塞 viewer 和控制循环。

## 2. 当前已实现能力

当前稳定 demo 已实现以下闭环：

- 五架无人机初始横向编队飞行；
- 动态窄洞/窄门的蛇形时序穿越；
- 动态洞口中心预测与 y/z 对齐；
- 实体箱体障碍侧绕；
- 超宽实体障碍顶部越障；
- 穿越型障碍与绕行型障碍混合过渡；
- 单机清障状态判断，避免全队错误同步切换；
- x 方向时空槽约束，降低追尾和挤洞风险；
- 近障高度保底机制，避免高度未对齐时硬冲；
- LLM 高层策略 JSON 接口；
- Qwen/DashScope 接入；
- 安全验证器与 fallback；
- 有限视野在线感知；
- 异步在线 LLM 重规划；
- 实时 SFSC + LLM 证据链面板，用于展示传感器融合结构化上下文、LLM 输入/输出、验证和执行反馈。

## 3. 仓库结构

```text
assets/skydio_x2/
    MuJoCo 飞行器模型、贴图和主场景 XML

src/
    无人机模型、控制器、几何工具、运动规划和算法模块

src/passage_planning/
    SV-STCP 的核心算法模块
    包括 SFSC 编码、可通行性分析、策略规划、LLM 接口、
    安全验证、在线感知、异步重规划、SFSC + LLM 证据链面板和指标统计

tests/
    当前稳定 demo、策略框架测试、LLM 接口测试、在线感知与重规划测试

ALGORITHM_SUMMARY.md
    算法阶段性总结

SV_STCP_DETAILED_ALGORITHM_REPORT.md
    更详细的方法报告
```

## 4. 环境准备

推荐使用 Python 3.10 或 3.11。

创建虚拟环境示例：

```bash
conda create -n quadrocopter python=3.10 -y
conda activate quadrocopter
pip install -r requirements.txt
```

如果使用 Ubuntu，并且需要打开 SFSC + LLM 证据链面板，建议确认系统具备 Tk：

```bash
sudo apt-get install python3-tk
```

MuJoCo viewer 需要图形显示环境。如果在服务器或远程终端运行，需要确保 X11/桌面显示可用。

## 5. Qwen API 配置

主 demo 默认启用 Qwen/DashScope 作为 LLM 策略生成器。请准备本地环境变量：

```bash
export DASHSCOPE_API_KEY='你的 DashScope API Key'
export DASHSCOPE_MODEL='qwen-plus'
export DASHSCOPE_VLM_MODEL='qwen3-vl-flash'
export DASHSCOPE_VLM_MAX_TOKENS='900'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
```

`DASHSCOPE_VLM_MODEL` 默认建议使用 flash 轻量版本，便于降低前视 RGB + SFSC 在线重规划的等待时间。如果本地 `.env.local` 中配置了其他模型，实际运行会优先使用 `.env.local` 或 shell 环境变量中的值。
`DASHSCOPE_VLM_MAX_TOKENS` 用于限制 VLM JSON 输出长度，避免模型生成过长解释拖慢在线返回。

也可以新建 `.env.local`：

```bash
cp .env.local.example .env.local
```

然后填入自己的 key。`.env.local` 已被 `.gitignore` 排除，不会提交到仓库。

如果没有配置 Qwen API，系统仍会触发 fallback 机制，但无法展示完整的真实在线 LLM 返回效果。为了复现本项目的完整演示，建议配置真实 API key。

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
- 策略 JSON 导入导出；
- LLM 输出解析；
- 安全验证与 fallback；
- 在线感知窗口；
- 在线局部重规划；
- 指标统计。

## 7. 运行主 demo：动态混合障碍 + Qwen + 语义面板

主 demo 命令：

```bash
python -m unittest tests.test_dynamic_mixed_bypass_obstacle_zone.TestDynamicMixedBypassObstacleZone.test_dynamic_mixed_bypass_obstacle_zone_viewer
```

运行后会看到：

1. MuJoCo viewer 中五架无人机通过动态混合障碍区；
2. SFSC + LLM 证据链面板同步显示传感器融合上下文、LLM 相关信息和执行反馈；
3. 在线重规划事件显示 fallback、LLM submitted、pending、accepted/rejected 等状态；
4. demo 完成后语义面板不会自动关闭，需要手动关闭，便于复盘 LLM 和底层执行状态。

主 demo 场景包含：

- 多个动态穿越型障碍；
- 动态实体侧绕障碍；
- 超宽实体顶部越障障碍；
- 不同通过高度的动态孔洞；
- 在线有限视野重规划；
- Qwen 高层策略生成；
- 安全验证与 fallback；
- 最终恢复编队。

## 8. SFSC + LLM 证据链面板说明

SFSC 是 Sensor-Fused Structured Context，中文为“传感器融合结构化上下文”。它不是直接把 XML 或仿真底层变量交给 LLM，也不是把视觉图像重新画成语义图后再喂给模型。在真实系统中，SFSC 对应的是 IMU/里程计、雷达/测距、机间通信、控制器状态和快速非 VLM 感知模块融合后的结构化上下文；在当前 MuJoCo demo 中，系统用仿真运行时状态构造同样格式的 SFSC，保证实验可复现。

后续多模态扩展的输入形式是：

```text
无人机机载前视 RGB 图像 + SFSC 非视觉结构化上下文
        ↓
VLM/LLM 高层策略分析
        ↓
安全验证器
        ↓
异步策略缓冲区
        ↓
底层稳定执行
```

当前视觉关键帧已经从 MuJoCo 场景俯视图切换为“当前 x 方向最前方 UAV”的机载前视 RGB 相机。系统会在 `uav0_front_rgb` 到 `uav4_front_rgb` 之间动态选择最前方无人机视角，避免固定后方无人机被前机遮挡。这类图像用于模拟真实探路无人机摄像头看到的局部障碍外观和通道形态，而不是从全局视角重新绘制语义图。

主 demo 当前采用统一的多模态 VLM 在线策略链路：系统在有限视野发现待规划障碍时，将最近的最前方 UAV 前视 RGB 和 SFSC 一起输入 Qwen VLM，由 VLM 输出标准 `PassagePlan` JSON。该策略必须通过 `StrategyValidator`，验证通过后才会合并到在线控制缓存；如果真实 VLM 超时、失败或输出不合法，确定性 fallback 会继续保持控制。

SFSC + LLM 证据链面板用于回答“LLM 到底接收了什么上下文、输出了什么策略、系统是否采用”：

```text
SFSC 上下文 / LLM 输入：
    非视觉传感器、无人机状态、队形通信、距离估计和规划状态被快速编码成什么结构化上下文

LLM 宏观策略 / 安全验证：
    LLM 输出了什么通行策略、是否通过验证、是否 fallback

底层执行反馈：
    无人机当前是在穿越、绕行、越顶、等待还是恢复编队

事件时间线：
    感知、fallback、LLM 请求、LLM 返回、策略合并、阶段切换的全过程
```

建议录屏时同时展示 MuJoCo viewer 和该面板，这样可以直观看到：

```text
SFSC 上下文形成 -> LLM 策略生成 -> 安全验证 -> 底层执行
```

## 9. 算法流程

整体流程如下：

```text
非视觉传感器融合 / 在线感知
        ↓
SFSC 编码
        ↓
可通行性分析
        ↓
确定性基准策略
        ↓
LLM 高层策略生成
        ↓
安全验证器
        ↓
确定性 fallback / LLM 策略合并
        ↓
异步在线策略缓冲区
        ↓
底层稳定执行内核
        ↓
动态穿越、实体绕行、顶部越障、恢复编队
```

## 10. 核心创新点

### 10.1 SFSC 障碍上下文建模

障碍不再只是几何体，而是由非视觉传感器融合得到的可规划上下文对象：

- 穿越型障碍；
- 实体绕行障碍；
- 动态孔洞；
- 可越顶障碍；
- 高风险窄洞；
- 厚实体障碍。

这使 LLM 能够理解障碍功能，而不是直接处理底层仿真变量。

### 10.2 多无人机时空协同

对窄洞类障碍，算法将多架无人机组织成蛇形时空队列：

- 每架无人机有通行顺序；
- 每架无人机有时间槽；
- 后机受前机清障状态约束；
- 避免多机同时挤入窄洞。

### 10.3 LLM under verification

LLM 只做高层策略建议，不直接控制飞行。策略必须通过验证器：

- 通过则进入执行；
- 不通过则 fallback；
- 超时或异常也 fallback。

### 10.4 异步在线 LLM 重规划

在线飞行时，系统不等待 LLM 返回：

```text
先执行确定性安全策略
后台请求 Qwen
返回后验证
验证通过再合并策略
```

这样既保留 LLM 的语义策略能力，又避免控制循环卡顿。

### 10.5 可解释语义呈现

SFSC + LLM 证据链面板将传感器融合结构化上下文、LLM 输入/输出、验证和执行反馈全部显示出来，使算法不只是“用了 LLM”，而是能直观看到 LLM 如何在非视觉传感器上下文约束下参与决策。

## 11. 常见问题

### 11.1 viewer 打不开

请确认当前机器具备图形显示环境。远程服务器需要配置 X11 转发或桌面环境。

### 11.2 Qwen 请求失败

检查：

```bash
echo $DASHSCOPE_API_KEY
echo $DASHSCOPE_MODEL
echo $DASHSCOPE_BASE_URL
```

如果没有配置 API key，系统会 fallback，但无法复现完整真实 LLM 在线效果。

### 11.3 demo 结束后 unittest 不退出

这是预期行为。为了方便复盘，SFSC + LLM 证据链面板在 demo 完成后会保持打开。手动关闭面板窗口后，unittest 会继续结束。

### 11.4 如何只验证算法、不打开界面

运行第 6 节中的非 viewer 单元测试即可。

## 12. 当前阶段结论

当前仓库已经形成一套完整的多无人机动态混合障碍通行算法闭环：

```text
语义建模
时空协同
LLM 高层策略
安全验证
异步在线重规划
底层稳定执行
SFSC + LLM 证据链展示
```

后续建议围绕该稳定版本开展系统实验和论文整理，包括：

- 无 LLM / 有 LLM 对比；
- 同步 LLM / 异步 LLM 对比；
- 有验证器 / 无验证器对比；
- 有时间槽 / 无时间槽对比；
- 成功率、总耗时、最小机间距、障碍净空、LLM 延迟和 fallback 次数统计。
