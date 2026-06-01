# SV-STCP Quadrocopter Avoidance

面向复杂动态混合障碍区的多无人机语义验证时空协同通行算法。

本仓库是一个基于 MuJoCo 的多无人机协同避障研究工程。当前核心工作不是普通单机避障，而是研究多架无人机在动态窄洞、实体绕行障碍、顶部越障障碍混合出现时，如何通过“语义规划 + 安全验证 + 底层稳定执行”完成协同通行。

算法名称：

```text
SV-STCP: Semantics-Verified Spatio-Temporal Cooperative Passage
```

中文可表述为：

```text
语义验证的多无人机动态混合障碍时空协同通行算法
```

## 1. 项目核心思想

本项目的核心不是让 LLM 直接控制无人机，而是将系统分成三层：

```text
LLM/语义层：理解障碍语义，生成高层通行策略
验证/规划层：检查策略合法性，提供确定性 fallback
执行/控制层：完成连续飞行控制、时序穿越、侧绕、越顶和恢复编队
```

也就是说：

- LLM 不输出电机推力；
- LLM 不直接输出每一帧速度；
- LLM 输出的是高层语义策略；
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
- 实时语义证据链面板，用于展示 LLM 输入、输出、验证和执行反馈。

## 3. 仓库结构

```text
assets/skydio_x2/
    MuJoCo 飞行器模型、贴图和主场景 XML

src/
    无人机模型、控制器、几何工具、运动规划和算法模块

src/passage_planning/
    SV-STCP 的核心算法模块
    包括语义场景编码、可通行性分析、策略规划、LLM 接口、
    安全验证、在线感知、异步重规划、语义证据链面板和指标统计

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

如果使用 Ubuntu，并且需要打开语义证据链面板，建议确认系统具备 Tk：

```bash
sudo apt-get install python3-tk
```

MuJoCo viewer 需要图形显示环境。如果在服务器或远程终端运行，需要确保 X11/桌面显示可用。

## 5. Qwen API 配置

主 demo 默认启用 Qwen/DashScope 作为 LLM 策略生成器。请准备本地环境变量：

```bash
export DASHSCOPE_API_KEY='你的 DashScope API Key'
export DASHSCOPE_MODEL='qwen-plus'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
```

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

- 障碍语义编码；
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
2. 语义证据链面板同步显示 LLM 相关信息；
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

## 8. 语义证据链面板说明

语义证据链面板用于回答“LLM 到底做了什么”：

```text
场景语义 / LLM 输入：
    当前看见了哪些障碍、障碍是什么类型、运动和尺寸语义是什么

LLM 宏观策略 / 安全验证：
    LLM 输出了什么通行策略、是否通过验证、是否 fallback

底层执行反馈：
    无人机当前是在穿越、绕行、越顶、等待还是恢复编队

事件时间线：
    感知、fallback、LLM 请求、LLM 返回、策略合并、阶段切换的全过程
```

建议录屏时同时展示 MuJoCo viewer 和该面板，这样可以直观看到：

```text
环境语义提取 -> LLM 策略生成 -> 安全验证 -> 底层执行
```

## 9. 算法流程

整体流程如下：

```text
MuJoCo 场景 / 在线感知
        ↓
语义场景编码
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

### 10.1 语义化障碍建模

障碍不再只是几何体，而是带有功能属性的语义对象：

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

语义证据链面板将 LLM 的输入、输出、验证和执行反馈全部显示出来，使算法不只是“用了 LLM”，而是能直观看到 LLM 如何参与决策。

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

这是预期行为。为了方便复盘，语义证据链面板在 demo 完成后会保持打开。手动关闭面板窗口后，unittest 会继续结束。

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
语义证据链展示
```

后续建议围绕该稳定版本开展系统实验和论文整理，包括：

- 无 LLM / 有 LLM 对比；
- 同步 LLM / 异步 LLM 对比；
- 有验证器 / 无验证器对比；
- 有时间槽 / 无时间槽对比；
- 成功率、总耗时、最小机间距、障碍净空、LLM 延迟和 fallback 次数统计。
