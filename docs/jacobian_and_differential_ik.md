# Jacobian 与 Differential IK 说明

本文档说明以下代码在当前项目中的作用与物理含义：

- [`src/nero_workcell/core/kinematics_model.py`](../src/nero_workcell/core/kinematics_model.py)
  中的 `forward_tcp_position()` 与 `compute_tcp_position_jacobian()`
- [`src/nero_workcell/core/differential_ik_follower.py`](../src/nero_workcell/core/differential_ik_follower.py)
  中的 `_solve_joint_velocity()`

重点解释以下问题：

- `pin.forwardKinematics(...)` 在做什么
- `pin.computeFrameJacobian(...)` 计算出来的是什么
- 为什么代码里只取雅可比的前 3 行
- 为什么要用阻尼最小二乘法（Damped Least Squares, DLS）求解关节速度
- 为什么还要加 nullspace 项

## 1. 这套代码整体在做什么

当前项目里的跟随控制不是直接求一个“末端到目标点的完整关节解”，而是做速度级控制：

1. 根据当前关节角 `q` 计算 TCP 当前的位置
2. 计算当前时刻想要的 TCP 线速度 `desired_velocity`
3. 用雅可比矩阵把这个 TCP 线速度映射成关节速度 $\dot{q}$
4. 再把 $\dot{q}$ 积分成下一拍的目标关节角

也就是说，这是一套 differential IK 流程。

它的核心关系是：

$$ v_{tcp} = J(q) \cdot \dot{q} $$

其中：

- $v_{tcp}$ 是 TCP 的瞬时线速度
- $J(q)$ 是当前关节配置下的雅可比矩阵
- $\dot{q}$ 是关节速度向量

## 2. `pin.forwardKinematics(...)` 有什么用

在 [`src/nero_workcell/core/kinematics_model.py`](../src/nero_workcell/core/kinematics_model.py)
里，代码会先调用：

```python
pin.forwardKinematics(self.model, self.data, q)
```

它的作用是：

- 根据当前关节配置 `q`
- 结合机器人 URDF 描述的连杆和关节结构
- 计算整台机械臂当前的运动学状态
- 并把结果写入 `self.data`

可以把它理解成：

“已知每个关节现在转到哪里，先把整台机器人当前摆成什么样算出来。”

后面的这些量都依赖这一步：

- TCP 当前位姿
- 特定 frame 的位置
- 对应的雅可比矩阵

如果不先做 forward kinematics，后面读取的 frame 位姿或 Jacobian 就不对应当前 `q`。

## 3. `pin.computeFrameJacobian(...)` 有什么用

在 `compute_tcp_position_jacobian()` 里，代码调用：

```python
jacobian = pin.computeFrameJacobian(
    self.model,
    self.data,
    q,
    self.tcp_frame_id,
    pin.LOCAL_WORLD_ALIGNED,
)
```

它的作用是：

- 计算 TCP 这个 frame 在当前姿态下的雅可比矩阵
- 描述“关节速度变化”会如何影响“TCP 的瞬时运动”

直观理解：

- 第 1 列表示“只让第 1 个关节以单位速度转动时，TCP 会产生怎样的瞬时速度”
- 第 2 列表示“只让第 2 个关节以单位速度转动时，TCP 会产生怎样的瞬时速度”
- 其余各列同理

所有列组合起来，就得到了关节速度到末端速度的线性映射。

## 4. 为什么只取前 3 行

`computeFrameJacobian(...)` 返回的是 6 维雅可比，通常包含：

- 前 3 行：线速度部分
- 后 3 行：角速度部分

但当前项目里，`DifferentialIKFollower` 只跟踪 TCP 的位置，不跟踪末端朝向，所以代码里只保留前 3 行：

```python
return np.array(jacobian[:3, :], dtype=float)
```

因此，这里真正使用的是“位置雅可比”。

它对应的近似关系是：
$$v_{tcp\_linear} \approx J_{pos}(q) \cdot \dot{q}$$
而不是完整的 6 维 twist 控制。

## 5. `pin.LOCAL_WORLD_ALIGNED` 是什么意思

这个参数决定雅可比结果用什么坐标系来表达。

当前这里使用 `pin.LOCAL_WORLD_ALIGNED`，可以近似理解为：

- 这个 Jacobian 针对的是 TCP 这个局部 frame
- 但它的轴方向和世界坐标系对齐

对本项目来说，这样做的好处是：

- `tcp_position`
- `target.position`
- `desired_velocity`

这些量都更容易在统一的 base/world 风格坐标系下解释和使用。

## 6. 为什么还需要雅可比

因为控制器先算出来的是“TCP 应该怎么动”，不是“关节应该怎么转”。

例如，控制器得到：

```text
desired_velocity = [0.00, 0.00, -0.05]
```

这表示 TCP 希望沿 z 负方向以 `0.05 m/s` 运动。

但机器人底层真正能执行的是关节速度命令，所以必须回答：

“为了让 TCP 这样运动，每个关节应该各转多快？”

这个从笛卡尔速度到关节速度的转换，就是雅可比和 differential IK 的任务。

## 7. `_solve_joint_velocity()` 在做什么

[`src/nero_workcell/core/differential_ik_follower.py`](../src/nero_workcell/core/differential_ik_follower.py)
里的 `_solve_joint_velocity()` 先调用：

```python
jacobian = self.model.compute_tcp_position_jacobian(q)
```

然后使用阻尼伪逆计算：

```python
jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_matrix)
dq = jacobian_pinv @ desired_velocity
```

这里采用的就是阻尼最小二乘法（Damped Least Squares, DLS）。在 differential IK 里，
它通常写成：

```text
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} v_{desired}
```

它的目标是：

- 尽量找到一组关节速度 `dq`
- 使 TCP 线速度接近 `desired_velocity`

这是因为当前系统通常是冗余的：

- TCP 位置速度只有 3 维
- 但机械臂可能有 7 个关节自由度

此时 `J` 一般是 `3 x 7`，不能直接做普通逆矩阵求解，所以要用伪逆。

### 7.1 为什么不能直接求 `J^{-1}`

普通逆矩阵只适用于：

- 方阵
- 且矩阵满秩可逆

而当前这里使用的是位置雅可比，只控制 TCP 的线速度，所以：

- 末端任务维度是 `3`
- 机械臂关节自由度通常是 `7`

于是 Jacobian 的形状一般是：

```text
J ∈ R^(3 x 7)
```

也就是 3 行 7 列，不是 `n x n` 方阵，因此不能直接写：

```python
dq = np.linalg.inv(J) @ desired_velocity
```

这是最直接的原因。

再往深一点看，当前系统本身也是：

```text
v = J dq
```

其中：

- `v` 是 3 维 TCP 线速度
- `dq` 是 7 维关节速度

这表示：

- 你只有 3 个任务约束
- 却有 7 个未知量

所以通常不会只有唯一解，而是可能有很多组不同的 `dq` 都能实现同一个 TCP 速度。

一个直观例子是：

- 你只要求末端向下移动一点
- 但 7 轴机械臂可以通过不同的肘部、腕部组合来做到
- 所以同一个 `v`，往往不止一组 `dq`

这时候问题就不是“求一个唯一逆解”，而是：

“在很多可行解里，选一个更合适的解。”

伪逆做的正是这件事。对于这种冗余系统，Moore-Penrose 伪逆通常会给出一个标准解，可以直观理解成：

- 在能完成当前 TCP 任务的解里
- 选一个范数较小、比较自然的关节速度解

而当前项目里使用的还是阻尼伪逆，它不仅解决“`J` 不是方阵”的问题，还能在接近奇异位形时提高数值稳定性。

### 7.2 这三行代码逐行在做什么

对应代码：

```python
damping_matrix = (self.damping ** 2) * np.eye(jacobian.shape[0])
jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_matrix)
dq = jacobian_pinv @ desired_velocity
```

可以按下面三步理解：

1. 构造阻尼矩阵：

```python
damping_matrix = (self.damping ** 2) * np.eye(jacobian.shape[0])
```

- `jacobian.shape[0]` 对当前实现来说是 `3`
- `np.eye(...)` 生成一个 `3 x 3` 单位矩阵
- 再乘上 `self.damping ** 2`，得到一个小的正定矩阵

它的主要用途是给 `J J^T` 加一个数值稳定项，避免矩阵在接近奇异位形时太接近不可逆。

2. 计算阻尼伪逆：

```python
jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_matrix)
```

这对应阻尼最小二乘（damped least squares）形式：

$$J^+ = J^T (J J^T + \lambda^2 I)^{-1}$$

从矩阵维度上看，如果当前 `J` 是 `3 x 7`，那么：

- $J^T$ 是 `7 x 3`
- $J J^T$ 是 `3 x 3`
- $J J^T + λ^2 I$ 仍然是 `3 x 3`
- 它的逆矩阵也是 `3 x 3`
- 最终 $J^T (J J^T + λ^2 I)^{-1}$ 的结果是 `7 x 3`

这正好可以把 3 维的 TCP 线速度映射成 7 维的关节速度。

从物理意义上看，这一行可以拆成两层理解：

- `np.linalg.inv(jacobian @ jacobian.T + damping_matrix)` 在任务空间里做一个“稳定的逆映射”，把期望 TCP 速度转换成一个经过阻尼修正的任务空间响应
- 左边再乘 `jacobian.T`，相当于把这个任务空间响应重新分配到各个关节自由度上

所以这行代码的目标不是“求一个精确逆”，而是：

- 尽量让 TCP 按期望速度运动
- 同时别让关节速度在接近奇异位形时变得过大

为了便于手算，可以看一个简化的 `2 x 3` 例子。假设：

```text
J =
[[1, 0, 0],
 [0, 1, 0]]
```

这表示：

- 第 1 个关节主要影响 x 方向速度
- 第 2 个关节主要影响 y 方向速度
- 第 3 个关节对当前任务几乎没有贡献

再设阻尼系数 `λ = 0.1`，那么：

$J J^T = I$

$J J^T + λ^2 I = 1.01 I$

$(J J^T + λ^2 I)^{-1} ≈ 0.9901 I$

于是：

$J^+ = J^T (J J^T + λ^2 I)^{-1}$ ≈
```text
[[0.9901, 0     ],
 [0,      0.9901],
 [0,      0     ]]
```

如果期望 TCP 速度是：

```text
desired_velocity = [0.20, -0.10]
```

那么：
dq = $J^+$ * desired_velocity ≈ [0.198, -0.099, 0.0]

这表示：

- 第 1 个关节提供 x 方向运动
- 第 2 个关节提供 y 方向运动
- 第 3 个关节在这个简化任务下不需要参与

这个例子虽然比当前项目的 `3 x 7` 情况简单，但它展示了同一件事：这行公式是在把“末端想怎么动”转换成“各个关节应该怎么动”。

它不是普通逆矩阵，而是在以下目标之间做折中：

- 尽量实现期望 TCP 线速度
- 同时避免关节速度因为奇异性或病态矩阵而爆大

3. 求解主任务关节速度：

```python
dq = jacobian_pinv @ desired_velocity
```

这一步把期望的 TCP 线速度 `desired_velocity` 投到关节速度空间中，得到一组可执行的关节速度 `dq`。

换句话说，它在求解：

```text
J(q) * dq ≈ desired_velocity
```

这里的“≈”很重要，因为：

- 这是一个冗余系统
- 解通常不是唯一的
- 引入阻尼后也不追求严格精确匹配，而是追求稳定、平滑、可执行的近似解

举个直观例子，如果：

```text
desired_velocity = [0.00, 0.00, -0.05]
```

表示 TCP 希望向下运动 `0.05 m/s`，那么这三行代码的任务就是：

- 根据当前姿态的 Jacobian
- 算出每个关节此刻应该分别转多快
- 让 TCP 尽量实现这个向下速度

## 8. 为什么要加 damping

阻尼项对应代码：

```python
damping_matrix = (self.damping ** 2) * np.eye(jacobian.shape[0])
```

它的作用主要是：

- 提高数值稳定性
- 在接近奇异位形时避免关节速度爆大
- 让解更平滑

如果没有阻尼，机械臂接近某些姿态时，雅可比会变得病态，此时为了实现一个很小的 TCP 速度，也可能需要非常大的关节速度。

加入阻尼之后，解会更保守一些，但工程上通常更稳。

## 9. 为什么要加 nullspace 项

当前主任务只控制 TCP 的位置，所以对冗余机械臂来说，往往存在很多组 `dq` 都能实现相同的 TCP 线速度。

如果只做主任务：

```python
dq = jacobian_pinv @ desired_velocity
```

可能会出现这些问题：

- 肘部和手腕姿态漂移
- 动作越来越别扭
- 更容易靠近关节极限
- 每次执行时姿态不够稳定

所以代码里又加了一个零空间次任务：

```python
nullspace = np.eye(self.model.nv) - jacobian_pinv @ jacobian
dq += nullspace @ (
    self.nullspace_gain * (self._reference_configuration - q)
)
```

它的意思是：

- 主任务仍然是实现 TCP 位置速度
- 但在尽量不影响主任务的前提下
- 把关节配置轻微拉回某个参考姿态

这里的参考姿态是开始跟踪时记录下来的关节配置。

这会让运动看起来更自然，也更稳定。

## 10. 一个直观例子

假设当前控制器希望 TCP 速度为：

```text
desired_velocity = [0.00, 0.00, -0.05]
```

同时当前位置下的雅可比是一个 `3 x 7` 矩阵：

```text
J(q) =
[[ 0.00, -0.21, -0.18,  0.00,  0.05,  0.00,  0.00],
 [ 0.32,  0.00,  0.00, -0.07,  0.00,  0.02,  0.00],
 [ 0.00,  0.31,  0.12,  0.00, -0.03,  0.00,  0.00]]
```

那么 `_solve_joint_velocity()` 的任务就是找出一组 `dq`，例如：

```text
dq = [0.01, -0.08, 0.03, 0.00, 0.01, 0.00, -0.01]
```

使得：

```text
J(q) * dq ≈ [0.00, 0.00, -0.05]
```

如果冗余自由度允许，还会顺便微调这组 `dq`，让机械臂姿态不要偏离参考配置太多。

## 11. 为什么还要把 `dq` 积分成 `q_target`

在 [`src/nero_workcell/core/differential_ik_follower.py`](../src/nero_workcell/core/differential_ik_follower.py)
里，求出关节速度后，代码还会执行：

```python
q_target = self.model.integrate_configuration(q, dq, self.control_period)
```

这一步的作用是：

- 把当前关节配置 `q`
- 和当前控制周期内的关节速度 `dq`
- 转换成下一拍的目标关节位置 `q_target`

它对应的近似物理关系是：

```text
q_target ≈ q + dq * dt
```

其中 `dt` 就是控制周期 `self.control_period`。

### 11.1 为什么不能直接用 `dq`

因为 `dq` 和 `q_target` 不是同一种物理量：

- `q` / `q_target` 是关节位置，单位通常是 `rad`
- `dq` 是关节速度，单位通常是 `rad/s`

而当前底层接口 `move_j(...)` 需要的是关节位置目标，不是关节速度目标。

因此，如果直接把 `dq` 传给 `move_j(...)`，就相当于把：

- “速度”

误当成了：

- “位置”

这在量纲上就是错的。

### 11.2 一个直观例子

假设当前关节角是：

```text
q = [0.50, -1.00, 0.20]
```

而这一个控制周期里求得的关节速度是：

```text
dq = [0.10, -0.20, 0.00]   rad/s
```

控制周期为：

```text
dt = 0.05 s
```

那么下一拍目标关节角应为：

```text
q_target = q + dq * dt
         = [0.505, -1.010, 0.20]
```

这表示：

- 第 1 关节在 0.05 秒内前进了 `0.005 rad`
- 第 2 关节在 0.05 秒内后退了 `0.010 rad`
- 第 3 关节不动

这才是可以发送给位置控制接口的目标关节角。

如果直接把 `dq` 发给 `move_j(...)`，机器人就会把：

```text
[0.10, -0.20, 0.00]
```

理解成一组绝对关节角目标，而不是关节速度，这显然不对。

### 11.3 为什么用 `integrate_configuration(...)`

当前项目调用的是：

```python
pin.integrate(self.model, q, dq * dt)
```

而不是简单手写 `q + dq * dt`。

这样做的好处是：

- 语义更清楚：这是在配置空间中做积分
- 和 Pinocchio 的运动学模型保持一致
- 如果以后引入更复杂的关节类型，控制器逻辑不需要重写

对当前这种普通转动关节机械臂来说，它和简单加法的结果通常很接近，但使用 `integrate(...)` 更规范。

## 12. 当前实现已经做了哪些保护

当前实现里已经包含这些基础保护：

- TCP 速度限幅
- 关节速度限幅
- 积分后的关节位置限幅
- 阻尼伪逆提高近奇异状态下的稳定性

但它还没有做完整的：

- 环境碰撞检查
- 显式的奇异性指标监控
- 末端姿态控制

所以这版实现更准确地说是：

- 一个面向 TCP 位置跟踪的 differential IK 控制器
- 而不是完整的全约束运动规划器

## 13. 在本项目中的调用链

当前主要调用链如下：

1. `follow_target()` 读取当前关节角 `q`
2. `forward_tcp_position(q)` 计算当前 TCP 位置
3. `_step_toward()` 构造 `desired_velocity`
4. `_solve_joint_velocity(q, desired_velocity)` 计算关节速度 `dq`
5. `integrate_configuration(q, dq, dt)` 得到下一拍目标关节角
6. 将目标关节角发送给机器人

因此，`forwardKinematics` 和 `computeFrameJacobian` 都是这条控制链上的基础步骤：

- 一个负责“当前 TCP 在哪”
- 一个负责“关节怎么动会让 TCP 怎么动”

两者缺一不可。
