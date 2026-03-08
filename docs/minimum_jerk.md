# minimum-jerk 原理说明

本文档说明 [`src/nero_workcell/core/cartesian_trajectory.py`](../src/nero_workcell/core/cartesian_trajectory.py)
中 minimum-jerk 轨迹的数学原理与工程意义，重点解释以下问题：

- 什么是 `jerk`
- 为什么机械臂末端轨迹常用 minimum-jerk
- 五次多项式中的 `10, -15, 6` 是怎么来的
- 为什么代码里还要计算 `blend_rate`
- 这套公式在当前项目里具体起什么作用

## 1. 什么是 jerk

对位置关于时间连续求导：

- 一阶导数是速度 `v(t)`
- 二阶导数是加速度 `a(t)`
- 三阶导数是 `jerk(t)`

也就是：

```text
v(t) = dx/dt
a(t) = d^2x/dt^2
jerk(t) = d^3x/dt^3
```

`jerk` 的物理意义是“加速度变化得有多快”。

如果 `jerk` 很大，通常意味着：

- 机械臂起步或刹车比较生硬
- 末端运动容易出现顿挫
- 执行器、减速器和结构件承受更明显的冲击
- 控制器更难平滑跟踪

因此，在满足起点和终点约束的前提下，让轨迹具备更平滑的速度和加速度变化，通常会得到更稳定的运动效果。

## 2. minimum-jerk 想解决什么问题

如果只做最简单的线性插值：

```text
p(t) = p0 + alpha(t) * (p1 - p0)
```

并让 `alpha(t)` 线性增长，那么位置虽然连续，但速度会在起点和终点附近发生突变：

- 起点瞬间从静止跳到常速
- 终点瞬间从常速跳到静止

这种轨迹对机械臂不够友好。

minimum-jerk 的核心思路是：

- 在起点和终点处让速度为 0
- 在起点和终点处让加速度也为 0
- 用一条足够平滑的时间缩放函数把直线路径参数化

这样做的结果是：

- 起步更柔和
- 结束更平稳
- 中间的加减速过程连续
- 更适合末端跟随、精细接近和抓取前对位

## 3. 当前代码中的轨迹形式

当前项目的 `CartesianTrajectory` 使用的是“空间直线 + minimum-jerk 时间缩放”。

设：

- 起点为 `start_position = p0`
- 终点为 `goal_position = p1`
- 位移为 `delta = p1 - p0`
- 归一化时间为 `tau in [0, 1]`

则轨迹位置写成：

```text
p(tau) = p0 + s(tau) * delta
```

其中：

- `delta` 决定空间路径方向和长度
- `s(tau)` 决定沿这条路径走了多少比例

在代码里，这个比例函数就是：

```text
s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
```

对应实现：

```python
blend = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
```

## 4. 为什么是五次多项式

为了保证轨迹在起点和终点足够平滑，通常对 `s(tau)` 加下面 6 个边界条件：

```text
s(0) = 0
s(1) = 1
s'(0) = 0
s'(1) = 0
s''(0) = 0
s''(1) = 0
```

这 6 个条件分别表示：

- 起点比例为 0
- 终点比例为 1
- 起点速度为 0
- 终点速度为 0
- 起点加速度为 0
- 终点加速度为 0

要满足 6 个独立条件，最自然的选择就是使用 5 次多项式：

```text
s(tau) = a0 + a1*tau + a2*tau^2 + a3*tau^3 + a4*tau^4 + a5*tau^5
```

## 5. `10, -15, 6` 是怎么推出来的

先代入 `tau = 0` 的三个条件：

```text
s(0) = 0
s'(0) = 0
s''(0) = 0
```

可以直接得到：

```text
a0 = 0
a1 = 0
a2 = 0
```

所以多项式可以简化成：

```text
s(tau) = a3*tau^3 + a4*tau^4 + a5*tau^5
```

再代入 `tau = 1` 的三个条件：

```text
a3 + a4 + a5 = 1
3*a3 + 4*a4 + 5*a5 = 0
6*a3 + 12*a4 + 20*a5 = 0
```

解这个方程组，得到：

```text
a3 = 10
a4 = -15
a5 = 6
```

所以最终：

```text
s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
```

这就是代码里 `blend` 的来源。它不是调参结果，而是由边界条件唯一确定的标准形式。

## 6. 为什么还要计算 `blend_rate`

代码里还有：

```python
blend_rate = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / self.duration
```

这是因为控制器不仅需要参考位置，还需要参考速度。

已知：

```text
s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
```

先对 `tau` 求导：

```text
ds/dtau = 30*tau^2 - 60*tau^3 + 30*tau^4
```

而归一化时间满足：

```text
tau = (now - start_time) / duration
```

所以：

```text
dtau/dnow = 1 / duration
```

根据链式法则：

```text
ds/dnow = ds/dtau * dtau/dnow
        = (30*tau^2 - 60*tau^3 + 30*tau^4) / duration
```

这就是 `blend_rate`。

于是：

```text
position = start_position + blend * delta
velocity = blend_rate * delta
```

含义分别是：

- `blend` 表示当前走到了整条路径的多少比例
- `blend_rate` 表示这个比例当前变化得有多快

## 7. 起点和终点为什么更平滑

minimum-jerk 轨迹满足：

- `s'(0) = 0`，所以起点速度为 0
- `s'(1) = 0`，所以终点速度为 0
- `s''(0) = 0`，所以起点加速度为 0
- `s''(1) = 0`，所以终点加速度为 0

这意味着：

- 不会一启动就猛冲
- 不会临近终点时硬刹车
- 轨迹在时间上更圆滑

对机械臂跟随控制来说，这种轨迹通常比简单线性插值更容易跟踪。

## 8. 在本项目中的作用

在 [`src/nero_workcell/core/differential_ik_follower.py`](../src/nero_workcell/core/differential_ik_follower.py)
里，`CartesianTrajectory.sample()` 的输出会被用来构造当前时刻的笛卡尔参考：

- `sample.position` 作为当前参考位置
- `sample.velocity` 作为前馈参考速度

随后控制器会结合当前 TCP 位置误差，求出期望末端速度，再通过 differential IK 计算对应的关节速度命令。

因此，minimum-jerk 在这里的作用不是“做全局路径规划”，而是：

- 在已知起点和目标点之间生成一条平滑的局部参考轨迹
- 让末端接近目标时更稳定
- 减少由参考信号突变带来的控制抖动

## 9. 一个具体数值例子

假设：

- `start_position = [0.0, 0.0, 0.0]`
- `goal_position = [0.3, 0.0, 0.0]`
- `duration = 2.0 s`
- 当前时刻距离开始已经过了 `1.0 s`

则：

```text
tau = 1.0 / 2.0 = 0.5
```

代入得到：

```text
blend = 10*(0.5)^3 - 15*(0.5)^4 + 6*(0.5)^5 = 0.5
```

因此当前位置为：

```text
position = [0.0, 0.0, 0.0] + 0.5 * [0.3, 0.0, 0.0]
         = [0.15, 0.0, 0.0]
```

这表示在轨迹进行到一半时，末端刚好到达中点附近，但速度变化是平滑产生的，而不是匀速硬切换。

## 10. 总结

当前代码中的 minimum-jerk 轨迹，本质上是：

- 用一条直线描述空间路径
- 用五次多项式描述时间缩放

其优点是：

- 起点和终点速度为 0
- 起点和终点加速度为 0
- 轨迹更容易被机械臂控制器稳定跟踪

因此，它非常适合当前项目这种“末端在笛卡尔空间中平滑接近目标点”的控制场景。
