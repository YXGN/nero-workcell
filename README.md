# agilex-workcell

基于 Nero 机械臂与 Intel RealSense 的工作站示例工程，包含：
- 视觉目标跟随（YOLO + RealSense + 机械臂相对运动）
- 眼在手上（Eye-in-Hand）标定数据采集与计算

## 目录结构

```text
.
├── requirements.txt
├── src/nero_workcell
│   ├── core                  # 相机与机械臂控制封装
│   └── eye_in_hand
│       ├── collect_data.py
│       ├── eye_in_hand.py
│       └── config.json
└── tools/get_realsense_serial.py
```

## 环境要求

- Python 3.10+（建议 3.10/3.11）
- Ubuntu 22.04（推荐）
- Intel RealSense 相机（D435i 等）
- Nero 机械臂与可用 CAN 通道（如 `can0`）

## 安装依赖

1. 创建并激活 Conda 虚拟环境（建议）：

```bash
conda create -n nero python=3.10 -y
conda activate nero
python -m pip install --upgrade pip
```

2. 安装 Python 依赖：

```bash
pip install -r requirements.txt
```

3. 安装项目本体（可编辑模式，推荐开发使用）：

```bash
pip install -e .
```

> 说明  
> - `python-can` 版本要求高于 `3.3.4`。  
> - `pyAgxArm` 来自 GitHub 源码安装，网络需可访问。  
> - 目标跟随任务现使用 Pinocchio 进行 Nero 机械臂的模型计算，请预先安装 `pinocchio`。  
> - 如果 `pyrealsense2` 安装失败，请先确认系统侧 RealSense 运行环境完整。

## 运行测试

先确保项目已按上文完成可编辑安装：

```bash
pip install -e .
```

1. 运行 YOLO + RealSense 真机测试：

```bash
YOLO_MODEL_PATH=/path/to/yolov8n.pt python tests/test_yolo_detector.py
```

要求：
- 已安装 `pyrealsense2` 与 `ultralytics`
- 已连接 RealSense 相机
- 画面中放入真实 `cup`

如已在本机缓存 `yolov8n.pt`，也可以省略 `YOLO_MODEL_PATH`：

```bash
python tests/test_yolo_detector.py
```

2. 运行 RealSense 真机测试：

```bash
python tests/test_realsense_camera.py
```

如有多台相机，可指定序列号：

```bash
REALSENSE_SERIAL=<serial> python tests/test_realsense_camera.py
```

3. 运行 Nero 机械臂真机测试：

```bash
python tests/test_arm_controller_real.py
```

这些都是真机测试。缺少 `pyAgxArm`、`pyrealsense2`、`ultralytics`，或未连接对应真实硬件时，会直接失败。`test_yolo_detector.py` 在连续多帧都检测不到真实目标物体时，也会直接失败。

## 快速检查

1. 检查 RealSense 是否可被识别：

```bash
python tools/get_realsense_serial.py
```

2. 若需要 JSON 输出：

```bash
python tools/get_realsense_serial.py --json
```

## 使用说明

### 1) 目标跟随任务

在仓库根目录执行：

```bash
python -m nero_workcell.tasks.follow_static_target \
  --target bottle \
  --model yolov8n.pt \
  --conf 0.5 \
  --urdf /Users/jianghaiping/robot/agx_arm_sim/robot_description/nero_description/urdf/nero_description.urdf
```

参数说明：
- `--target`：目标类别名（如 `bottle`、`cup`）
- `--model`：YOLO 模型路径（默认 `yolov8n.pt`）
- `--conf`：置信度阈值（默认 `0.5`）
- `--target-distance`：目标点的保持距离/接近距离，单位米（默认 `0.3`）
- `--urdf`：机器人 URDF 路径，必填
- `--tcp-frame`：URDF 中的末端 frame 名，默认 `end_effector`
- `--approach-dir AX AY AZ`：接近方向向量，默认 `0 0 -1`
- `--joint-command-mode`：关节命令接口，`js` 或 `j`（默认 `js`）

检测与三维反投影原理说明见：

- [`docs/yolo_detector.md`](/Users/jianghaiping/robot/nero-workcell/docs/yolo_detector.md)

运行中按 `q` 退出。

### 2) 眼在手上标定

先修改配置文件：

[`src/nero_workcell/eye_in_hand/config.json`](src/nero_workcell/eye_in_hand/config.json)

可参考（按当前脚本字段）：

```json
{
  "checkerboard": {
    "corner_point_long": 7,
    "corner_point_short": 10,
    "corner_point_size": 0.015
  },
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "robot": {
    "ip": "can0",
    "type": "nero"
  }
}
```

在仓库根目录执行数据采集与标定计算：

```bash
python -m nero_workcell.eye_in_hand.collect_data
python -m nero_workcell.eye_in_hand.eye_in_hand --no-show
```

说明：
- `collect_data.py` 运行时按键：`s` 采集、`q` 退出、`g` 切换井字格、`c` 切换角点显示。
- `eye_in_hand.py` 默认读取最新一次采集目录，也可用 `--data <路径>` 指定数据目录。

## 常见问题

1. 机械臂连接失败  
请确认 CAN 通道与权限设置正确（如 `can0` 可用）。

2. 相机未找到  
先检查 USB 连接与相机供电，再运行 `tools/get_realsense_serial.py` 验证。

3. YOLO 模型下载或加载失败  
请确认网络可用，或将模型文件提前下载到本地并通过 `--model` 指定路径。
