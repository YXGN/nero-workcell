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
│   ├── tasks/object_follower.py
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
> - 如果 `pyrealsense2` 安装失败，请先确认系统侧 RealSense 运行环境完整。

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
python -m nero_workcell.tasks.object_follower --target bottle --model yolov8n.pt --conf 0.5
```

参数说明：
- `--target`：目标类别名（如 `bottle`、`cup`）
- `--model`：YOLO 模型路径（默认 `yolov8n.pt`）
- `--conf`：置信度阈值（默认 `0.5`）

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
