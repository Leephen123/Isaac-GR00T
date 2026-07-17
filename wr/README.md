# WR

机器人数据采集、动捕数据处理与仿真工具库。支持从硬件采集多模态数据（相机、动捕、机器人本体感知），对动捕数据做标准化预处理，以及将处理后的数据回放到数字人、Mujoco 或真实机器人。

## 安装

首先创建并激活 Python 环境：

```bash
conda create -n wr python=3.12
conda activate wr
```

**机器人本体开发机（推荐安装方式）**（避免安装 torch/mujoco 等重依赖）：

```bash
pip install -e ".[scripts]"
```

**本地开发机（需要仿真环境时的安装方式）**（完整安装）：

```bash
pip install -e ".[all]"
```

部分依赖不在 PyPI，需按运行环境单独准备：

| 依赖 | 说明 |
| --- | --- |
| Unitree SDK | `scripts/collect_data.py` 从 `/home/unitree/unitree_sdk2_python` 加载 |
| teleop | 从 `/home/unitree/xr_teleoperate` 加载，用于机械臂控制 |
| westlake_sdkpy | `sim_res/mujoco/eman/sim2/` 中部分环境使用 |
| CycloneDDS 运行时 | DDS 通信底层，需要配置网络环境和机器人连接 |
| 相机设备 | `/dev/video*`，通过 `v4l2-ctl` 查询设备路径 |

## 数据格式

每条采集的 episode 保存在 `<root>/<task>/episode_<N>/` 目录下：

```text
episode_0/
├── data.json          # 原始采集数据（保留，不修改）
├── data_standard.json # 标准化后的动捕数据（用于训练）
└── images/
    ├── cam_left/      # 各相机图像序列
    └── cam_right/
```

`data.json` 是一个列表，每帧为一个字典：

```json
{
  "cam_left": "images/cam_left/00000.png",
  "cam_right": "images/cam_right/00000.png",
  "hand_joint": [...],
  "imu": [...],
  "mocap": [...],
  "task": ["task_name"],
  "fps": 120.0,
  "timestamp": 1700000000.0
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `hand_joint` | 14 维手部关节角 |
| `imu` | 4 维根节点四元数 `[w, x, y, z]` |
| `mocap` | `(15, 7)` 动捕数据 |

动捕数据每帧形状为 `(15, 7)`，每个关节为 `[x, y, z, qw, qx, qy, qz]`，共 15 个关节。

## 工作流程

### 1. 确认相机设备

```bash
python scripts/show_cameras.py
```

根据输出更新 `config/camera.yaml` 中各相机的设备路径。

### 2. 采集数据

```bash
python scripts/collect_data.py --root-path ./data --task <task_name>
```

运行后弹出 UI 窗口，键盘操作：

| 按键 | 功能 |
| --- | --- |
| `1` | 开始 / 停止当前 episode 采集 |
| `2` | 删除上一条 episode（仅在停止状态下可用） |
| `3` | 退出程序 |
| `l` | 切换 UI 语言（中/英） |

采集完成后数据保存至 `./data/<task_name>/episode_*/data.json`。

> 运行前需确认相机、动捕系统、机器人 SDK 及网络连接均正常。

### 3. 动捕数据标准化

```bash
python scripts/preprocess_data.py --input-dirs ./data/task_A ./data/task_B
```

`--input-dirs` 接受一个或多个根目录。脚本会遍历每个目录下所有 `episode_*` 子目录，读取 `data.json`，并将标准化后的动捕数据写入同目录的 `data_standard_6D.json`。

- `data.json`：原始采集数据，**保留不动**，供回放和调试使用
- `data_standard_6D.json`：标准化数据，**模型训练使用此文件**

### 4. 部署 / 回放

标准化数据在部署时需转换回世界坐标系。`data_res/transforms.py` 提供了互逆的两个函数：

- `compute_relative(ref, cur)`：将 `cur` 表达为相对于 `ref` 的位姿（预处理时使用）
- `compute_absolute(ref, rel)`：将相对位姿还原为世界坐标系位姿（部署时使用）

部署流程：读取机器人当前根节点位姿 → 用 `compute_absolute` 将模型输出转换回世界坐标系 → 通过 DDS 发送给数字人/机器人。

回放到真实机器人（通过 DDS 发布动捕数据）：

```bash
python inference/g1_replay.py \
    --replay-data-path ./data/task_A/episode_0/data_standard.json \
    --use-relative True \
    --send-fps 100
```

回放到 Mujoco 仿真：

```bash
python inference/g1_replay_mujoco.py \
    --replay-data-path ./path/to/data_standard.json
```

## 其他工具脚本

实时可视化动捕数据（3D 火柴人，从 DDS 订阅）：

```bash
python scripts/visualize_mocap.py
```

录制原始动捕数据到文件：

```bash
python scripts/record_mocap.py --output-path ./mocap_raw.npy --duration-s 30
```

## 目录结构

```text
wr/
├── config/
│   └── camera.yaml          # 相机设备路径配置
├── data_res/                # 核心数据处理模块（可作为库复用）
│   ├── camera.py            # 相机捕获
│   ├── dds.py               # CycloneDDS 消息定义、发布/订阅封装
│   ├── transforms.py        # 位姿变换（compute_relative/absolute、插值、四元数工具）
│   └── utils.py             # 数据加载、UI 渲染等工具函数
├── scripts/                 # 可直接运行的脚本
│   ├── collect_data.py      # 多模态数据采集
│   ├── preprocess_data.py   # 动捕数据标准化
│   ├── record_mocap.py      # 录制原始动捕数据
│   ├── show_cameras.py      # 查询可用相机设备
│   └── visualize_mocap.py   # 实时动捕可视化
├── inference/               # 推理与部署脚本
│   ├── g1_replay.py         # 回放到真实机器人
│   └── g1_replay_mujoco.py  # 回放到 Mujoco 仿真（未完成测试）
└── sim_res/                 # 仿真相关模块
    └── mujoco/eman/         # Mujoco 环境封装（G1、K2 等机型）
```

## 注意事项

- `sim_res/mujoco/eman/` 中部分环境依赖 `westlake_sdkpy` 和 `tasks` 包，这些包不在本仓库中。

## 数据详细规范标准

### 数据预处理

**X：`6 + 29`**

- `6` 表示机器人 IMU 去掉 yaw 的 quaternion 对应的 6D 表示。
- 所有时刻都不需要 yaw，不是减去 0 时刻的 yaw，而是直接不提供 yaw 信息。

**Y：`11 * 9`**

- 首先读取 `15 * 7`，其中 `7 = 3 + 4`，然后需要 `compute_relative`。
- 然后读取 `11 * 7`，`11` 表示从 `15` 进行切片。
- 切片顺序为 `[0, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14]`。
- 最后变成 `11 * 9`，其中 `9 = 3 + 6`，`6` 表示 quaternion 的 6D 值。

### 训练

- 训练输入：`6 + 29 + 图片`
- 训练输出：`11 * 9`

### 部署

- 部署输入：`6 + 29 + 图片`
- 部署输出：`11 * 9`
- 然后 `11 * 9` 变成 `11 * 7`。
- 然后 `11 * 7` 变成 `15 * 7`，按照切片顺序进行填充。
- 最后 `15 * 7` 进行 `compute_absolute`，发送机器人。

## TODO

- `inference/g1_replay_mujoco.py` 文件和 `inference/g1_replay.py` 文件应当合成为一个文件，只需要修改 topic 订阅便可以切换模式。
- `inference/g1_gr00t_infer_debug.py` 文件和 `inference/g1_gr00t_infer.py` 文件应当合成为一个文件，增加控制模式字段 `mode`，便可以完成模式切换。
- `inference/g1_gr00t_infer_debug.py` 文件应当支持在开环测试时，将最终发送给 GAE 的 mocap 数据进行保存的功能。同时，`scripts/visualize_mocap.py` 文件应当具备将保存的 mocap 数据进行可视化的功能，而不仅仅是目前只对 DDS 发送的实时数据进行可视化。
- `scripts/preprocess_data.py`文件当前在数据处理的输出侧还是(15, 7)形状，后续推荐改为(11, 7)。
