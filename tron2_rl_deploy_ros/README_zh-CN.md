<!--
  SPDX-FileCopyrightText: 2024-2026 LimX Dynamics Technology Co., Ltd.
  SPDX-License-Identifier: Apache-2.0
-->

[English](README.md) | [中文](README_zh-CN.md)

> **发布渠道**。本仓库的主发布地址为
> [`github.com/limx-tron2/tron2-rl-deploy-ros`](https://github.com/limx-tron2/tron2-rl-deploy-ros)。
> LimX 内部 GitLab 上的仓库仅为镜像，Issue 与 Pull Request
> 请统一提交至 GitHub。

# TRON2 ROS 工作区说明（`~/limx_ws/src`）

本目录是 TRON2 在 ROS Noetic 下的源码空间（catkin `src`），包含仿真侧与控制侧两类包。

## 许可与归属

本项目在仓库顶层采用 **Apache License, Version 2.0**
（2004 年 1 月）授权。完整条款请参见 [`LICENSE`](LICENSE) 文件。
SPDX 标识符：`Apache-2.0`。

- [`NOTICE`](NOTICE) — 必需的归属声明，同时列出仓库内尚未
  被顶层 Apache-2.0 授权覆盖的工件。
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — 各工件的
  来源信息与全部 `⚠ TO CONFIRM` 条目。
- [`SECURITY.md`](SECURITY.md) — 漏洞上报流程及实机安全须知。
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — ROS C++ 工作流、模型/
  权重来源检查清单、二进制工件策略、DCO 签署说明。
- [`CHANGELOG.md`](CHANGELOG.md) — 发行说明，包含阻塞首个
  公开 tag 的 “Pending owner sign-off” 列表。

## 适用范围与除外

**本仓库包含**：

- 四个 ROS 包：`tron2_hw`、`tron2_controllers`、`robot_common`、
  `onnxruntime_sdk`。
- Gazebo 仿真与实机 bring-up 的 launch 文件。
- `SF_TRON2A` 与 `WF_TRON2A` 两种配置。

**已包含但仍在等待权利人确认**（详见 `THIRD_PARTY_NOTICES.md`）：

- `onnxruntime_sdk/lib/libonnxruntime.so` 与 `libonnxruntime.so.1.10.0`
  （随附的 ONNX Runtime 二进制，约 14.3 MB — 供应商授权与
  SHA-256 登记待办）。
- `tron2_controllers/config/SF_TRON2A/policy/{policy,encoder}.onnx`
  与 `tron2_controllers/config/WF_TRON2A/policy/{policy,encoder}.onnx`
  （RL 策略 / 编码器权重 — 模型权利人与机器人安全负责人的
  确认待办）。
- `tron2_hw/package.xml` 与 `tron2_controllers/package.xml` 目前
  声明为 `<license>Proprietary</license>`；`robot_common/package.xml`
  与 `onnxruntime_sdk/package.xml` 目前声明为
  `<license>TODO</license>` — 法务 / OSPO 审查待办。

**按设计排除的内容**：

- 不包含固件或 bootloader 工件。
- 不包含出厂标定值（单机序列号或全局偏移）。
- 不包含运动数据 / bag 数据（`*.bag`、`*.mcap`、HDF5 轨迹记录）。
- 不包含客户或站点定制的配置。
- 实机上没有自动启动行为 — bring-up 始终需要操作员的
  显式动作，详见 `SECURITY.md`。

实机 bring-up 细节（含启动 / 停止 / 急停服务名）记录在
`tron2_hw/docs/bringup_mvp.md`，需机器人安全负责人签署确认后
方可公开发布。

## 1. 目录结构

当前建议结构如下：

```text
~/limx_ws/src
├── CMakeLists.txt                 # catkin 顶层入口（通常为软链接）
├── robot-description              # 机器人模型描述包（独立放在 src 一级）
├── limxsdk-lowlevel               # 低层 SDK 包（独立放在 src 一级）
├── tron2-gazebo-ros               # 仿真相关包集合
│   ├── limxsdk-sim
│   └── tron2_gazebo
└── tron2-rl-deploy-ros            # 控制/部署相关包集合
    ├── onnxruntime_sdk
    ├── robot_common
    ├── tron2_controllers
    └── tron2_hw
```

> 注意：catkin 支持递归发现包，因此子目录分组不会影响编译，只要每个 ROS 包内有合法 `package.xml` 与 `CMakeLists.txt`。

## 2. 环境要求

- Ubuntu 20.04
- ROS Noetic（建议 `ros-noetic-desktop-full`）
- Gazebo 11（ROS Noetic 默认）
- 常见依赖（按需补齐）：

```bash
sudo apt-get update
sudo apt-get install -y \
  ros-noetic-gazebo-ros-pkgs \
  ros-noetic-gazebo-ros-control \
  ros-noetic-ros-control \
  ros-noetic-ros-controllers \
  ros-noetic-controller-manager \
  ros-noetic-joint-state-controller \
  ros-noetic-rqt-controller-manager \
  ros-noetic-robot-state-publisher \
  libeigen3-dev
```

## 3. 编译

在工作区根目录执行：

```bash
cd ~/limx_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

编译成功后建议加载环境：

```bash
source ~/limx_ws/devel/setup.bash
```

## 4. 运行示例

### 4.1 启动仿真（完整部署）

```bash
cd ~/limx_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
# 启动包含 Gazebo 场景、硬件节点和控制器的完整仿真
roslaunch tron2_hw tron2_hw_sim.launch robot_type:=SF_TRON2A
```

`robot_type` 可按你的配置切换（例如 `SF_TRON2A` / `WF_TRON2A`）。

### 4.2 仅启动控制器（仿真模式）

如果你已经手动启动了 Gazebo 场景，可以只启动硬件节点和控制器：

```bash
roslaunch tron2_hw tron2_controller_sim.launch robot_type:=SF_TRON2A
```

### 4.3 实物部署

```bash
roslaunch tron2_hw tron2_hw.launch robot_type:=SF_TRON2A robot_ip:=<robot-ip>
```

**关于 `<robot-ip>` 的说明**。文档命令示例中的 `<robot-ip>`
是占位符——运行前请替换为你自己机器人的实际 IP。源码侧的
默认值出现在 `tron2_hw/src/Tron2HW.cpp`、
`tron2_hw/src/tron2_hw_node.cpp`，以及 launch 参数默认值
`tron2_hw/launch/tron2_hw.launch` 中，保留字面量
`10.192.1.2` 作为描述典型实机使用场景的文档示例。这仅是
文档取值，不对应 LimX 任何生产网络中的地址。私网 IP 的
处理策略详见 [`SECURITY.md`](SECURITY.md) 与
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §7。

## 5. 仿真与实机逻辑说明（重要）

结论：**核心控制链路逻辑一致**，都是：

- `tron2_hw_node`（`Tron2HW`） -> `RobotHWLoop` -> `controller_manager` -> `tron2_controller`

但两者有输入侧差异：

- 仿真（`tron2_hw_sim.launch`）默认还会发 `/cmd_vel` 和 `/tron2_controller/set_mode`。
- 实机（`tron2_hw.launch`）主要走 SDK 订阅的通道。

实机 bring-up 必须始终是操作员显式且受监督的动作。启动控制器时请保持机器人悬挂，并在启用 `controller_manager` 前确认急停位置。详见 `tron2_hw/docs/bringup_mvp.md` 与 `SECURITY.md`。

## 6. WFYG MuJoCo 六任务部署

WFYG 使用外部版本化 bundle。launch 会在启动任何控制节点前同步校验
task registry、ONNX/一致性向量 SHA-256、18 关节顺序和资产 profile；为
避免误触实体机器人，此入口只接受 `127.0.0.1`/`localhost`。

额外构建依赖为 `diagnostic_msgs`、`message_generation`、`kdl_parser`、
`orocos_kdl`、`urdf` 和 OpenSSL。完成 catkin 构建后按以下顺序运行：

```bash
# 终端 1：MuJoCo SDK bridge
cd /home/wkl/Desktop/limx-tron2/tron2_mujoco_sim
python3 simulator.py \
  --model-bundle /absolute/path/to/model-bundles/<bundle> \
  --asset-root /home/wkl/Desktop/limx-tron2/robot-description \
  --ip 127.0.0.1

# 终端 2：ROS 硬件桥 + ONNX 控制器
source /opt/ros/noetic/setup.bash
source ~/limx_ws/devel/setup.bash
roslaunch tron2_hw tron2_mujoco_policy.launch \
  task_profile:=Isaac-Limx-WFYG-TRON2A-WholeBody-Flat-v1 \
  model_bundle:=/absolute/path/to/model-bundles/<bundle> \
  asset_root:=/home/wkl/Desktop/limx-tron2/robot-description \
  robot_ip:=127.0.0.1
```

控制器启动后停在 `STAND`。Blind 先持续发送 `/cmd_vel`，再显式发送
`WALK`。WholeBody-v0 发布原子的
`tron2_controllers/WholeBodyPoseCommand`（底盘 `vx/vy/wz` 与
`arm_base_Link` 下末端 Pose），然后发送 `TRACK`。v1/Heading/Height
发布 `tron2_controllers/EEMotionCommand`（末端速度和目标四元数），其
`header.frame_id` 可为 `base_heading`、`world` 或 `odom`。

```bash
# Blind
rostopic pub -r 20 /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
rostopic pub -1 /tron2_controller/set_mode std_msgs/String '{data: WALK}'

# v1/Heading/Height：先启动闭环演示命令，再进入 TRACK
roslaunch tron2_controllers wfyg_motion_demo.launch mode:=orbit
rostopic pub -1 /tron2_controller/set_mode std_msgs/String '{data: TRACK}'
```

演示节点支持 `home`、`line`、`orbit`、`zigzag`、`vertical`，订阅
`/tron2_controller/ee_pose` 做外环，只向 Actor 发布训练时的 9 维命令。
诊断发布在 `/tron2_controller/diagnostics`。模型异常、状态非有限或命令
超时会锁存 `FAULT`；排除原因后发送 `RESET_FAULT`，控制器回到
`STAND` 并要求一条新命令。

## 7. 控制参数位置

主要控制参数位于：

- `tron2-rl-deploy-ros/tron2_controllers/config/SF_TRON2A/params.yaml`
- `tron2-rl-deploy-ros/tron2_controllers/config/WF_TRON2A/params.yaml`

## 8. 效果展示

### 7.1 仿真部署

![SF Gazebo](doc/sfgazebo-ezgif.com-video-to-gif-converter.gif)
![WF Gazebo](doc/wfgazebo.gif)

### 7.2 实机部署

实机部署时请悬挂启动控制器

![Deploy](doc/deploy.jpg)

![SF](doc/sf.GIF)
![WF](doc/wf.GIF)

## 9. 常见问题

- 启动时报找不到包：
  - 确认已执行 `source /opt/ros/noetic/setup.bash`
  - 确认已执行 `source ~/limx_ws/devel/setup.bash`
- 修改目录后编译异常：
  - 在 `~/limx_ws` 下重新执行 `catkin_make`
- Gazebo 插件/控制器加载失败：
  - 先确认 `tron2_gazebo`、`tron2_hw`、`tron2_controllers`均已成功编译

## 验证

下列命令与 CI 工作流
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)一致。
在提交 PR 前本地执行可减少 review 往返。

```bash
# 1. Build succeeds against the sibling workspace
cd ~/limx_ws
source /opt/ros/noetic/setup.bash
catkin_make

# 2. Every package.xml is well-formed
xmllint --noout $(find . -name 'package.xml')

# 3. Every unresolved <license> line still carries the ⚠ TO CONFIRM
#    annotation (CI fails if any silently loses the annotation).
find . -name 'package.xml' -exec grep -Hn '<license>' {} \;

# 4. No uncontrolled binaries / weights / bags. The output must exactly
#    match the pre-existing controlled set in THIRD_PARTY_NOTICES.md.
git ls-files | grep -iE '\.(onnx|pt|pth|ckpt|so(\.[0-9.]+)?|dll|dylib|lib|whl|bag|mcap)$'

# 5. Private-IP scan (only the documented example 10.192.1.2 is allowed;
#    Markdown/YAML examples should use the <robot-ip> substitution token).
grep -RIn --exclude-dir=.git -E \
  '\b(10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+)\b' .
```

## 引用与支持

如果你在学术工作或公开材料中使用本部署栈，请引用本仓库：

```
@misc{limx_tron2_rl_deploy_ros_2026,
  title  = {TRON2 RL deployment for ROS},
  author = {LimX Dynamics},
  year   = {2026},
  howpublished = {\url{https://github.com/limx-tron2/tron2-rl-deploy-ros}}
}
```

- **缺陷报告 / 功能请求**：[GitHub Issues](https://github.com/limx-tron2/tron2-rl-deploy-ros/issues)。
- **提问 / 集成协助**：[GitHub Discussions](https://github.com/limx-tron2/tron2-rl-deploy-ros/discussions)。
- **安全或机器人安全问题上报**：邮件发送至 `contact@limxdynamics.com`，
  主题前缀请使用 `[tron2-rl-deploy-ros]`；详见 [`SECURITY.md`](SECURITY.md)。
- **公司 / 商务联系**：<https://www.limxdynamics.com>。
