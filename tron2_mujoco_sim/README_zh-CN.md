# NANDU1 sim2sim 发布版

本仓库使用三目录结构：`tron2_mujoco_sim`（物理/传感器）、`tron2_rl_deploy_ros`（SDK＋ROS控制器）与`deployment_bundles`（13个正式部署包）。仅用于本机MuJoCo，不授权向实体机器人发送策略。

## 获取完整代码与权重

先安装Git LFS，再获取子模块及真实权重文件；ONNX的LFS指针不是可运行的模型。

```bash
git lfs install
git clone --recurse-submodules https://github.com/technoloboy/nandu-sim2sim.git
cd nandu-sim2sim
git lfs pull
```

## 环境与构建

激活Python 3.10环境（本机可继续使用`cambot-train`），准备好Docker及当前用户的Docker访问权限。脚本使用当前PATH中的`python3`，不依赖原作者的绝对路径；Tk需要由系统/Conda提供。

```bash
python -m pip install -r requirements-sim.txt
python -m pip install tron2_mujoco_sim/limxsdk-lowlevel/python3/amd64/*.whl
sg docker -c './tron2_rl_deploy_ros/scripts/nandu1_noetic.sh build'
```

发布版保留原四图模型指纹、课程阶段限幅、安全检查、键盘/摇杆接口。`deployment_bundles/catalogue.json`列出各任务的准确checkpoint身份及限制。bundle只移除了生成的Python缓存并重新计算清单；推理图及资产内容未改变。不包含训练checkpoint/优化器、训练仓、实验结果、个人说明Markdown或采集输出。

## 单终端：MuJoCo＋鼠标摇杆

在本仓库根目录、正常桌面终端运行，始终只运行一个SDK仿真实例。例如v1.3.1.2.2：

```bash
sg docker -c './tron2_mujoco_sim/scripts/run_nandu1_mouse.sh \
  --task Isaac-NANDU1-Blind-Flat-v1-3-1-2-2 \
  --model-bundle ./deployment_bundles/nandu1_v1_3_1_2_2_model_29500 \
  --telemetry-output ./tron2_mujoco_sim/outputs/v13122_tracking.csv'
```

松开摇杆回中，高度滑块保持。关闭窗口或Ctrl+C结束拥有的进程；原`--command-source external`双终端/无窗口入口仍保留。换任务时同时选用catalogue中匹配的bundle，不能互换同维度权重。输出目录运行时生成并被Git忽略，同名CSV会覆盖。

**v1.3.1.2.7/model_10000在零命令下多次失稳，未通过稳定性验收，只供复现排查；B/.3也存在高度偏差。** 部署正确性/数值一致性不等于全范围、长期或实机性能保证。

## 校验与测试

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python scripts/check_release_tree.py
```

普通测试无需训练仓；仅需要原始USD/URDF或重新导出的4项训练侧测试会明确skip，命令数学对照使用bundle中保存的训练参考代码，不冒称重新训练或重建。ROS构建会运行C++数值对照。GUI/长时间SDK实验可通过`tron2_mujoco_sim/tests/`及`scripts/`中的有限验收入口单独执行，结果不随仓库分发。

本发布版基于LimX官方MuJoCo/ROS工程，保留各目录的LICENSE、NOTICE及第三方许可说明；SDK、robot-description与joystick使用公开上游子模块固定版本。下方为原上游说明，历史演示媒体未随精简发布分发。

---

<!--
  SPDX-FileCopyrightText: 2024-2026 LimX Dynamics Technology Co., Ltd.
  SPDX-License-Identifier: Apache-2.0
-->

[English](README.md) | [中文](README_zh-CN.md)

> **发布渠道。** 本仓库的主发布点为 GitHub：
> <https://github.com/limx-tron2/tron2-mujoco-sim>。
> LimX 内部 GitLab 为镜像；Issue、PR 与安全报告请提交到 GitHub。

# tron2-mujoco-sim 使用说明

面向 TRON2A 人形平台的 MuJoCo 仿真器。`simulator.py` 将 LimX 底层
SDK（`RobotCmd` / `RobotState`，字段 `q` / `dq` / `tau` / `Kp` /
`Kd`）与 `mujoco.MjData` 打通，让驱动真实机器人的同一套控制器接口
可以直接在仿真环境中运行。

## 许可与归属

本项目采用 **Apache License, Version 2.0**（2004 年 1 月）授权。
完整许可文本见 [`LICENSE`](LICENSE)。SPDX 标识符：`Apache-2.0`。

- [`NOTICE`](NOTICE) — 必需的归属声明。
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — 各子模块与
  依赖的来源信息，以及文档媒体资源的审阅记录。
- [`SECURITY.md`](SECURITY.md) — 漏洞上报流程，以及仿真 vs 实机
  的边界说明。
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Python 与子模块工作流、
  DCO 签署要求、子模块 commit 升级流程。
- [`CHANGELOG.md`](CHANGELOG.md) — 发布记录，含所有待负责人签字
  确认的条目。

## 适用范围与除外

**本仓库包含：**

- `simulator.py` — MuJoCo 仿真器与 LimX SDK 桥接。
- 文档（`README.md`、`doc/*.jpg`、`doc/*.gif`、`doc/*.GIF`）。
- 子模块**声明**（通过 `.gitmodules`）— 共三个子模块按 commit 固定,
  需单独初始化：
  - `robot-description/` — URDF / MJCF 网格与模型。
  - `robot-joystick/` — 手柄辅助程序。
  - `limxsdk-lowlevel/` — LimX 底层 SDK（含预编译 wheel）。

**本仓库不包含（有意为之）：**

- 子模块本身。clone 后请执行
  `git submodule update --init --recursive`（或 clone 时加
  `--recurse-submodules`）。子模块的许可与再分发状态见
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §2
  （当前为 `⚠ TO CONFIRM`）。
- ONNX 控制策略（`.onnx`、`.pt`、`.pth`、`.ckpt`）。仿真器本身
  不直接加载它们；加载工作由姊妹仓库
  **`tron2-rl-deploy-python`** 中的部署栈完成 —
  见 [§3 模型文件位置](#3-模型文件位置)。
- SDK 的二进制（`.so`、`.dll`）或 wheel（`.whl`）不会直接进入
  本仓库；SDK wheel 位于 `limxsdk-lowlevel` 子模块中。
- 工厂标定数据、固件、运动 / bag 数据。

[§2 运行控制器](#2-运行控制器)中出现的 `<robot-ip>` 是**占位符**，
运行前请替换为您自己机器人或仿真器的实际 IP。本仓库源码不包含任何
硬编码的私网地址。仅内部使用的姊妹仓库 `tron2-rl-deploy-ros` 在其
源码 / launch 文件中保留了一处作为文档示例的字面量 `10.192.1.2`，
已在该仓库的 `SECURITY.md` 中声明；本仓库有意不含此类字面量。

## 1. 运行仿真

### Step 1: 打开终端

### Step 2: 克隆仓库

```bash
git clone --recurse-submodules https://github.com/limx-tron2/tron2-mujoco-sim.git
```

If you already cloned without submodules:

```bash
cd tron2-mujoco-sim
git submodule update --init --recursive
```

### Step 3: 安装 Python 依赖

```bash
pip install -U pip
pip install mujoco numpy scipy pyyaml onnxruntime pygame
```

### Step 4: 安装 LimX SDK（必须）

根据机器架构安装 SDK wheel（示例）：

```bash
# x86_64
pip install limxsdk-lowlevel/python3/amd64/limxsdk-*-py3-none-any.whl

# aarch64
pip install limxsdk-lowlevel/python3/aarch64/limxsdk-*-py3-none-any.whl
```

### Step 5: 设置机器人型号

当前支持：

- `SF_TRON2A`
- `WF_TRON2A`

示例：

```bash
export ROBOT_TYPE=SF_TRON2A
```

### Step 6: 启动 MuJoCo 仿真器

```bash
cd tron2-mujoco-sim
python3 simulator.py
```

可选：指定 SDK 通信 IP（默认 `127.0.0.1`）：

```bash
python3 simulator.py 127.0.0.1
```

---

## 2. 运行控制器

### Step 1: 打开新终端

### Step 2: 启动 ONNX 控制入口

```bash
cd tron2-rl-deploy-python
export ROBOT_TYPE=SF_TRON2A
python3 main.py
```

可选：指定机器人 IP（和 tron1 用法一致）：

```bash
# 说明：<robot-ip> 是占位符，运行前请替换为您自己机器人 /
# 仿真器的实际 IP。本仓库源码未硬编码任何私网地址。
# （仅内部使用的姊妹仓库 tron2-rl-deploy-ros 在其源码 /
# launch 文件中保留了作为文档示例的字面量 10.192.1.2，
# 已在该仓库的 SECURITY.md 中声明。）
python3 main.py <robot-ip>
```

### Step 3: 手柄控制说明

- `L1 + Y`：切换到 WALK
- `L1 + X`：切回 IDLE
- `R1`：清空速度指令
- 打开一个 Bash 终端。

- 运行 robot-joystick：

  ```
  ./robot-joystick/robot-joystick
  ```
---

## 3. WFYG 外部 bundle 模式

WFYG 的策略推理仍由 ROS 部署仓负责；本仓只读取 bundle 的 manifest，
校验 task、18 关节顺序和 MJCF SHA-256，然后启动对应的 nominal/3 kg
物理模型。策略权重不会被复制到本仓。

```bash
cd /home/wkl/Desktop/limx-tron2/tron2_mujoco_sim
python3 simulator.py \
  --model-bundle /absolute/path/to/model-bundles/<bundle> \
  --asset-root /home/wkl/Desktop/limx-tron2/robot-description \
  --ip 127.0.0.1
```

CI 可增加 `--headless --duration 10`。bundle 模式会自动选择
`WFYG_TRON2A`；如果同时设置了其他 `ROBOT_TYPE`，启动会被拒绝。收到
首条合法控制命令前维持 bundle 默认姿态；命令超时、长度错误、NaN/Inf
或过大增益会进入受限保持。原有 SF/WF 环境变量和位置 IP 调用保持兼容。

## 4. 模型文件位置


请将 ONNX 模型按机型放在：

- `tron2-rl-deploy-python/controllers/model/<ROBOT_TYPE>/policy.onnx`
- `tron2-rl-deploy-python/controllers/model/<ROBOT_TYPE>/encoder.onnx`
- `tron2-rl-deploy-python/controllers/model/<ROBOT_TYPE>/params.yaml`

例如：

- `tron2-rl-deploy-python/controller/model/SF_TRON2A/...`
- `tron2-rl-deploy-python/controller/model/WF_TRON2A/...`

---

## 5. 效果展示

### 仿真部署 (Simulation)

![SF Simulation](doc/sfmj-ezgif.com-video-to-gif-converter.gif)
![WF Simulation](doc/wfmj-ezgif.com-video-to-gif-converter.gif)

### 实机部署 (Real-world)

实机部署时请悬挂启动控制器

![Deploy](doc/deploy.jpg)

![SF Real-world](doc/sf.GIF)
![WF Real-world](doc/wf.GIF)


## 6. 常见问题

- `ROBOT_TYPE not set`：先执行 `export ROBOT_TYPE=...`
- `Model not found`：检查 `controller/model/<ROBOT_TYPE>/` 下模型文件是否齐全
- `No module named limxsdk`：SDK wheel 未安装到当前 Python 环境
- `RobotState has not been received yet`：通常是仿真器没启动，或仿真与控制器的 `ROBOT_TYPE` 不一致

## 验证

下列命令与 CI 保持一致（见
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）；在提交
PR 前本地跑一遍，可以省去大量来回。

```bash
# 1. 字节码编译仿真器入口
python -m py_compile simulator.py

# 2. 子模块已声明并（本地）已初始化
git submodule status --recursive

# 3. 代码风格检查（与 CI 一致）
pip install ruff
ruff check --exclude robot-description --exclude robot-joystick \
           --exclude limxsdk-lowlevel --exclude __pycache__ .

# 4. MuJoCo 导入检查（无 MuJoCo 环境可跳过）
pip install "mujoco>=2.3"
python -c "import mujoco; print('mujoco', mujoco.__version__)"

# 5. 无禁止入库的文件被暂存
git ls-files | grep -iE '(^|/)__pycache__(/|$)|\.(pyc|onnx|pt|pth|ckpt|so|dll|whl|bag|mcap)$' && echo BAD || echo OK
```

完整仿真 dry-run（加载 MuJoCo 模型，不接控制器 —
可视化窗口出现后按 Ctrl-C 结束）：

```bash
export ROBOT_TYPE=SF_TRON2A
python3 simulator.py 127.0.0.1
```

## 引用与支持

如果你在学术或公开工作中使用了本仿真器，请引用本仓库：

```
@misc{limx_tron2_mujoco_sim_2026,
  title  = {TRON2 MuJoCo simulator},
  author = {LimX Dynamics},
  year   = {2026},
  howpublished = {\url{https://github.com/limx-tron2/tron2-mujoco-sim}}
}
```

- **Bug 反馈 / 功能建议：**
  [GitHub Issues](https://github.com/limx-tron2/tron2-mujoco-sim/issues)。
- **使用问题 / 集成帮助：**
  [GitHub Discussions](https://github.com/limx-tron2/tron2-mujoco-sim/discussions)。
- **安全问题上报：** 邮箱 `contact@limxdynamics.com`；仿真 vs 实机
  的边界说明见 [`SECURITY.md`](SECURITY.md)。
- **公司 / 商务联系：** <https://www.limxdynamics.com>。
