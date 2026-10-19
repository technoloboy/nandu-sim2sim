# Contributing to `tron2-rl-deploy-ros`

Thanks for helping improve the TRON2 RL deployment stack. This
repository is different from `robot-description`: it contains
**executable control code that can move a real humanoid robot**.
Contributions are held to a higher bar for that reason.

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [What we do not accept](#what-we-do-not-accept)
- [Development setup](#development-setup)
- [Repository layout](#repository-layout)
- [Coding conventions](#coding-conventions)
- [Verification before opening a PR](#verification-before-opening-a-pr)
- [Model / weight provenance checklist](#model--weight-provenance-checklist)
- [Binary artifact policy](#binary-artifact-policy)
- [Real-hardware bring-up](#real-hardware-bring-up)
- [Commit messages](#commit-messages)
- [Pull request checklist](#pull-request-checklist)
- [Sign-off (DCO)](#sign-off-dco)
- [Code of conduct](#code-of-conduct)

## Ways to contribute

- Bug reports for the controller loop, launch files, or bring-up
  procedure.
- Simulation-only improvements (`tron2_controller_sim.launch`,
  `tron2_hw_sim.launch`) that do not affect the real-hardware code
  path.
- Refactors that reduce the surface area between simulation and real
  hardware.
- Documentation, verification snippets, sim examples.
- CI hardening (linters, static analysis, package.xml checks).

## What we do not accept

- **New ONNX weights** (`*.onnx`) or new SDK binaries (`*.so`, `*.dll`,
  `*.dylib`) **without prior written approval** from the model owner
  (weights) or SDK owner (binaries). See
  [Model / weight provenance checklist](#model--weight-provenance-checklist)
  and [Binary artifact policy](#binary-artifact-policy).
- Vendor CAD files (`.sldprt`, `.step`, `.iges`, `.blend`) — those
  belong in `robot-description`, and even there only if the license
  permits redistribution.
- Firmware, bootloader images, or field-calibration values.
- Rosbags, MCAP recordings, HDF5 trajectory captures — training and
  motion data belong in dedicated data repositories under signed
  redistribution terms.
- Customer-specific configuration, hostnames, credentials, or private
  IPs. Command examples use `<robot-ip>` as a substitution token;
  substitute your robot's actual IP before running. The only literal
  IP that may appear in-tree is the documentation example `10.192.1.2`
  used as the default value in `tron2_hw/src/Tron2HW.cpp`,
  `tron2_hw/src/tron2_hw_node.cpp`, and `tron2_hw/launch/tron2_hw.launch`
  (see `SECURITY.md` §"Known placeholders and internal identifiers").
- Images or media that disclose office locations, individuals, or
  non-public products.
- Auto-start behaviour on the real robot (systemd units, udev rules,
  `.desktop` autostart, etc.). The physical robot must always require
  an explicit operator action to start controllers.

## Development setup

Prerequisites:

- Ubuntu 20.04.
- ROS 1 Noetic (`ros-noetic-desktop-full`).
- Gazebo 11 (default with Noetic).
- Common ROS control packages (see `README.md` for the full apt
  install list).
- Sibling workspaces of `robot-description` and `limxsdk-lowlevel`
  checked out under `~/limx_ws/src/`. See `README.md` §1.

```bash
# ROS 1 workspace
cd ~/limx_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Repository layout

```
tron2-rl-deploy-ros/
├── onnxruntime_sdk/     # CMake wrapper + bundled libonnxruntime.so (⚠ pending clearance)
├── robot_common/        # Shared hardware-interface headers (⚠ license TODO)
├── tron2_controllers/   # RL controller: loads *.onnx, applies torques (⚠ pending clearance)
│   └── config/*/policy/ # *.onnx policy / encoder weights (⚠ pending clearance)
├── tron2_hw/            # Real-hardware interface + controller-manager loop
│   ├── launch/          # tron2_hw.launch, tron2_hw_sim.launch, tron2_controller_sim.launch
│   └── docs/            # bring-up MVP notes
├── doc/                 # Demo images / GIFs (⚠ pending EXIF + content review)
└── LICENSE / NOTICE / THIRD_PARTY_NOTICES.md / SECURITY.md / CHANGELOG.md
```

- Do not rename existing ROS packages — several downstream repositories
  `<depend>` on them by name.
- Do not commit editor / OS junk — see `.gitignore`.

## Coding conventions

- **C++:** C++14+, follow ROS style (roughly Google C++ with `snake_case`
  functions and `PascalCase` types). Prefer RAII; no naked `new`.
- **CMake:** target-based (`target_include_directories`,
  `target_link_libraries` — do not use global `include_directories` in
  new code).
- **ROS parameters:** every tunable that affects hardware behaviour
  (rates, gains, torque limits, mode transitions) must have a default
  in the YAML under `tron2_controllers/config/<VARIANT>/` and be
  overridable from a launch file.
- **Loop rates and dt:** the controller-manager `dt` and the
  ONNX-inference rate are safety-relevant. Any change requires a
  robot-safety owner in the review.

## Verification before opening a PR

Run all of the following and paste the summary into the PR
description. CI runs an equivalent subset — running locally saves
review round-trips.

```bash
# 1. Build succeeds
cd ~/limx_ws
source /opt/ros/noetic/setup.bash
catkin_make

# 2. Every package.xml is well-formed
xmllint --noout $(find . -name 'package.xml')

# 3. No new proprietary / TODO license without an annotation
find . -name 'package.xml' -exec grep -Hn '<license>' {} \;
# Every <license>Proprietary</license> and <license>TODO</license> line
# must be immediately preceded by an "⚠ TO CONFIRM" comment.

# 4. No new binaries or weights slipped in
git ls-files | grep -iE '\.(onnx|pt|pth|ckpt|so|dll|dylib|lib|whl|bag|mcap)$'
# The output must exactly match the pre-existing controlled set
# (see THIRD_PARTY_NOTICES.md §2, §3). Any new entry must have owner
# sign-off recorded in the PR description.

# 5. Private-IP / hostname scan
grep -RIn --exclude-dir=.git -E '\b10\.[0-9]+\.[0-9]+\.[0-9]+\b' .
# The only literal that may appear is the documentation example
# 10.192.1.2 (default in tron2_hw/src/Tron2HW.cpp,
# tron2_hw/src/tron2_hw_node.cpp, tron2_hw/launch/tron2_hw.launch —
# see SECURITY.md). All Markdown / YAML command examples must use
# the <robot-ip> substitution token instead of a literal address.
# Any other hit must be discussed in the PR.
```

## Model / weight provenance checklist

When adding, replacing, or updating any `*.onnx` file:

- [ ] Written approval from the model owner recorded in the PR.
- [ ] Training pipeline / commit / experiment ID logged in
      `THIRD_PARTY_NOTICES.md` §3.
- [ ] Training-data licensing confirmed (no third-party mocap /
      imitation data with incompatible terms).
- [ ] SHA-256 of the new file recorded.
- [ ] Model card: input / output shapes, observation frame, action
      bounds, known failure modes.
- [ ] Robot-safety owner has signed that the shipped weights are the
      ones cleared for public bring-up.

Missing any of the above is a blocker.

## Binary artifact policy

`onnxruntime_sdk/lib/libonnxruntime.so*` is the only pre-existing
binary. Do **not** replace it, and do not add other `.so` / `.dll` /
`.dylib` / `.lib` files, without:

- [ ] SDK owner sign-off recorded in the PR.
- [ ] Upstream project, exact tag / commit, build flags, target
      platform, and SHA-256 recorded in `THIRD_PARTY_NOTICES.md` §2.
- [ ] Upstream `LICENSE` and any required attribution files vendored
      alongside the binary.

## Real-hardware bring-up

Anything under `tron2_hw/launch/tron2_hw.launch` and the
`tron2_hw/docs/bringup_mvp.md` procedure is safety-relevant. Changes
require:

- [ ] Robot-safety owner review.
- [ ] Test on a **hoisted** robot, not free-standing, unless the change
      is proven simulation-only.
- [ ] No new auto-start behaviour (see
      [What we do not accept](#what-we-do-not-accept)).
- [ ] Emergency-stop service still halts motion within the documented
      time budget.

## Commit messages

Follow Conventional Commits:

```
type(scope): short imperative summary

Longer explanation if needed.

Signed-off-by: Your Name <you@example.com>
```

`type` ∈ `feat | fix | docs | refactor | chore | ci | test | safety`.
`scope` is usually a package folder (`tron2_hw`, `tron2_controllers`,
`robot_common`, `onnxruntime_sdk`) or `meta` for repo-wide changes.

## Pull request checklist

- [ ] `catkin_make` succeeds locally against `~/limx_ws` with the
      sibling `robot-description` and `limxsdk-lowlevel` checked out.
- [ ] `xmllint --noout` passes for every `package.xml`.
- [ ] `<license>TODO` / `<license>Proprietary` lines are still each
      preceded by the `⚠ TO CONFIRM` annotation comment.
- [ ] No new `*.onnx` / `*.so` / `*.dll` / `*.pt` / `*.bag` /
      `*.mcap` without owner approval (see the two checklists above).
- [ ] `THIRD_PARTY_NOTICES.md` updated if any provenance changed.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`.
- [ ] No calibration values, firmware, credentials, hostnames, or
      customer-specific configuration.
- [ ] No media that discloses individuals or sites (EXIF scan clean).
- [ ] DCO sign-off on every commit.

## Sign-off (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/).
Every commit must be signed off:

```bash
git commit -s -m "your message"
```

Signing off certifies that you have the right to submit the change
under the repository's license.

## Code of conduct

Be respectful and constructive. Reports to
`contact@limxdynamics.com`.
