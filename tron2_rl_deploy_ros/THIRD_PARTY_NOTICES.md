# Third-Party Notices

`tron2-rl-deploy-ros` (TRON2 RL deployment for ROS) is distributed under
the Apache License 2.0 at the repository top level (see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE)).

However, several in-tree artifacts and ROS packages are **not** yet
cleared under that license. This file lists every such artifact, its
provenance status, and the owner action required before a public
release.

> **Status:** items marked `⚠ TO CONFIRM` are pending sign-off from
> the responsible legal / hardware / model / SDK / safety owner. Do
> not cut a public release while any `⚠ TO CONFIRM` entry remains.

---

## 1. ROS packages in this repository — resolved (2026-07-14)

All four ROS packages now declare `<license>Apache-2.0</license>`,
consistent with the top-level [`LICENSE`](LICENSE).

| ROS package | Path | `<license>` | Status |
|-------------|------|-------------|--------|
| `tron2_hw`          | `tron2_hw/package.xml`          | `Apache-2.0` | ✅ resolved |
| `tron2_controllers` | `tron2_controllers/package.xml` | `Apache-2.0` | ✅ resolved |
| `robot_common`      | `robot_common/package.xml`      | `Apache-2.0` | ✅ resolved |
| `onnxruntime_sdk`   | `onnxruntime_sdk/package.xml`   | `Apache-2.0` | ✅ resolved — covers the LimX-authored CMake wrapper; the bundled ONNX Runtime binary in §2 keeps its own `⚠ TO CONFIRM` row for provenance / checksum / redistribution notice. |

The `⚠ TO CONFIRM` inline comments in each `package.xml` have been
removed. The CI guard remains: any future `<license>TODO` or
`<license>Proprietary` value regresses the sign-off and fails CI unless
paired with a fresh `⚠ TO CONFIRM` annotation.

---

## 2. Bundled ONNX Runtime binary

| Path | Size (bytes) | SHA-256 | Kind (`file` output) |
|------|-------------:|---------|----------------------|
| `onnxruntime_sdk/lib/libonnxruntime.so`        | 14 320 080 | `0db7bb6201cc86370e54cb9ff0baf768a75660a872f4732822d325c84ac3d9d9` | ELF 64-bit LSB shared object, x86-64, SYSV, dynamically linked |
| `onnxruntime_sdk/lib/libonnxruntime.so.1.10.0` | 14 320 080 | `0db7bb6201cc86370e54cb9ff0baf768a75660a872f4732822d325c84ac3d9d9` | Byte-identical soname alias of the file above |
| `onnxruntime_sdk/include/*`                    | headers    | —                                                                    | Public C / C++ headers for the same version |

Evidence collected 2026-07-16: SHA-256 and file-type strings computed
against the tracked working-tree files (reproduce with the block in
§2.1 below). The two `.so` files are **byte-identical** (same SHA-256);
one is a soname alias of the other, not a second architecture. Only a
**single Linux x86-64 SYSV build** is shipped — there is no aarch64
variant tracked in this repository.

- **Upstream project:** Microsoft ONNX Runtime — <https://github.com/microsoft/onnxruntime>
- **Upstream license:** MIT (see the ONNX Runtime `LICENSE`).
- **Bundled version:** filename claims **1.10.0**. ⚠ TO CONFIRM by
  SDK owner: exact upstream tag / commit, build flags (CPU / CUDA /
  TensorRT / …), and match the recorded SHA-256 to an authorised build.
- **Redistribution clearance:** ⚠ TO CONFIRM. MIT permits
  redistribution with attribution, but this repo does **not** yet
  ship the ONNX Runtime `LICENSE` and `ThirdPartyNotices.txt` next to
  the binary as MIT requires. Before release, either:
  - **(A)** vendor the upstream `LICENSE` / `ThirdPartyNotices.txt`
    into `onnxruntime_sdk/`, cite the exact upstream tag matching the
    SHA-256 above, and keep the recorded SHA-256 here; **or**
  - **(B)** remove the binary and headers, and require downstream
    users to install `onnxruntime` from upstream releases (documented
    in `README.md`).

### 2.1 Reproduce the evidence

```bash
sha256sum onnxruntime_sdk/lib/libonnxruntime.so \
          onnxruntime_sdk/lib/libonnxruntime.so.1.10.0
file      onnxruntime_sdk/lib/libonnxruntime.so
```

Owner action: **SDK lead + legal / OSPO**.

---

## 3. RL policy / encoder ONNX weights

Four ONNX files are checked into the tree:

| Path | Size (bytes) | SHA-256 |
|------|-------------:|---------|
| `tron2_controllers/config/SF_TRON2A/policy/policy.onnx`  | 791 050 | `0b353a087c912c33b9ba690560f3501cf7bf2bf25fde91c07ee2bdfb36502d3a` |
| `tron2_controllers/config/SF_TRON2A/policy/encoder.onnx` | 588 998 | `5f7e2b8865fda7c284f0dd98b79f5c1d78935c83cbf43bf311d768154937111e` |
| `tron2_controllers/config/WF_TRON2A/policy/policy.onnx`  | 770 148 | `3000df452681056738a15b46fa67f4f8436b34bbb6dcc6b22fa08b1b1f8dd071` |
| `tron2_controllers/config/WF_TRON2A/policy/encoder.onnx` | 503 276 | `507d0630d78873f7aabfeab4eae9d7669610d709fcc903c4296d1908da54b3e7` |

Evidence collected 2026-07-16: all four files are **byte-identical**
to the corresponding files in the sibling `tron2-rl-deploy-python` repo
(`controllers/model/{SF,WF}_TRON2A/{policy,encoder}.onnx`). Any owner
decision made about these four models must be applied to both repos
consistently — they are one artifact under two paths.

For **each** of the four files, the model owner must resolve the
following before a public release:

- ⚠ TO CONFIRM — training pipeline, commit hash / experiment ID.
- ⚠ TO CONFIRM — training data provenance and licensing (was any
  third-party motion capture, imitation data, or asset used?).
- ⚠ TO CONFIRM — public-release approval (may LimX distribute these
  weights under Apache-2.0, under a separate model licence, or not
  at all?).
- ⚠ TO CONFIRM — model card: input / output tensor shapes, expected
  observation frame, action bounds, safety envelope, and known
  failure modes.
- ⚠ TO CONFIRM — real-hardware safety review: the same policy is
  loaded on the physical robot by `tron2_hw.launch`; the robot-safety
  owner must sign that the shipped weights are the ones cleared for
  public bring-up.

If any of the four cannot be cleared, remove the file (and add its
path to `.gitignore`) **in both repositories** before release. Do
**not** silently swap in different weights — record the substitution
here and in the sibling `tron2-rl-deploy-python/MODEL_CARD.md`.

Owner action: **model lead + robot-safety lead + legal**.

---

## 4. External repository dependencies (not vendored)

`tron2-rl-deploy-ros` depends on the following LimX repositories at
build time. They are **not** vendored here; the workspace layout in
`README.md` expects sibling checkouts.

| Repository | Role | License status |
|------------|------|----------------|
| `robot-description`  | URDF / xacro / MuJoCo / STL / USD for TRON2A. Referenced via `<depend>robot_description</depend>` in `tron2_controllers/package.xml`. | Apache-2.0 (see the sibling repo's `LICENSE` and `NOTICE`). ⚠ TO CONFIRM per-asset mesh / USD provenance is tracked in that repo's `THIRD_PARTY_NOTICES.md`. |
| `limxsdk-lowlevel`   | Low-level SDK used by `tron2_hw`. Referenced via `<depend>limxsdk_lowlevel</depend>` in `tron2_hw/package.xml`. | ⚠ TO CONFIRM — SDK owner must publish the license before this repo can be built from a public clone. |

---

## 5. ROS runtime dependencies (not redistributed)

The following ROS / third-party packages are declared as `<depend>` in
this repo's `package.xml` files and are **runtime dependencies only** —
they are neither vendored nor shipped here.

| Dependency | Where declared | License | Where obtained |
|------------|----------------|---------|----------------|
| `roscpp`, `std_msgs`, `std_srvs`, `geometry_msgs`, `urdf`, `realtime_tools` | `tron2_hw`, `tron2_controllers`, `robot_common` | BSD-3-Clause | ROS Noetic distribution |
| `controller_manager`, `controller_manager_msgs`, `controller_interface`, `hardware_interface` | `tron2_hw`, `tron2_controllers`, `robot_common` | BSD-3-Clause | `ros-noetic-ros-control` / `ros-noetic-ros-controllers` |
| `eigen`     | `tron2_controllers` (`<build_depend>`) | MPL-2.0 / BSD-3-Clause | `libeigen3-dev` |
| `rclcpp`    | `onnxruntime_sdk` (ROS 2 branch) | Apache-2.0 | ROS 2 distribution |
| ONNX Runtime (linked, C++) | `tron2_controllers/CMakeLists.txt` links `${ONNXRUNTIME_SDK_LIB_FILES}` | MIT | See §2 above |

Downstream users must obtain and license these independently.

---

## 6. Documentation media

| Path | Kind | Provenance | License |
|------|------|------------|---------|
| `doc/deploy.jpg`                                     | Photo of a real-hardware bring-up (robot on a hoist) | ⚠ TO CONFIRM — must not disclose office interior, individuals, or non-public products. | ⚠ TO CONFIRM |
| `doc/sf.GIF`, `doc/wf.GIF`                           | Real-hardware locomotion capture           | ⚠ TO CONFIRM | ⚠ TO CONFIRM |
| `doc/sfgazebo-ezgif.com-video-to-gif-converter.gif`  | Screen-recorded Gazebo simulation          | ⚠ TO CONFIRM | ⚠ TO CONFIRM |
| `doc/wfgazebo.gif`                                   | Screen-recorded Gazebo simulation          | ⚠ TO CONFIRM | ⚠ TO CONFIRM |

Before release, run:

```bash
exiftool doc/* | grep -iE '(gps|serial|make|model|software|author|artist|copyright)'
```

and strip anything that discloses office locations, camera serials, or
individual contributors' names, unless intentionally kept:

```bash
exiftool -all= doc/*
```

Also visually re-review every frame of the GIFs for people, whiteboards,
badges, and screens containing internal URLs / hostnames.

---

## 7. Sensitive on-robot configuration referenced in-tree

Not shipped as separate files, but present in text and therefore
subject to legal / safety review before publication:

- **Private-network robot IP.** All Markdown / YAML command examples
  use `<robot-ip>` as a placeholder token; substitute your robot's
  actual IP before running. The source-side defaults in
  `tron2_hw/src/Tron2HW.cpp:19`, `tron2_hw/src/tron2_hw_node.cpp:23`,
  and the launch argument default in `tron2_hw/launch/tron2_hw.launch:3`
  retain the literal `10.192.1.2` as a documentation example
  describing typical real-hardware usage. This is a documentation
  value only and is not the address of any LimX production network;
  the private-IP handling policy is declared in `SECURITY.md` under
  "Known placeholders and internal identifiers". CI's private-IP
  scan allow-lists that one literal — any other RFC 1918 / RFC 6598
  address in-tree fails the build.
- Emergency-stop / start / stop service names and topics documented
  in `tron2_hw/docs/bringup_mvp.md`. ⚠ TO CONFIRM by robot-safety
  owner: publishing these interface names is safe (they are already
  visible over the ROS graph on a bring-up network).

---

## 8. Update procedure

Whenever a package, binary, model, image, or upstream reference is
added or changed:

1. Update the corresponding section of this file.
2. Re-run the EXIF strip (§6) on any changed doc media.
3. If the change touches an `⚠ TO CONFIRM` row, block the merge on
   written sign-off from the responsible owner (see `.github/CODEOWNERS`).
4. Record any SHA-256 change of `libonnxruntime.so*` and re-run the
   SDK owner's clearance.
