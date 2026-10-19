# Changelog

All notable changes to `tron2-rl-deploy-ros` will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source scaffolding: `NOTICE`, `THIRD_PARTY_NOTICES.md`,
  `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- GitHub CI workflow (`.github/workflows/ci.yml`):
  - `xmllint` well-formedness check on every `package.xml`.
  - Deny-list scan for `*.onnx`, `*.pt`, `*.pth`, `*.ckpt`, `*.so`,
    `*.dll`, `*.dylib`, `*.lib`, `*.whl`, `*.bag`, `*.mcap` — CI
    passes only with the pre-existing controlled set explicitly
    allow-listed in the workflow; any newcomer fails the build.
  - `<license>TODO` / `<license>Proprietary` scan — CI fails **unless**
    each such line is immediately preceded by an `⚠ TO CONFIRM`
    annotation comment (so the risk cannot be silently removed).
  - Private-IP scan with an allow-list for the documented example
    `<robot-ip>` — any other private address fails the build.
  - EXIF sanity for `doc/` media (GPS / serial / author / artist).
- Issue templates (`bug_report.md`, `feature_request.md`, `config.yml`)
  and pull-request template under `.github/`. PR template includes an
  explicit checkbox: **"This PR does not add ONNX weights or new SDK
  binaries without a NOTICE update."**
- `CODEOWNERS` mapping default, safety, model, SDK, legal, hardware,
  content, and maintainer teams.
- `README.md`:
  - SPDX header (`Apache-2.0`).
  - "License & attribution" section linking `LICENSE`, `NOTICE`,
    `THIRD_PARTY_NOTICES.md`, `SECURITY.md`, `CONTRIBUTING.md`,
    `CHANGELOG.md`.
  - "Scope / not included" section stating explicitly that the four
    ONNX policy / encoder files, `libonnxruntime.so`, and the four
    proprietary / TODO-licensed packages are pending clearance.
  - "Verification" section listing the CI-equivalent local commands
    (`catkin_make`, `xmllint`, package.xml license scan, binary scan,
    private-IP scan).
  - "Cite & support" section with `contact@limxdynamics.com`.
- `.gitignore`:
  - ROS build outputs (`build/`, `devel/`, `install/`, `log/`,
    `.catkin_tools/`).
  - `__pycache__/`, editor / OS junk.
  - Explicit deny-list for controlled artifact types (`*.onnx`,
    `*.pt`, `*.pth`, `*.ckpt`, `*.so`, `*.dll`, `*.dylib`, `*.lib`,
    `*.whl`, `*.bag`, `*.mcap`). Controlled artifacts already in the
    tree are grandfathered by explicit path allow-list entries.
- `package.xml` (all four packages):
  - SPDX header comment (`Apache-2.0` for the metadata itself).
  - `<url>` entries (website / repository / bugtracker).
  - `<author email="contact@limxdynamics.com">LimX Dynamics</author>`.
  - `<maintainer>` email normalised to `contact@limxdynamics.com`.
  - `⚠ TO CONFIRM` annotation comment immediately before every
    unresolved `<license>` line — the underlying value is **not**
    changed (see "Pending owner sign-off" below).

### Changed
- `<version>` bumped to `0.1.0` across all four `package.xml` files
  (previously `0.0.0` for `tron2_controllers` per the task brief,
  `0.0.1` for `tron2_hw`, `robot_common`, and `onnxruntime_sdk`) so
  that the first public tag has a coherent minor version. ROS package
  **names** are preserved (`tron2_hw`, `tron2_controllers`,
  `robot_common`, `onnxruntime_sdk`) for downstream `<depend>`
  compatibility.

### Pending owner sign-off (blocks first public tag)

Every one of the following must be resolved by the named owner before
this repository can cut a public release. Each corresponds to a
### Resolved (2026-07-14)
- **`package.xml` `<license>` fields — resolved to `Apache-2.0`:**
  - `tron2_hw/package.xml` — was `Proprietary`, now `Apache-2.0`
  - `tron2_controllers/package.xml` — was `Proprietary`, now `Apache-2.0`
  - `robot_common/package.xml` — was `TODO`, now `Apache-2.0`
  - `onnxruntime_sdk/package.xml` — was `TODO`, now `Apache-2.0`
  All four `⚠ TO CONFIRM` inline comments have been removed. The
  Apache-2.0 grant covers LimX-authored sources in these packages; the
  bundled ONNX Runtime binary and the RL policy weights below remain
  separately gated.

`⚠ TO CONFIRM` row in `THIRD_PARTY_NOTICES.md` and is intentionally
listed here so that release engineering has a single blocking
checklist.

- **SDK owner — bundled ONNX Runtime binary:**
  - `onnxruntime_sdk/lib/libonnxruntime.so`
  - `onnxruntime_sdk/lib/libonnxruntime.so.1.10.0`
  Record exact upstream tag, build flags, target platform, and
  SHA-256; vendor upstream `LICENSE` and `ThirdPartyNotices.txt` next
  to the binary, or remove the binary and require users to install
  ONNX Runtime independently.
- **Model owner + robot-safety owner — RL policy / encoder weights:**
  - `tron2_controllers/config/SF_TRON2A/policy/policy.onnx`
  - `tron2_controllers/config/SF_TRON2A/policy/encoder.onnx`
  - `tron2_controllers/config/WF_TRON2A/policy/policy.onnx`
  - `tron2_controllers/config/WF_TRON2A/policy/encoder.onnx`
  For **each** file: training pipeline / experiment ID, training-data
  licensing, model card, SHA-256, and public-release approval.
- **Robot-safety owner — emergency-stop and real-hardware bring-up:**
  Review `tron2_hw/docs/bringup_mvp.md` (start / stop / emergency-stop
  services) and `tron2_hw/launch/tron2_hw.launch` (policy / encoder
  loading, controller-manager `dt`). Confirm that publishing service
  names and topics on a bring-up network is safe, and that no
  auto-start behaviour is triggered from the shipped launch files.
- **Content / legal — `doc/` media:**
  EXIF strip and visual re-review of `doc/deploy.jpg`, `doc/sf.GIF`,
  `doc/wf.GIF`, `doc/sfgazebo-ezgif.com-video-to-gif-converter.gif`,
  `doc/wfgazebo.gif` for individuals, office locations, badges,
  whiteboards, and internal hostnames.

### Resolved (2026-07-16)

- **Private-network IP** — resolved 2026-07-16 per owner decision.
  All Markdown / YAML command examples now use `<robot-ip>` as a
  placeholder token. Source-side defaults in
  `tron2_hw/src/Tron2HW.cpp:19`, `tron2_hw/src/tron2_hw_node.cpp:23`,
  and `tron2_hw/launch/tron2_hw.launch:3` retain the literal
  `10.192.1.2` as a documentation example; this is a documentation
  value only, declared in `SECURITY.md`. CI's private-IP scan
  continues to allowlist that one value.

## [0.1.0] — TBD

First public release. Contents (subject to the sign-offs above):

- Four ROS packages: `tron2_hw`, `tron2_controllers`, `robot_common`,
  `onnxruntime_sdk`.
- Launch files for Gazebo simulation and real-hardware bring-up.
- Configuration for the `SF_TRON2A` and `WF_TRON2A` variants.

[Unreleased]: https://github.com/limx-tron2/tron2-rl-deploy-ros/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/limx-tron2/tron2-rl-deploy-ros/releases/tag/v0.1.0
