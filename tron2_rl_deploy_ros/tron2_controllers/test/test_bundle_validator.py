#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
VALIDATOR = PACKAGE / "scripts" / "validate_wfyg_bundle.py"
REGISTRY = json.loads((PACKAGE / "config" / "WFYG_TRON2A" / "task_profiles.yaml").read_text())["WFYGTaskRegistry"]["profiles"]
JOINTS = [
    "proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint", "knee_R_Joint", "wheel_R_Joint",
    "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joint", "arm5_Joint", "arm6_Joint",
    "gripper1_Joint", "gripper2_Joint",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BundleValidatorTest(unittest.TestCase):
    def make_bundle(self, root: Path, profile: dict) -> Path:
        bundle = root / profile["task_id"].replace("/", "_")
        bundle.mkdir()
        (bundle / "encoder.onnx").write_bytes(b"encoder")
        (bundle / "policy.onnx").write_bytes(b"policy")
        (bundle / "conformance.npz").write_bytes(b"golden")
        asset_root = root / "asset-root"
        asset_root.mkdir(exist_ok=True)
        asset_files = {}
        for kind in ("mjcf", "urdf", "urdf_rl", "usd"):
            asset_path = asset_root / f"{kind}.asset"
            asset_path.write_bytes(("<robot/>" if kind == "urdf_rl" else kind).encode())
            asset_files[kind] = {"relative_path": asset_path.name, "sha256": digest(asset_path)}
        manifest = {
            "schema": "limx.wfyg-deployment-bundle/v1",
            "task": {
                **profile,
                "robot_variant": "WFYG_TRON2A",
                "hardware_joint_names": JOINTS,
            },
            "assets": {"profile": "legacy-nominal", "files": asset_files},
            "models": {
                "encoder": {"file": "encoder.onnx", "sha256": digest(bundle / "encoder.onnx")},
                "policy": {"file": "policy.onnx", "sha256": digest(bundle / "policy.onnx")},
            },
            "conformance": {"file": "conformance.npz", "sha256": digest(bundle / "conformance.npz")},
        }
        (bundle / "manifest.yaml").write_text(json.dumps(manifest))
        return bundle

    def invoke(
        self, bundle: Path, task: str, robot_ip: str = "127.0.0.1", emit_urdf: bool = False
    ) -> subprocess.CompletedProcess:
        command = [
            sys.executable, str(VALIDATOR), "--bundle", str(bundle), "--task", task,
            "--robot-ip", robot_ip, "--asset-root", str(bundle.parent / "asset-root"),
        ]
        if emit_urdf:
            command.append("--emit-urdf")
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_all_training_ids_and_play_aliases_are_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in REGISTRY:
                bundle = self.make_bundle(root, profile)
                self.assertEqual(self.invoke(bundle, profile["task_id"]).returncode, 0)
                self.assertEqual(self.invoke(bundle, profile["play_alias"]).returncode, 0)
                emitted = self.invoke(bundle, profile["task_id"], emit_urdf=True)
                self.assertEqual(emitted.stdout, "<robot/>")

    def test_hash_mismatch_and_real_robot_ip_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory), REGISTRY[0])
            (bundle / "policy.onnx").write_bytes(b"tampered")
            self.assertNotEqual(self.invoke(bundle, REGISTRY[0]["task_id"]).returncode, 0)
            self.assertNotEqual(self.invoke(bundle, REGISTRY[0]["task_id"], "10.0.0.2").returncode, 0)


if __name__ == "__main__":
    unittest.main()
