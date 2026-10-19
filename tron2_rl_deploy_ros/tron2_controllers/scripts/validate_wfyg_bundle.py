#!/usr/bin/env python3
"""Fail-fast validator used synchronously by the WFYG roslaunch file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SCHEMA = "limx.wfyg-deployment-bundle/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_yaml(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot parse canonical JSON/YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Root of {path} must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--emit-urdf", action="store_true")
    args = parser.parse_args()
    if args.robot_ip not in ("127.0.0.1", "localhost"):
        raise ValueError("WFYG policy launch is sim2sim-only and refuses a non-loopback robot IP")

    bundle = args.bundle.expanduser().resolve()
    manifest = load_json_yaml(bundle / "manifest.yaml")
    registry_path = Path(__file__).resolve().parents[1] / "config" / "WFYG_TRON2A" / "task_profiles.yaml"
    registry = load_json_yaml(registry_path)["WFYGTaskRegistry"]["profiles"]
    by_name = {}
    for profile in registry:
        by_name[profile["task_id"]] = profile
        by_name[profile["play_alias"]] = profile
    if args.task not in by_name:
        raise ValueError(f"Task is not registered for WFYG deployment: {args.task}")
    expected = by_name[args.task]
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported bundle schema: {manifest.get('schema')!r}")
    task = manifest.get("task", {})
    if task.get("task_id") != expected["task_id"]:
        raise ValueError(f"Bundle task {task.get('task_id')} does not match requested task {expected['task_id']}")
    for key in (
        "runtime_profile", "observation_size", "command_size", "action_size",
        "encoder_input_size", "policy_input_size", "action_decoder",
    ):
        if task.get(key) != expected[key]:
            raise ValueError(f"Bundle task contract mismatch for {key}")
    if task.get("robot_variant") != "WFYG_TRON2A" or len(task.get("hardware_joint_names", [])) != 18:
        raise ValueError("Bundle does not contain the WFYG 18-joint hardware contract")
    assets = manifest.get("assets", {})
    if assets.get("profile") not in ("legacy-nominal", "payload-3kg"):
        raise ValueError("Bundle asset profile is missing or unsupported")
    for name in ("encoder", "policy"):
        metadata = manifest.get("models", {}).get(name, {})
        model_path = bundle / metadata.get("file", "")
        if not model_path.is_file() or sha256(model_path) != metadata.get("sha256"):
            raise ValueError(f"{name} ONNX checksum mismatch")
    conformance = manifest.get("conformance", {})
    conformance_path = bundle / conformance.get("file", "")
    if not conformance_path.is_file() or sha256(conformance_path) != conformance.get("sha256"):
        raise ValueError("Conformance-vector checksum mismatch")
    verified_assets = {}
    if args.asset_root is not None:
        asset_root = args.asset_root.expanduser().resolve()
        for kind in ("mjcf", "urdf", "urdf_rl", "usd"):
            metadata = assets.get("files", {}).get(kind, {})
            relative_path = metadata.get("relative_path", "")
            path = (asset_root / relative_path).resolve()
            try:
                path.relative_to(asset_root)
            except ValueError as exc:
                raise ValueError(f"{kind} asset path escapes asset root") from exc
            if not path.is_file() or sha256(path) != metadata.get("sha256"):
                raise ValueError(f"{kind} asset checksum mismatch")
            verified_assets[kind] = path
    elif args.emit_urdf:
        raise ValueError("--emit-urdf requires --asset-root")
    if args.emit_urdf:
        sys.stdout.write(verified_assets["urdf_rl"].read_text(encoding="utf-8"))
    else:
        print("validated")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - roslaunch needs one concise failure
        print(f"WFYG bundle validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
