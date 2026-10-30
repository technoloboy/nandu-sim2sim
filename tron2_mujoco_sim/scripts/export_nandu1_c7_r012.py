"""Create three task-isolated ONNX/MuJoCo bundles from real 30000 checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nandu1_c7_sim2sim import VERSIONS, bundle_for, validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=VERSIONS, nargs="+", default=list(VERSIONS))
    parser.add_argument("--training-root", type=Path, default=ROOT.parent / "tron2_rl_lab")
    parser.add_argument("--asset-dir", type=Path,
        default=ROOT.parent / "deployment_bundles/nandu1_v1_3_1_model_30000/assets")
    args = parser.parse_args()
    sys.path.insert(0, str(args.training_root / "rsl_rl"))
    import torch
    from rsl_rl.nandu_drl.contract import DRLTaskSpec
    from rsl_rl.nandu_drl.checkpoint import validate_checkpoint
    from rsl_rl.nandu_drl.export import export_controller
    for version in args.version:
        checkpoint = args.training_root / f"logs/rsl_rl/autodl/v1-6-C7-P-1-{version}-261029-30000/model_30000.pt"
        output = bundle_for(version)
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        spec = DRLTaskSpec.from_task(f"Isaac-NANDU1-Blind-Flat-v1-6-C7-P-1-{version}")
        validate_checkpoint(data, spec)
        if data["iteration"] != 30000:
            raise ValueError("This export requires iteration 30000")
        if output.exists():
            raise FileExistsError(f"Refusing to replace bundle: {output}")
        if not (args.asset_dir / "robot.xml").is_file():
            raise FileNotFoundError(args.asset_dir / "robot.xml")
        meta = export_controller(checkpoint, output)
        shutil.copyfile(args.training_root / "rsl_rl/rsl_rl/nandu_drl/mujoco_adapter.py",
                        output / "processor/mujoco_adapter.py")
        sources = output / "simulator_sources"
        sources.mkdir()
        for name in ("nandu1_contract.py", "nandu1_v131.py"):
            shutil.copyfile(ROOT / name, sources / name)
        shutil.copytree(args.asset_dir, output / "assets")
        meta.update(sim2sim_format="nandu_c7_r012_mujoco_v1", checkpoint_iteration=30000,
                    source_checkpoint=str(checkpoint.resolve()),
                    simulator_asset_source=str(args.asset_dir.resolve()),
                    simulation_clocks_s=dict(policy=.02, pd=.005, physics=.001),
                    note="Frozen nominal MuJoCo plant. Contact/friction solver differs from Isaac; not hardware qualification.")
        meta["files"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in output.rglob("*") if p.is_file() and p != output / "manifest.json"}
        (output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        validate_bundle(output, version)
        print(f"{version}: {checkpoint} -> {output}", flush=True)


if __name__ == "__main__":
    main()
