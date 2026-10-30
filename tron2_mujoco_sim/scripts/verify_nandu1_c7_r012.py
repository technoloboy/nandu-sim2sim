"""Replay 1000 causal ticks through checkpoint and frozen ONNX controllers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from nandu1_c7_sim2sim import C7Simulation, VERSIONS, bundle_for


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(ROOT.parent / "tron2_rl_lab/rsl_rl"))
    from rsl_rl.nandu_drl.deployment import DeploymentController
    torch.set_num_threads(1)
    results = {}
    for version in VERSIONS:
        sim = C7Simulation(bundle_for(version), version)
        frozen = sim.controller
        checkpoint = DeploymentController.from_checkpoint(sim.metadata["source_checkpoint"])
        policy, history = sim.policy.clone(), sim.history.clone()
        initial = torch.tensor([sim.initial_height])
        checkpoint.reset(policy, history, initial_height=initial)
        frozen.reset(policy, history, initial_height=initial)
        frozen.complete_motor_records = False
        maxima = dict(estimate=0., issued_action=0., integral_mapping=0., command=0., integral=0.)
        for tick in range(1000):
            # Same observed physical trajectory for both controllers, not independent plants.
            policy[:, :3] = torch.tensor([[.01*np.sin(tick*.01), .01*np.cos(tick*.01), .02*np.sin(tick*.02)]])
            history = torch.cat((history.reshape(1, 10, 49)[:, 1:], policy[:, None]), 1).flatten(1)
            request = torch.tensor([[.12*np.sin(tick*.025), .05*np.sin(tick*.013), .9+.05*np.sin(tick*.009)]], dtype=torch.float32)
            estimated = [c.encoder(history) for c in (checkpoint, frozen)]
            actions = [c.step(policy, history, request, 4, tick) for c in (checkpoint, frozen)]
            pairs = dict(estimate=estimated,
                         issued_action=actions,
                         integral_mapping=[c.packet_context["mapping"] for c in (checkpoint, frozen)],
                         command=[c.processor.command for c in (checkpoint, frozen)],
                         integral=[c.state.integral for c in (checkpoint, frozen)])
            for key, (a, b) in pairs.items():
                maximum = float((a-b).abs().max())
                maxima[key] = max(maxima[key], maximum)
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5, msg=f"{version}/{tick}/{key}")
            policy[:, -13:] = actions[1]
        results[version] = dict(ticks=1000, passed=True, max_abs_error=maxima,
            checkpoint_sha256=sim.metadata["checkpoint_sha256"],
            task=sim.metadata["task_name"], iteration=sim.metadata["checkpoint_iteration"])
        print(version, json.dumps(maxima), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
