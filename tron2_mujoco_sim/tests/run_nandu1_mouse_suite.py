"""Sequential real GUI/ROS acceptance for the seven deployed local tasks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SIM = Path(__file__).resolve().parents[1]
MODELS = [('1', 'nandu1_v1_1_model_30000'), ('2', 'nandu1_v1_2_model_40000'),
          ('3', 'nandu1_v1_3_model_30000'), ('2-1', 'nandu1_v1_2_1_model_29000'),
          ('3-1', 'nandu1_v1_3_1_model_30000'), ('4', 'nandu1_v1_4_model_24000'),
          ('4-1', 'nandu1_v1_4_1_model_14000')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for version, model in MODELS:
        task = 'Isaac-NANDU1-Blind-Flat-v1-' + version
        destination = args.output / ('v1_' + version.replace('-', '_'))
        command = [sys.executable, str(SIM / 'tests/run_nandu1_mouse_e2e.py'),
                   '--task', task, '--bundle', str(SIM.parent / 'deployment_bundles' / model),
                   '--output', str(destination)]
        print('Starting ' + task, flush=True)
        completed = subprocess.run(command, capture_output=True, text=True,
                                   env=dict(os.environ, PYTHONUNBUFFERED='1'))
        (args.output / (destination.name + '.log')).write_text(completed.stdout + completed.stderr)
        results.append(dict(task=task, exit_code=completed.returncode,
                            result='pass' if completed.returncode == 0 else 'fail'))
        print(json.dumps(results[-1]), flush=True)
        (args.output / 'suite.json').write_text(json.dumps(results, indent=2))
    return 0 if all(result['exit_code'] == 0 for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
