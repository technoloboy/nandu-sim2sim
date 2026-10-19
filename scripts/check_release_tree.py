#!/usr/bin/env python3
"""Audit exactly the staged Git blobs, including LFS pointers and gitlinks."""
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT
ORIGINAL_MD = {'CHANGELOG.md', 'CONTRIBUTING.md', 'README.md', 'README_zh-CN.md',
               'SECURITY.md', 'THIRD_PARTY_NOTICES.md'}
BAD_PARTS = {'__pycache__', '.pytest_cache', '.superpowers', 'outputs', 'output',
             'logs', 'cache', '.cache', '.venv', 'venv', '.ssh',
             '.nandu1_noetic_ws', 'build', 'devel', 'install', 'docs'}
LEGACY_MODELS = {
    f'tron2_rl_deploy_ros/tron2_controllers/config/{robot}/policy/{graph}.onnx'
    for robot in ('SF_TRON2A', 'WF_TRON2A') for graph in ('encoder', 'policy')}
SECRET = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})|\bsshpass\s+-p\s')


def payload_paths(contents):
    """Approved catalogue -> exact sealed payload paths, never whole folders."""
    data = contents.get('deployment_bundles/catalogue.json')
    if data is None:return set()
    result = {'deployment_bundles/catalogue.json'}
    for bundle in json.loads(data)['bundles']:
        name = bundle['directory']
        if Path(name).name != name:raise ValueError('Unsafe catalogue directory')
        prefix = 'deployment_bundles/' + name + '/'
        manifest_name = prefix + 'manifest.yaml'
        if manifest_name not in contents:continue
        manifest = json.loads(contents[manifest_name])
        result.update((manifest_name, prefix + 'SHA256SUMS'))
        result.update(prefix + relative for relative in manifest['files'])
    return result


def local_payload_paths():
    # Only the pure path-classification API uses this fallback. main() below
    # always passes the catalogue/manifests from the Git index itself.
    catalogue = RELEASE_ROOT / 'deployment_bundles/catalogue.json'
    contents = {'deployment_bundles/catalogue.json': catalogue.read_bytes()}
    for b in json.loads(contents['deployment_bundles/catalogue.json'])['bundles']:
        name = 'deployment_bundles/' + b['directory'] + '/manifest.yaml'
        contents[name] = (RELEASE_ROOT / name).read_bytes()
    return payload_paths(contents)


def violations(paths, allowed_payloads=None):
    allowed = local_payload_paths() if allowed_payloads is None else allowed_payloads
    result = []
    for relative in paths:
        p = Path(relative)
        golden = relative == 'tron2_rl_deploy_ros/tron2_controllers/test/golden/wfyg_math.csv'
        if BAD_PARTS.intersection(p.parts) or p.suffix.lower() in ('.pyc', '.pyo', '.bag', '.mcap', '.mp4', '.log') or (p.suffix == '.csv' and not golden):
            result.append(relative + ': generated/output artifact')
        if p.suffix.lower() == '.md' and (len(p.parts) != 2 or p.parts[0] not in ('tron2_mujoco_sim', 'tron2_rl_deploy_ros') or p.name not in ORIGINAL_MD):
            result.append(relative + ': non-original Markdown')
        if p.name in ('.nandu1_launch.lock', 'docker-local-proxy.conf') or p.suffix in ('.sock', '.ckpt', '.pth'):
            result.append(relative + ': local-only artifact')
        if p.name in ('.env', '.git-credentials', '.netrc', 'credentials.json', 'id_rsa', 'id_ed25519') or p.name.startswith('.env.') or p.suffix.lower() in ('.pem', '.key'):
            result.append(relative + ': credential-like filename')
        if p.parts[0] == 'deployment_bundles' and relative not in allowed:
            result.append(relative + ': unapproved bundle payload')
        if p.suffix in ('.onnx', '.pt') and p.parts[0] != 'deployment_bundles' and relative not in LEGACY_MODELS:
            result.append(relative + ': unapproved model location')
        if re.fullmatch(r'model_\d+\.pt', p.name):
            result.append(relative + ': training checkpoint')
    return result


def staged_snapshot(root):
    records = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root)
    paths, entries = [], {}
    for row in records.split(b'\0'):
        if not row:continue
        fields, name = row.split(b'\t', 1)
        mode, oid, stage = fields.split()
        if stage != b'0':raise ValueError('Unmerged index cannot be published')
        path = name.decode('utf-8')
        paths.append(path)
        if mode != b'160000':entries[path] = oid
    ids = sorted(set(entries.values()))
    reply = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=root,
        input=b''.join(oid + b'\n' for oid in ids))
    stream = io.BytesIO(reply)
    blobs = {}
    for oid in ids:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b'blob':raise ValueError('Invalid staged blob')
        size = int(header[2]);blobs[oid] = stream.read(size)
        if stream.read(1) != b'\n':raise ValueError('Truncated staged blob')
    return paths, {path:blobs[oid] for path, oid in entries.items()}


def main():
    paths, contents = staged_snapshot(ROOT)
    bad = violations(paths, payload_paths(contents))
    # Scan all staged bytes, not only selected filename suffixes and never the
    # potentially different working tree. Do not echo any matched secret.
    bad.extend(path + ': credential-like content' for path, body in contents.items() if SECRET.search(body))
    if bad:raise SystemExit('\n'.join(bad))
    print('Published tree policy passed')


if __name__ == '__main__':main()
