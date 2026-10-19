import os
from pathlib import Path
import subprocess
import unittest
import tempfile

def check_keyboard_wrapper(tmp_path):
    """Real launcher against a docker executable recording its CLI arguments."""
    root=Path(__file__).resolve().parents[2]
    docker=tmp_path/'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    env=dict(os.environ,PATH=str(tmp_path)+':'+os.environ['PATH'])
    master,slave=os.openpty()
    try:
        result=subprocess.run(['bash',str(root/'scripts/nandu1_noetic.sh'),'run','true'],stdin=slave,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,text=True)
        assert result.returncode==0,result.stderr
        assert '-it' in result.stdout.splitlines()
    finally:
        os.close(master);os.close(slave)

class ContainerTest(unittest.TestCase):
    def test_supervised_run_has_explicit_container_identity(self):
        with tempfile.TemporaryDirectory() as path:
            docker=Path(path)/'docker';docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');docker.chmod(0o755)
            root=Path(__file__).resolve().parents[2]
            env=dict(os.environ,PATH=path+':'+os.environ['PATH'],NANDU1_CONTAINER_NAME='nandu1-local-test123')
            result=subprocess.run(['bash',str(root/'scripts/nandu1_noetic.sh'),'run','true'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--name',result.stdout.splitlines())
            self.assertIn('nandu1-local-test123',result.stdout.splitlines())

    def test_keyboard_wrapper_preserves_tty(self):
        with tempfile.TemporaryDirectory() as path:
            check_keyboard_wrapper(Path(path))

    def test_build_proxy_stays_explicit_and_host_local(self):
        with tempfile.TemporaryDirectory() as path:
            docker=Path(path)/'docker'
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            docker.chmod(0o755)
            root=Path(__file__).resolve().parents[2]
            env=dict(os.environ,PATH=path+':'+os.environ['PATH'],NANDU1_BUILD_PROXY='http://127.0.0.1:7897')
            result=subprocess.run(['bash',str(root/'scripts/nandu1_noetic.sh'),'build'],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('http_proxy=http://127.0.0.1:7897',result.stdout.splitlines())
            self.assertIn('https_proxy=http://127.0.0.1:7897',result.stdout.splitlines())

if __name__=='__main__':unittest.main()
