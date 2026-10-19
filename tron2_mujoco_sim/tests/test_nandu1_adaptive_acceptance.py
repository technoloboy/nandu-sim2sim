import importlib.util
from pathlib import Path
import socket
import pytest


def load():
    path=Path(__file__).resolve().parents[1]/'scripts/check_nandu1_adaptive_ros.py'
    assert path.is_file(), 'Fresh-output localhost acceptance runner missing'
    spec=importlib.util.spec_from_file_location('adaptive_acceptance',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_acceptance_refuses_existing_output_and_remote_ip(tmp_path):
    m=load()
    with pytest.raises(ValueError):m.preflight(tmp_path,'127.0.0.1')
    with pytest.raises(ValueError):m.preflight(tmp_path/'new','192.168.1.4')


def test_acceptance_refuses_an_occupied_auxiliary_port(tmp_path):
    m=load()
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with pytest.raises(RuntimeError):m.preflight(tmp_path/'fresh','127.0.0.1',ports=[(socket.SOCK_DGRAM,port)])
