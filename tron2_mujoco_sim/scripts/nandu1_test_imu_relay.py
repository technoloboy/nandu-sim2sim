#!/usr/bin/env python3
"""Acceptance-only localhost relay: 15013 -> 15012, control on 15015.

Modes drop/stale/duplicate/nan perturb ONLY the auxiliary IMU transport.
Never run with a real robot, and never use this relay as the production bridge.
"""
import argparse
import json
import select
import socket
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--duration',type=float,default=180)
    args=parser.parse_args();incoming=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    incoming.bind(('127.0.0.1',15013));control=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    control.bind(('127.0.0.1',15015));out=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    mode='normal';start=time.monotonic();last_packet={}
    try:
        while time.monotonic()-start<args.duration:
            readable,_,_=select.select([incoming,control],[],[],.1)
            for sock in readable:
                data,address=sock.recvfrom(65535)
                if address[0]!='127.0.0.1':continue
                if sock is control:
                    desired=data.decode()
                    if desired=='status':
                        control.sendto(json.dumps({key:last_packet.get(key) for key in ['stamp_ns','applied_kp_max','applied_dq_max']}).encode(),address)
                    if desired in ['normal','drop','stale','duplicate','nan']:mode=desired
                    continue
                last_packet=json.loads(data)
                if mode=='drop':continue
                if mode!='normal':
                    packet=json.loads(data)
                    if mode=='stale':packet['stamp_ns']-=200_000_000
                    if mode=='duplicate':packet['seq']=0
                    if mode=='nan':packet['up3']['gyro'][0]=float('nan')
                    data=json.dumps(packet).encode()
                out.sendto(data,('127.0.0.1',15012))
    finally:
        incoming.close();control.close();out.close()

if __name__=='__main__':main()
