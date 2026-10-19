#!/usr/bin/env python3
"""Observe physical restart recovery while the same controller remains alive."""
import argparse
import json
from pathlib import Path
import time
import rospy
from std_msgs.msg import Float64MultiArray,UInt64

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(rospy.myargv()[1:]);rospy.init_node('nandu1_epoch_acceptance')
    epoch=[None];tracking=[None]
    rospy.Subscriber('/nandu1/reset_epoch',UInt64,lambda m:epoch.__setitem__(0,m.data),queue_size=1)
    rospy.Subscriber('/nandu1/tracking',Float64MultiArray,lambda m:tracking.__setitem__(0,m.data),queue_size=1)
    end=time.monotonic()+30
    while epoch[0] is None and time.monotonic()<end:time.sleep(.01)
    old=epoch[0];print('Waiting for physical epoch change from',old,flush=True)
    while time.monotonic()<end:
        t=tracking[0]
        if epoch[0]!=old and t and int(t[1])==epoch[0]:
            assert abs(t[2])<1e-6 and abs(t[3])<1e-6
            assert abs(t[4]-.8856453118528085)<1e-6
            report={'passed':True,'old_epoch':old,'new_epoch':epoch[0],
                    'command_reset':list(t[2:5]),'policy_resumed_without_controller_restart':True}
            args.output.write_text(json.dumps(report,indent=2)+'\n');print(report);return
        time.sleep(.005)
    raise AssertionError('Physical epoch did not resume POLICY')

if __name__=='__main__':main()
