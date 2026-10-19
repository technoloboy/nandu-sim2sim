#!/usr/bin/env python3
"""Versioned raw IMU + control + evaluation logger; never supplies policy inputs."""
import csv
import json
import math
from pathlib import Path
import threading
from collections import deque

SCHEMA='nandu1.adaptive-telemetry/e-r2-v1'
EVENT_SCHEMA='nandu1.adaptive-state-events/v1'
TRACKING_FIELDS=['tracking_stamp','tracking_epoch']+[f'command_{i}' for i in range(3)]+[f'estimate_{i}' for i in range(7)]+[f'action_{i}' for i in range(13)]
TRACKING_FIELDS += [f'{prefix}_{i}' for prefix in ['q','dq','q_target','dq_target','torque_actual'] for i in range(13)]
TRACKING_FIELDS += [f'{prefix}_{i}' for i in range(13) for prefix in ['torque_limited','torque_saturated']]
TRACKING_FIELDS += ['relative_heading','heading_target']+[f'external_intent_{i}' for i in range(3)]
TRACKING_FIELDS += ['policy_step','policy_period','command_timed_out','target_projection_count','base_tilt']
TRACKING_FIELDS += [f'requested_{i}' for i in range(3)]+[f'dynamic_{i}' for i in range(12)]+[f'hidden_{i}' for i in range(64)]+[f'eta_{i}' for i in range(8)]+[f'actor_input_{i}' for i in range(71)]
SENSOR_FIELDS=[f'{body}_{kind}_{i}' for body in ('base','up3') for kind,n in [('quat',4),('gyro',3),('acc',3)] for i in range(n)]
FIELDS=['stamp','sim_time','epoch','sensor_seq','tracking_age','velocity_projection_events']+[f'truth_{i}' for i in range(7)]+[f'command_truth_{i}' for i in range(3)]+SENSOR_FIELDS+TRACKING_FIELDS
FIELDS += [f'raw_{kind}_{i}' for kind in ('q','dq','tau') for i in range(13)]
FIELDS += [f'{kind}_{i}' for kind in ('tcp_world','up3_origin_world','up3_com_world') for i in range(3)]


def tracking_fields(layout='adaptive'):
    if layout in ('adaptive','ablation_b','ablation_d','ablation_f'):return list(TRACKING_FIELDS)
    if layout=='adaptive_balance':
        return TRACKING_FIELDS[:129]+[f'dynamic_{i}' for i in range(11)]+[f'hidden_{i}' for i in range(64)]+[f'eta_{i}' for i in range(8)]+[f'actor_input_{i}' for i in range(70)]
    if layout=='network_e_tcp':return list(TRACKING_FIELDS[:121])
    raise ValueError('Unsupported explicit NANDU1 telemetry layout: '+str(layout))


def data_fields(layout='adaptive'):
    start=FIELDS.index('tracking_stamp')
    return FIELDS[:start]+tracking_fields(layout)+FIELDS[start+len(TRACKING_FIELDS):]


def align_packet(packet, tracking, layout='adaptive'):
    fields=tracking_fields(layout)
    if len(tracking)!=len(fields) or not all(math.isfinite(x) for x in tracking):return None
    stamp=packet['stamp_ns']/1e9
    if packet['epoch']!=int(tracking[1]) or abs(stamp-tracking[0])>=.04:return None
    row=dict(zip(fields,tracking))
    row.update(stamp=stamp,sim_time=packet['sim_time'],epoch=packet['epoch'],sensor_seq=packet['seq'],
               tracking_age=stamp-tracking[0],velocity_projection_events=packet.get('velocity_projection_events',0))
    for body in ('base','up3'):
        for kind,n in [('quat',4),('gyro',3),('acc',3)]:
            values=packet[body][kind]
            if len(values)!=n:return None
            row.update({f'{body}_{kind}_{i}':v for i,v in enumerate(values)})
    for key,n in [('truth',7),('command_truth',3)]:
        if len(packet[key])!=n:return None
        row.update({f'{key}_{i}':v for i,v in enumerate(packet[key])})
    for key in ('q','dq','tau'):
        if len(packet[key])!=13:return None
        row.update({f'raw_{key}_{i}':v for i,v in enumerate(packet[key])})
    for key in ('tcp_world','up3_origin_world','up3_com_world'):
        values=packet['point_diagnostics'][key]
        if len(values)!=3:return None
        row.update({f'{key}_{i}':v for i,v in enumerate(values)})
    return row if all(math.isfinite(v) for v in row.values()) else None


class Pairer:
    """Pair every control cycle once with nearest retained sensor timestamp."""
    def __init__(self,layout='adaptive'):
        self.layout=layout;self.fields=tracking_fields(layout)
        self.packets=deque(maxlen=512);self.pending=deque(maxlen=256)
        self.seen=deque(maxlen=1024)

    def tracking(self, values):
        if len(values)!=len(self.fields) or not all(math.isfinite(v) for v in values):return []
        key=(int(values[1]),values[0])
        if key in self.seen:return []
        self.seen.append(key);self.pending.append(list(values));return self.drain()

    def packet(self, value):
        self.packets.append(value);return self.drain()

    def drain(self):
        rows=[];waiting=deque(maxlen=256)
        for track in self.pending:
            candidates=[p for p in self.packets if p['epoch']==int(track[1])]
            if not candidates or max(p['stamp_ns']/1e9 for p in candidates)<track[0]+.005:
                waiting.append(track);continue
            nearest=min(candidates,key=lambda p:abs(p['stamp_ns']/1e9-track[0]))
            row=align_packet(nearest,track,self.layout)
            if row is not None:rows.append(row)
        self.pending=waiting;return rows


class StateEvents:
    """Deduplicate diagnostic heartbeats, not distinct controller sessions."""
    def __init__(self,schema=EVENT_SCHEMA):self.last=None;self.schema=schema

    def update(self, stamp, level, message, values):
        mode=values.get('mode','FAULT' if level>=2 else message)
        if not math.isfinite(stamp) or mode not in ('POLICY','PASSIVE','FAULT'):return None
        epoch=int(values.get('epoch',0));generation=int(values.get('generation',0))
        key=(mode,int(level),message,epoch,generation)
        if key==self.last:return None
        self.last=key
        return dict(schema=self.schema,source='/nandu1/diagnostics',stamp=stamp,
            mode=mode,level=int(level),reason=message if mode=='FAULT' else '',
            message=message,epoch=epoch,generation=generation)


def main():
    import rospy
    from std_msgs.msg import Float64MultiArray,String
    from diagnostic_msgs.msg import DiagnosticArray
    from nandu1_bundle import load_bundle
    rospy.init_node('nandu1_adaptive_telemetry')
    bundle=Path(rospy.get_param('~bundle'));manifest,_=load_bundle(bundle)
    layout=rospy.get_param('~layout','adaptive');fields=data_fields(layout);track_fields=tracking_fields(layout)
    expected={'adaptive':'nandu1_adaptive_ablation_e_r2','ablation_b':'nandu1_adaptive_ablation_b_r2','ablation_d':'nandu1_adaptive_ablation_d_r2','ablation_f':'nandu1_adaptive_ablation_f_r2','adaptive_balance':'nandu1_adaptive_balance_r2','network_e_tcp':'nandu1_network_v1_3_1_2_e_r1'}[layout]
    if manifest['task'].get('runtime_profile')!=expected:raise ValueError('Telemetry layout differs from validated runtime')
    schema=({'adaptive':SCHEMA,'adaptive_balance':'nandu1.adaptive-balance-telemetry/r2-v1','network_e_tcp':'nandu1.network-e-telemetry/v1'}.get(layout)
        or 'nandu1.adaptive-telemetry/'+layout.replace('_','-')+'-r2-v1')
    event_schema=({'adaptive':EVENT_SCHEMA,'adaptive_balance':'nandu1.adaptive-balance-state-events/v1','network_e_tcp':'nandu1.network-e-state-events/v1'}.get(layout)
        or 'nandu1.'+layout.replace('_','-')+'-state-events/v1')
    output=rospy.get_param('~output','')
    if not output:
        rospy.spin();return
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema=schema,layout=layout,task=manifest['task']['task_id'],checkpoint=manifest['checkpoint'],
        evaluation_stage=manifest['task']['evaluation_stage'],command_min=manifest['task']['command_min'],command_max=manifest['task']['command_max'],dimensions=manifest['dimensions'],
        bundle=str(bundle),fields=fields,tracking_fields=track_fields,
        alignment='one row per policy cycle; nearest buffered sensor stamp; same epoch; absolute delta <40ms; full 200Hz packets in .raw.jsonl; no truth in controller',
        units=dict(velocity='m/s',height='m',gyro='rad/s',q='rad',dq='rad/s',torque='Nm'))
    metadata['state_events']=dict(schema=event_schema,path=path.with_suffix('.events.jsonl').name,
        semantics='controller diagnostic transitions, including PASSIVE/FAULT without policy tracking; reason, epoch and reset generation retained')
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lock=threading.Lock();pairer=Pairer(layout);last_key=[(-1,-1)];events=StateEvents(event_schema)
    with path.open('w',newline='') as stream,path.with_suffix('.reset.csv').open('w',newline='') as reset_stream,path.with_suffix('.raw.jsonl').open('w') as raw_stream,path.with_suffix('.events.jsonl').open('w') as event_stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();stream.flush()
        from nandu1_reset_evidence import FIELDS as RESET_FIELDS
        reset_writer=csv.writer(reset_stream);reset_writer.writerow(RESET_FIELDS)
        def write(rows):
            for row in rows:writer.writerow(row)
            if rows:stream.flush()
        def tracking(message):
            with lock:write(pairer.tracking(message.data))
        def sensors(message):
            try:
                p=json.loads(message.data);key=(int(p['epoch']),int(p['seq']))
                with lock:
                    if key<=last_key[0]:return
                    last_key[0]=key;raw_stream.write(json.dumps(p,separators=(',',':'))+'\n');raw_stream.flush()
                    write(pairer.packet(p))
            except (KeyError,ValueError,TypeError) as exc:rospy.logwarn_throttle(1,'Adaptive logger rejects packet: %s',exc)
        def reset(message):
            if len(message.data)==len(RESET_FIELDS) and all(math.isfinite(v) for v in message.data):
                with lock:reset_writer.writerow(message.data);reset_stream.flush()
        def diagnostic(message):
            with lock:
                for status in message.status:
                    if status.name!='nandu1/policy':continue
                    event=events.update(message.header.stamp.to_sec(),status.level,status.message,
                        {v.key:v.value for v in status.values})
                    if event is not None:
                        event['received_stamp']=rospy.Time.now().to_sec()
                        event_stream.write(json.dumps(event,separators=(',',':'))+'\n');event_stream.flush()
        rospy.Subscriber(rospy.get_param('~tracking_topic','/nandu1/adaptive/tracking'),Float64MultiArray,tracking,queue_size=100,tcp_nodelay=True)
        rospy.Subscriber(rospy.get_param('~raw_topic','/nandu1/adaptive/raw_sensors'),String,sensors,queue_size=100,tcp_nodelay=True)
        rospy.Subscriber('/nandu1/reset_evidence',Float64MultiArray,reset,queue_size=10)
        rospy.Subscriber('/nandu1/diagnostics',DiagnosticArray,diagnostic,queue_size=100,tcp_nodelay=True)
        rospy.spin()


if __name__=='__main__':main()
