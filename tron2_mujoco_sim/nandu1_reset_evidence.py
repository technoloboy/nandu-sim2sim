"""Evidence gate for an actual controller reset, not merely continuing telemetry."""
import math

FIELDS=('stamp','epoch','generation','last_action_inf','history_repeat_error','heading','target','cue_inf',
        'command_yaw','command_vx','command_height','initial_height')

def verify_reset_evidence(rows,reset_wall):
    if not math.isfinite(reset_wall):raise ValueError('Non-finite reset timestamp')
    before=[r for r in rows if r['stamp']<reset_wall]
    after=[r for r in rows if reset_wall<=r['stamp']<reset_wall+.5]
    if not before or not after or max(r['generation'] for r in after)<=max(r['generation'] for r in before):
        raise ValueError('Reset generation did not advance (RESET missing or ignored)')
    old=max(r['generation'] for r in before)
    evidence=min((r for r in after if r['generation']>old),key=lambda r:r['stamp'])
    if not all(math.isfinite(float(evidence[name])) for name in FIELDS):raise ValueError('Non-finite reset state')
    zeros=('last_action_inf','history_repeat_error','heading','target','cue_inf','command_yaw','command_vx')
    if any(abs(evidence[key])>1e-7 for key in zeros) or abs(evidence['command_height']-evidence['initial_height'])>1e-6:
        raise ValueError('Controller reset state was not cleared')
    return evidence
