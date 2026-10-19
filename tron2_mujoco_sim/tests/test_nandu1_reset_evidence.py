import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def row(stamp,generation,**changes):
    result=dict(stamp=stamp,epoch=1,generation=generation,last_action_inf=0.,history_repeat_error=0.,
                heading=0.,target=0.,cue_inf=0.,command_yaw=0.,command_vx=0.,command_height=.9,initial_height=.9)
    result.update(changes);return result

def test_reset_needs_new_generation_not_just_post_reset_telemetry():
    from nandu1_reset_evidence import verify_reset_evidence
    with pytest.raises(ValueError,match='generation'):verify_reset_evidence([row(1,3),row(2.1,3)],2)
    assert verify_reset_evidence([row(1,3),row(2.1,4)],2)['generation']==4

@pytest.mark.parametrize('field',['last_action_inf','history_repeat_error','heading','target','cue_inf','command_yaw','command_vx','command_height'])
def test_ack_does_not_hide_uncleared_policy_state(field):
    from nandu1_reset_evidence import verify_reset_evidence
    bad=row(2.1,4,**{field:.2})
    with pytest.raises(ValueError,match='state'):verify_reset_evidence([row(1,3),bad],2)
