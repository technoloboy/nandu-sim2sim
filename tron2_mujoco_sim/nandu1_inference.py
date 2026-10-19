"""Sensor-only command/Actor adapter used by offline conformance probes.

The ROS controller is the deployment path; this independent implementation
exists to compare numerical contracts, not to bypass the SDK transport.
"""
import numpy as np
import nandu1_contract as c

class CommandPipeline:
    def __init__(self,task):
        self.task=task;self.profile=c.task_profile(task['task_id']);self.stage=task.get('evaluation_stage',4)
        self.height=task['initial_up3_height'];self.reset()

    def reset(self):
        self.applied=np.array([0.,0.,self.height]);self.external=self.applied.copy()
        self.processor=None
        if self.profile['task_id'] in (c.V14_TASK,c.V141_TASK):
            if self.profile['task_id']==c.V141_TASK:from nandu1_v141 import CommandProcessor
            else:from nandu1_v14 import CommandProcessor
            self.processor=CommandProcessor(self.stage,self.height)
        elif self.profile['task_id'] in (c.V131_TASK,c.V1312_TASK):
            from nandu1_v131 import CommandProcessor
            self.processor=CommandProcessor()

    def step(self,desired,observation,gyro,dt=.02):
        desired=np.clip(c.finite(desired,3),self.task['command_min'],self.task['command_max'])
        obs=c.finite(observation,49);gyro=c.finite(gyro,3)
        if not np.isfinite(dt) or not 1e-6<=dt<=.1:raise ValueError('Invalid policy dt')
        if self.profile['command']==6:
            features=self.processor.step(desired,obs[3:6],gyro,dt)
        else:
            rate=np.array(self.task.get('command_rate',[2,.8,.5]))
            self.external+=np.clip(desired-self.external,-rate*dt,rate*dt)
            features=self.processor.step(dt,obs[3:6],gyro,self.external,self.stage) if self.processor is not None else self.external.copy()
        self.applied=np.asarray(features[:3]).copy()
        return np.asarray(features,dtype=np.float32)

    def actor_input(self,estimated,observation,features):
        if self.profile['command']==6:
            from nandu1_v14 import actor_input
            return actor_input(estimated,observation,features)
        return np.r_[c.finite(estimated,7),c.finite(observation,49),c.finite(features,3)].astype(np.float32)
