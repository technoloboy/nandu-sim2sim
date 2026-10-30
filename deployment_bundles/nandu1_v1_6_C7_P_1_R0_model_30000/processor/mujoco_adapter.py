"""Read-only existing asset/PD adapter for independent evaluation, never training."""
from pathlib import Path
import sys
import importlib
import numpy as np
import mujoco
from .contract import TCP


class _MovedAttachmentPayload:
    """Rebuild from immutable link-frame full inertia; keep the TCP separate."""
    def __init__(self, model, baseline):
        self.model, self.baseline = model, baseline
        self.nominal = baseline.nominal
        self.parameters = baseline.parameters.copy()

    def apply(self, data, up3_mass=.67148, rear_mass=0., rear_x=-.275):
        from .c7_p1 import ATTACHMENT_COM
        self.baseline.apply(data, up3_mass, rear_mass, rear_x)
        mass, com, inertia = self.nominal["up3"]
        if up3_mass > mass and up3_mass != .67148:
            extra = up3_mass - mass
            attachment = np.asarray(ATTACHMENT_COM)
            center = (mass * com + extra * attachment) / up3_mass
            tensor = inertia + np.diag(extra * np.array([
                .06**2 + .05**2, .05**2 + .05**2, .05**2 + .06**2]) / 12)
            for body_mass, point in ((mass, com), (extra, attachment)):
                offset = point - center
                tensor += body_mass * (offset @ offset * np.eye(3) - np.outer(offset, offset))
            diagonal, axes = np.linalg.eigh(tensor)
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, axes.reshape(9))
            bid = self.baseline.ids["up3"]
            self.model.body_mass[bid] = up3_mass
            self.model.body_ipos[bid] = center
            self.model.body_inertia[bid], self.model.body_iquat[bid] = diagonal, quat
            mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
            mujoco.mj_forward(self.model, data)
        self.parameters = self.baseline.parameters.copy()


class Physics:
    def __init__(self, source_root, xml, up3_mass=.67148, rear_mass=0., rear_x=-.275, *, spec=None):
        self.spec = spec
        self.revised = spec is not None and spec.physics_revision in (1, 2)
        self.r4 = spec is not None and bool(spec.revision_profile)
        source_root = Path(source_root).resolve()
        sys.path.insert(0, str(source_root))
        self.c = importlib.import_module("nandu1_contract")
        payload_module = importlib.import_module("nandu1_v131")
        if Path(self.c.__file__).resolve().parent != source_root:
            raise ValueError("Ambiguous imported MuJoCo contract")
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml).resolve()))
        self.data = mujoco.MjData(self.model)
        self.ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.c.JOINTS]
        self.actuators = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.c.JOINTS]
        if min(self.ids+self.actuators) < 0: raise ValueError("Missing required named joint/actuator")
        self.qadr, self.vadr = self.model.jnt_qposadr[self.ids], self.model.jnt_dofadr[self.ids]
        self.reset_q = self.c.DEFAULT_Q.copy()
        self.reset_root_z = .65337
        if self.revised:
            import torch
            from .c7_p1 import ARM3_UPPER, KNEE_UPPER, RESET_ROOT_Z, reset_joint_positions
            self.reset_q = reset_joint_positions(torch.from_numpy(self.reset_q), self.c.JOINTS).numpy()
            self.reset_root_z = RESET_ROOT_Z
            for name, upper in (("arm3", ARM3_UPPER), ("knee_L_Joint", KNEE_UPPER),
                                ("knee_R_Joint", KNEE_UPPER)):
                self.model.jnt_range[self.ids[self.c.JOINTS.index(name)], 1] = upper
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.up3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "up3")
        self.sensors = {}
        for prefix in ("base_imu", "up3_imu"):
            for suffix, width in (("quat", 4), ("gyro", 3)):
                name = prefix+"_"+suffix
                i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if i < 0 or self.model.sensor_dim[i] != width: raise ValueError(name)
                self.sensors[name] = slice(self.model.sensor_adr[i], self.model.sensor_adr[i]+width)
        self.substeps = round(.02/self.model.opt.timestep)
        if abs(self.substeps*self.model.opt.timestep-.02) > 1e-9: raise ValueError("Nonintegral 50 Hz period")
        # Use existing payload composition, no FK, no asset regeneration.
        self.payload = payload_module.FixedPayload(self.model)
        if self.revised:
            self.payload = _MovedAttachmentPayload(self.model, self.payload)
        self.payload.apply(self.data, up3_mass, rear_mass, rear_x)
        if self.r4:
            from .c7_p1_r4_wheels import WheelPacketState
            self.wheel_state=WheelPacketState(1)
            self.wheel_indices=np.array([self.c.JOINTS.index(n) for n in ('wheel_L_Joint','wheel_R_Joint')])
            for name in ('knee_L_Joint','knee_R_Joint'):
                self.model.jnt_range[self.ids[self.c.JOINTS.index(name)],0]=.2
            for j in self.wheel_indices:
                self.model.jnt_actfrcrange[self.ids[j]]=[-22,22]
                self.model.actuator_ctrlrange[self.actuators[j]]=[-22,22]
                self.model.actuator_forcerange[self.actuators[j]]=[-22,22]
            self.pd_substeps=round(.005/self.model.opt.timestep)
            if abs(self.pd_substeps*self.model.opt.timestep-.005)>1e-9:
                raise ValueError('R4 requires an integral 200Hz PD clock')
            self.friction_approximation=('MuJoCo native frictionloss uses 0.6 Nm within 0.001 rad/s and '
                '0.3 Nm otherwise; solver approximation, NOT an identical PhysX static/dynamic friction law')
        self.reset()

    def reset(self):
        if self.revised:
            params = self.payload.parameters
            self.payload.apply(self.data, params["up3_total_mass"], params["rear_mass"], params["rear_x"])
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = [0., 0., self.reset_root_z, 1., 0., 0., 0.]
        self.data.qpos[self.qadr] = self.reset_q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)  # native simulator initialization, no separate FK solver
        self.velocity_projection_events = self.torque_saturation_events = 0
        self.position_target = self.c.DEFAULT_Q.copy()
        self.velocity_target = np.zeros(13)
        if self.r4:
            import torch
            q=torch.as_tensor(self.data.qpos[self.qadr[self.wheel_indices]],dtype=torch.float32)[None]
            self.wheel_state.reset([0],q,delay_steps=torch.zeros(1,dtype=torch.long))
            self.executed_action=np.zeros(13)
            self.motor_records=[]

    def measurement(self, previous_action):
        s = self.data.sensordata
        return self.c.observe(s[self.sensors["base_imu_quat"]], s[self.sensors["base_imu_gyro"]],
                              s[self.sensors["up3_imu_quat"]], s[self.sensors["up3_imu_gyro"]],
                              self.data.qpos[self.qadr], self.data.qvel[self.vadr], previous_action)

    def control(self, action, packet_context=None):
        if self.r4:return self._control_r4(action,packet_context)
        q, dq = self.c.decode(action)
        # Preserve the existing E MuJoCo bridge: physical-limit projection, PD each 1 ms.
        q = self.c.clip_position_targets(q, self.model.jnt_range[self.ids, 0], self.model.jnt_range[self.ids, 1])
        self.position_target, self.velocity_target = q.copy(), dq.copy()
        for _ in range(self.substeps):
            torque = self.c.KP*(q-self.data.qpos[self.qadr])+self.c.KD*(dq-self.data.qvel[self.vadr])
            self.torque_saturation_events += int(np.count_nonzero(np.abs(torque) > self.c.EFFORT))
            self.data.ctrl[self.actuators] = np.clip(torque, -self.c.EFFORT, self.c.EFFORT)
            mujoco.mj_step(self.model, self.data)
            speed = self.data.qvel[self.vadr]
            excessive = np.abs(speed) > self.c.VELOCITY
            self.velocity_projection_events += int(excessive.sum())
            if excessive.any():
                self.data.qvel[self.vadr] = np.clip(speed, -self.c.VELOCITY, self.c.VELOCITY)
                mujoco.mj_forward(self.model, self.data)

    def _control_r4(self,action,packet_context):
        import torch
        from .c7_p1_r4_state import motor_packet_record
        q,_=self.c.decode(action)
        self.position_target=q.copy();self.velocity_target=np.zeros(13)
        wi=self.wheel_indices
        def tensor(x):return torch.as_tensor(x.copy(),dtype=torch.float32)[None]
        self.wheel_state.capture(tensor(self.data.qpos[self.qadr[wi]]),tensor(np.asarray(action)[-2:]),packet_context)
        limit=self.c.EFFORT.copy();limit[wi]=22.
        self.motor_records=[]
        for tick in range(self.substeps):
            dq=self.data.qvel[self.vadr]
            if tick%self.pd_substeps==0:
                torque=self.c.KP*(q-self.data.qpos[self.qadr])-self.c.KD*dq
                if packet_context is not None:
                    positions=self.c.POSITION_IDS
                    requested=tensor(torque[positions])
                    emitted=tensor(np.clip(torque,-limit,limit)[positions])
                    self.motor_records.append(motor_packet_record(packet_context,list(range(11)),
                        tensor(self.data.qpos[self.qadr[positions]]),requested,emitted))
                _,target,requested,emitted=self.wheel_state.step(
                    tensor(self.data.qpos[self.qadr[wi]]),tensor(dq[wi]))
                torque[wi]=emitted.numpy()[0]
                self.position_target[wi]=target.numpy()[0]
                self.torque_saturation_events+=int((np.abs(torque)>limit).sum())
                self.data.ctrl[self.actuators]=np.clip(torque,-limit,limit)
            # Use one native solver friction term, never add another torque model.
            self.model.dof_frictionloss[self.vadr[wi]]=np.where(np.abs(dq[wi])<.001,.6,.3)
            mujoco.mj_step(self.model,self.data)
            speed=self.data.qvel[self.vadr]
            excessive=np.abs(speed)>self.c.VELOCITY
            self.velocity_projection_events+=int(excessive.sum())
            if excessive.any():
                self.data.qvel[self.vadr]=np.clip(speed,-self.c.VELOCITY,self.c.VELOCITY)
                mujoco.mj_forward(self.model,self.data)
        self.executed_action=np.asarray(action).copy()
        self.executed_action[-2:]=self.wheel_state.mean_executed().numpy()[0]
        self.motor_records+=self.wheel_state.saturation_records

    def truth(self):
        d = self.data
        up_v, base_v = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.up3, up_v, 0)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY, self.base, base_v, 0)
        point = d.xpos[self.up3] + d.xmat[self.up3].reshape(3, 3) @ np.asarray(TCP)
        velocity = up_v[3:] + np.cross(up_v[:3], point-d.xipos[self.up3])
        rotation = d.xmat[self.base].reshape(3, 3)
        return dict(tcp=point.copy(), tcp_velocity_api=velocity.copy(), tcp_velocity_body=rotation.T @ velocity,
                    base_velocity_body=rotation.T @ base_v[3:],
                    base_origin=d.xpos[self.base].copy(), base_com=d.xipos[self.base].copy(),
                    up3_origin=d.xpos[self.up3].copy(), up3_com=d.xipos[self.up3].copy(),
                    yaw_rate=base_v[2], base_quaternion=d.xquat[self.base].copy(),
                    up3_quaternion=d.xquat[self.up3].copy(),
                    position_target=self.position_target.copy(), velocity_target=self.velocity_target.copy(),
                    joint_position=d.qpos[self.qadr].copy(), joint_velocity=d.qvel[self.vadr].copy(),
                    torque=d.qfrc_actuator[self.vadr].copy())
