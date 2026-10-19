"""Sensor-only A/r2 and E/r2 references with explicit hashed training math."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import nandu1_contract as base
from nandu1_adaptive_bundle import load_adaptive_bundle


class Graph:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    def __call__(self, *values):
        arrays = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32) for x in values]
        result = self.session.run(None, dict(zip([x.name for x in self.session.get_inputs()], arrays)))
        if any(not np.isfinite(x).all() for x in result): raise ValueError('Nonfinite model output')
        tensors = tuple(torch.from_numpy(x.copy()) for x in result)
        return tensors[0] if len(tensors) == 1 else tensors


class AdaptiveRuntime:
    def __init__(self, bundle, *, initial_height, stage=None):
        self.manifest, _ = load_adaptive_bundle(bundle)
        self.bundle = Path(bundle).resolve()
        if stage is None:stage=self.manifest['task']['evaluation_stage']
        if type(stage) is not int or not 0 <= stage <= 4: raise ValueError('Invalid stage')
        self.stage = stage
        spec = importlib.util.spec_from_file_location('nandu_e_r2_reference', self.bundle / 'processing_reference.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.processor = module.StatefulProcessor()
        self.encoder, self.policy, self.context = [Graph(self.bundle / (name + '.onnx'))
                                                   for name in ('encoder', 'policy', 'context_step')]
        self.reset(initial_height=initial_height)

    def reset(self, *, initial_height):
        if not np.isfinite(initial_height): raise ValueError('Nonfinite reset height')
        self.processor.reset(height=initial_height)
        self.history = base.History()
        self.previous = torch.zeros(1, 13)
        self.last_id = None
        self.cached = None
        self.known_clipping = self.target_clipping = False

    def report_interval_clipping(self, *, action_clipped, target_clipped):
        self.known_clipping = bool(action_clipped)
        self.target_clipping = bool(target_clipped)

    @torch.inference_mode()
    def step(self, observation, raw_intent, *, step_id):
        if type(step_id) is not int or step_id < 0: raise ValueError('Invalid step ID')
        if self.last_id is not None:
            if step_id < self.last_id: raise ValueError('step_id cannot move backwards')
            if step_id == self.last_id: return copy.deepcopy(self.cached)
        obs = torch.from_numpy(base.finite(observation, 49).astype(np.float32)).reshape(1, 49)
        goal = base.finite(raw_intent, 3).astype(np.float32)
        yaw, vx = base.STAGE_LIMITS[self.stage]
        lo, hi = base.STAGE_HEIGHT_RANGES[self.stage]
        goal = torch.from_numpy(np.clip(goal, [-yaw,-vx,lo], [yaw,vx,hi]).astype(np.float32)).reshape(1,3)
        obs[:,36:] = self.previous
        saved = copy.deepcopy((self.processor, self.history))
        try:
            # r2 manager_based_rl_env computes next commands BEFORE returning
            # observations; update_runtime then observes these same commands.
            command = self.processor.command_step(goal, obs, self.stage, step_id)
            history = torch.from_numpy(self.history.append(obs[0].numpy())).reshape(1,245)
            encoded = self.encoder(history)
            if encoded.shape != (1,7): raise ValueError('Wrong encoder shape')
            if self.last_id is not None:
                self.processor.observe(obs, encoded, self.previous, self.context,
                                       known_clipping=self.known_clipping, step_id=step_id)
            dynamic = self.processor.dynamic
            actor_input = torch.cat((encoded, obs, command, dynamic), -1)
            action = self.policy(actor_input)
            if action.shape != (1,13) or not action.isfinite().all(): raise ValueError('Invalid action')
            numpy = lambda x: x.detach().numpy().reshape(-1).copy()
            result = dict(observation=numpy(obs), history=numpy(history), encoded=numpy(encoded),
                commands=numpy(command), dynamic=numpy(dynamic), actor_input=numpy(actor_input),
                action=numpy(action), raw_action=numpy(action), hidden=numpy(self.processor.hidden),
                eta=numpy(self.processor.filtered_eta), memory=numpy(self.processor.memory.normalized()),
                memory_filter=numpy(self.processor.memory.error_lpf),
                height_filter=(numpy(self.processor.height_memory.error_lpf) if hasattr(self.processor,'height_memory') else np.empty(0,np.float32)),
                height_rate=float(self.processor.commands.height_rate[0]),
                heading=float(self.processor.commands.heading[0]),
                heading_target=float(self.processor.commands.heading_target[0]))
            if any(not np.isfinite(v).all() for v in result.values()): raise ValueError('Nonfinite runtime state')
        except Exception:
            self.processor, self.history = saved
            raise
        self.previous = action.clone()
        self.last_id, self.cached = step_id, result
        return copy.deepcopy(result)

    def state_dict(self):
        state=dict(previous=self.previous, hidden=self.processor.hidden,
            eta=self.processor.filtered_eta, memory=self.processor.memory.state_dict(),
            commands=self.processor.commands.state_dict(),last_id=self.last_id)
        if hasattr(self.processor,'height_memory'):state['height_memory']=self.processor.height_memory.state_dict()
        return copy.deepcopy(state)
