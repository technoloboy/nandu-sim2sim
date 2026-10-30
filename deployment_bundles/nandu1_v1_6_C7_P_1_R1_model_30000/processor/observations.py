"""Named feature assembly. Actor never receives the diagnostic filter state."""
import torch
from .state import physical_errors


def actor_input(policy, estimate, command, state):
    error = (physical_errors(policy, estimate, command) / policy.new_tensor([.25, .12, .10])).clamp(-8, 8)
    result = torch.cat((estimate, policy, error, command, state.integral, state.rate, state.slow), -1)
    if result.shape[-1] != 86: raise ValueError("Expected Actor86")
    return result.detach()


def critic_input(baseline111, state, diagnostic27, rates3, command_context7, extension=None):
    result = torch.cat((baseline111, state.integral, state.error_lp, diagnostic27, state.slow,
                        rates3, command_context7, state.stop_clock, state.previous_previous_action), -1)
    if result.shape[-1] != 186: raise ValueError("Expected common Critic186")
    return result if extension is None else torch.cat((result, extension), -1)


def branch_context(actor86):
    return torch.cat((actor86[..., 68:86], actor86[..., 61:62]), -1)


def refresh_actor_estimate(saved_actor, estimate):
    """Current encoder in replay; preserve behavior temporal states and issued reference."""
    policy, command = saved_actor[..., 7:56], saved_actor[..., 59:62]
    errors = physical_errors(policy, estimate, command)
    error = (errors / errors.new_tensor([.25, .12, .10])).clamp(-8, 8)
    return torch.cat((estimate, policy, error, saved_actor[..., 59:]), -1).detach()
