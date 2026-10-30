"""Revision-aware task profiles; P2 shares P1 hardware, not task identity."""


def physical_profile_for(spec):
    """Return a fresh profile without rewriting historical P1 metadata."""
    if spec.revision_profile:
        from .c7_p1_r4_actions import physical_profile
        return physical_profile(spec)
    if spec.physics_revision == 1:
        from .c7_p1 import physical_profile
    elif spec.physics_revision == 2:
        from .c7_p2 import physical_profile
    else:
        raise ValueError("No physical profile for this task revision")
    return physical_profile()
