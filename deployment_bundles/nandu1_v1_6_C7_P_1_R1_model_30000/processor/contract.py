"""Versioned interfaces; task identity is not inferred from checkpoint tensor shapes."""
from dataclasses import dataclass
import re


FACTORS = {
    "C0": (False, False, False), "C1": (True, False, False),
    "C2": (False, True, False), "C3": (False, False, True),
    "C4": (True, True, False), "C5": (True, False, True),
    "C6": (False, True, True), "C7": (True, True, True),
    "L": (True, True, True), "R": (True, False, True),
}
PREFIX = "Isaac-NANDU1-Blind-Flat-"
POLICY_DIM, HISTORY_DIM, ENCODER_DIM, ACTION_DIM = 49, 245, 7, 13
TCP = (-0.066473051764, 0.161623719479, -0.000621484902)
# Actor: [estimated state7, policy49, instantaneous error3, issued command3,
#         integral3, issued derivative3, slow measured configuration18].
ACTOR_FIELDS = dict(estimate=slice(0, 7), policy=slice(7, 56),
                    error=slice(56, 59), command=slice(59, 62),
                    integral=slice(62, 65), rate=slice(65, 68),
                    slow=slice(68, 86))


@dataclass(frozen=True)
class DRLTaskSpec:
    combination: str
    learner: str
    physics_revision: int = 0
    numerical_revision: int | None = None
    revision_profile: str | None = None

    def __post_init__(self):
        if self.combination not in FACTORS or self.learner not in ("P", "Q"):
            raise ValueError(f"Unsupported online experiment: {self.combination}-{self.learner}")
        if (type(self.physics_revision) is not int or self.physics_revision not in (0, 1, 2)
                or (self.physics_revision and (self.combination, self.learner) != ("C7", "P"))):
            raise ValueError("Only C7-P supports physical/stance revisions 1 and 2")
        if self.revision_profile is not None and (
                self.revision_profile not in ('R0', 'R1', 'R2') or
                (self.combination, self.learner, self.physics_revision) != ('C7', 'P', 1)):
            raise ValueError('Revision profiles are independent C7-P-1 R0/R1/R2 tasks')
        if self.numerical_revision is None:
            object.__setattr__(self, "numerical_revision", 4 if self.revision_profile else 3 if self.physics_revision else 2)
        if (type(self.numerical_revision) is not int or self.numerical_revision not in (2, 3, 4)
                or (self.numerical_revision == 3 and not self.physics_revision)
                or (self.physics_revision == 2 and self.numerical_revision != 3)
                or ((self.numerical_revision == 4) != bool(self.revision_profile))):
            raise ValueError("C7-P-1 supports r2/r3; C7-P-2 supports only numerical revision 3")

    @classmethod
    def from_task(cls, name):
        match = re.fullmatch(re.escape(PREFIX) + r"(?:Play-)?v1-6-(C[0-7]|L|R)-(P|Q)(-[12])?(?:-(R[012]))?", name)
        if not match:
            raise ValueError(f"Not a supported v1.6 task: {name}")
        combination, learner, revision, profile = match.groups()
        return cls(combination, learner, int(revision[1:]) if revision else 0, revision_profile=profile)

    @classmethod
    def from_metadata(cls, data):
        """Resolve legacy Play explicitly; new training still uses from_task."""
        current = cls.from_task(data.get("task_name", ""))
        candidates = [current]
        if current.physics_revision == 1 and not current.revision_profile:
            candidates.append(cls(current.combination, current.learner, 1, 2))
        for candidate in candidates:
            if data.get("checkpoint_schema") == candidate.schema:
                candidate.validate_metadata(data)
                return candidate
        raise ValueError("Checkpoint task/schema mismatch; unknown numerical revision")

    @property
    def task(self):
        suffix = f"-{self.physics_revision}" if self.physics_revision else ""
        if self.revision_profile: suffix += '-' + self.revision_profile
        return f"{PREFIX}v1-6-{self.combination}-{self.learner}{suffix}"

    @property
    def play_task(self):
        return self.task.replace("Flat-", "Flat-Play-", 1)

    @property
    def schema(self):
        suffix = f"_{self.physics_revision}" if self.physics_revision else ""
        if self.revision_profile: suffix += '_' + self.revision_profile
        return f"nandu_drl_v1_6_{self.combination}_{self.learner}{suffix}_online_r{self.numerical_revision}"

    @property
    def log_family(self):
        suffix = f"_{self.physics_revision}" if self.physics_revision else ""
        if self.revision_profile: suffix += '_' + self.revision_profile
        numerics = f"_stable_r{self.numerical_revision}" if self.stable_numerics else ""
        return f"nandu1_drl_{self.combination}_{self.learner}{suffix}{numerics}"

    @property
    def stable_numerics(self): return self.physics_revision in (1, 2) and self.numerical_revision in (3, 4)

    @property
    def history_frames(self): return 10 if self.revision_profile else 5
    @property
    def history_dim(self): return self.history_frames * POLICY_DIM

    @property
    def use_trim(self): return FACTORS[self.combination][0]
    @property
    def use_integral(self): return FACTORS[self.combination][1]
    @property
    def use_frequency(self): return FACTORS[self.combination][2]
    @property
    def recurrent(self): return self.combination == "R"
    @property
    def descent(self): return self.combination == "L"
    @property
    def actor_dim(self): return 86
    @property
    def state_dim(self): return 186 + (4 if self.revision_profile else 32 if self.recurrent else 5 if self.descent else 0)
    @property
    def critic_input_dim(self): return self.state_dim + (13 if self.learner == "Q" else 0)
    @property
    def rollout_steps(self): return 72 if self.recurrent else 24

    def validate_metadata(self, data):
        if data.get("checkpoint_schema") != self.schema or data.get("task_name") != self.task:
            raise ValueError("Checkpoint task/schema mismatch; old weights are not ordinary resumes.")


def matrix():
    return [DRLTaskSpec(c, a) for c in FACTORS for a in ("P", "Q")]


def registered_specs():
    """Physical children do not change the original factorial experiment matrix."""
    return [*matrix(), DRLTaskSpec("C7", "P", physics_revision=1),
            DRLTaskSpec("C7", "P", physics_revision=2),
            *(DRLTaskSpec('C7', 'P', 1, revision_profile=p) for p in ('R0', 'R1', 'R2'))]
