"""C7 Play physics settings, separate from the frozen historical collector."""
from nandu1_c7_sim2sim import C7Simulation as HistoricalC7Simulation
from nandu1_c7_sim2sim import VERSIONS, bundle_for, validate_bundle


PHYSICS_PROFILES = {
    "robojudo": (.005, "mjINT_EULER"),
    "legacy": (.001, "mjINT_IMPLICITFAST"),
}


class C7Simulation(HistoricalC7Simulation):
    """Same policy/PD logic, with the explicitly selected live physics model."""

    def __init__(self, *args, physics_profile="robojudo", **kwargs):
        if physics_profile not in PHYSICS_PROFILES:
            raise ValueError("Unknown C7 physics profile")
        self.physics_profile = physics_profile
        super().__init__(*args, **kwargs)
        # Construction above only resets/forwards; no physics step has run.
        # Apply to the instance so original bundle files/hashes stay immutable.
        timestep, integrator = PHYSICS_PROFILES[physics_profile]
        self.physics.model.opt.timestep = timestep
        self.physics.model.opt.integrator = getattr(self.mujoco.mjtIntegrator, integrator)
        self.physics.substeps = round(.02 / timestep)
        self.physics.pd_substeps = round(.005 / timestep)
        self.reset()

    def physics_metadata(self):
        model = self.physics.model
        timestep = float(model.opt.timestep)
        return dict(profile=self.physics_profile,
                    clocks_s=dict(policy=timestep * self.physics.substeps,
                                  pd=timestep * self.physics.pd_substeps, physics=timestep),
                    integrator=self.mujoco.mjtIntegrator(model.opt.integrator).name,
                    solver=self.mujoco.mjtSolver(model.opt.solver).name,
                    solver_iterations=int(model.opt.iterations),
                    physics_steps_per_pd=int(self.physics.pd_substeps),
                    physics_steps_per_policy=int(self.physics.substeps),
                    mujoco_version=self.mujoco.__version__,
                    bundle_declared_clocks_s=self.metadata.get("simulation_clocks_s"))
