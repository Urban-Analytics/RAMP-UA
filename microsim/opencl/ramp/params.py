import numpy as np


class Params:
    """Convenience class for setting simulator parameters. Also holds the default values."""

    def __init__(self):
        """Create a simulator with the default parameters."""
        self.symptomatic_multiplier = 0.5
        self.proportion_asymptomatic = 0.5
        self.exposed_scale = 2.82
        self.exposed_shape = 3.93
        self.presymptomatic_scale = 2.45
        self.presymptomatic_shape = 7.12
        self.infection_scale = 3.0
        self.infection_location = 16.0
        self.lockdown_multiplier = 1.0
        self.place_hazard_multipliers = np.array(
            [0.0165, 0.0165, 0.0165, 0.0165, 0.00], dtype=np.float32)
        self.recovery_probs = np.array(
            [0.9999839, 0.9999305, 0.999691, 0.999156,
             0.99839, 0.99405, 0.9807, 0.9572, 0.922],
            dtype=np.float32)

    def asarray(self):
        """Pack the parameters into a flat array for uploading."""
        return np.concatenate([
            np.array(
                [
                    self.symptomatic_multiplier,
                    self.proportion_asymptomatic,
                    self.exposed_scale,
                    self.exposed_shape,
                    self.presymptomatic_scale,
                    self.presymptomatic_shape,
                    self.infection_scale,
                    self.infection_location,
                    self.lockdown_multiplier,
                ],
                dtype=np.float32,
            ),
            self.place_hazard_multipliers,
            self.recovery_probs,
        ])

    @classmethod
    def fromarray(cls, array):
        p = cls()
        p.symptomatic_multiplier = array[0]
        p.proportion_asymptomatic = array[1]
        p.exposed_scale = array[2]
        p.exposed_shape = array[3]
        p.presymptomatic_scale = array[4]
        p.presymptomatic_shape = array[5]
        p.infection_scale = array[6]
        p.infection_location = array[7]
        p.lockdown_multiplier = array[8]
        p.place_hazard_multipliers = array[9:14]
        p.recovery_probs = array[14:23]
        return p

    def set_lockdown_multiplier(self, lockdown_multipliers, timestep):
        """Update the lockdown multiplier based on the current time."""
        self.lockdown_multiplier = lockdown_multipliers[np.minimum(lockdown_multipliers.shape[0]-1, timestep)]

    def num_bytes(self):
        return 4*self.asarray().size
