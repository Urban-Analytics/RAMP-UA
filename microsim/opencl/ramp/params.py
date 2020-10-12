import numpy as np
from collections import namedtuple

LocationHazardMultipliers = namedtuple(
    "LocationHazardMultipliers",
    [
        "retail",
        "primary_school",
        "secondary_school",
        "home",
        "work"
    ]
)

IndividualHazardMultipliers = namedtuple(
    "IndividualHazardMultipliers",
    [
        "presymptomatic",
        "asymptomatic",
        "symptomatic"
    ]
)


class Params:
    """Convenience class for setting simulator parameters. Also holds the default values."""

    def __init__(self,
                 location_hazard_multipliers=LocationHazardMultipliers(
                        retail=0.0165,
                        primary_school=0.0165,
                        secondary_school=0.0165,
                        home=0.0165,
                        work=0.0
                    ),
                 individual_hazard_multipliers=IndividualHazardMultipliers(
                        presymptomatic=1.0,
                        asymptomatic=0.75,
                        symptomatic=1.0
                    ),
                 proportion_asymptomatic=0.4
                 ):
        """Create a simulator with the default parameters."""
        self.symptomatic_multiplier = 0.5
        self.proportion_asymptomatic = proportion_asymptomatic
        self.exposed_scale = 2.82
        self.exposed_shape = 3.93
        self.presymptomatic_scale = 2.45
        self.presymptomatic_shape = 7.12
        self.infection_scale = 3.0
        self.infection_location = 16.0
        self.lockdown_multiplier = 1.0
        self.place_hazard_multipliers = np.array([location_hazard_multipliers.retail,
                                                  location_hazard_multipliers.primary_school,
                                                  location_hazard_multipliers.secondary_school,
                                                  location_hazard_multipliers.home,
                                                  location_hazard_multipliers.work], dtype=np.float32)

        self.individual_hazard_multipliers = np.array([individual_hazard_multipliers.presymptomatic,
                                                       individual_hazard_multipliers.asymptomatic,
                                                       individual_hazard_multipliers.symptomatic], dtype=np.float32)

        self.recovery_probs = np.array([0.9999839, 0.9999305, 0.999691, 0.999156,
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
            self.individual_hazard_multipliers,
            self.recovery_probs,
        ])

    @classmethod
    def fromarray(cls, params_array):
        location_hazard_multipliers = LocationHazardMultipliers(
            retail=params_array[9],
            primary_school=params_array[10],
            secondary_school=params_array[11],
            home=params_array[12],
            work=params_array[13]
        )
        individual_hazard_multipliers = IndividualHazardMultipliers(
            presymptomatic=params_array[14],
            asymptomatic=params_array[15],
            symptomatic=params_array[16]
        )
        p = cls(location_hazard_multipliers, individual_hazard_multipliers)
        p.symptomatic_multiplier = params_array[0]
        p.proportion_asymptomatic = params_array[1]
        p.exposed_scale = params_array[2]
        p.exposed_shape = params_array[3]
        p.presymptomatic_scale = params_array[4]
        p.presymptomatic_shape = params_array[5]
        p.infection_scale = params_array[6]
        p.infection_location = params_array[7]
        p.lockdown_multiplier = params_array[8]
        p.recovery_probs = params_array[17:26]
        return p

    def set_lockdown_multiplier(self, lockdown_multipliers, timestep):
        """Update the lockdown multiplier based on the current time."""
        self.lockdown_multiplier = lockdown_multipliers[np.minimum(lockdown_multipliers.shape[0] - 1, timestep)]

    def num_bytes(self):
        return 4 * self.asarray().size
