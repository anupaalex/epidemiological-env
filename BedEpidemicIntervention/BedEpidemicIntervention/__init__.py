from gym.envs.registration import register

register(
    id='bed-epidemic-intervention-v0',
    entry_point='BedEpidemicIntervention.envs:BedEpidemicInterventionEnv',
)
