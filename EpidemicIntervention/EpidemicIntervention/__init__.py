from gym.envs.registration import register

register(
    id='epidemic-intervention-v0',
    entry_point='EpidemicIntervention.envs:EpidemicInterventionEnv',
)
