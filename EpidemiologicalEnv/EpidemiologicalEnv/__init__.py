from gym.envs.registration import register

register(
    id='epidemiological-env-v0',
    entry_point='EpidemiologicalEnv.envs:EpidemiologicalEnv',
)
