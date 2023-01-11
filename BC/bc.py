import gym
from gym.spaces import Box, Dict
from imitation.algorithms import bc
import numpy as np
# Define env and action spaces

env_space = Dict(
                {
                    "tool_pos": Box(low=-1., high=1.0, shape=(3,), dtype=np.float32),
                    "tool_quat": Box(low=-1., high=1.0, shape=(4,), dtype=np.float32),
                    "tool_theta":Box(low=-1., high=1.0, shape=(1,), dtype=np.float32),
                    'cube0_pos': Box(low=-1., high=1.0, shape=(3,), dtype=np.float32),
                    'cube0_quat': Box(low=-1., high=1.0, shape=(4,), dtype=np.float32),
                    'cube1_pos':Box(low=-1., high=1.0, shape=(3,), dtype=np.float32),
                    'cube1_quat': Box(low=-1., high=1.0, shape=(4,), dtype=np.float32),
                    'goal0_pos': Box(low=-1., high=1.0, shape=(3,), dtype=np.float32),
                    'goal1_pos': Box(low=-1., high=1.0, shape=(3,), dtype=np.float32),
                }
            )

action_space = Dict(
    {
        "xy_linear_velocity" : Box(low=-1., high=1.0, shape=(3,), dtype=np.float32)
    }

)

transitions = np.load('transitions.npy', allow_pickle=True)
rng = np.random.default_rng()

bc_trainer = bc.BC(
    observation_space=env_space,
    action_space=action_space,
    demonstrations=transitions,
    rng=rng,
)