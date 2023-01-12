import gym
from gym.spaces import Box, Dict
from imitation.algorithms import bc
from imitation.data.types import Trajectory
import numpy as np
# Define env and action spaces
size_env = 3+4+1+3+4+3+4+3+3
env_space = Box(low=-1., high=1.0, shape=(size_env,), dtype=np.float32)
                    

action_space = Box(low=-1., high=1.0, shape=(3,), dtype=np.float32)
 

# Construct transition 
transitions = []

folder = 'trajs/'
n_episodes = 10
for i in range(n_episodes):
    filename = 'traj'+str(i)+'.npy'
    traj = np.load(folder+filename, allow_pickle=True)
    n_step,d = traj.shape
    obs = []
    acts = []
    for x in traj:
        
        acts.append(x[1]['xy_linear_velocity'])

        env = x[0]
        field_env = ['tool_pos', 'tool_quat', 'tool_theta', 'cube0_pos', 'cube0_quat', 'cube1_pos', 'cube1_quat', 'goal0_pos', 'goal1_pos']
        obs_=[]
        for f in field_env:
            if f == 'tool_theta':
                obs_.append(env[f])
                continue
            for y in env[f]:
                obs_.append(y)
        obs.append(np.array(obs_))
    obs.append(obs[-1])
    transitions.append(Trajectory(obs=obs, acts=acts, infos=np.zeros(len(acts)), terminal=True))

    


rng = np.random.default_rng()

bc_trainer = bc.BC(
    observation_space=env_space,
    action_space=action_space,
    demonstrations=transitions,
    rng=rng,
)