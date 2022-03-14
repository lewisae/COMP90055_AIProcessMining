#COMP90055: Test code to load different Open AI Gym environments

import gym

from gym import envs
print(envs.registry.all())

env = gym.make('Ant-v2')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()
