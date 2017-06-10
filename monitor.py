import gym
import universe
from gym import wrappers
env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)
# env = wrappers.Monitor(env, 'flashgames.DuskDrive-v0', force=True)

for i_episode in range(20):
    observation = env.reset()
    action_n = [[('KeyEvent', 'ArrowUp', False)] for ob in observation]  # your agent here
    for t in range(20000):
        print(t)
        env.render()
        action = env.action_space.sample()
        print(type(action))
        observation, reward, done, info = env.step(action_n)
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break
