import sys

import gym
import universe  # register the universe environments
import getch # to take continuous input with getch
import agent

NUM_KEY_SAMPLE = 10
TICK = 0
env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)
observation_n = env.reset()
action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
action_dict = {"A":[('KeyEvent', 'ArrowUp', True)],
               "D":[('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)],
               "B":[('KeyEvent', 'ArrowDown', True)],
               "C":[('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowRight', True),('KeyEvent', 'ArrowLeft', False)]}

while True:
    print(agent.action)
#    observation_n, reward_n, done_n, info = env.step(action_n)
    # if observation_n[0] != None and (TICK % NUM_KEY_SAMPLE == 0):
    #     #print(observation_n[0]['vision'])
    #     #print(observation_n[0]['vision'].shape)
    #     k = getch.getch()
    #     tup = action_dict.get(k)
    #     if tup != None:
    #         action_n = [tup for ob in observation_n]  # your agent here
    #         print("This is your key val" + str(k) + str(tup))
    #         sys.stdin.flush()
    #     print("The tick is " + str(TICK))
    # #print(reward_n)
    # #print(done_n)
    env.render(mode='human')
    TICK += 1
