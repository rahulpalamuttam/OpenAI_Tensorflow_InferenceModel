import sys

import gym
import universe  # register the universe environments
from gym import wrappers
import getch # to take continuous input with getch

import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NUM_KEY_SAMPLE = 30
TICK = 0
env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)

observation_n = env.reset()
action_n = [[('KeyEvent', 'ArrowUp', False)] for ob in observation_n]  # your agent here
action_dict = {"w":[('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)],
               "a":[('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)],
               "s":[('KeyEvent', 'ArrowDown', True)],
               "d":[('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowRight', True),('KeyEvent', 'ArrowLeft', False)],
               "x":[('KeyEvent', 'x', True)]}
encoding_dict = {"w":[1,0,0,0,0], "a":[0,1,0,0,0], "s":[0,0,1,0,0],"d":[0,0,0,1,0],"x":[0,0,0,0,1]}
saved  = []
output = open('test_data.pkl', 'wb')
env.render()
done = [False]
d = 0
while d < 2 or done[0] == False:
    print(observation_n[0])
    print("ob len " + str(len(observation_n)))
    print(type(observation_n[0]))
    observation_n, reward, done, info = env.step(action_n)
    print(done)
    if observation_n != None and observation_n[0] != None:
        k = getch.getch()
        act = action_dict.get(k)
        encoding = encoding_dict.get(k)
        saved.append((observation_n[0]['vision'].copy(), encoding))
        print("Print the key : " +  str(k) + " " + str(act) + " " + str(len(saved)))
        print(d)
        action_n = [act]
    if done[0] == True:
        d += 1
    env.render()

print("It's DONEDONEDONEDONEDONEDONEDONEDONEDONEDONEDONEDONEDONEDONEDONEDONE")
pickle.dump(saved, output)
output.close()

#im = plt.imshow(saved[421])
#plt.hist(images[num].ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
#plt.show()
