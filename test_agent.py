import sys

import gym
import universe  # register the universe environments
from gym import wrappers
import getch # to take continuous input with getch

import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np

import cv2

def _process_frame_flash(frame):
    frame = frame[90:600, 15:815, :]
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame

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
decoding_dict = {"[1, 0, 0, 0, 0]":"w", "[0, 1, 0, 0, 0]":"a", "[0, 0, 1, 0, 0]":"s","[0, 0, 0, 1, 0]":"d","[0, 0, 0, 0, 1]":"x"}
saved  = []
output = open('recorded.pkl', 'wb')
env.render()
done = [False]
d = 0

checkpoint_path = 'temp/conv2d8x4_average_pooling_simple_affine_lr1e-4-1496924405/checkpoint-228'
sess = tf.Session()
saver = tf.train.import_meta_graph('temp/conv2d8x4_average_pooling_simple_affine_lr1e-4-1496924405/checkpoint-228.meta')
saver.restore(sess, tf.train.latest_checkpoint('temp/conv2d8x4_average_pooling_simple_affine_lr1e-4-1496924405/'))
graph = tf.get_default_graph()

print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y_:0")
softmax = graph.get_tensor_by_name("softmax:0")
while d < 2 or done[0] == False:
    print(observation_n[0])
    print("ob len " + str(len(observation_n)))
    print(type(observation_n[0]))
    observation_n, reward, done, info = env.step(action_n)
    print(done)
    if observation_n != None and observation_n[0] != None:
        frame = _process_frame_flash(observation_n[0]['vision'])
        feed_dict = {x:[frame], y:[np.asarray([1,0,0,0,0])]}
        k = sess.run(softmax, feed_dict)
        print(k)
        k[np.where(k==np.max(k))] = 1
        k[np.where(k!=np.max(k))] = 0
        k = str([int(t) for t in k[0].tolist()])
        print(k)
        decoding = decoding_dict.get(k)
        print("Decoding : " + decoding)
        act = action_dict.get(decoding)
        #saved.append((observation_n[0]['vision'].copy(), encoding))
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
