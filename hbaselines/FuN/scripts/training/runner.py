"""
#############################################
#Class of scripts to run the Neural Networks#
#############################################
"""


import gym
import tensorflow as tf
import numpy as np
import cv2

from hbaselines.FuN.scripts.training.feudal_networks.policies.feudal_policy \
    import FeudalPolicy

env = gym.make('PongDeterministic-v0')

# print last_c_g
length = 0
rewards = 0


def process_frame42(frame):
    """
    Function for processing the frames

    Parameters
    ----------
    frame : object
        Frame object to be processed
    """
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


with tf.Session() as sess, sess.as_default():
    pi = FeudalPolicy([42, 42, 1], env.action_space.n, 0)
    last_state = env.reset()
    last_c_g, last_features = pi.get_initial_features()

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint('/tmp/pong/train/'))
    while True:
        terminal_end = False

        for _ in range(1000):
            last_state = process_frame42(last_state)
            fetched = pi.act(last_state, last_c_g, *last_features)
            action, value_, g, s, last_c_g, features = \
                fetched[0], fetched[1], fetched[2],\
                fetched[3], fetched[4], fetched[5:]
            action_to_take = action.argmax()
            state, reward, terminal, _ = env.step(action_to_take)

            # collect the experience
            length += 1
            rewards += reward

            last_state = state
            last_features = features

            timestep_limit =\
                env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or \
                        not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                last_c_g, last_features = pi.get_initial_features()
                print("Episode finished. Sum of rewards: %f. Length: %d"
                      % (rewards, length))
                length = 0
                rewards = 0
                break
