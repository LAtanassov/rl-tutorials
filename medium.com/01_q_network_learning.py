import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

net_state = tf.placeholder(shape=[1, 16], dtype=tf.float32) # state as one hot vector
net_weigth = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
net_Q = tf.matmul(net_state, net_weigth)
net_action = tf.argmax(net_Q, 1)

net_Q_ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
net_loss = tf.reduce_sum(tf.square(net_Q_ - net_Q))
net_trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
net_model = net_trainer.minimize(net_loss)

disc_fact = .99
epsilon = 0.1
max_episodes = 2000
max_states = 99
n_states = []
rewards = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        for j in range(max_states):

            # get action and current Q
            action, Q = sess.run([net_action, net_Q], feed_dict={net_state: np.identity(16)[state:state + 1]})
            if np.random.rand(1) < epsilon: # exploration
                action[0] = env.action_space.sample()

            # simulate next step and get next Q_
            state_, reward, done, _ = env.step(action[0])
            Q_ = sess.run(net_Q, feed_dict={net_state: np.identity(16)[state_:state_ + 1]})

            Q_[0, action[0]] = reward + disc_fact * np.max(Q_)
            _, W1 = sess.run([net_model, net_weigth], feed_dict={net_state: np.identity(16)[state:state + 1], net_Q_: Q_})
            total_reward += reward
            state = state_

            if done == True:
                break
        rewards.append(total_reward)

print "Percent of succesful episodes: " + str(sum(rewards) / max_episodes) + "%"
plt.plot(rewards)
plt.show()