import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

hidden_layer_size = 10
learning_rate = 1e-2
discount_factor = 0.99
observation_size = 4

net_observation = tf.placeholder(tf.float32, [None, observation_size])
net_weights_1 = tf.get_variable("weights_1", shape=[observation_size, hidden_layer_size], initializer=tf.contrib.layers.xavier_initializer())
net_hidden_1 = tf.nn.relu(tf.matmul(net_observation, net_weights_1))
net_weights_2 = tf.get_variable("weights_2", shape=[hidden_layer_size, 1], initializer=tf.contrib.layers.xavier_initializer())
net_pred_actions = tf.nn.sigmoid(tf.matmul(net_hidden_1, net_weights_2))

net_taken_actions = tf.placeholder(tf.float32, [None, 1], name="taken_actions")
advantages = tf.placeholder(tf.float32,name="rewards")

net_weights = tf.trainable_variables()
net_grad_1 = tf.placeholder(tf.float32, name="batch_grad1")
net_grad_2 = tf.placeholder(tf.float32, name="batch_grad2")

net_log_like = tf.log(net_taken_actions * (net_taken_actions - net_pred_actions) + (1 - net_taken_actions) * (net_taken_actions + net_pred_actions))
net_loss = -tf.reduce_mean(net_log_like * advantages)
net_new_grad = tf.gradients(net_loss, net_weights)

net_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
net_update_grad = net_trainer.apply_gradients(zip([net_grad_1, net_grad_2], net_weights))

def discount_rewards(rewards):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        running_add = running_add * discount_factor + rewards[t]
        discounted_r[t] = running_add
    return discounted_r



def reset_gradients(gradients):
    for ix, gradient in enumerate(gradients):
        gradients[ix] = gradient * 0
    return gradients

running_reward = None
total_reward = 0
batch_size = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gradBuffer = reset_gradients(sess.run(net_weights))

    for i in range(10000): # running episodes
        observation = env.reset()
        done = False
        observations, hs, dlogps, rewards, actions = [], [], [], [], []

        while not done: # sample episode and stack it
            observation = np.reshape(observation, [1, observation_size])
            action = 1 if np.random.uniform() < sess.run(net_pred_actions, feed_dict={net_observation: observation}) else 0

            observations.append(observation)
            actions.append(1 if action == 0 else 0)

            observation, reward, done, _ = env.step(action)
            total_reward += reward
            rewards.append(reward)

        discounted_epr = discount_rewards(np.vstack(rewards))
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        tGrad = sess.run(net_new_grad, feed_dict={
            net_observation: np.vstack(observations),
            net_taken_actions: np.vstack(actions),
            advantages: discounted_epr})

        for ix, grad in enumerate(tGrad):
            gradBuffer[ix] += grad

        if i % batch_size == 0:
            sess.run(net_update_grad, feed_dict={net_grad_1: gradBuffer[0], net_grad_2: gradBuffer[1]})
            gradBuffer = reset_gradients(gradBuffer)

            running_reward = total_reward if running_reward is None else running_reward * 0.99 + total_reward * 0.01
            print 'Average reward for episode %f.  Total average reward %f.' % (total_reward / batch_size, running_reward / batch_size)

            if total_reward / batch_size > 200:
                print "Task solved in", i, 'episodes!'
                break

            total_reward = 0

print i, 'Episodes completed.'