import tensorflow as tf
import numpy as np


bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)
def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

net_weights = tf.Variable(tf.ones([num_bandits])) #
net_max_action = tf.argmax(net_weights, 0)

net_reward = tf.placeholder(shape=[1], dtype=tf.float32)
net_action = tf.placeholder(shape=[1], dtype=tf.int32)

net_advantage = tf.slice(net_weights, net_action, [1]) #

net_loss = -(tf.log(net_advantage) * net_reward)
net_trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
net_model = net_trainer.minimize(net_loss)

max_episodes = 1000
rewards = np.zeros(num_bandits)
epsilon = 0.1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_episodes):

        # exploration vs. exploitation
        if np.random.rand(1) < epsilon:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(net_max_action)

        reward = pullBandit(bandits[action])
        _, _, weights = sess.run([net_model, net_advantage, net_weights], feed_dict={net_reward:[reward], net_action:[action]})

        rewards[action] += reward
        if i % 50 == 0:
            print "Running reward for the " + str(num_bandits) + " bandits: " + str(rewards)

print "The agent thinks bandit " + str(np.argmax(weights) + 1) + " is the most promising...."
if np.argmax(weights) == np.argmax(-np.array(bandits)):
    print "...and it was right!"
else:
    print "...and it was wrong!"

