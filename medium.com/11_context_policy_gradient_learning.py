import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class ContextualBandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pullArm(self, action):
        if np.random.randn(1) > self.bandits[self.state, action]:
            return 1
        else:
            return -1

class Agent():
    def __init__(self, learn_rate, num_states, num_actions):

        self.net_state = tf.placeholder(shape=[1], dtype=tf.int32)
        self.net_reward = tf.placeholder(shape=[1], dtype=tf.float32)
        self.net_action = tf.placeholder(shape=[1], dtype=tf.int32)

        self.net_output = tf.reshape(slim.fully_connected(
            inputs=slim.one_hot_encoding(self.net_state, num_states),
            num_outputs=num_actions,
            biases_initializer=None,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.ones_initializer()),[-1])

        self.net_max_action = tf.argmax(self.net_output, 0)
        self.net_advantage = tf.slice(self.net_output, self.net_action, [1])
        self.net_loss = -(tf.log(self.net_advantage) * self.net_reward)
        self.net_model = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(self.net_loss)
        self.net_weights = tf.trainable_variables()[0]


bandit = ContextualBandit()
agent = Agent(learn_rate=0.001, num_states=bandit.num_bandits, num_actions=bandit.num_actions)

max_episodes = 10000
rewards = np.zeros([bandit.num_bandits, bandit.num_actions])
epsilon = 0.1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_episodes):
        state = bandit.getBandit()

        # exploration vs. exploitation
        if np.random.rand(1) < epsilon:
            action = np.random.randint(bandit.num_actions)
        else:
            action = sess.run([agent.net_max_action], feed_dict={agent.net_state: [state]})[0]

        reward = bandit.pullArm(action)
        _, weights = sess.run([agent.net_model, agent.net_weights], feed_dict={agent.net_reward: [reward], agent.net_action: [action], agent.net_state: [state]})

        rewards[state, action] += reward
        if i % 500 == 0:
            print "Mean reward for each of the " + str(bandit.num_bandits) + " bandits: " + str(np.mean(rewards, axis=1))

for a in range(bandit.num_bandits):
    print "The agent thinks action " + str(np.argmax(weights[a]) + 1) + " for bandit " + str(a + 1) + " is the most promising...."
    if np.argmax(weights[a]) == np.argmin(bandit.bandits[a]):
        print "...and it was right!"
    else:
        print "...and it was wrong!"