import gym
import numpy as np

env = gym.make("FrozenLake-v0")
Q = np.zeros([env.observation_space.n,env.action_space.n])

learn_rate = .85 # learning rate
disc_fact = .99 # discount factor
max_steps = 99
max_episodes = 2000
rewards = []

for i in range(max_episodes): # each episode
    state = env.reset()
    total_reward = 0
    done = False
    for j in range(max_steps): # each state in episode
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        state_, reward, done, _ = env.step(action)
        Q[state, action] = Q[state, action] + learn_rate * (reward + disc_fact * np.max(Q[state_, :]) - Q[state, action])
        total_reward += reward
        state = state_
        if done == True:
            break
    rewards.append(total_reward)

print "Score over time: " +  str(sum(rewards) / max_episodes)
print "Final Q-Table Values"
print Q