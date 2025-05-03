import random
import gymnasium as gym
import numpy as np 


#training
env = gym.make("Taxi-v3")

alpha = 0.9
gamma = 0.95
epsilon = 1
epsilon_decay = 0.999
min_epsilon = 0.01
max_steps = 100
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action (state):
    if random.uniform(0,1)<epsilon : 
        return env.action_space.sample()
    else: 
        return np.argmax(q_table[state,:]) 

for episode in range (1000):
    state, _ = env.reset()
    done = False

    for step in range (max_steps): 
        action = choose_action(state) 
        next_state, reward, done, truncated, info  = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        state = next_state 
        if done or truncated:  
            break 

    #epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-0.001 * episode)


np.save("q_table.npy", q_table)
