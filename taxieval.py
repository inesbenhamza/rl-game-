
import numpy as np
import gymnasium as gym


env = gym.make("Taxi-v3", render_mode="human")

q_table = np.load ("q_table.npy")

for episode in range (5): 
    state, _ = env.reset()
    done = False

    print('Episode:', episode)

    for step in range (100): #max step is 100 
        env.render()
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print ('finished episode', episode, 'with reward', reward)
            break 


env.close()



