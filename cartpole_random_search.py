import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def run_episode(env, parameters,epochs, penalties,total_epochs, total_penalties):

    observation = env.reset()
    totalreward = 0
    for epoch in range(epochs):
        #env.render()
        total_epochs+=1
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        print('Reward:', reward)
        if reward != 1:
            penalties += 1
            print('penalties:', penalties)
        epochs += 1
        total_penalties += penalties
        total_epochs += epochs
        if done:
            break
    return totalreward,total_penalties,total_epochs

def train(submit):
    env = gym.make('CartPole-v1')
    if submit:
        env.monitor.start('cartpole-experiments/', force=True)
    counter = 0
    bestparams = None
    bestreward = 0
    penalties, episode_reward = 0, 0
    epochs=200
    total_epochs, total_penalties = 0, 0
    episodes=0
    n_episode=1000
    # counter = 0
    scores = deque(maxlen=200)
    episodes_result = deque(maxlen=200)
    penalties_result = deque(maxlen=100)
    for episode in range(n_episode):
        episodes+=1
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward,total_penalties,total_epochs = run_episode(env, parameters,epochs, penalties,total_epochs, total_penalties)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                break
        scores.append(reward)
        episodes_result.append(epochs)
        penalties_result.append(total_penalties)
        mean_reward = np.mean(scores)
    if submit:
        for _ in range(100):
            run_episode(env, parameters, epochs, penalties, total_epochs, total_penalties)
    print(f"Results after {episode} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episode}")
    print(f"Average Rewards per episode: {np.mean(scores)}")
    print(f"Average penalties per episode: {total_penalties / episode}")
    print("Training finished.\n")
    plt.hist(episodes_result, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Episodes required to reach Goal')
    plt.ylabel('Frequency')
    plt.title('Episode Histogram of Cart problem solving by Random Search')
    plt.show()
    plt.hist(scores, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Rewards Achieved Per Episode')
    plt.ylabel('Frequency')
    plt.title('Rewards Histogram of Taxi problem solving by Random Search')
    plt.show()
    plt.hist(penalties_result, 50, normed=1, facecolor='g', alpha=0.75)
    plt.xlabel('Penalties Per Episode')
    plt.ylabel('Frequency')
    plt.title('Penalties Histogram of Taxi problem solving by Random Search')
    plt.show()
    env.monitor.close()

    return counter,episodes_result,scores,penalties_result

# train an agent to submit to openai gym
train(submit=False)

# create graphs
results = []
#for _ in range(1000):
#  results.append(train(submit=False))

#plt.hist(results,50,normed=1, facecolor='g', alpha=0.75)
#plt.xlabel('Episodes required to reach 200')
#plt.ylabel('Frequency')
#plt.title('Histogram of Random Search')
#plt.show()

#print ( np.sum(results) / 1000.0)