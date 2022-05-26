import numpy as np
from typing import List

class QLearningModel:

    # Training parameters
    n_training_episodes: int = 1000 
    learning_rate: float = 0.8

    # Evaluation parameters
    n_eval_episode: int = 1000

    # Environment parameters
    env_id: str = "FrozenLake-v1"
    max_steps: int = 99
    gamma: float = 0.95
    eval_seed: List = []

    # Exploration parameters
    epsilon: float = 1.0
    max_epsilon: float = 1.0
    min_epsilon: float = 0.05
    decay_rate: float = 0.01

    def __init__(self, env) -> None:
        self.env = env

    def initialize_q_table(self):
        state_space = self.env.observation_space.n
        action_space = self.env.action_space.n
        Qtable = np.zeros((state_space, action_space))
        return Qtable

    def epsilon_greedy_policy(self, Qtable, state):
        # Randomly generate a number between 0 and 1
        random_int = np.random.random_sample()
        # if random_int > greater than epsilon --> exploitation
        if random_int > self.epsilon:
            # Take the action with the highest value given a state
            # np.argmax can be useful here
            action = np.argmax(Qtable[state, :])
        # else --> exploration
        else:
            action = self.env.action_space.sample()
            # Take a random action
        
        return action


    def greedy_policy(self, Qtable, state):
        # Exploitation: take the action with the highest state, action value
        action = np.argmax(Qtable[state, :])
        
        return action


    def train(self, Qtable):
        for episode in range(self.n_training_episodes):
            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            # Reset the environment
            state = self.env.reset()
            step = 0
            done = False

            # repeat
            for step in range(self.max_steps):
            # Choose the action At using epsilon greedy policy
                action = self.epsilon_greedy_policy(Qtable, state)

                # Take action At and observe Rt+1 and St+1
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = self.env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                Qtable[state][action] = Qtable[state][action] + self.learning_rate * (reward + self.gamma * np.max(Qtable[state, :]) - Qtable[state][action])

                # If done, finish the episode
                if done:
                    break
                
                # Our state is the new state
                state = new_state
        
        return Qtable