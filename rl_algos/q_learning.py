import numpy as np
from typing import List
from omegaconf import DictConfig

class QLearningModel:

    n_training_episodes: int 
    learning_rate: float
    n_eval_episode: int
    max_steps: int
    gamma: float
    eval_seed: List = []
    epsilon: float
    max_epsilon: float
    min_epsilon: float
    decay_rate: float

    def __init__(self, env, conf) -> None:
        self.env = env
        self.conf = conf
        self.n_training_episodes = self.conf["n_training_episodes"]
        self.learning_rate = self.conf["learning_rate"]
        self.n_eval_episode = self.conf["n_eval_episode"]
        self.max_steps = self.conf["max_steps"]
        self.gamma = self.conf["gamma"]
        self.eval_seed = self.conf["eval_seed"]
        self.epsilon = self.conf["epsilon"]
        self.max_epsilon = self.conf["max_epsilon"]
        self.min_epsilon = self.conf["min_epsilon"]
        self.decay_rate = self.conf["decay_rate"]


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