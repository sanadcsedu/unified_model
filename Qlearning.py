import misc
import numpy as np
from collections import defaultdict
import itertools
import environment5 as environment5
import multiprocessing
from multiprocessing import Pool
import time
import random
from pathlib import Path
import glob
from tqdm import tqdm 
import os 

class Qlearning:
    def __init__(self):
        pass


    def epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action. Float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        """

        def policy_fnc(state):
            coin = random.random()
            if coin < epsilon:
                best_action = random.randint(0, 3)
                print("random")
            else:
                best_action = np.argmax(Q[state])
                print("best")
            return best_action

        return policy_fnc



    def q_learning(self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: setting the environment as local fnc by importing env earlier
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            epsilon: Chance to sample a random action. Float between 0 and 1.

        Returns:
            A tuple (Q, episode_lengths).
            Q is the optimal action-value function, a dictionary mapping state -> action values.
            stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """
        Q = defaultdict(lambda: np.zeros(len(env.action_space)))

        for i_episode in range(num_episodes):
            # The policy we're following
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            # Reset the environment and pick the first state
            state = env.reset(all = False, test = False)
            # action_probs = policy(state)
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action = policy(state)

            training_accuracy=[]
            for t in itertools.count():
                # Take a step
                print(Q[state])
                next_state, reward, done, info = env.step(state, action, False)

                training_accuracy.append(info)

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                print("Debug",  best_next_action)
                td_target = reward*info + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                # Q[state][action] += alpha * (td_delta + info)
                Q[state][action] += alpha * td_delta

                state = next_state
                action = best_next_action
                if done:
                    break


        return Q, np.mean(training_accuracy)


    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):

            state = env.reset(all=False, test=True)
            stats = []
            model_actions = []
            
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
            # action_probs = policy(state)
            # action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action = policy(state)

            for t in itertools.count():
            
                next_state, reward, done, prediction  = env.step(state, action, True)
            
                stats.append(prediction)
            
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                print("Debug",  best_next_action)
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                # Q[state][action] += alpha * (td_delta + prediction)
                Q[state][action] += alpha * td_delta

                state = next_state
                action = best_next_action
                if done:
                    break

        return np.mean(stats)

if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_faa
    # user_list = env.user_list_brightkite
    obj2 = misc.misc(len(user_list))

    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[:2], 'Qlearn',5, result_queue,))
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[2:4], 'Qlearn',5, result_queue,))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[4:6], 'Qlearn',5, result_queue,))
    p4 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[6:], 'Qlearn',5, result_queue,))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype = float)
    p1.join()
    # temp = result_queue.get()
    final_result = np.add(final_result, result_queue.get())
    p2.join()
    # print(result_queue.get())
    final_result = np.add(final_result, result_queue.get())
    p3.join()
    # print(result_queue.get())
    final_result = np.add(final_result, result_queue.get())
    p4.join()
    # print(result_queue.get())
    final_result = np.add(final_result, result_queue.get())
    final_result /= 4
    print("Q-Learning ", ", ".join(f"{x:.2f}" for x in final_result))

    # faa: 0.58, 0.53, 0.57, 0.61, 0.63, 0.63, 0.68, 0.66, 0.68