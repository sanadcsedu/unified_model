import pdb
import misc
import numpy as np
from collections import defaultdict
import pandas as pd
import itertools
# import matplotlib.pyplot as plt
import sys
# import plotting
import environment5 as environment5
import random
import multiprocessing
import time
from pathlib import Path
import glob


class TD_SARSA:
    def __init__(self):
        pass

    # @jit(target ="cuda")
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
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fnc

    def sarsa(
        self, env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.5
    ):
        """
               SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

               Args:
                   env: OpenAI environment.
                   num_episodes: Number of episodes to run for.
                   discount_factor: Gamma discount factor.
                   alpha: TD learning rate.
                   epsilon: Chance the sample a random action. Float betwen 0 and 1.

               Returns:
                   A tuple (Q, stats).
                   Q is the optimal action-value function, a dictionary mapping state -> action values.
                   stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
               """
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        # Define the valid actions for each state

        Q = defaultdict(lambda: np.zeros(len(env.action_space)))
        
        # The policy we're following
        policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))
        
        for i_episode in range(num_episodes):
            state = env.reset(all = False, test = False)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            training_accuracy=[]

            # One step in the environment
            for t in itertools.count():
                # Take a step
                next_state, reward, done, pred = env.step(state, action, False)
                training_accuracy.append(pred)

                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
                
                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
        
                if done:
                    break
                    
                action = next_action
                state = next_state        
        
        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for i_episode in range(1):
            # Reset the environment and pick the first action
            state = env.reset(all=False, test=True)

            stats = []
            
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction = env.step(state, action, True)
                stats.append(prediction)
            
                # Pick the next action
                next_action_probs = policy(next_state)
                next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
                # TD Update
                td_target = reward + discount_factor * Q[next_state][next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                action = next_action
                state = next_state
                if done:
                    break

        return np.mean(stats)

if __name__ == "__main__":
    env = environment5.environment5()
    user_list = env.user_list_faa
    obj2 = misc.misc(len(user_list))

    result_queue = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[:2], 'SARSA',15, result_queue,))
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[2:4], 'SARSA',15, result_queue,))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[4:6], 'SARSA',15, result_queue,))
    p4 = multiprocessing.Process(target=obj2.hyper_param, args=(env,user_list[6:], 'SARSA',15, result_queue,))
    
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
    print("SARSA ", ", ".join(f"{x:.2f}" for x in final_result))
    