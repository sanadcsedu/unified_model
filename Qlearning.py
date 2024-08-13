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
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[state])
            A[best_action] += (1.0 - epsilon)
            return A

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
            state = env.reset(all=False, test=False)
            training_accuracy = []
            for t in itertools.count():
                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, prediction, _ = env.step(state, action, False)

                training_accuracy.append(prediction)

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                state = next_state
                if done:
                    break

        return Q, np.mean(training_accuracy)

    def test(self, env, Q, discount_factor, alpha, epsilon, num_episodes=1):
        epsilon = epsilon

        for _ in range(1):
            state = env.reset(all=False, test=True)
            stats = []
            # model_actions = []
            insight = defaultdict(list)
            policy = self.epsilon_greedy_policy(Q, epsilon, len(env.action_space))

            for t in itertools.count():
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                # model_actions.append(action)
                next_state, reward, done, prediction, ground_action = env.step(state, action, True)

                stats.append(prediction)
                insight[ground_action].append(prediction)
                # Turning off the Q-Learning update when testing, the prediction is based on the Learned model from first x% interactions
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                state = next_state
                if done:
                    break
        
        #Calculating the number of occurance and prediction rate for each action
        granular_prediction = defaultdict()
        for keys, values in insight.items():
            granular_prediction[keys] = (len(values), np.mean(values))

        return np.mean(stats), granular_prediction


if __name__ == "__main__":
    final_output = []
    env = environment5.environment5()
    # user_list = env.user_list_faa
    user_list = env.user_list_brightkite
    obj2 = misc.misc(len(user_list))

    result_queue = multiprocessing.Queue()
    info = multiprocessing.Queue()
    info_split = multiprocessing.Queue()
    info_split_cnt = multiprocessing.Queue() 

    p1 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list[:2], 'Qlearn', 10, result_queue, info, info_split, info_split_cnt))
    p2 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list[2:4], 'Qlearn', 10, result_queue, info, info_split, info_split_cnt))
    p3 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list[4:6], 'Qlearn', 10, result_queue, info, info_split, info_split_cnt))
    p4 = multiprocessing.Process(target=obj2.hyper_param, args=(env, user_list[6:], 'Qlearn', 10, result_queue, info, info_split, info_split_cnt))

    split_final = np.zeros((4, 9), dtype = float)
    split_final_cnt = np.zeros((4, 9), dtype = float)

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    final_result = np.zeros(9, dtype=float)
    p1.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())
    
    p2.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    p3.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    p4.join()
    final_output.extend(info.get())
    final_result = np.add(final_result, result_queue.get())
    split_final = np.add(split_final, info_split.get())
    split_final_cnt = np.add(split_final_cnt, info_split_cnt.get())

    final_result /= 4
    split_final /= 4
    split_final_cnt /= 4

    print("Q-Learning ", ", ".join(f"{x:.2f}" for x in final_result))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final[ii]))

    for ii in range(4):
        print("Action ", ii, ", ".join(f"{x:.2f}" for x in split_final_cnt[ii]))

    # output = open("output.txt", 'w')
    # for idx, l in enumerate(final_output):
    #     output.write(l + '\n')

# FAA DATASET

# u6 0.74, 0.71, 0.72, 0.72, 0.74, 0.63, 0.57, 0.57, 0.66
# u4 0.64, 0.44, 0.52, 0.53, 0.56, 0.67, 0.72, 0.66, 0.57
# u7 0.69, 0.78, 0.76, 0.75, 0.48, 0.25, 0.60, 0.74, 0.68
# u3 0.65, 0.59, 0.65, 0.35, 0.60, 0.78, 0.67, 0.91, 0.93
# u1 0.61, 0.57, 0.62, 0.60, 0.65, 0.58, 0.64, 0.71, 0.82
# u2 0.58, 0.46, 0.57, 0.51, 0.52, 0.34, 0.67, 0.15, 0.31
# u5 0.74, 0.76, 0.81, 0.84, 0.74, 0.70, 0.78, 0.85, 0.80
# u8 0.74, 0.60, 0.75, 0.59, 0.66, 0.70, 0.57, 0.49, 0.37

# Q-Learning  0.67, 0.62, 0.67, 0.61, 0.62, 0.65, 0.65, 0.63, 0.64

# Accuracy of actions over different thresholds
# Action  0 0.58, 0.67, 0.54, 0.59, 0.60, 0.61, 0.49, 0.57, 0.51
# Action  1 0.02, 0.01, 0.03, 0.04, 0.07, 0.06, 0.04, 0.02, 0.10
# Action  2 0.73, 0.71, 0.68, 0.58, 0.63, 0.60, 0.51, 0.55, 0.34
# Action  3 0.48, 0.34, 0.52, 0.45, 0.46, 0.43, 0.51, 0.31, 0.25

# Average Count of Actions over different thresholds
# Action  0 57.62, 56.38, 51.75, 45.12, 40.00, 34.12, 25.75, 17.50, 8.25
# Action  1 12.12, 11.50, 10.38, 9.38, 8.00, 7.12, 5.00, 3.12, 1.50
# Action  2 128.12, 106.25, 92.00, 78.12, 62.00, 43.88, 33.00, 25.50, 12.25
# Action  3 120.00, 108.25, 92.75, 78.88, 65.62, 55.12, 41.12, 23.25, 11.88

# Brightkite Dataset

# u14 0.38, 0.67, 0.65, 0.64, 0.67, 0.63, 0.67, 0.72, 0.94
# u11 0.53, 0.75, 0.78, 0.81, 0.75, 0.86, 0.82, 0.90, 0.83
# u9 0.68, 0.76, 0.26, 0.89, 0.84, 0.93, 0.88, 0.86, 0.79
# u15 0.60, 0.59, 0.54, 0.52, 0.58, 0.58, 0.52, 0.71, 0.75
# u13 0.81, 0.81, 0.80, 0.85, 0.86, 0.83, 0.89, 0.85, 0.86
# u10 0.79, 0.71, 0.81, 0.68, 0.82, 0.74, 0.79, 0.77, 0.73
# u12 0.86, 0.76, 0.76, 0.88, 0.89, 0.63, 0.86, 0.80, 0.87
# u16 0.58, 0.53, 0.54, 0.64, 0.73, 0.69, 0.71, 0.64, 0.80

# Q-Learning  0.65, 0.70, 0.64, 0.74, 0.77, 0.74, 0.77, 0.78, 0.82

# Action  0 0.78, 0.91, 0.91, 0.84, 0.90, 0.85, 0.88, 0.84, 0.70
# Action  1 0.04, 0.02, 0.01, 0.02, 0.02, 0.01, 0.00, 0.01, 0.00
# Action  2 0.55, 0.61, 0.57, 0.70, 0.76, 0.71, 0.69, 0.66, 0.54
# Action  3 0.41, 0.27, 0.31, 0.49, 0.38, 0.47, 0.45, 0.40, 0.17
# Action  0 211.38, 184.00, 162.38, 136.75, 105.12, 80.25, 61.00, 45.12, 27.75
# Action  1 9.12, 7.25, 6.00, 4.62, 3.88, 2.50, 2.12, 1.88, 1.12
# Action  2 147.75, 139.12, 122.75, 110.62, 101.50, 88.50, 66.25, 37.38, 16.00
# Action  3 115.62, 99.38, 84.75, 69.88, 57.38, 42.75, 30.75, 21.75, 7.25